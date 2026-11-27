"""Differentiable framework P32/PM on mathematical KDA producer caches.

Uses the frozen upstream recurrent reference and its unmodified assert_close.
This validates the precision policy, not the production producer or NPU backward.
"""
import argparse
import json
from pathlib import Path
import torch
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close
from numerical_gate import metrics


def chunked(q, k, v, raw, beta, alog, bias, initial, mode, weak=False):
    q, k, v, raw, beta = [x.float() for x in (q, k, v, raw, beta)]
    delta = -.001*torch.nn.functional.softplus(raw+bias) if weak else -5*torch.sigmoid(alog.exp()*(raw+bias))
    state = initial.float()
    outputs, caches = [], []
    quant = lambda x: x.bfloat16().float() if mode == 'PM' else x
    for start in range(0, q.shape[0], 64):
        sl = slice(start, start+64)
        qc, kc, vc, bc = q[sl], k[sl], v[sl], beta[sl]
        g = delta[sl].cumsum(0); m = len(g)
        causal = torch.ones(m, m, dtype=torch.bool).tril()
        diff = torch.where(causal[..., None], g[:, None]-g[None, :], 0.)
        decay = diff.exp()*causal[..., None]
        akk = (kc[:, None]*kc[None, :]*decay).sum(-1)
        a = torch.eye(m)+(bc[:, None]*akk).tril(-1)
        w = torch.linalg.solve_triangular(a, bc[:, None]*kc*g.exp(), upper=False)
        u = torch.linalg.solve_triangular(a, bc[:, None]*vc, upper=False)
        kg = kc*(g[-1]-g).exp()
        p = quant(w) @ quant(state)
        value = u-p
        score = (qc[:, None]*kc[None, :]*decay).sum(-1)*.37
        output = (qc*g.exp()*.37)@state + score@value
        new = g[-1].exp()[:, None]*state + quant(kg).T@quant(value)
        caches.append(dict(w=w.detach(), u=u.detach(), kg=kg.detach(), g=g.detach(),
                           initial=state.detach(), p=p.detach(), value=value.detach(), final=new.detach()))
        state = new; outputs.append(output)
    return torch.cat(outputs), state, caches, delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4); records=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for tokens, weak in ((65,False),(1024,False),(1024,True)):
        torch.manual_seed(1127)
        shape=(tokens,256)
        data=[torch.nn.functional.normalize(torch.randn(shape),dim=-1).bfloat16(),
              torch.nn.functional.normalize(torch.randn(shape),dim=-1).bfloat16(),
              torch.randn(shape).bfloat16(), torch.randn(shape),
              torch.rand(tokens).bfloat16(), torch.tensor(.1), torch.randn(256)*.01,
              torch.randn(256,256)*.1]
        do=torch.randn(shape); dh=torch.randn(256,256)*.1
        for mode in ('P32','PM'):
            xs=[x.clone().requires_grad_() for x in data]
            o,h,caches,delta=chunked(*xs,mode,weak)
            # Weak mode intentionally does not use A_log; keep its grad absent.
            grads=torch.autograd.grad((o*do).sum()+(h*dh).sum(),xs,allow_unused=True)
            ys=[x.clone().requires_grad_() for x in data]
            q,k,v,raw,beta,al,bias,h0=ys
            gate=-.001*torch.nn.functional.softplus(raw+bias) if weak else -5*torch.sigmoid(al.exp()*(raw+bias))
            ro,rh=naive_recurrent_kda(q.float()[None,:,None],k.float()[None,:,None],v.float()[None,:,None],
                 gate[None,:,None],beta.float()[None,:,None],scale=.37,
                 initial_state=h0[None,None],output_final_state=True)
            rg=torch.autograd.grad((ro[0,:,0]*do).sum()+(rh[0,0]*dh).sum(),ys,allow_unused=True)
            values=[('o',o,ro[0,:,0],.005,False),('ht',h,rh[0,0],.005,False)]
            values += [(n,x,y,t,n=='dA') for n,x,y,t in zip(
                ('dq','dk','dv','dg','db','dA','dbias','dh0'),grads,rg,
                (.008,.008,.008,.02,.02,.003,.008,.008)) if x is not None]
            for name,actual,expected,tol,warn in values:
                result=metrics(actual,expected)
                try:
                    assert_close(name,expected,actual,tol,warning=warn)
                    assert result['nonfinite']==0
                    passed=True
                except AssertionError:
                    passed=False
                records.append(dict(tokens=tokens,weak=weak,mode=mode,name=name,
                                    passed=passed,ratio_threshold=tol,warning=warn,**result))
            if mode=='P32':torch.save(caches,args.output/f'cache-t{tokens}-weak{int(weak)}.pt')
            print(tokens,weak,mode,'failed',[(r['name'],r['rms_ratio']) for r in records
                  if r['tokens']==tokens and r['weak']==weak and r['mode']==mode and not r['passed']],flush=True)
            (args.output/'producer-gate.json').write_text(json.dumps(dict(scope='framework_only',records=records),indent=2))
    assert all(r['passed'] for r in records if r['mode']=='P32')


if __name__=='__main__':main()
