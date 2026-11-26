"""CPU design gate. No ATK; original upstream recurrent is loaded without FLA imports."""
import argparse
import ast
import itertools
import json
import math
from pathlib import Path

import torch


def forward(q, k, v, delta, beta, h0, chunk, exp2):
    h = h0
    records, outputs = [], []
    cbase = 1 / math.log(2) if exp2 else 1
    for start in range(0, len(q), chunk):
        qs, ks, vs, ds, bs = [x[start:start + chunk] for x in (q, k, v, delta, beta)]
        m = len(qs)
        g = (ds * cbase).cumsum(0)
        e = (g * (math.log(2) if exp2 else 1)).exp()
        diff = ((g[:, None] - g[None, :]) * (math.log(2) if exp2 else 1)).exp()
        r = (ks[:, None] * ks[None, :] * diff).sum(-1)
        l = torch.eye(m, dtype=q.dtype) + (bs[:, None] * r).tril(-1)
        a = torch.linalg.solve_triangular(l, torch.eye(m, dtype=q.dtype), upper=False)
        score = ((qs[:, None] * ks[None, :] * diff).sum(-1)).tril()
        rk = ks * e * bs[:, None]
        rv = vs * bs[:, None]
        w, u = a @ rk, a @ rv
        qg = qs * e
        kg = ks * (e[-1] / e)
        vn = u - w @ h
        outputs.append(.37 * (qg @ h + score @ vn))
        records.append((start, qs, ks, vs, bs, g, e, diff, a, rk, rv, w, vn, h, qg, kg, score))
        h = e[-1, :, None] * h + kg.T @ vn
    return torch.cat(outputs), h, records


def backward(records, do, dht, exp2):
    dq, dk, dv, dd, db = [torch.zeros_like(x) for x in (do.new_empty(len(do), records[0][1].shape[-1]),) * 2
                          + (do, do.new_empty(len(do), records[0][1].shape[-1]), do[:, 0])]
    dn = dht.clone()
    base = math.log(2) if exp2 else 1
    cbase = 1 / base
    for rec in reversed(records):
        start, q, k, v, beta, g, e, diff, a, rk, rv, w, vn, h, qg, kg, score = rec
        m = len(q)
        upstream = do[start:start+m]
        ds = (.37 * (upstream @ vn.T)).tril()
        dvn = .37 * score.T @ upstream + kg @ dn
        dprev = .37 * (qg.T @ upstream) + e[-1, :, None] * dn - w.T @ dvn
        dw = -dvn @ h.T
        dkg = vn @ dn.T
        da = dw @ rk.T + dvn @ rv.T
        dl = (-a.T @ da @ a.T).tril(-1)
        assert torch.count_nonzero(dl.triu()) == 0
        drk, drv = a.T @ dw, a.T @ dvn
        qgrad = .37 * (upstream @ h.T) * e
        kgrad = drk * beta[:, None] * e + dkg * (e[-1] / e)
        ggrad = base * (.37 * (upstream @ h.T) * qg + drk * rk)
        tail = base * dkg * kg
        ggrad = ggrad - tail
        ggrad[-1] += tail.sum(0) + base * e[-1] * (h * dn).sum(1)
        bgrad = (drk * k * e).sum(-1) + (drv * v).sum(-1)
        for i in range(m):
            for j in range(i+1):
                ef = diff[i, j]
                kval = dl[i, j] * beta[i] if j < i else 0
                qgrad[i] += ds[i, j] * k[j] * ef
                kgrad[j] += (ds[i, j] * q[i] + kval * k[i]) * ef
                if j < i:
                    kgrad[i] += kval * k[j] * ef
                    contrib = base * (ds[i, j] * q[i] + kval * k[i]) * k[j] * ef
                    ggrad[i] += contrib
                    ggrad[j] -= contrib
                    bgrad[i] += dl[i, j] * (k[i] * k[j] * ef).sum()
        dq[start:start+m], dk[start:start+m] = qgrad, kgrad
        dv[start:start+m] = beta[:, None] * drv
        dd[start:start+m] = ggrad.flip(0).cumsum(0).flip(0) * cbase
        db[start:start+m] = bgrad
        dn = dprev
    return dq, dk, dv, dd, db, dn


def load_upstream(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'naive_recurrent_kda')
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {'torch': torch}
    exec(compile(module, str(path), 'exec'), scope)
    return scope[fn.name]


def run(upstream):
    torch.set_num_threads(1)
    torch.manual_seed(33)
    worst = 0.
    count = 0
    for chunk, exp2, mode, norm, beta_mode in itertools.product((32,64), (False,True), ('pre','safe','softplus'), (False,True), (0,1,2)):
        t, K, V = 65, 4, 3
        q0 = (torch.randn(t,K,dtype=torch.double)*.2).requires_grad_()
        k0 = (torch.randn(t,K,dtype=torch.double)*.2).requires_grad_()
        v = torch.randn(t,V,dtype=torch.double,requires_grad=True)
        raw = (torch.randn(t,K,dtype=torch.double)*.1-2).requires_grad_()
        log = torch.full((1,),-1.,dtype=torch.double,requires_grad=True)
        bias = torch.zeros(K,dtype=torch.double,requires_grad=True)
        br = (torch.randn(t,dtype=torch.double)*.2).requires_grad_()
        h0 = (torch.randn(K,V,dtype=torch.double)*.1).requires_grad_()
        q = q0 * (q0.square().sum(-1,keepdim=True)+1e-6).rsqrt() if norm else q0
        k = k0 * (k0.square().sum(-1,keepdim=True)+1e-6).rsqrt() if norm else k0
        x, a = raw+bias, log.exp()
        delta = raw if mode=='pre' else -2*torch.sigmoid(a*x) if mode=='safe' else -a*torch.nn.functional.softplus(x)
        beta = br if beta_mode==0 else torch.sigmoid(br)*beta_mode
        do, dht = torch.randn(t,V,dtype=torch.double), torch.randn(K,V,dtype=torch.double)
        o, ht, records = forward(q,k,v,delta,beta,h0,chunk,exp2)
        nodes = (q,k,v,delta,beta,h0)
        expected = torch.autograd.grad((o*do).sum()+(ht*dht).sum(),nodes,retain_graph=True)
        with torch.no_grad(): actual = backward(records,do,dht,exp2)
        for got, ref in zip(actual,expected):
            torch.testing.assert_close(got,ref,atol=1e-10,rtol=1e-9)
            worst=max(worst,(got-ref).abs().max().item())
        # Explicitly validate the public-input normalization/beta branches.
        qgrad,kgrad,bgrad=actual[0],actual[1],actual[4]
        if norm:
            qgrad=(qgrad-q*(q*qgrad).sum(-1,keepdim=True))*(q0.square().sum(-1,keepdim=True)+1e-6).rsqrt()
            kgrad=(kgrad-k*(k*kgrad).sum(-1,keepdim=True))*(k0.square().sum(-1,keepdim=True)+1e-6).rsqrt()
        if beta_mode:
            bgrad=bgrad*beta*(1-beta/beta_mode)
        chained=torch.autograd.grad((o*do).sum()+(ht*dht).sum(),(q0,k0,br),retain_graph=True)
        for got,ref in zip((qgrad,kgrad,bgrad),chained):torch.testing.assert_close(got,ref,atol=1e-10,rtol=1e-9)
        # Original upstream recurrent, unchanged float32 arithmetic and autograd.
        oq,ok,ov,og,ob,oh=[x.detach().float().requires_grad_() for x in nodes]
        ro,rh=upstream(oq[None,:,None],ok[None,:,None],ov[None,:,None],og[None,:,None],ob[None,:,None],scale=.37,initial_state=oh[None,None],output_final_state=True)
        torch.testing.assert_close(o.float(),ro[0,:,0],atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(ht.float(),rh[0,0],atol=2e-5,rtol=2e-4)
        rg=torch.autograd.grad((ro[0,:,0]*do.float()).sum()+(rh[0,0]*dht.float()).sum(),(oq,ok,ov,og,ob,oh))
        for got,ref in zip(actual,rg):torch.testing.assert_close(got.float(),ref,atol=3e-5,rtol=3e-4)
        # Verify raw gate and norm chains separately through independent autograd.
        dr=actual[3]
        if mode!='pre':
            safe_t=torch.exp(-torch.abs(a*x))
            dx=dr*(-2*a*safe_t/(1+safe_t).square() if mode=='safe' else -a*torch.sigmoid(x))
            dlog=(dx*x).sum().reshape_as(log) if mode=='safe' else (dr*delta).sum().reshape_as(log)
            gold=torch.autograd.grad(delta,(raw,log,bias),grad_outputs=dr,retain_graph=True)
            for got,ref in zip((dx,dlog,dx.sum(0)),gold):torch.testing.assert_close(got,ref,atol=1e-10,rtol=1e-9)
        count+=1
    print(json.dumps({'cases':count,'manual_vs_chunk_autograd_max_abs':worst,'upstream_recurrent':'passed','status':'CPU formula gate only; not device acceptance'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream',type=Path,required=True)
    args=parser.parse_args()
    run(load_upstream(args.upstream/'fla/ops/kda/naive.py'))
