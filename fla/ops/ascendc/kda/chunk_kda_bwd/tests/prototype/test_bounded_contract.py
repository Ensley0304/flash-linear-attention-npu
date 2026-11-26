"""Development checks of production device stages against the upstream recurrent.

This direct-launch harness is not a substitute for the original upstream pytest
suite through ACLNN. It isolates packed/state/workspace ownership before binding.
"""
import argparse
import ast
import ctypes as C
import itertools
import json
from pathlib import Path

import torch
import torch_npu


class Config(C.Structure):
    _fields_ = [(s, C.c_uint32) for s in (
        'batch heads tokens keys values sequences chunks packed timeMajor stateVFirst '
        'normalize betaMode gateMode useExp2 recompute').split()] + [
        (s, C.c_float) for s in ('scale', 'epsilon', 'lowerBound')] + [
        ('gateTimeMajor',C.c_uint32),('betaTimeMajor',C.c_uint32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(1)
    path = args.upstream / 'fla/ops/kda/naive.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'naive_recurrent_kda')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    reference = scope[fn.name]
    lib = C.CDLL(str(Path(__file__).parent / 'build/libkda_bounded_owner.so'))
    lib.bounded_workspace_size.argtypes = [C.POINTER(Config), C.c_int]
    lib.bounded_workspace_size.restype = C.c_uint64
    lib.run_bounded.argtypes = [C.POINTER(C.c_void_p), C.POINTER(Config), C.c_int, C.c_int, C.c_uint32, C.c_void_p]
    lib.run_bounded.restype = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    T, H, K, V, blocks = 65, 3, 64, 32, 2  # H>blocks exercises slot/accumulator reuse.
    offsets = [0, 0, 1, 34, 34, 65]
    seq = []
    chunks = 0
    for start, end in zip(offsets, offsets[1:]):
        seq.append([start, end-start, chunks])
        chunks += (end-start+31)//32
    S = len(seq)
    shape = (T, H, K)
    raw_inputs = [torch.randn(shape)*.15, torch.randn(shape)*.15, torch.randn(T,H,V)*.15,
                  torch.randn(shape)*.15-2, torch.randn(T,H)*.15,
                  torch.randn(H)*.1-.5, torch.randn(H,K)*.1, torch.randn(S,H,K,V)*.1]
    dO = (torch.randn(T,H,V)*.1).to(dtype).float()
    dht = torch.randn(S,H,K,V)*.1
    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, device='npu', dtype=dtype)
    def upload(x, dt=torch.float32):
        return x.to(device='npu', dtype=dt).contiguous()
    cases = itertools.product((False, True), (False, True), (0,1,2), (0,1,2), (False, True)) if args.full else [
        (False,False,2,0,True), (True,True,1,2,False), (True,False,0,1,True), (False,True,1,0,True)]
    count = 0
    for norm, vf, gate, bm, exp2 in cases:
        tm = count % 2 == 0
        have_a, have_bias = count % 3 != 1, count % 3 != 2
        q,k,v,g,beta,alog,bias,h0 = [x.clone().requires_grad_() for x in raw_inputs]
        # Match the public input rounding; reference then uses FP32 as upstream does.
        q,k,v = [x.detach().to(dtype).float().requires_grad_() for x in (q,k,v)]
        qh = q*(q.square().sum(-1,keepdim=True)+1e-6).rsqrt() if norm else q
        kh = k*(k.square().sum(-1,keepdim=True)+1e-6).rsqrt() if norm else k
        x = g + bias if have_bias else g
        a = alog.exp()[None,:,None] if have_a else 1
        delta = g if gate == 0 else -2*torch.sigmoid(a*x) if gate == 2 else -a*torch.nn.functional.softplus(x)
        be = beta if bm == 0 else bm*torch.sigmoid(beta)
        outputs, states = [], []
        for s,(start,end) in enumerate(zip(offsets,offsets[1:])):
            if start == end:
                outputs.append(v[start:end]);states.append(h0[s]);continue
            o,ht = reference(qh[start:end][None],kh[start:end][None],v[start:end][None],
                delta[start:end][None],be[start:end][None],scale=.37,
                initial_state=h0[s:s+1],output_final_state=True)
            outputs.append(o[0]);states.append(ht[0])
        ro,rht = torch.cat(outputs),torch.stack(states)
        grads = torch.autograd.grad((ro*dO).sum()+(rht*dht).sum(),
            (q,k,v,g,beta,h0,alog,bias),allow_unused=True)
        p = [None]*47
        def layout(z):return z if tm else z.transpose(0,1)
        p[0],p[1],p[2] = [upload(layout(z),dtype) for z in (q,k,v)]
        p[3],p[4] = upload(g),upload(beta)
        p[5],p[6] = upload(alog) if have_a else None,upload(bias) if have_bias else None
        for i in (7,10,12,13,14,15,16):p[i]=empty(H,T,K)
        for i in (8,9):p[i]=empty(H,T,32)
        p[11]=empty(H,T,V)
        for i in (17,18,19):p[i]=empty(H,T)
        def state(z):return z.transpose(-2,-1) if vf else z
        p[20],p[28] = upload(state(h0)),upload(state(dht))
        p[21]=empty(chunks,H,V,K) if vf else empty(chunks,H,K,V)
        p[22],p[23]=empty(H,T,V),empty(*state(h0).shape)
        p[24]=upload(dO,dtype)
        p[31]=empty(*state(h0).shape)
        p[37],p[38],p[39]=[empty(*layout(z).shape,dtype=dtype) for z in (q,k,v)]
        p[40],p[41]=empty(T,H,K),empty(T,H)
        p[42],p[43]=empty(H) if have_a else None,empty(H,K) if have_bias else None
        p[44],p[45]=empty(T,H,V,dtype=dtype),upload(torch.tensor(seq),torch.int64)
        cfg=Config(1,H,T,K,V,S,chunks,1,tm,vf,norm,bm,gate,exp2,1,.37,1e-6,-2.,1,1)
        nbytes=lib.bounded_workspace_size(C.byref(cfg),1)*blocks
        p[46]=torch.full((nbytes,),211,device='npu',dtype=torch.uint8)
        pointers=(C.c_void_p*len(p))(*[t.data_ptr() if t is not None else 0 for t in p])
        def launch(backward):
            lib.run_bounded(pointers,C.byref(cfg),backward,int(args.bf16),blocks,torch.npu.current_stream().npu_stream)
            torch.npu.synchronize()
        launch(0)
        tolerance={'atol':3e-4,'rtol':.02 if args.bf16 else .005}
        torch.testing.assert_close(p[44].cpu().float(),ro,**tolerance)
        torch.testing.assert_close(p[23].cpu(),state(rht),**tolerance)
        saved=None
        for recompute in (False,True):
            cfg.recompute=recompute
            p[46].fill_(173)
            launch(1)
            got=[p[i].cpu() if p[i] is not None else None for i in (37,38,39,40,41,31,42,43)]
            for i,(value,ref) in enumerate(zip(got,grads)):
                if value is None:continue
                if ref is None:ref=torch.zeros_like(value)
                elif i<3:ref=layout(ref)
                elif i==5:ref=state(ref)
                torch.testing.assert_close(value.float(),ref,**tolerance)
            if saved is not None:
                for x,y in zip(got,saved):
                    if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
            saved=got
        torch.testing.assert_close(p[20].cpu(),state(h0),atol=0,rtol=0)
        count+=1
        print(json.dumps({'case':count,'norm':norm,'v_first':vf,'gate':gate,'beta':bm,'exp2':exp2,
                          'time_major':tm,'a_log':have_a,'bias':have_bias,'status':'PASS'}),flush=True)
    print(json.dumps({'cases':count,'result':'PASS','scope':'device stages; public ACLNN/upstream pytest pending'}))


if __name__=='__main__':main()
