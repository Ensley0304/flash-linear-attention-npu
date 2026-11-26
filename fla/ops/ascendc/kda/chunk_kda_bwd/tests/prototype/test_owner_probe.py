"""Single-launch full-stage workload prototype; full caches, H=1, safe gate only."""
import argparse
import ast
import ctypes as C
import json
import time
from pathlib import Path
import torch
import torch_npu

ap=argparse.ArgumentParser()
ap.add_argument('--tokens',type=int,default=65)
ap.add_argument('--dim',type=int,default=64)
ap.add_argument('--value-dim',type=int)
ap.add_argument('--bf16',action='store_true')
ap.add_argument('--bounded',action='store_true')
ap.add_argument('--upstream',type=Path,required=True)
args=ap.parse_args()
T,K,V=args.tokens,args.dim,args.value_dim or args.dim
N=(T+31)//32
dtype=torch.bfloat16 if args.bf16 else torch.float16
torch.manual_seed(9)
torch.set_num_threads(1)
f32=torch.float32
def empty(*shape,dt=f32):return torch.empty(shape,device='npu',dtype=dt)
def rand(*shape,dt=f32,scale=.1):return (torch.randn(shape,device='npu',dtype=f32)*scale).to(dt)
p=[rand(T,K,dt=dtype),rand(T,K,dt=dtype),rand(T,V,dt=dtype),rand(T,K)-2,rand(T)+.2,rand(1),rand(K)]
p += [empty(T,K),empty(T,32),empty(T,32)]
p += [empty(T,K),empty(T,V),empty(T,K),empty(T,K),empty(T,K),empty(T,K),empty(T,K),empty(T),empty(T),empty(T)]
p += [rand(K,V),empty(N,K,V),empty(T,V),empty(K,V),rand(T,V,dt=dtype),empty(T,32),empty(T,V),empty(T,K),rand(K,V),empty(N,K,V),empty(T,V),empty(K,V)]
p += [empty(T,K) for _ in range(4)]+[empty(T)]
p += [empty(T,K,dt=dtype),empty(T,K,dt=dtype),empty(T,V,dt=dtype),empty(T,K),empty(T),empty(N),empty(N,K)]
assert len(p)==44
lib=C.CDLL(str(Path(__file__).parent/('build/libkda_bounded_owner.so' if args.bounded else 'build/libkda_owner_probe.so')))
if args.bounded:
    lib.owner_workspace_size.argtypes=[C.c_uint32]*3
    lib.owner_workspace_size.restype=C.c_uint64
    p.extend([empty(T,V,dt=dtype),torch.tensor([0,T,0],device='npu',dtype=torch.int64),
              empty(lib.owner_workspace_size(T,K,V),dt=torch.uint8)])
lib.run_owner_probe.argtypes=[C.POINTER(C.c_void_p),C.c_uint32,C.c_uint32,C.c_uint32,C.c_int,C.c_int,C.c_void_p]
lib.run_owner_probe.restype=None
ptrs=(C.c_void_p*len(p))(*[x.data_ptr() for x in p])
def launch(mode):
    torch.npu.synchronize();start=time.perf_counter()
    lib.run_owner_probe(ptrs,T,K,V,mode,int(args.bf16),torch.npu.current_stream().npu_stream)
    torch.npu.synchronize()
    return time.perf_counter()-start
print(json.dumps({'phase':'prepare','seconds':launch(0),'T':T,'K':K}),flush=True)
results=[]
saved=None
cache_before={i:p[i].cpu().clone() for i in range(7,24)}
for mode in (1,2):
    elapsed=launch(mode)
    actual=[p[i].cpu() for i in (37,38,39,40,41,31,42,43)]
    if saved is None:saved=actual
    else:
        for i,old in cache_before.items():
            new=p[i].cpu()
            if not torch.equal(new,old):
                print(json.dumps({'cache':i,'max_abs':(new.float()-old.float()).abs().max().item()}),flush=True)
        for i,(x,y) in enumerate(zip(actual,saved)):
            print(json.dumps({'gradient':i,'save_recompute_max_abs':(x.float()-y.float()).abs().max().item()}),flush=True)
    results.append({'mode':'save' if mode==1 else 'recompute','seconds':elapsed})
    print(json.dumps(results[-1]),flush=True)
# Original upstream recurrent function, unchanged, for small numerical canary.
if T<=129:
    path=args.upstream/'fla/ops/kda/naive.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='naive_recurrent_kda')
    scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    q,k,v,g,beta,log,bias,h0=[p[i].cpu().float().detach().requires_grad_() for i in (0,1,2,3,4,5,6,20)]
    delta=-2*torch.sigmoid(log.exp()*(g+bias))
    o,ht=scope['naive_recurrent_kda'](q[None,:,None],k[None,:,None],v[None,:,None],delta[None,:,None],beta[None,:,None],scale=.37,initial_state=h0[None,None],output_final_state=True)
    if args.bounded:
        torch.testing.assert_close(p[44].cpu().float(),o[0,:,0],atol=3e-4,rtol=.02 if args.bf16 else .005)
        torch.testing.assert_close(p[23].cpu(),ht[0,0],atol=3e-4,rtol=.02 if args.bf16 else .005)
    loss=(o[0,:,0]*p[24].cpu().float()).sum()+(ht[0,0]*p[28].cpu()).sum()
    refs=torch.autograd.grad(loss,(q,k,v,g,beta,h0,log,bias))
    for i,(got,ref) in enumerate(zip(saved,refs)):
        if i==6:got=got[:1]
        if i==7:got=got[0]
        print(json.dumps({'reference_gradient':i,'max_abs':(got.float()-ref).abs().max().item()}),flush=True)
        torch.testing.assert_close(got.float(),ref,atol=3e-4,rtol=.02 if args.bf16 else .005)
    print('upstream recurrent/autograd canary PASS',flush=True)
for x,y in zip(actual,saved):torch.testing.assert_close(x,y,atol=0,rtol=0)
print(json.dumps({'T':T,'K':K,'dtype':str(dtype),'timings':results,'save_recompute_exact':True,'scope':'single-owner prototype; not public-interface acceptance'}))
