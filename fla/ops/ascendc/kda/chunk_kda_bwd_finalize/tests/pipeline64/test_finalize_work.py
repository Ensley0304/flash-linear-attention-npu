"""Single-launch complete record against an independent FP64 autograd graph."""
import ctypes
import json
import os
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3310)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
if os.getenv('HEAD_PAIR'):
 lib.c64_workspace_bytes=lib.c64_pair_workspace_bytes
 lib.run_c64_finalize_work=lib.run_c64_finalize_pair
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2
lib.c64_workspace_bytes.restype=ctypes.c_uint64
run=lib.run_c64_finalize_work
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*9+[ctypes.c_void_p]
run.restype=None
cases=[]
for k,v in [(60,60),(64,32),(100,100),(128,128),(256,256)]:
 for m in [1,16,64]:
  for case in range(6):
   if os.getenv('ADVERSARIAL') and (k not in (60,100,256) or m!=64 or case not in (4,5)):continue
   if os.getenv('PROFILE_WORK') and (k,v,m,case)!=(256,256,64,0):continue
   norm=case%2;gate=case//2;bm=case%3;exp2=case%2;dtype=1+case%2
   dt=torch.float16 if dtype==1 else torch.bfloat16
   print('BEGIN',k,v,m,case,flush=True)
   def leaf(shape,scale=.1,typed=False):
    x=torch.randn(shape)*scale
    return (x.to(dt) if typed else x).double().requires_grad_()
   q,kx=leaf((m,k),typed=True),leaf((m,k),typed=True)
   value=leaf((m,v),typed=True);raw=leaf((m,k),10. if os.getenv('ADVERSARIAL') else .02)
   bias=leaf((k,),.01);alog=leaf((),.01);beta=leaf((m,),.2,True)
   rstdq=(q.square().sum(-1)+1e-6).rsqrt();rstdk=(kx.square().sum(-1)+1e-6).rsqrt()
   qh=q*rstdq[:,None] if norm else q
   kh=kx*rstdk[:,None] if norm else kx
   be=beta if bm==0 else beta.sigmoid()*(2 if bm==2 else 1)
   if gate==0:delta=raw-.1
   elif gate==1:delta=-alog.exp()*torch.nn.functional.softplus(raw+bias)
   else:delta=-5*torch.sigmoid(alog.exp()*(raw+bias))
   gn=delta.cumsum(0)
   G=gn/(.6931471805599453 if exp2 else 1.)
   mask=torch.ones(m,m,dtype=torch.bool).tril()
   ex=torch.where(mask[:,:,None],gn[:,None]-gn[None,:],torch.zeros_like(gn[:,None]-gn[None,:])).exp()
   rkk=(kh[:,None]*kh[None,:]*ex).sum(-1)
   A=torch.linalg.inv(torch.eye(m,dtype=torch.float64)+(be[:,None]*rkk).tril(-1))
   rk=be[:,None]*kh*gn.exp();rv=be[:,None]*value
   h,dh=torch.randn(k,v,dtype=torch.float64)*.01,torch.randn(k,v,dtype=torch.float64)*.01
   du,vn=torch.randn(m,v,dtype=torch.float64)*.01,torch.randn(m,v,dtype=torch.float64)*.01
   dqr,ds=torch.randn(m,k,dtype=torch.float64)*.01,torch.randn(m,m,dtype=torch.float64)*.01
   loss=((A@(rv-rk@h))*du).sum()
   kg=kh*(gn[-1]-gn).exp()
   loss+=(kg.T@vn*dh).sum()+(.37*qh*gn.exp()*dqr).sum()
   loss+=((qh[:,None]*kh[None,:]*ex).sum(-1)*ds.tril()).sum()
   loss+=(gn[-1].exp()[:,None]*h*dh).sum()
   expected=list(torch.autograd.grad(loss,(q,kx,value,raw,beta,bias,alog),allow_unused=True))
   expected[5]=torch.zeros(k,dtype=torch.float64) if expected[5] is None else expected[5]
   expected[6]=torch.tensor(0.,dtype=torch.float64) if expected[6] is None else expected[6]
   def dev(t,typ=torch.float32):return t.detach().to(typ).npu()
   ws=torch.full((lib.c64_workspace_bytes(k,v)//4,),-937.25,device='npu')
   akk=torch.eye(64,device='npu');akk[:m,:m].copy_(A.detach())
   out=[torch.empty_like(dev(t,dt if i in [0,1,2,4] else torch.float32)) for i,t in enumerate(expected[:5])]
   kp=(k+15)//16*16
   partial=torch.full((kp+1,),-937.25,device='npu')
   tensors=[ws,dev(du),dev(h),dev(dh),dev(vn),akk,dev(qh),dev(kh),dev(G),dev(value,dt),dev(be),
            dev(dqr),dev(ds),dev(raw),dev(bias),dev(alog),dev(rstdq),dev(rstdk),*out,
            torch.full((2,64),-937.25,device='npu'),torch.full((16,),-937.25,device='npu'),partial]
   other=[t.clone() for t in tensors] if os.getenv('HEAD_PAIR') else []
   ptrs=(ctypes.c_void_p*(26+len(other)))(*[t.data_ptr() for t in tensors+other])
   run(ptrs,m,k,v,norm,gate,bm,exp2,dtype,int(os.getenv('CUBE_SCORE','1')),torch.npu.current_stream().npu_stream)
   torch.npu.synchronize()
   for actual,reference in zip(out,expected[:5]):
    torch.testing.assert_close(actual.cpu(),reference.to(actual.dtype),atol=3e-5,rtol=1e-2 if dtype==2 else 3e-3)
   torch.testing.assert_close(partial[:k].cpu().double(),expected[5],atol=3e-5,rtol=3e-4)
   torch.testing.assert_close(partial[kp].cpu().double(),expected[6],atol=3e-5,rtol=3e-4)
   if other:
    for actual,reference in zip(other[18:23],expected[:5]):
     torch.testing.assert_close(actual.cpu(),reference.to(actual.dtype),atol=3e-5,rtol=1e-2 if dtype==2 else 3e-3)
   cases.append(dict(k=k,v=v,m=m,case=case))
   if os.getenv('PROFILE_WORK'):
    def call():run(ptrs,m,k,v,norm,gate,bm,exp2,dtype,int(os.getenv('CUBE_SCORE','1')),torch.npu.current_stream().npu_stream)
    for _ in range(2):call()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ['PROFILE_WORK'])) as prof:
     for _ in range(10):call();torch.npu.synchronize();prof.step()
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
