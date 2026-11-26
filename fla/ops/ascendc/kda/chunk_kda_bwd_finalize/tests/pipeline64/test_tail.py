import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3309)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2;lib.c64_workspace_bytes.restype=ctypes.c_uint64
lib.c64_offset.argtypes=[ctypes.c_uint32]*4;lib.c64_offset.restype=ctypes.c_uint64
run=lib.run_c64_tail
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*7+[ctypes.c_void_p]
run.restype=None
cases=[]
for k in [60,64,100,128,256]:
 for m in [1,16,64]:
  for case in range(18):
   norm=case%2;gate=(case//2)%3;bm=case//6;exp2=case%2;dtype=1+case%2
   dt=torch.float16 if dtype==1 else torch.bfloat16
   print('BEGIN',k,m,norm,gate,bm,flush=True)
   rq=(torch.randn(m,k)*.2).to(dt).double().requires_grad_()
   rk=(torch.randn(m,k)*.2).to(dt).double().requires_grad_()
   raw=torch.randn(m,k).float().double().requires_grad_()
   bias=(torch.randn(k)*.1).double().requires_grad_();alog=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
   logits=torch.randn(m).to(dt).double().requires_grad_()
   qgrad,kgrad,ggrad=[torch.randn(m,k).float()*.02 for _ in range(3)]
   bg1,bg2=torch.randn(m)*.02,torch.randn(m)*.02
   if norm:
    rstdq=(rq.square().sum(-1)+1e-6).rsqrt();rstdk=(rk.square().sum(-1)+1e-6).rsqrt()
    qhat=rq*rstdq[:,None];khat=rk*rstdk[:,None]
   else:qhat,khat=rq,rk;rstdq=rstdk=None
   be=logits if bm==0 else logits.sigmoid()*(2 if bm==2 else 1)
   x=raw+bias
   if gate==0:delta=raw
   elif gate==1:delta=-alog.exp()*torch.nn.functional.softplus(x)
   else:delta=-5*torch.sigmoid(alog.exp()*x)
   G=delta.cumsum(0)/(0.6931471805599453 if exp2 else 1.)
   loss=(qhat*qgrad+khat*kgrad+G*ggrad).sum()+(be*(bg1.double()+bg2.double())).sum()
   expected=torch.autograd.grad(loss,(rq,rk,raw,logits,bias,alog),allow_unused=True)
   expected=list(expected);expected[4]=expected[4] if expected[4] is not None else torch.zeros(k,dtype=torch.float64)
   expected[5]=expected[5] if expected[5] is not None else torch.tensor(0.,dtype=torch.float64)
   kp=(k+15)//16*16
   w=torch.full((lib.c64_workspace_bytes(k,k)//4,),-937.25,device='npu')
   for index,value in [(4,qgrad),(5,kgrad),(6,ggrad)]:
    off=lib.c64_offset(k,k,1,index)//4;w[off:off+64*kp].view(64,kp)[:m,:k].copy_(value)
   def dev(t):return t.detach().float().npu() if t is not None else None
   oq,ok=torch.empty(m,k,device='npu',dtype=dt),torch.empty(m,k,device='npu',dtype=dt)
   og,ob=torch.empty(m,k,device='npu'),torch.empty(m,device='npu',dtype=dt)
   bp=torch.zeros(2,64,device='npu');bp[0,:m].copy_(bg1);bp[1,:m].copy_(bg2)
   exchange=torch.full((16,),-937.25,device='npu');params=torch.full((kp+1,),-937.25,device='npu')
   tensors=[w,dev(qhat) if norm else None,dev(khat) if norm else None,dev(raw),dev(be),dev(rstdq),dev(rstdk),
            oq,ok,og,ob,dev(bias),dev(alog),bp,exchange,params]
   pointers=(ctypes.c_void_p*16)(*[t.data_ptr() if t is not None else 0 for t in tensors])
   run(pointers,m,k,norm,gate,bm,exp2,dtype,torch.npu.current_stream().npu_stream)
   torch.npu.synchronize()
   for out,reference in zip([oq,ok,og,ob],expected[:4]):
    torch.testing.assert_close(out.cpu(),reference.to(out.dtype),atol=2e-5,rtol=1e-2 if dtype==2 and out.dtype==dt else 2e-3)
   torch.testing.assert_close(params[:k].cpu().double(),expected[4],atol=2e-5,rtol=2e-4)
   torch.testing.assert_close(params[kp].cpu().double(),expected[5],atol=2e-5,rtol=2e-4)
   assert torch.all(params[k:kp]==0)
   cases.append(dict(k=k,m=m,norm=norm,gate=gate,beta=bm,dtype=dtype))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
