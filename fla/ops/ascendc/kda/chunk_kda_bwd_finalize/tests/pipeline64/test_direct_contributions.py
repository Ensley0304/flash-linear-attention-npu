import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3305)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2;lib.c64_workspace_bytes.restype=ctypes.c_uint64
lib.c64_offset.argtypes=[ctypes.c_uint32]*4;lib.c64_offset.restype=ctypes.c_uint64
run=lib.run_c64_direct_contributions
run.argtypes=[ctypes.c_void_p]*11+[ctypes.c_uint32]*6+[ctypes.c_float,ctypes.c_void_p]
run.restype=None
cases=[]
for k,v in [(60,60),(64,64),(100,100),(128,128),(256,256),(64,32)]:
 for m in [1,63,64]:
  for variant in range(6):
   vf,dtype=variant%2,variant//2
   print('BEGIN',m,k,v,vf,dtype,flush=True)
   dt=[torch.float32,torch.float16,torch.bfloat16][dtype]
   q,kh,qr,drk,dkg=[torch.randn(m,k)*.05 for _ in range(5)]
   val=(torch.randn(m,v)*.05).to(dt);drv=torch.randn(m,v)*.05
   g=-torch.rand(m,k).cumsum(0)*.1
   beta=torch.rand(m);beta[0]=0
   h,dh=torch.randn(k,v)*.05,torch.randn(k,v)*.05
   base=0.6931471805599453 if vf else 1.
   if vf:g=g/base
   rq,rk,rg,rb,rv=[x.double().requires_grad_() for x in [q,kh,g,beta,val]]
   scale=ctypes.c_float(.37).value
   loss=(scale*qr.double()*rq*(base*rg).exp()).sum()
   loss+=(drk.double()*rb[:,None]*rk*(base*rg).exp()).sum()
   loss+=(dkg.double()*rk*(base*(rg[-1:]-rg)).exp()).sum()
   loss+=(h.double()*dh.double()*(base*rg[-1]).exp()[:,None]).sum()
   loss+=(drv.double()*rb[:,None]*rv).sum()
   expected=torch.autograd.grad(loss,(rq,rk,rg,rb,rv))
   w=torch.full((lib.c64_workspace_bytes(k,v)//4,),-937.25,device='npu')
   kp,vp=(k+15)//16*16,(v+15)//16*16
   def view(kind,index,width):
    off=lib.c64_offset(k,v,kind,index)//4
    return w[off:off+64*width].view(64,width)
   view(1,2,kp)[:m,:k].copy_(drk);view(1,3,kp)[:m,:k].copy_(dkg);view(2,1,vp)[:m,:v].copy_(drv)
   tensors=[x.npu() for x in [q,kh,g,qr,val,h.T.contiguous() if vf else h,dh.T.contiguous() if vf else dh,beta]]
   dv=torch.full((m,v),float('nan'),device='npu',dtype=dt)
   bp=torch.full((2,64),-937.25,device='npu')
   run(w.data_ptr(),*[x.data_ptr() for x in tensors],dv.data_ptr(),bp.data_ptr(),m,k,v,vf,vf,dtype,scale,
       torch.npu.current_stream().npu_stream)
   torch.npu.synchronize()
   actual=[view(1,i,kp)[:m,:k].cpu().double() for i in [4,5,6]]+[bp[:,:m].cpu().double().sum(0)]
   for x,y in zip(actual,expected[:4]):torch.testing.assert_close(x,y,atol=2e-6,rtol=4e-4)
   torch.testing.assert_close(dv.cpu(),expected[4].to(dt),atol=2e-6,rtol=1e-2 if dtype==2 else 1e-3)
   assert torch.all(dv[0]==0)
   assert torch.all(bp[:,m:]==-937.25)
   for i in [4,5,6]:
    assert torch.all(view(1,i,kp)[m:]==-937.25)
    assert torch.all(view(1,i,kp)[:m,k:]==-937.25)
   cases.append(dict(m=m,k=k,v=v,v_first=vf,dtype=dtype,max_abs=max((x-y).abs().max().item() for x,y in zip(actual,expected))))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
