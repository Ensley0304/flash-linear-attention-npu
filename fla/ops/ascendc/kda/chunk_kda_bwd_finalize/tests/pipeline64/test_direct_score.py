import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3307)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
run=lib.run_c64_direct_score
run.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*2+[ctypes.c_float,ctypes.c_void_p]
run.restype=None
cases=[]
for m in [1,16,63,64]:
 for d in [28,32,60,64,100,128,256]:
  for strong in [False,True]:
   print('BEGIN',m,d,strong,flush=True)
   q,k=torch.randn(m,d)*.1,torch.randn(m,d)*.1
   g=-torch.arange(1,m+1)[:,None].expand(m,d).float()*(5 if strong else .1)
   ds,dl=torch.randn(m,m)*.1,torch.randn(m,m)*.1
   beta=torch.rand(m);beta[0]=0
   if m>1:beta[1]=0
   base=0.6931471805599453 if strong else 1.
   # Quantize the stored exp2 cache exactly as passed to the device.
   if strong:g=g/base
   qr,kr,gr,br=[x.double().requires_grad_() for x in [q,k,g,beta]]
   mask=torch.ones(m,m,dtype=torch.bool).tril()
   diff=(gr[:,None,:]-gr[None,:,:])*base
   ex=torch.where(mask[:,:,None],diff,torch.zeros_like(diff)).exp()
   loss=((qr[:,None,:]*kr[None,:,:]*ex).sum(-1)*ds.double().tril()).sum()
   loss+=((kr[:,None,:]*kr[None,:,:]*ex).sum(-1)*br[:,None]*dl.double().tril(-1)).sum()
   expected=torch.autograd.grad(loss,(qr,kr,gr,br))
   src=[x.npu() for x in [q,k,g,ds,dl,beta]]
   seeds=[torch.randn(m,d)*.001 for _ in range(3)]
   outputs=[x.npu() for x in seeds]
   db=torch.full((2,64),-937.25,device='npu')
   run(*[x.data_ptr() for x in src],*[x.data_ptr() for x in outputs],db.data_ptr(),
       m,d,base,torch.npu.current_stream().npu_stream)
   torch.npu.synchronize()
   actual=[x.cpu().double()-seed.double() for x,seed in zip(outputs,seeds)]+[db[:,:m].cpu().double().sum(0)]
   for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=2e-6,rtol=4e-4)
   assert torch.isfinite(db[:,:m]).all()
   assert torch.all(db[:,m:]==-937.25)
   cases.append(dict(m=m,d=d,strong=strong,max_abs=max((x-y).abs().max().item() for x,y in zip(actual,expected))))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
