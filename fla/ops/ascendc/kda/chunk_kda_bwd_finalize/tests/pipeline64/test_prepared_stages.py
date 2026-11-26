"""F1--F4 device intermediate checks; full acceptance still uses fla-org."""
import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401

lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2
lib.c64_workspace_bytes.restype=ctypes.c_uint64
lib.c64_offset.argtypes=[ctypes.c_uint32]*4
lib.c64_offset.restype=ctypes.c_uint64
run=lib.run_c64_prepared_stages
run.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*6+[ctypes.c_void_p]
run.restype=None
torch.npu.set_device(0)
torch.manual_seed(3301)
report=[]
for k,v in [(60,60),(64,64),(100,100),(128,128),(256,256),(64,32)]:
 for m in [1,63,64]:
  for variant in range(6):
   vf,dtype=variant%2,variant//2
   print('BEGIN',m,k,v,vf,flush=True)
   kp,vp=(k+15)//16*16,(v+15)//16*16
   du,h,dh,vn=[torch.randn(*s)*.05 for s in [(m,v),(k,v),(k,v),(m,v)]]
   akk=torch.eye(m)+torch.tril(torch.randn(m,m)*.01,-1)
   khat,gate,value,beta=torch.randn(m,k)*.05,-torch.rand(m,k),torch.randn(m,v)*.05,torch.rand(m)
   value=value.to([torch.float32,torch.float16,torch.bfloat16][dtype])
   beta[0]=0
   rk,rv=khat*gate.exp()*beta[:,None],value.float()*beta[:,None]
   dw=-du.double()@h.double().T
   dkg=vn.double()@dh.double().T
   drv=akk.double().T@du.double()
   da=dw@rk.double().T+du.double()@rv.double().T
   middle=akk.double().T@da
   dl=torch.tril(-middle@akk.double().T,-1)
   drk=akk.double().T@dw
   size=lib.c64_workspace_bytes(k,v)//4
   workspace=torch.full((size,),-937.25,device='npu')
   def view(kind,index,width):
    start=lib.c64_offset(k,v,kind,index)//4
    return workspace[start:start+64*width].view(64,width)

   a=torch.zeros(64,64);a[:m,:m]=akk
   args=[du.npu(),(h.T.contiguous() if vf else h).npu(),
         (dh.T.contiguous() if vf else dh).npu(),vn.npu(),a.npu(),khat.npu(),(gate/0.6931471805599453 if vf else gate).npu(),value.npu(),beta.npu()]
   for repeat in range(3):
    run(workspace.data_ptr(),*[x.data_ptr() for x in args],m,k,v,vf,vf,dtype,
        torch.npu.current_stream().npu_stream)
    torch.npu.synchronize()
    checks=[(1,0,kp,k,rk),(2,0,vp,v,rv),(1,1,kp,k,dw),(1,3,kp,k,dkg),(2,1,vp,v,drv),
            (0,0,64,m,da),(0,1,64,m,middle),(0,2,64,m,dl),(1,2,kp,k,drk)]
    error=0.
    for kind,index,stride,cols,expected in checks:
     actual=view(kind,index,stride).cpu()
     torch.testing.assert_close(actual[:m,:cols],expected.float(),atol=1e-5,rtol=2e-4)
     assert torch.all(actual[m:]==-937.25)
     assert torch.all(actual[:m,cols:]==-937.25)
     error=max(error,(actual[:m,:cols]-expected.float()).abs().max().item())
   report.append(dict(m=m,k=k,v=v,v_first=vf,input_dtype=dtype,repeats=3,max_abs=error))
print(json.dumps(dict(result='PASS',cases=report)),flush=True)
