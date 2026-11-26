import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0);torch.manual_seed(3312)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_dhu_workspace.argtypes=[ctypes.c_uint32]*2;lib.c64_dhu_workspace.restype=ctypes.c_uint64
run=lib.run_c64_dhu_work;run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*5+[ctypes.c_void_p];run.restype=None
cases=[]
for k,v in [(60,60),(64,64),(64,32),(100,100),(128,128),(256,256)]:
 for t in [1,65,129]:
  for case in range(4):
   dtype=1+case%2;exp2=case%2;initial=case//2
   dt=torch.float16 if dtype==1 else torch.bfloat16
   print('BEGIN',k,v,t,case,flush=True)
   q,kg,w=[torch.randn(t,k)*.02 for _ in range(3)];g=-torch.rand(t,k)*.1
   for begin in range(0,t,64):g[begin:begin+64]=g[begin:begin+64].cumsum(0)
   if exp2:g/=.6931471805599453
   grad=torch.randn(t,v).to(dt);local=torch.randn(t,v)*.01
   state=torch.randn(k,v).double()*.1 if initial else torch.zeros(k,v,dtype=torch.float64)
   last=state.clone();chunks=(t+63)//64;expected=torch.empty(t,v,dtype=torch.float64);snapshots=torch.empty(chunks,k,v,dtype=torch.float64)
   for c in reversed(range(chunks)):
    begin=c*64;end=min(t,begin+64);snapshots[c]=state
    du=local[begin:end].double()+kg[begin:end].double()@state;expected[begin:end]=du
    decay=(g[end-1].double()*(.6931471805599453 if exp2 else 1)).exp()
    state=decay[:,None]*state+.37*(q[begin:end].double().T@grad[begin:end].double())-w[begin:end].double().T@du
   outputs=[torch.full((t,v),-937.25,device='npu'),torch.empty(k,v,device='npu'),torch.empty(chunks,k,v,device='npu')]
   inputs=[x.npu() for x in [q,kg,w,g,grad,local]]
   tensors=[torch.full((lib.c64_dhu_workspace(k,v)//4,),-937.25,device='npu'),*inputs,outputs[0],last.float().npu() if initial else None,outputs[1],outputs[2]]
   ptrs=(ctypes.c_void_p*11)(*[x.data_ptr() if x is not None else 0 for x in tensors])
   for _ in range(3):
    run(ptrs,t,k,v,exp2,dtype,torch.npu.current_stream().npu_stream);torch.npu.synchronize()
    for x,y in zip(outputs,[expected,state,snapshots]):torch.testing.assert_close(x.cpu().double(),y,atol=3e-6,rtol=3e-4)
   cases.append(dict(k=k,v=v,t=t,case=case))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
