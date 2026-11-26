import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3311)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2;lib.c64_workspace_bytes.restype=ctypes.c_uint64
run=lib.run_c64_prepare_work
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*5+[ctypes.c_void_p]
run.restype=None
cases=[]
for k,v in [(60,60),(64,64),(64,32),(100,100),(128,128),(256,256)]:
 for m in [1,63,64]:
  for vf in [0,1]:
   for dtype in [1,2]:
    dt=torch.float16 if dtype==1 else torch.bfloat16
    print('BEGIN',k,v,m,vf,dtype,flush=True)
    aqk=torch.randn(m,m).tril();vn=torch.randn(m,v);grad=torch.randn(m,v).to(dt);h=torch.randn(k,v)
    expected=[.37*(grad.double()@vn.double().T).tril(),aqk.double().T@grad.double(),grad.double()@h.double().T]
    a=torch.zeros(m,64,device='npu');a[:,:m].copy_(aqk)
    outputs=[torch.full(shape,-937.25,device='npu') for shape in [(m,64),(m,v),(m,k)]]
    tensors=[torch.full((lib.c64_workspace_bytes(k,v)//4,),-937.25,device='npu'),a,vn.npu(),grad.npu(),
             (h.T.contiguous() if vf else h).npu(),*outputs]
    ptrs=(ctypes.c_void_p*8)(*[x.data_ptr() for x in tensors])
    for _ in range(3):
     run(ptrs,m,k,v,vf,dtype,torch.npu.current_stream().npu_stream);torch.npu.synchronize()
     for i,(x,y) in enumerate(zip(outputs,expected)):
      torch.testing.assert_close((x[:,:m] if i==0 else x).cpu().double(),y,atol=3e-5,rtol=2e-5)
     assert torch.all(outputs[0][:,m:]==0)
    cases.append(dict(k=k,v=v,m=m,vf=vf,dtype=dtype))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
