"""Exactly one direct MIX launch, with no device output reads (simulator safe)."""
import ctypes
import sys
import torch
import torch_npu
torch.npu.set_device(0)
torch.manual_seed(1127)
lib=ctypes.CDLL(sys.argv[1]);run=lib.run_mix
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*6+[ctypes.c_void_p]
run.restype=None
tokens=65;heads=int(sys.argv[2]) if len(sys.argv)>2 else 2;group=heads;blocks=4
inputs=[(torch.randn(1,heads,tokens,256)*.01).npu() for _ in range(4)]
initial=(torch.randn(1,heads,256,256)*.1).npu()
h=torch.empty(1,2,heads,256,256,device='npu');v=torch.empty_like(inputs[0])
final=torch.empty_like(initial);scratch=torch.empty(blocks*group*24576,device='npu')
ptrs=(ctypes.c_void_p*9)(*[x.data_ptr() for x in inputs],initial.data_ptr(),h.data_ptr(),
                         v.data_ptr(),final.data_ptr(),scratch.data_ptr())
torch.npu.synchronize()
run(ptrs,1,heads,tokens,group,0,blocks,torch.npu.current_stream().npu_stream)
torch.npu.synchronize()
print('MIX_LAUNCH_FINISHED',flush=True)
