"""Device-only protocol gate; not a replacement for fla-org numerical acceptance."""
import ctypes as C
import json
from pathlib import Path
import torch
import torch_npu

lib=C.CDLL(str(Path(__file__).parent/'build/libkda_gm_roundtrip.so'))
lib.run_gm_roundtrip.argtypes=[C.c_void_p,C.c_void_p]+[C.c_uint32]*4+[C.c_int,C.c_void_p]
lib.run_gm_roundtrip.restype=None
cases=0
for cache in (0,1):
    for width in (1,60,100,128,256):
        heads,blocks,loops=5,2,67
        scratch=torch.full((blocks*(loops+2)*width,),13.25,device='npu',dtype=torch.float32)
        output=torch.empty((heads,width),device='npu',dtype=torch.float32)
        for repeat in range(4):
            scratch.fill_(-991.0-repeat)
            lib.run_gm_roundtrip(scratch.data_ptr(),output.data_ptr(),heads,width,loops,blocks,cache,torch.npu.current_stream().npu_stream)
            torch.npu.synchronize()
            expected=(torch.arange(heads,dtype=torch.float32)*100+3)[:,None].expand(heads,width)
            torch.testing.assert_close(output.cpu(),expected,atol=0,rtol=0)
            cases+=1
print(json.dumps({'cases':cases,'result':'PASS','protocol':'DataCopyPad + TQue + MTE3_MTE2 SetFlag/WaitFlag','cache_modes':'default and disabled','scope':'GM protocol only'}))
