"""FwdH probe on cache tensors emitted by the unmodified NPU producer."""
import ctypes
import json
import os
import sys
import threading
from pathlib import Path
import torch
import torch_npu
from numerical_gate import metrics,recurrence

torch.set_num_threads(4);torch.npu.set_device(0)
data=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
w,u,kg,g=[data[n] for n in ('w','u','kg','G')]
batch,heads,tokens,_=w.shape;chunks=(tokens+63)//64
lib=ctypes.CDLL(sys.argv[1]);run=lib.run_mix
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*6+[ctypes.c_void_p];run.restype=None
records=[]
for has_initial in (False,True):
    torch.manual_seed(1127)
    state=torch.randn(batch,heads,256,256)*.1 if has_initial else torch.zeros(batch,heads,256,256)
    expected_h=torch.empty(batch,chunks,heads,256,256);expected_v=torch.empty_like(u);expected_f=torch.empty_like(state)
    for b in range(batch):
        for h in range(heads):
            trace=recurrence(w[b,h],u[b,h],kg[b,h],g[b,h],state[b,h],'FP64')
            for c,r in enumerate(trace):
                expected_h[b,c,h]=r['H_snapshot'];expected_v[b,h,c*64:c*64+len(r['Vnew'])]=r['Vnew']
            expected_f[b,h]=trace[-1]['R_run']
    for group in (1,2):
        blocks=min(20,batch*((heads+group-1)//group)*4)
        values=[x.npu() for x in (w,u,kg,g,state)]
        outputs=[torch.empty_like(t,device='npu') for t in (expected_h,expected_v,expected_f)]
        scratch=torch.empty(blocks*group*24576,device='npu')
        ptrs=(ctypes.c_void_p*9)(*[x.data_ptr() for x in values[:4]],values[4].data_ptr() if has_initial else 0,
                               *[x.data_ptr() for x in outputs],scratch.data_ptr())
        for repeat in range(5):
            timer=threading.Timer(60,lambda:os._exit(124));timer.daemon=True;timer.start()
            run(ptrs,batch,heads,tokens,group,0,blocks,torch.npu.current_stream().npu_stream)
            torch.npu.synchronize();timer.cancel()
            for name,actual,expected in zip(('H_snapshot','Vnew','H_final'),outputs,(expected_h,expected_v,expected_f)):
                actual=actual.cpu();torch.testing.assert_close(actual,expected,atol=3e-6,rtol=1e-4)
                records.append(dict(name=name,initial=has_initial,group=group,repeat=repeat,**metrics(actual,expected)))
        print('PASS',has_initial,group,flush=True)
Path(sys.argv[3]).write_text(json.dumps(dict(scope='actual_npu_producer_cache_fwdh_stage',records=records),indent=2))
