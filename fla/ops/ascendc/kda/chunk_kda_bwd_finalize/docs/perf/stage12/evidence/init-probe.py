import faulthandler
faulthandler.dump_traceback_later(30,repeat=True)
print('IMPORT',flush=True)
import torch
import torch_npu
print('SET_DEVICE',flush=True)
torch.npu.set_device(3)
print('ALLOCATE',flush=True)
x=torch.zeros(1,device='npu:3')
torch.npu.synchronize()
print('PASS',flush=True)
faulthandler.cancel_dump_traceback_later()
