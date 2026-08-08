import torch
import torch_npu

torch.npu.set_device(0)
x = torch.ones(1, device="npu:0")
torch.npu.synchronize()
print("NPU_OK", x.cpu().item(), flush=True)
