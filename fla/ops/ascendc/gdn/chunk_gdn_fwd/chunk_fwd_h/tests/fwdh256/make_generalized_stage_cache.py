"""Synthetic FP32 FwdH stage inputs, not NPU Prepare outputs."""
import argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
for b,t,h in ((2,65,8),(2,1024,4)):
    torch.manual_seed(1127);shape=(b,h,t,256)
    data=dict(w=torch.randn(shape)*.01,u=torch.randn(shape)*.05,
              kg=torch.randn(shape)*.01,G=-torch.randn(shape).abs()*.005)
    torch.save(data,a.output/f'b{b}t{t}h{h}.pt')
