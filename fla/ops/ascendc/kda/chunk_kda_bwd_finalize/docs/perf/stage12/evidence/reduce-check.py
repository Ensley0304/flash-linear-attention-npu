import sys
import argparse
from pathlib import Path
import torch
sys.path.insert(0, 'examples/fast_kernel_launch_example/tests/kda_finalize_stage01')
from check_stage01 import check_case
import signal, os
p=argparse.ArgumentParser()
p.add_argument('library')
a=p.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(3)
torch.ops.load_library(str(Path(a.library).resolve()))
signal.signal(signal.SIGALRM,lambda *_: os._exit(124))
args=argparse.Namespace(stage=12,q_norm=False,beta_mode=0,lower_bound=-5.,tail_golden_dir=None,
                       scale=128**-.5,device=3,cores=1,repeat=2,workspace_only=False,gate_mode='cumulative')
for case in [(1,96,1,None),(1,96,65,None),(1,3,1089,None),(1,3,2113,None)]:
    check_case(args,*case)
