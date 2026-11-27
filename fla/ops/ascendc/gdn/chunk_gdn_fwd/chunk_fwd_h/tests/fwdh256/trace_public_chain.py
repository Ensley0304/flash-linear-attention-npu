"""Small public-chain route trace; precision is checked separately by upstream tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import torch
import torch_npu

p=argparse.ArgumentParser()
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--adapter',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--recompute',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def check(label):
    info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',os.environ['ASCEND_RT_VISIBLE_DEVICES'],'-c','0'],text=True)
    (a.output/(label+'.txt')).write_text(info)
    ids=[int(x) for x in re.findall(r'Process id:\s*(\d+)',info)]
    assert 'No process in device' in info or ids
    assert all(pid==os.getpid() for pid in ids),info
check('before');torch.npu.set_device(0)
spec=importlib.util.spec_from_file_location('public_chain_adapter',a.adapter)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
cpu=torch.load(a.inputs,map_location='cpu',weights_only=True)
names=('q','k','v','g','beta','A_log','dt_bias')
xs=[cpu[name].npu().requires_grad_(True) for name in names]
opts=dict(scale=.37,chunk_size=64,disable_recompute=not a.recompute,
          initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,
          use_gate_in_kernel=True,safe_gate=True,lower_bound=-5.,state_v_first=False)
print('CHAIN_FORWARD_BEGIN',flush=True)
out=adapter.public_chunk(**dict(zip(names,xs)),**opts)[0]
torch.npu.synchronize();print('CHAIN_FORWARD_PASS',flush=True)
print('CHAIN_BACKWARD_BEGIN',flush=True)
grads=torch.autograd.grad(out,xs,cpu['do'].npu())
torch.npu.synchronize();print('CHAIN_BACKWARD_PASS',flush=True)
assert all(torch.isfinite(x).all().item() for x in (out,*grads))
check('after');(a.output/'results.json').write_text(json.dumps(dict(
    status='passed',scope='route_only',recompute=a.recompute,shape=list(cpu['q'].shape))))
