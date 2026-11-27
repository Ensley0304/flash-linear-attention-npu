"""Original fla-org test_chunk assertions through public saved/recompute calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
# This precision-only run also captures actual forward/recompute dispatch.
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT']='1'
import torch
import torch_npu

p=argparse.ArgumentParser()
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--adapter',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def check(label):
    info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',os.environ['ASCEND_RT_VISIBLE_DEVICES'],'-c','0'],text=True)
    (a.output/(label+'.txt')).write_text(info)
    ids=[int(x) for x in re.findall(r'Process id:\s*(\d+)',info)]
    assert 'No process in device' in info or ids
    assert all(pid==os.getpid() for pid in ids),info
check('before');torch.npu.set_device(0)
up=load('original_test_kda',a.upstream/'tests/ops/test_kda.py')
up.chunk_kda=load('public_chain_adapter',a.adapter).public_chunk
results=[]
for batch,tokens,heads in ((2,65,3),(1,129,8)):
    for saved in (True,False):
        label=f'b{batch}t{tokens}h{heads}-saved{int(saved)}'
        print('PUBLIC_GENERALIZED_BEGIN',label,flush=True)
        torch.manual_seed(21)
        up.test_chunk(B=batch,T=tokens,H=heads,HV=heads,D=256,scale=.37,
            gate_logit_normalizer=1.,mask_p=0.,use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,has_a_log=True,dtype=torch.bfloat16,
            safe_gate=True,disable_recompute=saved,chunk_size=64)
        torch.npu.synchronize();check(label)
        results.append(dict(case=label,status='original_assertions_passed'))
        (a.output/'results.json').write_text(json.dumps(results,indent=2))
        print('PUBLIC_GENERALIZED_PASS',label,flush=True)
