import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import torch
import torch_npu
sys.path.insert(0,str(Path('examples/fast_kernel_launch_example/tests/kda_finalize_stage01').resolve()))
from check_stage01 import check_case
p=argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('action',choices=['smoke','check'])
p.add_argument('--perf',action='store_true')
p.add_argument('--device',type=int,default=3)
a=p.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(a.device)
torch.ops.load_library(str(Path(f'stage12/{a.tag}-'+('perf' if a.perf else 'capture')+'.so').resolve()))
signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
smoke=[(1,3,65,None),(1,3,99,[0,1,0,33,65,0])]
tails=[(1,3,t,None) for t in (1,16,31,32,33,63,64,129)]
tails += [(2,3,65,None),(1,96,65,None),(1,3,289,[0,1,31,32,33,63,64,65,0])]
if a.action=='smoke':
    jobs=[(s,'smoke','cumulative',n,m,-5.) for s in (11,12) for n in (False,True) for m in (0,1,2)]
else:
    jobs=[(12,'tails','cumulative',n,m,-5.) for n in (False,True) for m in (0,1,2)]
    if not a.perf:
        jobs += [(12,'tails',g,n,0,-5.) for g in ('flat','extreme') for n in (False,True)]
        jobs += [(s,'smoke','cumulative',False,0,-5.) for s in range(11)]
        jobs += [(12,'smoke','cumulative',True,2,-.5)]
results=[]
tag=f'{a.tag}-{a.action}-batch-'+('perf' if a.perf else 'capture')
for stage,suite,gate,norm,beta,lower in jobs:
    check_args=SimpleNamespace(stage=stage,q_norm=norm,beta_mode=beta,lower_bound=lower,
                               tail_golden_dir=Path(f'stage12/{a.tag}-golden'),scale=128**-.5,
                               device=a.device,cores=1,repeat=2,workspace_only=a.perf,gate_mode=gate)
    print('BEGIN',stage,suite,gate,norm,beta,lower,flush=True)
    start=time.time()
    cases=smoke if suite=='smoke' else tails
    for case in cases: check_case(check_args,*case)
    results.append(dict(stage=stage,suite=suite,gate=gate,norm=norm,beta=beta,lower=lower,
                        checks=len(cases)*2,seconds=time.time()-start,status='PASS'))
    Path('stage12',tag+'.json').write_text(json.dumps(results,indent=2))
    print('PASS_GROUP',results[-1],flush=True)
Path('stage12',tag+'.status').write_text('PASS\n')
