import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import torch
import torch_npu
sys.path.insert(0, str(Path('examples/fast_kernel_launch_example/tests/kda_finalize_stage01').resolve()))
from check_stage01 import make_case
from check_stage11 import make_tail, check_stage11
p=argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--perf',action='store_true')
p.add_argument('--device',type=int,default=3)
args=p.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(args.device)
lib=Path(f'stage12/{args.tag}-'+('perf' if args.perf else 'capture')+'.so')
torch.ops.load_library(str(lib.resolve()))
signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
out=Path('stage12')
records=[]
def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
for norm in (False,True):
 for variable in (False,True):
  name=('varlen' if variable else 'fixed')+f'-norm{int(norm)}'
  inputs,cu,chunks=make_case(1,96,8192,[0,1,31,64,65,4096,3935,0] if variable else None,gate_mode='cumulative')
  if norm: inputs=(torch.nn.functional.normalize(inputs[0].float(),dim=-1).bfloat16(),*inputs[1:])
  beta=torch.rand(inputs[0].shape[:-1],generator=torch.Generator().manual_seed(137))
  beta.flatten()[:4]=torch.tensor([0.,1.,.5,.99966466])
  gen=torch.Generator().manual_seed(593)
  q=(torch.randn(inputs[0].shape,generator=gen)*.125).bfloat16()
  dq_raw=torch.randn(inputs[0].shape,generator=gen)*.125
  aq=torch.randn((*inputs[0].shape[:-1],64),generator=gen)*.125
  rstd=torch.rand(q.shape[:-1],generator=gen)+.5 if norm else None
  if norm: q=torch.nn.functional.normalize(q.float(),dim=-1).bfloat16()
  device=f'npu:{args.device}'
  dev=[x.to(device) for x in inputs]
  dq=torch.full_like(dev[0],float('nan'));dv=torch.full_like(dq,float('nan'))
  dbeta,dq_input,raw,daq=[x.to(device) for x in (beta,q,dq_raw,aq)]
  rst=rstd.to(device) if norm else None
  tail_cpu,tail=make_tail(inputs,chunks,norm,device,1)
  torch.npu.synchronize()
  def launch(stage):
      signal.alarm(60)
      cap,trace,ws=torch.ops.ascend_ops.kda_finalize_stage01(*dev,cu,0,stage,dbeta,dv,dq_input,raw,128**-.5,daq,dq,rst,**(tail if stage>=11 else {}))
      torch.npu.synchronize();signal.alarm(0)
      return cap.cpu(),trace.cpu(),ws.cpu()
  prefix={}
  if not args.perf:
      cap,trace,ws=launch(10)
      prefix=dict(capture=digest(cap[:,:,:816*1024]),dq=digest(dq.cpu()),dv=digest(dv.cpu()))
      del cap,trace,ws
  cap,trace,ws=launch(12)
  print('DEVICE_DONE',name,flush=True)
  record=dict(case=name,outputs={key:digest(value.cpu()) for key,value in tail.items() if key.endswith('_out')})
  record['outputs'].update(dq=digest(dq.cpu()),dv=digest(dv.cpu()))
  if args.perf:
      gold=json.loads((out/(args.tag+'-model-capture.json')).read_text())
      wanted=next(x for x in gold if x['case']==name)
      assert record['outputs']==wanted['outputs'],(name,'capture/perf mismatch')
  else:
      assert digest(cap[:,:,:816*1024])==prefix['capture'],(name,'Stage 0-10 capture changed')
      assert record['outputs']['dq']==prefix['dq'] and record['outputs']['dv']==prefix['dv']
      check_stage11(inputs,q,beta,aq,chunks,variable,cap,tail_cpu,tail,
                    SimpleNamespace(stage=12,beta_mode=1,lower_bound=-5.))
      record['fp64_stage11_stage12']='PASS'
      record['prefix_stage10']='bitwise identical'
  records.append(record)
  (out/(args.tag+'-model-'+('perf' if args.perf else 'capture')+'.json')).write_text(json.dumps(records,indent=2))
  print('PASS',name,flush=True)
  del dev,cap,trace,ws,dq,dv,dbeta,dq_input,raw,daq,rst,inputs,q,dq_raw,beta,aq,rstd,tail_cpu,tail
