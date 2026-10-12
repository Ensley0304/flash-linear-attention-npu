import argparse,hashlib,json,signal,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch,torch_npu
sys.path.insert(0,str(Path('examples/fast_kernel_launch_example/tests/kda_finalize_stage01').resolve()))
from check_stage01 import make_case
from check_stage10 import check_stage10

p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--tag',required=True)
p.add_argument('--device',type=int,default=2)
p.add_argument('--gold',default='')
p.add_argument('--perf',action='store_true')
args=p.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(args.device)
torch.ops.load_library(str(Path(args.library).resolve()))
signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
out=Path('stage10')
records=[]
for norm in (False,True):
 for variable in (False,True):
  name=('varlen' if variable else 'fixed')+f'-norm{int(norm)}'
  inputs,cu,chunks=make_case(1,96,8192,[0,1,31,64,65,4096,3935,0] if variable else None,gate_mode='cumulative')
  beta=torch.rand(inputs[0].shape[:-1],generator=torch.Generator().manual_seed(137))*2
  beta.flatten()[:4]=torch.tensor([0.,1.,2.,.99966466])
  gen=torch.Generator().manual_seed(593)
  q=(torch.randn(inputs[0].shape,generator=gen)*.125).bfloat16()
  dq_raw=torch.randn(inputs[0].shape,generator=gen)*.125
  aq=torch.randn((*inputs[0].shape[:-1],64),generator=gen)*.125
  rstd=torch.rand(q.shape[:-1],generator=gen)+.5 if norm else None
  if norm:q=torch.nn.functional.normalize(q.float(),dim=-1).bfloat16()
  device=f'npu:{args.device}'
  dev=[x.to(device) for x in inputs]
  dq=torch.full_like(dev[0],float('nan'));dv=torch.full_like(dq,float('nan'))
  dbeta,dq_input,raw,daq=[x.to(device) for x in (beta,q,dq_raw,aq)]
  rst=rstd.to(device) if norm else None
  torch.npu.synchronize()
  signal.alarm(60)
  cap,trace,ws=torch.ops.ascend_ops.kda_finalize_stage01(*dev,cu,0,10,dbeta,dv,dq_input,raw,128**-.5,daq,dq,rst)
  torch.npu.synchronize();signal.alarm(0)
  cap,trace,ws,dq,dv=cap.cpu(),trace.cpu(),ws.cpu(),dq.cpu(),dv.cpu()
  print('DEVICE_DONE',args.tag,name,flush=True)
  def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  record=dict(case=name,dq=digest(dq),dv=digest(dv))
  if not args.perf:
   record['capture']=digest(cap[:, :, :784*1024])
   trace[:, 9] = 9  # Prefix comparison uses the Stage 9 trace tag.
   record['trace']=digest(trace)
  # Compare only live Stage 9 records; other bytes in performance workspace are unspecified.
  from check_stage01 import workspace_works
  cores=ws.numel()//(1280*1024);live=[]
  for core,work in workspace_works(96,len(chunks),cores,9):
   for owner in range(2):
    base=(2*core+owner)*640*1024+((work//cores)&1)*320*1024
    live.append(ws[base+168*1024:base+200*1024])
  record['workspace_dg']=digest(torch.cat(live))
  if args.gold:
   gold=json.loads((out/args.gold).read_text())
   wanted=next(x for x in gold if x['case']==name)
   for key in ('dq', 'dv', 'workspace_dg'):
    assert record[key]==wanted[key],(name,key,'equivalence mismatch')
  else:
   # Reuse the previously FP64-validated r2 fixture; do not generate a new golden.
   known=json.loads(Path('stage9/model-golden.json').read_text())
   wanted=next(x for x in known if x['case']==name)
   for key in ('dq','dv','capture','trace'):
    assert record[key]==wanted[key],(name,key,'baseline identity mismatch')
  check_args=SimpleNamespace(workspace_only=args.perf, stage=10)
  check_stage10(inputs,q,beta,aq,chunks,variable,cap,ws,check_args)
  right=[]
  for core,work in workspace_works(96,len(chunks),cores,10):
   for owner in range(2):
    base=(2*core+owner)*640*1024+((work//cores)&1)*320*1024
    right.append(ws[base+96*1024:base+112*1024])
  record['workspace_right']=digest(torch.cat(right))
  if args.gold:
   assert record['workspace_right']==wanted['workspace_right'],(name,'right capture/perf mismatch')
  record['right_fp64']='PASS'
  records.append(record)
  (out/(args.tag+'.json')).write_text(json.dumps(records,indent=2))
  print('PASS',args.tag,name,flush=True)
  del dev,cap,trace,ws,dq,dv,dbeta,dq_input,raw,daq,rst,live,inputs,q,dq_raw,beta,aq,rstd
