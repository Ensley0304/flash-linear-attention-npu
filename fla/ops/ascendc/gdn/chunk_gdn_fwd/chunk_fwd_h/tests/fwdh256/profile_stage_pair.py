"""Same FP32 cache values for public FwdH, direct prototype and NPU Triton.

The canonical input is BNSD. Triton receives BSND views; any contiguous copies
and gate preparation inside its wrapper are included in the timing boundary.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import torch
import torch_npu
from numerical_gate import recurrence,metrics
from fla.utils import assert_close

p=argparse.ArgumentParser();p.add_argument('--implementation',choices=('ours','probe','triton'),required=True)
p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--library');p.add_argument('--initial',action='store_true');p.add_argument('--final',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.npu.set_device(0)
data=torch.load(a.inputs,map_location='cpu',weights_only=True)
w,u,kg,g=[data[name].float() for name in ('w','u','kg','G')]
batch,heads,tokens,dim=w.shape;assert dim==256
torch.manual_seed(1127);state=torch.randn(batch,heads,256,256)*.1 if a.initial else torch.zeros(batch,heads,256,256)
eh=torch.empty(batch,(tokens+63)//64,heads,256,256);ev=torch.empty_like(u);ef=torch.empty_like(state)
for b in range(batch):
    for head in range(heads):
        trace=recurrence(w[b,head],u[b,head],kg[b,head],g[b,head],state[b,head],'FP64')
        for chunk,row in enumerate(trace):
            eh[b,chunk,head]=row['H_snapshot'];ev[b,head,chunk*64:chunk*64+len(row['Vnew'])]=row['Vnew']
        ef[b,head]=trace[-1]['R_run']
# All implementations consume the same FP32 log2 gate values.
wn,un,kn,gn=[x.npu() for x in (w,u,kg,g/torch.log(torch.tensor(2.)))]
sn=state.npu() if a.initial else None
if a.implementation=='triton':
    from fla.ops.common.backends.triton_ascend.chunk_delta_h import chunk_gated_delta_rule_fwd_h_npu
    def call():
        h,v,f=chunk_gated_delta_rule_fwd_h_npu(kn.transpose(1,2),wn.transpose(1,2),un.transpose(1,2),
            gk=gn.transpose(1,2),initial_state=sn,output_final_state=a.final,chunk_size=64)
        return h,v.transpose(1,2),f
elif a.implementation=='ours':
    from fla_npu.ops.ascendc import npu_chunk_fwd_h
    def call():return npu_chunk_fwd_h(kn,wn,un,gk=gn,initial_state=sn,output_final_state=a.final,use_exp2=True)
else:
    lib=ctypes.CDLL(a.library);run=lib.run_mix
    run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*6+[ctypes.c_void_p];run.restype=None
    hn=torch.empty_like(eh,device='npu');vn=torch.empty_like(ev,device='npu');fn=torch.empty_like(ef,device='npu')
    blocks=min(20,batch*heads*4);scratch=torch.empty(blocks*24576,device='npu')
    ptrs=(ctypes.c_void_p*9)(wn.data_ptr(),un.data_ptr(),kn.data_ptr(),gn.data_ptr(),sn.data_ptr() if sn is not None else 0,
        hn.data_ptr(),vn.data_ptr(),fn.data_ptr() if a.final else 0,scratch.data_ptr())
    def call():
        run(ptrs,batch,heads,tokens,1,1,blocks,torch.npu.current_stream().npu_stream)
        return hn,vn,fn if a.final else None
def check(label):
    info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',os.environ['ASCEND_RT_VISIBLE_DEVICES'],'-c','0'],text=True)
    (a.output/(label+'.txt')).write_text(info)
    ids=[int(x) for x in re.findall(r'Process id:\s*(\d+)',info)]
    assert 'No process in device' in info or ids
    assert all(pid==os.getpid() for pid in ids),info
manifest=dict(implementation=a.implementation,shape=list(w.shape),dtype=str(w.dtype),initial=a.initial,final=a.final,
    input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),boundary='BNSD cache entry including wrapper conversions',status='started')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
try:
    check('before');actual=call();torch.npu.synchronize()
    results={}
    for name,got,ref in [('H',actual[0],eh),('Vnew',actual[1],ev)]+([('final',actual[2],ef)] if a.final else []):
        value=got.cpu();assert_close(name,ref,value,.005);results[name]=metrics(value,ref)
    for _ in range(10):call()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'profile'))) as prof:
        for _ in range(10):call();torch.npu.synchronize();prof.step()
    check('after');files=list((a.output/'profile').rglob('kernel_details.csv'));assert len(files)==1
    rows=list(csv.DictReader(files[0].open()));groups={}
    if a.implementation=='probe' and 'Step Id' not in rows[0]:
        assert len(rows)==10 and all('fwdh256_mix_probe' in row['Name'] for row in rows)
        groups={i:[row] for i,row in enumerate(rows)}
    else:
        for row in rows:groups.setdefault(row['Step Id'],[]).append(row)
    assert len(groups)==10
    samples=[sum(float(row['Duration(us)']) for row in group) for group in groups.values()]
    summary=dict(median_us=statistics.median(samples),samples=samples,results=results,csv=str(files[0]),
        names=sorted({row['Name'] for row in rows}),tasks=[len(group) for group in groups.values()])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));manifest['status']='passed'
    print(json.dumps(summary),flush=True)
except Exception as error:
    manifest['status']='failed';manifest['error']=str(error);raise
finally:(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
