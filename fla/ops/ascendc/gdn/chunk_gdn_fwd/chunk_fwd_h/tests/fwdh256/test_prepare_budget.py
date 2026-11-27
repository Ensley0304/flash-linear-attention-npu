"""Unmodified existing producer: cache correctness and a non-FwdH budget bound."""
import argparse
import ctypes
import csv
import json
import os
import statistics
import threading
from pathlib import Path
import torch
import torch_npu
from producer_gate import chunked
from numerical_gate import metrics

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.npu.set_device(0)
cpu=torch.load(a.inputs,map_location='cpu',weights_only=True)
_,tokens,heads,dim=cpu['q'].shape;assert dim==256
lib=ctypes.CDLL(a.library);run=lib.run_prepare_budget
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*3+[ctypes.c_void_p];run.restype=None
names=['q','k','v','g','beta','A_log','dt_bias']
tensors=[cpu[n].npu() for n in names]
output_names=['G','Aqk','Akk','w','u','qg','kg','qg_scaled','qhat','khat','qrstd','krstd','beta_eff']
for name in output_names:
    cols=64 if name in ('Aqk','Akk') else 1 if name in ('qrstd','krstd','beta_eff') else 256
    tensors.append(torch.full((1,heads,tokens,cols),-937.25,device='npu'))
ptrs=(ctypes.c_void_p*20)(*[x.data_ptr() for x in tensors]);blocks=min(40,heads*((tokens+63)//64))
def call():run(ptrs,tokens,heads,blocks,torch.npu.current_stream().npu_stream)
timer=threading.Timer(60,lambda:os._exit(124));timer.daemon=True;timer.start()
print('LAUNCH existing producer',tokens,heads,flush=True);call();torch.npu.synchronize();timer.cancel()
outputs={n:t.cpu() for n,t in zip(output_names,tensors[7:])};results=[]
for head in range(heads):
    inputs=[cpu[n][0,:,head] for n in ('q','k','v','g','beta')]
    _,_,caches,_=chunked(*inputs,cpu['A_log'][head],cpu['dt_bias'].reshape(heads,256)[head],
                        torch.zeros(256,256),'P32')
    for name,key in [('G','g'),('w','w'),('u','u'),('kg','kg')]:
        expected=torch.cat([c[key] for c in caches]);actual=outputs[name][0,head]
        result=metrics(actual,expected)
        try:torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-3);passed=True
        except AssertionError:passed=False
        results.append(dict(head=head,name=name,passed=passed,**result))
torch.save({k:outputs[k] for k in ('G','w','u','kg')},a.output/'actual-producer-caches.pt')
print('cache checks',[(r['name'],r['passed'],r['max_abs']) for r in results],flush=True)
for _ in range(10):call()
torch.npu.synchronize()
with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'profile'))) as prof:
    for _ in range(10):call();torch.npu.synchronize();prof.step()
files=list((a.output/'profile').rglob('kernel_details.csv'));assert len(files)==1
rows=list(csv.DictReader(files[0].open()));assert len(rows)==10
assert all('fwdh256_existing_prepare_budget' in r['Name'] for r in rows)
samples=[float(r['Duration(us)']) for r in rows]
summary=dict(scope='existing_producer_direct_probe_not_public_chain',shape=[1,tokens,heads,256],
    cache_checks=results,median_us=statistics.median(samples),samples=samples,csv=str(files[0]))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
print('PRODUCER_MEDIAN_US',summary['median_us'],flush=True)
