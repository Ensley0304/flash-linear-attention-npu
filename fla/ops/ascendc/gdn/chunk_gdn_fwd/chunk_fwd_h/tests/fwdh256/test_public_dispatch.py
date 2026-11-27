"""Public ACLNN FwdH: independent FP64 control, route markers and optional profiling."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import torch
import torch_npu
from fla.utils import assert_close
from fla_npu.ops.ascendc import npu_chunk_fwd_h
from numerical_gate import recurrence, metrics

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--suite',choices=('smoke','matrix','fallback','invalid'),default='smoke')
p.add_argument('--profile',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.npu.set_device(0)
def check(label):
    info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',os.environ['ASCEND_RT_VISIBLE_DEVICES'],'-c','0'],text=True)
    (a.output/(label+'.txt')).write_text(info)
    ids=[int(x) for x in re.findall(r'Process id:\s*(\d+)',info)]
    assert 'No process in device' in info or ids
    assert all(pid==os.getpid() for pid in ids),info

if a.suite=='invalid':
    x=torch.zeros(1,1,65,256,device='npu');g=torch.zeros(1,1,65,device='npu')
    cases=[('both_gates',x,x,dict(g=g,gk=x)),('no_gate',x,x,{}),
           ('chunk32',x,x,dict(gk=x,chunk_size=32)),
           ('no_vnew',x,x,dict(gk=x,save_new_value=False)),
           ('mixed_cache_dtype',x,x.bfloat16(),dict(gk=x))]
    records=[];check('invalid-before')
    for label,k,w,options in cases:
        try:npu_chunk_fwd_h(k,w,x,**options)
        except RuntimeError as error:
            assert 'GetWorkspaceSize failed' in str(error),str(error)
            records.append(dict(case=label,expected='ACLNN workspace query rejection',error=str(error)))
        else:raise AssertionError('Invalid input accepted: '+label)
    check('invalid-after');(a.output/'results.json').write_text(json.dumps(records,indent=2))
    print('INVALID_CASES_PASSED',len(records),flush=True);raise SystemExit(0)

# B,T,H,I,F,exp2,state-v-first,state-bf16,packed
cases=[(1,65,2,i,f,e,False,False,False) for e in (False,True) for i,f in ((False,False),(True,True))]
if a.suite=='matrix':
    cases=[(1,t,1,i,f,e,False,False,False) for t in (1,65)
           for e in (False,True) for i in (False,True) for f in (False,True)]
    cases += [(1,t,h,i,i,e,False,False,False) for t in (7,63,64,129,1024,4096)
              for h in (1,2) for i,e in ((False,True),(True,False))]
if a.suite=='fallback':
    cases=[(1,65,3,True,True,True,False,False,False),
           (2,65,1,True,True,True,False,False,False),
           (1,4097,1,False,False,False,False,False,False),
           (1,65,1,True,True,True,True,False,False),
           (1,65,1,True,True,True,False,True,False),
           (1,65,1,True,True,True,False,False,True)]
records=[]
for index,(batch,tokens,heads,initial,final,exp2,vfirst,bfstate,packed) in enumerate(cases):
    key=f'{index}-b{batch}-t{tokens}-h{heads}-i{int(initial)}f{int(final)}-e{int(exp2)}-vf{int(vfirst)}-bf{int(bfstate)}-pack{int(packed)}'
    torch.manual_seed(1127)
    shape=(batch,heads,tokens,256)
    w=torch.randn(shape)*.01;u=torch.randn(shape)*.05;kg=torch.randn(shape)*.01
    gate=-torch.randn(shape).abs()*.005
    bounds=[0,31,tokens] if packed else [0,tokens]
    ns=len(bounds)-1 if packed else batch
    state=torch.randn(ns,heads,256,256)*.1 if initial else torch.zeros(ns,heads,256,256)
    if bfstate:state=state.bfloat16().float()
    chunks=sum((end-start+63)//64 for start,end in zip(bounds,bounds[1:])) if packed else (tokens+63)//64
    eh=torch.empty(batch,chunks,heads,256,256);ev=torch.empty_like(u);ef=torch.empty_like(state)
    for b in range(batch):
        offset=0
        for seq,(begin,end) in enumerate(zip(bounds,bounds[1:])):
            si=seq if packed else b
            for head in range(heads):
                trace=recurrence(w[b,head,begin:end],u[b,head,begin:end],kg[b,head,begin:end],
                                 gate[b,head,begin:end],state[si,head],'FP64')
                for c,row in enumerate(trace):
                    eh[b,offset+c,head]=row['H_snapshot']
                    ev[b,head,begin+c*64:begin+c*64+len(row['Vnew'])]=row['Vnew']
                ef[si,head]=trace[-1]['R_run']
            offset+=(end-begin+63)//64
    g_input=gate/torch.log(torch.tensor(2.)) if exp2 else gate
    h0=state.transpose(-1,-2).contiguous() if vfirst else state
    if bfstate:h0=h0.bfloat16();ef=ef.bfloat16()
    if vfirst:eh=eh.transpose(-1,-2).contiguous();ef=ef.transpose(-1,-2).contiguous()
    kn,wn,un,gn=[x.npu() for x in (kg,w,u,g_input)]
    sn=h0.npu() if initial else None
    opts=dict(gk=gn,initial_state=sn,output_final_state=final,use_exp2=exp2,
              state_v_first=vfirst,cu_seqlens=bounds if packed else None)
    def call():return npu_chunk_fwd_h(kn,wn,un,**opts)
    check(key+'-before');print('ROUTE_BEGIN',key,flush=True)
    timer=threading.Timer(90,lambda:os._exit(124));timer.daemon=True;timer.start()
    actual=call();torch.npu.synchronize();timer.cancel()
    rec=dict(case=key,expected_fast=a.suite!='fallback',results={})
    for name,got,ref in [('H',actual[0],eh),('Vnew',actual[1],ev)]+([('final',actual[2],ef)] if final else []):
        value=got.cpu();rec['results'][name]=metrics(value,ref)
        assert_close(name,ref,value,.005)
        if not bfstate:torch.testing.assert_close(value,ref,atol=3e-6,rtol=1e-4)
    if a.profile:
        for _ in range(10):call()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/key))) as prof:
            for _ in range(10):call();torch.npu.synchronize();prof.step()
        files=list((a.output/key).rglob('kernel_details.csv'));assert len(files)==1
        groups={}
        for row in csv.DictReader(files[0].open()):groups.setdefault(row['Step Id'],[]).append(row)
        assert len(groups)==10
        values=[sum(float(row['Duration(us)']) for row in rows) for rows in groups.values()]
        rec['performance']=dict(median_us=statistics.median(values),samples=values,csv=str(files[0]),
            kernels=sorted({row['Name'] for rows in groups.values() for row in rows}))
    check(key+'-after');records.append(rec)
    (a.output/'results.json').write_text(json.dumps(records,indent=2))
    print('ROUTE_PASS',key,rec.get('performance',{}).get('median_us'),flush=True)
