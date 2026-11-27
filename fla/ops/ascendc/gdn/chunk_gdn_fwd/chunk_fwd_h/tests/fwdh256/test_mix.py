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
from numerical_gate import metrics, recurrence

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',action='store_true');p.add_argument('--quick',action='store_true');args=p.parse_args()
torch.npu.set_device(0);torch.set_num_threads(4)
lib=ctypes.CDLL(args.library);run=lib.run_mix
run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_uint32]*6+[ctypes.c_void_p];run.restype=None
args.output.mkdir(parents=True,exist_ok=True);records=[]
cases=[(1,t,h,g,i,f) for t,h in ((65,1),(65,2),(256,1),(1024,1),(1024,2))
       for g in (1,2) for i,f in ((False,False),(True,True))]
if args.quick:cases=cases[:4]
for batch,tokens,heads,group,initial,final in cases:
    torch.manual_seed(1127)
    shape=(batch,heads,tokens,256)
    w=torch.randn(shape)*.01;u=torch.randn(shape)*.05;kg=torch.randn(shape)*.01
    gate=torch.randn(shape).abs()*-.005
    state=torch.randn(batch,heads,256,256)*.1 if initial else torch.zeros(batch,heads,256,256)
    chunks=(tokens+63)//64
    h_expected=torch.empty(batch,chunks,heads,256,256)
    v_expected=torch.empty_like(u);f_expected=torch.empty_like(state)
    for b in range(batch):
        for h in range(heads):
            trace=recurrence(w[b,h],u[b,h],kg[b,h],gate[b,h],state[b,h],'FP64')
            for c,r in enumerate(trace):
                h_expected[b,c,h]=r['H_snapshot'];v_expected[b,h,c*64:c*64+len(r['Vnew'])]=r['Vnew']
            f_expected[b,h]=trace[-1]['R_run']
    blocks=min(20,batch*((heads+group-1)//group)*4)
    hn=torch.full(h_expected.shape,-937.25,device='npu')
    vn=torch.full_like(u,-937.25,device='npu');fn=torch.full_like(state,-937.25,device='npu')
    scratch=torch.empty(blocks*group*24576,device='npu')
    inputs=[x.npu() for x in (w,u,kg,gate,state)]
    ptrs=(ctypes.c_void_p*9)(*[x.data_ptr() for x in inputs[:4]],
         inputs[4].data_ptr() if initial else 0,hn.data_ptr(),vn.data_ptr(),
         fn.data_ptr() if final else 0,scratch.data_ptr())
    def call():run(ptrs,batch,heads,tokens,group,0,blocks,torch.npu.current_stream().npu_stream)
    key=f'b{batch}-t{tokens}-h{heads}-gh{group}-i{int(initial)}-f{int(final)}'
    print('LAUNCH',key,flush=True)
    # Runtime calls may block in C; a Python signal handler cannot reliably
    # interrupt them. Terminate this dedicated test process after 60 seconds.
    watchdog=threading.Timer(60,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    call();torch.npu.synchronize();watchdog.cancel()
    rec=dict(case=key,blocks=blocks,results={})
    for name,actual,expected in [('H_snapshot',hn,h_expected),('Vnew',vn,v_expected)]+([('H_final',fn,f_expected)] if final else []):
        value=actual.cpu();result=metrics(value,expected);rec['results'][name]=result
        torch.testing.assert_close(value,expected,atol=3e-6,rtol=1e-4)
        assert result['nonfinite']==0
    if args.profile:
        for _ in range(10):call()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output/key))) as prof:
            for _ in range(10):call();torch.npu.synchronize();prof.step()
        files=list((args.output/key).rglob('kernel_details.csv'));assert len(files)==1
        groups={}
        rows=list(csv.DictReader(files[0].open()))
        # Direct runtime launches have no PyTorch op/Step flow. The probe call
        # consists of exactly one kernel; require all 10 tasks, never filter.
        if rows and 'Step Id' not in rows[0]:
            assert len(rows)==10 and all('fwdh256_mix_probe' in r['Name'] for r in rows)
            groups={i:[r] for i,r in enumerate(rows)}
        else:
            for r in rows:groups.setdefault(r['Step Id'],[]).append(r)
        samples=[sum(float(r['Duration(us)']) for r in rows) for rows in groups.values()]
        rec['performance']=dict(samples=samples,median_us=statistics.median(samples),csv=str(files[0]))
    records.append(rec);(args.output/'results.json').write_text(json.dumps(records,indent=2))
    print('PASS',key,rec.get('performance',{}).get('median_us'),flush=True)
