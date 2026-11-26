"""Device-task timing of the transport component, NOT full KDA backward."""
import argparse
import ctypes
import csv
import json
import statistics
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device(0)
torch.manual_seed(3203)
lib = ctypes.CDLL(str(Path(__file__).parent / 'build/libkda_saved_transport.so'))
run = lib.run_saved_transport
run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 5 + [ctypes.c_void_p]
run.restype = None
report = []
for m,n,k,count,mode in [(32,64,128,3,0), (32,64,128,67,0),
                         (64,64,256,3,3), (64,64,256,67,3)]:
    x = torch.randn(count,m,k)
    y = torch.randn(count,k,n)
    expected = (x.double() @ y.double()).float()
    x = (x.transpose(-1,-2).contiguous() if mode & 1 else x).npu()
    y = (y.transpose(-1,-2).contiguous() if mode & 2 else y).npu()
    scratch = torch.full((192*1024//4,), -937.25, device='npu')
    out = torch.empty(count,m,n,device='npu')
    def call():
        run(x.data_ptr(),y.data_ptr(),scratch.data_ptr(),out.data_ptr(),
            m,n,k,count,mode,torch.npu.current_stream().npu_stream)
    for _ in range(10): call()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(),expected,atol=5e-5,rtol=2e-5)
    dest = a.output / f'm{m}_n{n}_k{k}_count{count}_mode{mode}'
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(dest)),
    ):
        for _ in range(20):
            call()
            torch.npu.synchronize()
    files = list(dest.rglob('kernel_details.csv'))
    assert len(files)==1, files
    with files[0].open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==20, [(r['Name'],r['Duration(us)']) for r in rows]
    durations=[float(r['Duration(us)']) for r in rows]
    report.append(dict(m=m,n=n,k=k,count=count,transpose_mode=mode,
                       samples=20,median_us=statistics.median(durations),
                       min_us=min(durations),max_us=max(durations),
                       per_product_us=statistics.median(durations)/count,
                       task_names=sorted({r['Name'] for r in rows}),
                       duration_us=durations))
    print(json.dumps(report[-1]),flush=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
