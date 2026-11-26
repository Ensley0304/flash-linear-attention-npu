"""Measure primitive cost, not end-to-end KDA speedup."""
import argparse
import ctypes as C
from pathlib import Path
import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
lib = C.CDLL(str(Path(__file__).parent / 'build/libkda_fp32_cube.so'))
lib.run_fp32_mixed.argtypes = [C.c_void_p] * 4 + [C.c_uint32] * 5 + [C.c_void_p]
lib.run_fp32_mixed.restype = None
for m, n, k in ((32, 32, 32), (32, 128, 32), (32, 128, 128)):
    for repeats in (1, 67):
        a = torch.randn(repeats, m, k, device='npu')
        b = torch.randn(repeats, k, n, device='npu')
        scratch = torch.empty(m, n, device='npu')
        output = torch.empty(repeats, m, n, device='npu')
        def call():
            lib.run_fp32_mixed(a.data_ptr(), b.data_ptr(), scratch.data_ptr(), output.data_ptr(),
                              m, n, k, repeats, 0, torch.npu.current_stream().npu_stream)
        for _ in range(2):
            call()
        torch.npu.synchronize()
        name = f'm{m}_n{n}_k{k}_r{repeats}'
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / name)),
        ) as prof:
            for _ in range(10):
                call()
                torch.npu.synchronize()
                prof.step()
        print(name, 'captured', flush=True)
