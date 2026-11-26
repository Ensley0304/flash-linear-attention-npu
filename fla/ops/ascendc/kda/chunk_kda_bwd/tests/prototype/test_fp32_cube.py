"""Primitive feasibility only; fla-org remains the operator acceptance reference."""
import ctypes as C
import json
from pathlib import Path
import torch
import torch_npu

lib = C.CDLL(str(Path(__file__).parent / 'build/libkda_fp32_cube.so'))
lib.run_fp32_cube.argtypes = [C.c_void_p] * 3 + [C.c_uint32] * 5 + [C.c_void_p]
lib.run_fp32_cube.restype = None
torch.manual_seed(329)
results = []
for m, n, k in ((32, 32, 32), (32, 128, 32), (32, 32, 128), (32, 128, 128), (128, 128, 32)):
    for repeats in (1, 17):
        print(f'BEGIN m={m} n={n} k={k} repeats={repeats}', flush=True)
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        expected = (a.double() @ b.double()).float()
        da, db = a.npu(), b.npu()
        dc = torch.full((m, n), float('nan'), device='npu')
        lib.run_fp32_cube(da.data_ptr(), db.data_ptr(), dc.data_ptr(), m, n, k,
                          repeats, 0, torch.npu.current_stream().npu_stream)
        torch.npu.synchronize()
        actual = dc.cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        results.append(dict(m=m, n=n, k=k, repeats=repeats,
                            max_abs=float((actual - expected).abs().max())))
print(json.dumps(dict(result='PASS', hf32=False, cases=results)))

lib.run_fp32_mixed.argtypes = [C.c_void_p] * 4 + [C.c_uint32] * 5 + [C.c_void_p]
lib.run_fp32_mixed.restype = None
mixed_results = []
for m, n, k in ((32, 32, 32), (32, 128, 32), (32, 32, 128), (32, 128, 128), (128, 128, 32)):
    repeats = 67
    print(f'BEGIN mixed m={m} n={n} k={k} repeats={repeats}', flush=True)
    a = torch.randn(repeats, m, k)
    b = torch.randn(repeats, k, n)
    expected = (a.double() @ b.double()).float()
    da, db = a.npu(), b.npu()
    scratch = torch.full((m, n), -991.25, device='npu')
    output = torch.full((repeats, m, n), float('nan'), device='npu')
    lib.run_fp32_mixed(da.data_ptr(), db.data_ptr(), scratch.data_ptr(), output.data_ptr(),
                      m, n, k, repeats, 0, torch.npu.current_stream().npu_stream)
    torch.npu.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    mixed_results.append(dict(m=m, n=n, k=k, repeats=repeats,
                              max_abs=float((actual - expected).abs().max())))
print(json.dumps(dict(result='PASS', protocol='MIX_AIC_1_2 single result slot', cases=mixed_results)))

transpose_results = []
for mode in (1, 2, 3):
    for m, n, k in ((32, 32, 32), (32, 128, 32), (32, 32, 128), (32, 128, 128), (128, 128, 32)):
        repeats = 17
        print(f'BEGIN transpose mode={mode} m={m} n={n} k={k}', flush=True)
        a = torch.randn(repeats, m, k)
        b = torch.randn(repeats, k, n)
        expected = (a.double() @ b.double()).float()
        da = (a.transpose(-1, -2).contiguous() if mode & 1 else a).npu()
        db = (b.transpose(-1, -2).contiguous() if mode & 2 else b).npu()
        scratch = torch.full((m, n), -137.0, device='npu')
        output = torch.full((repeats, m, n), float('nan'), device='npu')
        lib.run_fp32_mixed(da.data_ptr(), db.data_ptr(), scratch.data_ptr(), output.data_ptr(),
                          m, n, k, repeats, mode, torch.npu.current_stream().npu_stream)
        torch.npu.synchronize()
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        transpose_results.append(dict(mode=mode, m=m, n=n, k=k,
                                      max_abs=float((actual - expected).abs().max())))
print(json.dumps(dict(result='PASS', transposes=transpose_results)))
