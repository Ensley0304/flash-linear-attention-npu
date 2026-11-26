"""Transport/resource gate only. Full KDA acceptance remains fla-org's tests."""
import ctypes
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

lib = ctypes.CDLL(str(Path(__file__).parent / 'build/libkda_saved_transport.so'))
run = lib.run_saved_io if os.environ.get("KDA_CHECK_IO") == "1" else lib.run_saved_transport
run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 5 + [ctypes.c_void_p]
run.restype = None
torch.npu.set_device(0)
acl = ctypes.CDLL("libascendcl.so")
acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
acl.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
assert acl.aclrtSetOpExecuteTimeOut(10) == 0
if os.environ.get("KDA_CHECK_HANDSHAKE") == "1":
    lib.run_saved_handshake.argtypes = [ctypes.c_void_p]
    lib.run_saved_handshake(torch.npu.current_stream().npu_stream)
    torch.npu.synchronize()
    print("HANDSHAKE PASS", flush=True)
    raise SystemExit(0)
torch.manual_seed(3203)
cases = []
for mode in range(4):
  for m, n, k in ((32, 32, 32), (32, 64, 128), (64, 64, 256), (16, 16, 8)):
    for count in (1, 2, 3, 7, 67):
        print('BEGIN', mode, m, n, k, count, flush=True)
        a, b = torch.randn(count, m, k), torch.randn(count, k, n)
        expected = (a.double() @ b.double()).float()
        if os.environ.get("KDA_CHECK_IO") == "1":
            expected.fill_(-937.25)
        da = (a.transpose(-1, -2).contiguous() if mode & 1 else a).npu()
        db = (b.transpose(-1, -2).contiguous() if mode & 2 else b).npu()
        scratch = torch.full((192 * 1024 // 4,), -937.25, device='npu')
        result = torch.full((count, m, n), float('nan'), device='npu')
        torch.npu.synchronize()
        print("H2D DONE", flush=True)
        run(da.data_ptr(), db.data_ptr(), scratch.data_ptr(), result.data_ptr(),
            m, n, k, count, mode, torch.npu.current_stream().npu_stream)
        print("LAUNCH DONE", flush=True)
        torch.npu.synchronize()
        actual = result.cpu()
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=2e-5)
        cases.append(dict(mode=mode, m=m, n=n, k=k, outputs=count,
                          packets=count * ((k + 127) // 128),
                          max_abs=(actual - expected).abs().max().item()))
print(json.dumps(dict(result='PASS', cases=cases)))
