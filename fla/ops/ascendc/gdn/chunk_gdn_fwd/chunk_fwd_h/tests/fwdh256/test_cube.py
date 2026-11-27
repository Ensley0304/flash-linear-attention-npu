import ctypes
import json
import os
import sys
from pathlib import Path
import torch
import torch_npu
from numerical_gate import metrics

torch.manual_seed(1127)
torch.npu.set_device(0)
lib = ctypes.CDLL(sys.argv[1])
run = lib.run_cube
run.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_uint32]*6 + [ctypes.c_void_p]
run.restype = None
records = []
for stage in ('S0', 'S2'):
    for length in (1, 7, 15, 16, 17, 63, 64):
        for support in ('K0', 'K1', 'all') if stage == 'S0' else ('all',):
            m, k, ta = (length, 256, 0) if stage == 'S0' else (64, length, 1)
            # Noncompact input rows ensure ND pitch is tested.
            a = torch.randn((length, 256))*0.07
            b = torch.randn((k, 256))*0.09
            if stage == 'S0':
                if support == 'K0': a[:, 128:] = 0
                if support == 'K1': a[:, :128] = 0
                expected = a.double() @ b[:, 128:192].double()
            else:
                expected = a[:, 64:128].double().T @ b[:, 128:192].double()
            an, bn = a.npu(), b.npu()
            out = torch.full((5, 64, 64), -937.25, device='npu')
            print('LAUNCH', stage, length, support, flush=True)
            run(an.data_ptr()+(64*4 if ta else 0), bn.data_ptr()+128*4, out.data_ptr(),
                m, k, 256, 256, ta, 5, torch.npu.current_stream().npu_stream)
            torch.npu.synchronize()
            for repeat in range(5):
                actual = out[repeat, :m].cpu().double()
                result = metrics(actual, expected)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                assert result['nonfinite'] == 0
                records.append(dict(stage=stage, length=length, support=support, repeat=repeat, **result))
Path(sys.argv[2]).write_text(json.dumps(dict(status='PASS', records=records), indent=2))
print('PASS', len(records), flush=True)
