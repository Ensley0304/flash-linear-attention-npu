"""Native layout variants must be exactly equal after logical permutation."""
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_chunk_kda_bwd_finalize import run_npu
from gen_chunk_kda_bwd_finalize import PROFILES
from reference import NAMES, build_inputs

torch.npu.set_device(int(sys.argv[1]))
device = torch.device(f'npu:{sys.argv[1]}')
records = []
for index in (7, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183):
    spec = dict(PROFILES[index])
    data, cu, chunks = build_inputs(spec, 20261013 + index)
    original = run_npu(data, spec, cu, chunks, device)
    spec['output_layout'] = 'TND' if spec['lengths'] is not None else 'BSND'
    changed = run_npu(data, spec, cu, chunks, device)
    for n, left, right in zip(NAMES, original, changed):
        if n in ('dq', 'dk', 'dv'):
            left = left.transpose(-3, -2).contiguous()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    records.append(dict(case=index, outputs=7, bitwise='PASS'))
print(json.dumps(records, indent=2))

# Reject a layout from the other rank family before launching a kernel.
for index, invalid in ((0, 'TND'), (96, 'BSND')):
    spec = dict(PROFILES[index], output_layout=invalid)
    data, cu, chunks = build_inputs(spec, 20261013 + index)
    try:
        run_npu(data, spec, cu, chunks, device)
    except RuntimeError as exc:
        assert 'invalid output_layout' in str(exc), str(exc)
    else:
        raise AssertionError(f'{invalid} accepted for case {index}')
print('Invalid layout families rejected: PASS')

for index in (7, 103):
    spec = dict(PROFILES[index])
    data, cu, chunks = build_inputs(spec, 20261013 + index)
    explicit = run_npu(data, spec, cu, chunks, device)
    spec['output_layout'] = ''
    default = run_npu(data, spec, cu, chunks, device)
    for left, right in zip(explicit, default):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
print('Default layout remains BNSD/NTD: PASS')
