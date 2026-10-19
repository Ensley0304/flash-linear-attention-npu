"""Validate the frozen CPU contract before implementing a device kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import torch

OP = Path(__file__).resolve().parents[1]
REPO = OP.parents[1]
ATK = REPO / 'tests/atk/chunk_kda_bwd_finalize'
sys.path.insert(0, str(ATK))
from reference import NAMES, build_inputs, reference

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True, type=Path)
parser.add_argument('--report', required=True, type=Path)
parser.add_argument('--cases', default='0,4,7,23,39,47,55,63,71,87,95,103,119,135,143,159,175,191,192,199')
args = parser.parse_args()
torch.set_num_threads(1)
old = runpy.run_path(str(args.baseline))
manifest = json.loads((OP / 'docs/baseline.json').read_text())
for name, expected in manifest['assets'].items():
    content = (REPO / name).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(content).hexdigest() == expected, name
assert args.baseline.read_text() == (ATK / 'reference.py').read_text()
profiles = json.loads((ATK / 'case_matrix.json').read_text())
report = {'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
          'canonical_sha256': hashlib.sha256((ATK / 'reference.py').read_bytes()).hexdigest(),
          'purpose': 'CPU relocation parity and FP32-vs-FP64 calibration; not DUT acceptance',
          'cases': []}
args.report.parent.mkdir(parents=True, exist_ok=True)
for case_id in map(int, args.cases.split(',')):
    started = time.time()
    spec = profiles[case_id]
    data, _, chunks = build_inputs(spec, 20261013 + case_id)
    high = reference(data, spec, chunks, True)
    low = reference(data, spec, chunks, False)
    # Independently import the frozen source, including both precision paths.
    for precision, actual in ((True, high), (False, low)):
        expected = old['reference'](data, spec, chunks, precision)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), case_id
    outputs = {}
    for name, hi, lo in zip(NAMES, high, low):
        assert hi.shape == lo.shape and hi.dtype == lo.dtype
        assert torch.isfinite(hi).all() and torch.isfinite(lo).all(), (case_id, name)
        error = (lo.double() - hi.double()).abs()
        atol, rtol = ((.0156, .0156) if hi.dtype == torch.bfloat16 else (.0000153, .000977))
        matched = error <= atol + rtol * hi.double().abs()
        outputs[name] = {'shape': list(hi.shape), 'dtype': str(hi.dtype),
                         'range_abs': hi.double().abs().max().item(),
                         'max_abs': error.max().item(),
                         'matched_ratio_default_policy': matched.double().mean().item(),
                         'default_error_count': (~matched).sum().item(), 'frozen_parity': True}
    report['cases'].append({'case_id': case_id, 'spec': spec, 'outputs': outputs,
                            'elapsed_s': time.time() - started})
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'case_id': case_id, 'parity': 'PASS', 'elapsed_s': time.time() - started}), flush=True)
report['status'] = 'PASS'
args.report.write_text(json.dumps(report, indent=2) + '\n')
