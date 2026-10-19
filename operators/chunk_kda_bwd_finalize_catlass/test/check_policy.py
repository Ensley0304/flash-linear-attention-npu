"""Exercise the shared comparator with the frozen CPU quantization budget."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--skill-root', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
op = Path(__file__).resolve().parents[1]
atk = op.parents[1] / 'tests/atk/chunk_kda_bwd_finalize'
sys.path.insert(0, str(atk))
sys.path.insert(0, str(args.skill_root / 'skills/catlass-cpp-test/scripts'))
from compare_precision import compare_case
from reference import NAMES, build_inputs, reference

torch.set_num_threads(1)
profiles = json.loads((atk / 'case_matrix.json').read_text())
policy = json.loads((op / 'reference/precision-policy.json').read_text())
reports = []
for i, (rows, norm, mode) in enumerate(itertools.product((1, 7, 33, 64), (False, True), (0, 1, 2))):
    variable = i % 2 == 1
    spec = dict(profiles[0], T=rows, H=1, B=1, norm=norm, beta_mode=mode,
                lengths=[0, rows, 0] if variable else None,
                output_layout=('TND' if i % 4 == 1 else 'NTD') if variable else
                              ('BSND' if i % 4 == 2 else 'BNSD'))
    data, _, chunks = build_inputs(spec, 20261019 + i)
    actual = dict(zip(NAMES, reference(data, spec, chunks, False)))
    golden = dict(zip(NAMES, reference(data, spec, chunks, True)))
    masks = {name: np.ones(value.shape, dtype=bool) for name, value in golden.items()}
    # These one-head, one-chunk cases are entirely boundary regions. Requiring
    # 100% here is stricter than the global 99.9% fraction.
    expected = {name: {'shape': list(value.shape), 'dtype': str(value.dtype).split('.')[-1],
                       'required_regions': ['boundary_all']}
                for name, value in golden.items()}
    result = compare_case(actual, golden, case=f'cpu_calibration_{i}', policy=policy,
                          expected_outputs=expected, valid_masks=masks,
                          regions={name: {'boundary_all': mask} for name, mask in masks.items()})
    reports.append({'spec': spec, 'result': result})
    print(i, result['status'], flush=True)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps({'purpose': 'CPU policy calibration, not device acceptance',
                                  'pass': all(item['result']['pass'] for item in reports),
                                  'cases': reports}, indent=2) + '\n')
raise SystemExit(0 if all(item['result']['pass'] for item in reports) else 1)
