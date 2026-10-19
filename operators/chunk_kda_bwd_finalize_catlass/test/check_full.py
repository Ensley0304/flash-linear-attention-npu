"""Direct complete-kernel verification against the unchanged seven-output contract."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import ct
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--executable', type=Path, required=True)
parser.add_argument('--skill-root', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--cores', type=int, default=20)
parser.add_argument('--cases', default='smoke')
parser.add_argument('--seed', type=int, default=20261013)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--smoke-ids', default='0,1,2,3,4,5')
parser.add_argument('--work-dir', type=Path, help='Optional reusable binary input directory; reports remain per run')
parser.add_argument('--reference-cache', type=Path, help='Cache CPU results by source hash, spec and seed')
args = parser.parse_args()
op = Path(__file__).resolve().parents[1]
atk = op.parents[1] / 'tests/atk/chunk_kda_bwd_finalize'
sys.path.insert(0, str(atk))
sys.path.insert(0, str(args.skill_root / 'skills/catlass-cpp-test/scripts'))
from reference import NAMES, build_inputs, reference
from compare_precision import compare_case

torch.set_num_threads(1)
policy = json.loads((op / 'reference/precision-policy.json').read_text())
matrix = json.loads((atk / 'case_matrix.json').read_text())
if args.cases == 'smoke':
    cases = [
        dict(matrix[0], B=1, H=1, T=1, lengths=None, norm=False, beta_mode=0, output_layout='BNSD'),
        dict(matrix[0], B=1, H=3, T=65, lengths=None, norm=False, beta_mode=0, output_layout='BNSD'),
        dict(matrix[0], B=1, H=3, T=65, lengths=None, norm=False, beta_mode=1, output_layout='BSND'),
        dict(matrix[0], B=2, H=3, T=129, lengths=None, norm=True, beta_mode=2, output_layout='BSND'),
        dict(matrix[0], B=1, H=3, T=128, lengths=[0, 1, 64, 0, 63], norm=False, beta_mode=0, output_layout='NTD'),
        dict(matrix[0], B=1, H=3, T=128, lengths=[0, 1, 64, 0, 63], norm=True, beta_mode=2, output_layout='TND'),
    ]
    selected = [(i, cases[i]) for i in map(int, args.smoke_ids.split(','))]
else:
    indices = range(len(matrix)) if args.cases == 'all' else map(int, args.cases.split(','))
    selected = [(i, matrix[i]) for i in indices]
args.artifacts.mkdir(parents=True, exist_ok=True)
results = []
for case_id, spec in selected:
    started = time.time()
    print(f'case={case_id} START cores={args.cores}', flush=True)
    directory = args.work_dir if args.work_dir is not None else args.artifacts / f'case-{case_id}'
    directory.mkdir(parents=True, exist_ok=True)
    data, cu, chunks = build_inputs(spec, args.seed + case_id)
    variable = spec['lengths'] is not None
    layout = spec.get('output_layout', spec.get('outputLayout', 'NTD' if variable else 'BNSD'))
    token_major = layout in ('BSND', 'TND')
    batches = len(spec['lengths']) if variable else spec['B']
    config = (batches, spec['H'], spec['T'], len(chunks), int(variable), int(token_major),
              int(spec['norm']), spec['beta_mode'], spec['scale'], spec['lower_bound'])
    (directory / 'configuration.bin').write_bytes(struct.pack('<8Q2f', *config))
    metadata = [(seq, first, rows, index) for index, (seq, _, first, rows) in enumerate(chunks)]
    (directory / 'metadata.bin').write_bytes(np.asarray(metadata, dtype=np.uint64).tobytes())
    if variable:
        (directory / 'cu_seqlens.bin').write_bytes(np.asarray(cu, dtype=np.int64).tobytes())
    for name, value in data.items():
        binary = value.contiguous().view(torch.uint16) if value.dtype == torch.bfloat16 else value.contiguous()
        (directory / f'{name}.bin').write_bytes(binary.numpy().tobytes())
    launch_cores = min(args.cores, spec.get('cores', 0) or args.cores)
    completed = subprocess.run([str(args.executable), str(directory), str(args.device), str(launch_cores),
                                '0', str(args.repeat)], capture_output=True, text=True, timeout=60)
    (directory / 'host.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, (case_id, completed.stderr)
    print(f'case={case_id} NPU done; CPU golden', flush=True)
    reference_hash = hashlib.sha256((atk / 'reference.py').read_bytes()).hexdigest()
    cache_key = hashlib.sha256(json.dumps([reference_hash, spec, args.seed + case_id], sort_keys=True).encode()).hexdigest()
    cache_file = args.reference_cache / (cache_key + '.pt') if args.reference_cache else None
    if cache_file is not None and cache_file.exists():
        saved = torch.load(cache_file, weights_only=False, map_location='cpu')
        assert saved['key'] == cache_key
        high, low = saved['high'], saved['low']
    else:
        high = reference(data, spec, chunks, True)
        low = reference(data, spec, chunks, False)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(key=cache_key, high=high, low=low), cache_file)
    actual, golden, benchmark = {}, {}, {}
    for index, name in enumerate(NAMES):
        dtype = torch.bfloat16 if index < 3 else torch.float32
        raw = (directory / f'output_{name}.bin').read_bytes()
        value = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        expected = high[index]
        # The reference already returns the public layouts; compare the native bytes directly.
        actual[name] = value.reshape(expected.shape)
        golden[name], benchmark[name] = expected, low[index]
    dual = {}
    for name in NAMES:
        result = ct.dual(actual[name], golden[name], benchmark[name], level='L1',
                         dtype=str(actual[name].dtype).split('.')[-1])
        dual[name] = {'pass': bool(result['success']), 'ratios': result['ratios']}
    masks = {name: np.ones(value.shape, dtype=bool) for name, value in actual.items()}
    expected_outputs = {name: {'shape': list(value.shape), 'dtype': str(value.dtype).split('.')[-1],
                               'required_regions': ['all_valid']} for name, value in actual.items()}
    shared = compare_case(actual, golden, case=f'full_{case_id}', policy=policy,
                          expected_outputs=expected_outputs, valid_masks=masks,
                          regions={name: {'all_valid': mask} for name, mask in masks.items()})
    passed = shared['pass'] and all(item['pass'] for item in dual.values())
    results.append({'case_id': case_id, 'spec': spec, 'seed': args.seed + case_id,
                    'requested_cores': launch_cores, 'repeats': args.repeat, 'pass': passed,
                    'ct': dual, 'shared': shared, 'elapsed_s': time.time() - started,
                    'host': completed.stdout,
                    'binary_sha256': hashlib.sha256(args.executable.read_bytes()).hexdigest()})
    (args.artifacts / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'case={case_id} {"PASS" if passed else "FAIL"}', flush=True)
    if not passed:
        raise AssertionError(f'Full case {case_id} failed; stopping before larger cases')
