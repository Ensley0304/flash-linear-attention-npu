"""Direct C++ Stage 0 test, using snapshots from the unique CPU reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import ct

parser = argparse.ArgumentParser()
parser.add_argument('--executable', type=Path, required=True)
parser.add_argument('--skill-root', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--prefix', type=int, choices=range(7), default=0)
args = parser.parse_args()
op = Path(__file__).resolve().parents[1]
atk = op.parents[1] / 'tests/atk/chunk_kda_bwd_finalize'
sys.path.insert(0, str(atk))
sys.path.insert(0, str(args.skill_root / 'skills/catlass-cpp-test/scripts'))
from reference import build_inputs, reference, round_bfloat16
from compare_precision import compare_case

torch.set_num_threads(1)
policy = json.loads((op / 'reference/precision-policy.json').read_text())
base = json.loads((atk / 'case_matrix.json').read_text())[0]
results = []
args.artifacts.mkdir(parents=True, exist_ok=True)
for rows in (1, 7, 31, 32, 33, 63, 64):
    spec = dict(base, B=1, H=1, T=rows, norm=args.prefix >= 1 and rows in (33, 64),
                beta_mode=rows % 3, gate_mode='saturated' if rows == 64 else 'ordinary')
    data, _, chunks = build_inputs(spec, 20261019 + rows)
    high, low = [], []
    reference(data, spec, chunks, True, high)
    reference(data, spec, chunks, False, low)
    assert len(high) == len(low) == 1
    directory = args.artifacts / f'rows-{rows}'
    directory.mkdir(exist_ok=True)
    for name in ('v', 'akk', 'v_new', 'h', 'dh', 'dv_scan'):
        columns = 64 if name == 'akk' else 128
        size = 128 if name in ('h', 'dh') else 64
        padded = torch.zeros(size, columns, dtype=torch.bfloat16)
        value = data[name].reshape(-1, columns)
        padded[:value.shape[0]] = value
        (directory / f'{name}.bin').write_bytes(padded.view(torch.uint16).numpy().tobytes())
    if args.prefix >= 1:
        for name in ('q', 'k', 'gk', 'raw_g', 'dq_raw'):
            padded = torch.zeros(64, 128, dtype=data[name].dtype)
            padded[:rows] = data[name].reshape(rows, 128)
            binary = padded.view(torch.uint16) if padded.dtype == torch.bfloat16 else padded
            (directory / f'{name}.bin').write_bytes(binary.numpy().tobytes())
        for name in ('beta', 'q_rstd', 'k_rstd'):
            padded = torch.zeros(64)
            padded[:rows] = data.get(name, torch.ones(rows)).reshape(rows)
            (directory / f'{name}.bin').write_bytes(padded.numpy().tobytes())
        for name in ('a_log', 'dt_bias'):
            (directory / f'{name}.bin').write_bytes(data[name].contiguous().numpy().tobytes())
    completed = subprocess.run([str(args.executable), str(directory), str(rows), str(args.device), str(args.prefix)],
                               capture_output=True, text=True, timeout=60)
    (directory / 'host.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, (rows, completed.stderr)
    actual, golden, benchmark = {}, {}, {}
    names = ['s0_state_raw', 's0_dv_b', 's0_zv', 's0_dw']
    if args.prefix >= 1:
        names += ['s1_ke', 's1_exp', 's1_rh', 's1_g_last', 's1_gate_factor', 's1_raw_x']
    if args.prefix >= 2:
        names += ['s2_dkg', 's2_zw']
    if args.prefix >= 3:
        names += ['s3_zb', 's3_dv', 's3_db_v']
    if args.prefix >= 4:
        names += ['s4_tza']
    if args.prefix >= 5:
        names += ['s5_dq_base', 's5_dk_base', 's5_dg_base', 's5_db_base']
    if args.prefix >= 6:
        names += ['s6_dakk_raw']
    bf16_names = ('s0_dw', 's1_ke', 's3_zb', 's3_dv', 's4_tza')
    square_names = ('s0_zv', 's2_zw', 's3_zb', 's4_tza', 's6_dakk_raw')
    for name in names:
        columns = 1 if name in ('s3_db_v', 's5_db_base') else (64 if name in square_names else 128)
        dtype = torch.bfloat16 if name in bf16_names else torch.float32
        one_row = name in ('s1_rh', 's1_g_last')
        physical_rows = 1 if one_row else 64
        raw = (directory / f'{name}.bin').read_bytes()
        assert len(raw) == physical_rows * columns * (2 if dtype == torch.bfloat16 else 4)
        output = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(physical_rows, columns).clone()
        assert torch.isfinite(output).all(), (rows, name, 'finite')
        if not one_row:
            assert (output[rows:] == 0).all(), (rows, name, 'tail')
        if name in square_names:
            assert (output[:, rows:] == 0).all(), (rows, name, 'column tail')
        actual[name] = (output.reshape(128) if one_row else
                        output[:rows, :rows if name in square_names else columns].contiguous())
        if name in ('s3_db_v', 's5_db_base'):
            actual[name] = actual[name].reshape(rows)
        golden[name] = high[0]['values'][name]
        benchmark[name] = low[0]['values'][name]
        if dtype == torch.bfloat16:
            golden[name] = round_bfloat16(golden[name])
            benchmark[name] = benchmark[name].bfloat16()
    masks = {name: np.ones(value.shape, dtype=bool) for name, value in actual.items()}
    expected = {name: {'shape': list(value.shape),
                       'dtype': 'bfloat16' if name in bf16_names else 'float32',
                       'required_regions': ['boundary_all']} for name, value in golden.items()}
    shared = compare_case(actual, golden, case=f'stage0_rows_{rows}', policy=policy,
                          expected_outputs=expected, valid_masks=masks,
                          regions={name: {'boundary_all': mask} for name, mask in masks.items()})
    dual = {}
    for name in actual:
        comparison = ct.dual(actual[name], golden[name], benchmark[name], level='L1',
                             dtype=str(actual[name].dtype).split('.')[-1],
                             out_dir=str(directory / name))
        dual[name] = {'pass': bool(comparison['success']), 'ratios': comparison['ratios']}
    passed = shared['pass'] and all(value['pass'] for value in dual.values())
    results.append({'rows': rows, 'pass': passed, 'shared': shared, 'ct': dual})
    print(rows, 'PASS' if passed else 'FAIL', flush=True)
    (args.artifacts / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
assert all(value['pass'] for value in results)
