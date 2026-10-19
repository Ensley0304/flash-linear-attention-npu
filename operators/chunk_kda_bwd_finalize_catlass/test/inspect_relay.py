"""Locate the first divergent retained stage in a one-work-per-core diagnostic dump."""
import argparse
from pathlib import Path
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tests/atk/chunk_kda_bwd_finalize'))
from reference import build_inputs, reference

torch.set_num_threads(1)
spec = dict(B=1, H=3, T=128, lengths=[0, 1, 64, 0, 63], norm=True, beta_mode=2,
            output_layout='TND', lower_bound=-5.0, scale=0.08838834764831845, gate_mode='ordinary')
data, _, chunks = build_inputs(spec, 20261018)
capture = []
reference(data, spec, chunks, False, capture)
raw = (args.directory / 'debug_relay.bin').read_bytes()
stages = [('s0_state_raw', 0, 128, torch.float32), ('s0_dv_b', 32, 128, torch.float32),
          ('s1_ke', 80, 128, torch.bfloat16), ('s2_dkg', 96, 128, torch.float32),
          ('s2_zw', 128, 64, torch.float32), ('s3_zb', 144, 64, torch.bfloat16),
          ('s7_ak', 160, 64, torch.float32)]
for work, observation in enumerate(capture):
    rows = chunks[observation['chunk']][-1]
    for name, kib, columns, dtype in stages:
        start = work * 512 * 1024 + kib * 1024
        size = 64 * columns * (2 if dtype == torch.bfloat16 else 4)
        actual = torch.frombuffer(bytearray(raw[start:start + size]), dtype=dtype).reshape(64, columns)
        actual = actual[:rows, :rows if columns == 64 else columns].float()
        expected = observation['values'][name].to(dtype).float()
        error = (actual - expected).abs()
        significant = error > 1e-5 + 1e-4 * expected.abs()
        if significant.any():
            locations = significant.nonzero()
            print(work, name, 'max_abs', error.max().item(), 'count', significant.sum().item(),
                  'first', locations[0].tolist(), 'rows', locations[:, 0].unique().tolist(), flush=True)
