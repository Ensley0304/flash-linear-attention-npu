import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--perf', required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--executable', default='finalize')
parser.add_argument('--scratch-root', type=Path)
parser.add_argument('--binary-root', type=Path)
parser.add_argument('--device', type=int, default=3)
args = parser.parse_args()
root = Path('/data/wys/kda-finalize-a2-cannbot-20261019')
scratch = args.scratch_root or root
records = json.loads((scratch / 'reports' / args.perf / 'summary.json').read_text())
assert len(records) == 32
assert all(r.get('device', 3) == args.device for r in records)
gains = []
for round_id in range(2):
    for tokens in (8192, 16384):
        for norm, beta in ((0, 0), (1, 0), (0, 1), (1, 1)):
            selected = [r for r in records if (r['round'], r['T'], r['norm'], r['beta']) == (round_id, tokens, norm, beta)]
            assert len(selected) == 2
            times = {r['label']: r['combined_us'] for r in selected}
            gains.append((times['A'] - times['B']) / times['A'] * 100)
assert min(gains) >= -.3 and statistics.mean(gains) > .5, gains
print(json.dumps(dict(min_gain_pct=min(gains), mean_gain_pct=statistics.mean(gains))), flush=True)
subprocess.run([sys.executable, '-u', str(root / 'source-r1/operators/chunk_kda_bwd_finalize_catlass/test/check_full.py'),
    '--executable', str((args.binary_root or root / 'build-r1') / args.executable),
    '--skill-root', str(root / 'plugins-official/catlass-op-generator/catlass-cpp-generator'),
    '--artifacts', str(scratch / 'reports' / args.label), '--work-dir', str(scratch / 'precision-work'),
    '--reference-cache', str(root / 'reference-cache'), '--device', str(args.device), '--cores', '20', '--repeat', '3',
    '--cases', '0,4,7,23,39,47,55,63,71,87,95,103,119,135,143,159,175,191,192,199'], check=True)
