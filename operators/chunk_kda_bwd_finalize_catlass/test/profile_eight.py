"""Two alternating A/B rounds, eight configurations, unchanged shared inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import struct
import subprocess

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--scratch-root', type=Path)
parser.add_argument('--binary-root', type=Path)
parser.add_argument('--device', type=int, default=3)
args = parser.parse_args()
root = Path('/data/wys/kda-finalize-a2-cannbot-20261019')
scratch = args.scratch_root or root
out = scratch / 'reports' / args.label
out.mkdir(parents=True, exist_ok=args.resume)
data_root = scratch / 'perf-inputs'
data_root.mkdir(exist_ok=True)
torch.set_num_threads(1)
for tokens in (8192, 16384):
    directory = data_root / str(tokens)
    directory.mkdir(exist_ok=True)
    if (directory / 'ready.json').exists():
        continue
    existing = root / 'perf-inputs' / str(tokens)
    if scratch != root and (existing / 'ready.json').exists():
        for file in existing.glob('*.bin'):
            if not file.name.startswith('output_') and file.name != 'configuration.bin':
                (directory / file.name).symlink_to(file)
        shutil.copyfile(existing / 'ready.json', directory / 'ready.json')
        continue
    if shutil.disk_usage(scratch).free < 16 * 1024**3:
        raise RuntimeError('Need 16 GiB free to prepare inputs and outputs')
    cache = Path('/data/wys/kda-finalize-a2-stage9-20261012/perf-finalize-20261013') / f'inputs-{tokens}.pt'
    data, cu, chunks = torch.load(cache, weights_only=False, mmap=True, map_location='cpu')
    for name, tensor in data.items():
        value = tensor.contiguous()
        if value.dtype == torch.bfloat16:
            value = value.view(torch.uint16)
        value.numpy().tofile(directory / f'{name}.bin')
    metadata = [(seq, first, rows, index) for index, (seq, _, first, rows) in enumerate(chunks)]
    (directory / 'metadata.bin').write_bytes(b''.join(struct.pack('<4Q', *row) for row in metadata))
    (directory / 'ready.json').write_text(json.dumps(dict(cache=str(cache), chunks=len(chunks), tokens=tokens)))
    del data
    print(f'inputs T={tokens} ready', flush=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

binaries = {label: (args.binary_root or root / 'build-r1') / name for label, name in [('A', args.baseline), ('B', args.candidate)]}
hashes = {label: digest(path) for label, path in binaries.items()}
records = json.loads((out / 'summary.json').read_text()) if args.resume else []
assert all(r['sha256'] == hashes[r['label']] and r.get('device', 3) == args.device for r in records)
completed = {(r['round'], r['T'], r['norm'], r['beta'], r['label']) for r in records}
assert len(completed) == len(records)
for round_id in range(2):
    for tokens in (8192, 16384):
        directory = data_root / str(tokens)
        for norm, beta in ((0, 0), (1, 0), (0, 1), (1, 1)):
            config = (1, 96, tokens, tokens // 64, 0, 0, norm, beta, 128 ** -.5, -5.)
            (directory / 'configuration.bin').write_bytes(struct.pack('<8Q2f', *config))
            for label in ('A', 'B') if round_id == 0 else ('B', 'A'):
                if (round_id, tokens, norm, beta, label) in completed:
                    continue
                name = f'r{round_id}-t{tokens}-n{norm}-b{beta}-{label}'
                base_name = name
                attempt = 0
                while (out / name).exists():
                    attempt += 1
                    name = f'{base_name}-retry{attempt}'
                exe = binaries[label]
                assert digest(exe) == hashes[label]
                application = f'{exe} {directory} {args.device} 20 20 20'
                with (out / f'{name}.log').open('w') as log:
                    subprocess.run(['msprof', f'--application={application}', f'--output={out / name}',
                                    '--aic-metrics=PipeUtilization', '--runtime-api=on'],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
                rows = [row for file in (out / name).rglob('op_summary_*.csv')
                        for row in csv.DictReader(file.open())]
                groups = {}
                for row in rows:
                    op = row['Op Name']
                    if 'FinalizeCatlass' in op:
                        groups.setdefault(op, []).append(row)
                assert len(groups) == 2, name
                measured = {}
                for op, items in groups.items():
                    assert len(items) == 40, (name, op, len(items))
                    values = [float(item['Task Duration(us)']) for item in items[-20:]]
                    measured[op] = dict(mean_us=statistics.mean(values), median_us=statistics.median(values),
                                        min_us=min(values), max_us=max(values), samples=20)
                    for key in ('aiv_vec_ratio', 'aiv_scalar_ratio', 'aic_mac_ratio', 'aic_mte2_ratio'):
                        measured[op][key] = statistics.mean(float(item[key]) for item in items[-20:])
                record = dict(device=args.device, round=round_id, T=tokens, norm=norm, beta=beta, label=label,
                              sha256=hashes[label], kernels=measured,
                              combined_us=sum(item['mean_us'] for item in measured.values()))
                records.append(record)
                (out / 'summary.json').write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)

gains = []
for tokens in (8192, 16384):
    for norm, beta in ((0, 0), (1, 0), (0, 1), (1, 1)):
        selected = [r for r in records if (r['T'], r['norm'], r['beta']) == (tokens, norm, beta)]
        times = {label: statistics.mean(r['combined_us'] for r in selected if r['label'] == label) for label in ('A', 'B')}
        gain = (times['A'] - times['B']) / times['A'] * 100
        gains.append(dict(T=tokens, norm=norm, beta=beta, baseline_us=times['A'], candidate_us=times['B'], gain_pct=gain))
(out / 'gate.json').write_text(json.dumps(dict(configurations=gains,
    pass_gate=min(g['gain_pct'] for g in gains) >= -.3 and statistics.mean(g['gain_pct'] for g in gains) > .5), indent=2))
