"""Run identical C32/C64 workloads and retain every device task in the report.

These are exploratory performance cases, not substitutes for fla-org accuracy
tests or a user-provided production workload list. No ATK dependency.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def summarize(directory, samples):
    result = {}
    for mode in ('backward_save', 'backward_recompute'):
        files = list((directory / mode).rglob('kernel_details.csv'))
        if len(files) != 1:
            raise RuntimeError(f'{mode}: expected one fresh capture, got {len(files)}')
        with files[0].open() as source:
            rows = sorted(csv.DictReader(source), key=lambda r: float(r['Start Time(us)']))
        if not rows or len(rows) % samples:
            raise RuntimeError(f'{mode}: incomplete task count {len(rows)}')
        count = len(rows) // samples
        groups = [rows[n * count:(n + 1) * count] for n in range(samples)]
        names = [row['Name'] for row in groups[0]]
        if any([row['Name'] for row in group] != names for group in groups):
            raise RuntimeError(f'{mode}: task sequences differ between calls')
        spans = [max(float(r['Start Time(us)']) + float(r['Duration(us)']) for r in group)
                 - float(group[0]['Start Time(us)']) for group in groups]
        sums = [sum(float(r['Duration(us)']) for r in group) for group in groups]
        result[mode] = dict(tasks_per_call=count, names=names, span_us=spans,
                            sum_us=sums, metric='sum of all device task durations',
                            median_us=statistics.median(sums),
                            min_us=min(sums), max_us=max(sums),
                            span_median_us=statistics.median(spans))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--only', help='comma-separated case names')
    args = parser.parse_args()
    cases = [(f'fp16_k{k}_v{v}', k, v, 65, 2, 'float16', False)
             for k, v in ((60, 60), (64, 64), (100, 100), (128, 128), (256, 256), (64, 32))]
    cases += [('bf16_128_state', 128, 128, 65, 2, 'bfloat16', False),
              ('bf16_128_no_state', 128, 128, 65, 2, 'bfloat16', True),
              ('bf16_128_long_no_state', 128, 128, 1024, 8, 'bfloat16', True)]
    # Establish the existing specialized C64 route before general-path cases.
    cases.sort(key=lambda c: not c[-1])
    if args.only:
        selected = set(args.only.split(','))
        if selected - {c[0] for c in cases}:
            parser.error('unknown case name')
        cases = [c for c in cases if c[0] in selected]
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    driver = Path(__file__).with_name('profile_public_c32.py')
    for name, keys, values, tokens, heads, dtype, no_state in cases:
        results[name] = dict(keys=keys, values=values, tokens=tokens, heads=heads,
                             dtype=dtype, no_state=no_state, raw_safe=no_state)
        # Alternate order to avoid always measuring C32 first.
        for chunk in ((64, 32) if len(results) % 2 else (32, 64)):
            output = args.output / name / f'c{chunk}'
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(driver), '--output', str(output),
                       '--chunk-size', str(chunk), '--samples', str(args.samples),
                       '--keys', str(keys), '--values', str(values), '--tokens', str(tokens),
                       '--heads', str(heads), '--dtype', dtype, '--backward-only']
            if no_state:
                # The existing BF16/128 specialized C64 backward requires raw
                # safe gate. Give both chunk sizes the same legal inputs.
                command.extend(('--no-state', '--raw-safe'))
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = summarize(output, args.samples)
            if chunk == 32 and any(v['tasks_per_call'] != 1 for v in result.values()):
                raise RuntimeError('C32 single-kernel contract failed')
            results[name][f'c{chunk}'] = result
        ratios = {mode: results[name]['c64'][mode]['median_us'] / results[name]['c32'][mode]['median_us']
                  for mode in ('backward_save', 'backward_recompute')}
        results[name]['speedup'] = ratios
        print(name, ratios, flush=True)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
