import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

root = Path('/data/wys/kda-finalize-a2-stage9-20261012')
os.chdir(root)
out = root / 'bound-opt'
tests = 'examples/fast_kernel_launch_example/tests/kda_finalize_stage01/'
p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('action', choices=['smoke', 'check', 'profile'])
p.add_argument('--baseline', action='store_true')
p.add_argument('--library-tag', default='')
args = p.parse_args()
libdir = 'stage9' if args.baseline else 'bound-opt'
library_tag = args.library_tag or args.tag
results = []

def run(name, cmd):
    start = time.time()
    with (out / (name + '.log')).open('w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    results.append(dict(name=name, rc=rc, seconds=time.time()-start, command=cmd))
    (out / (args.tag+'-'+args.action+'.json')).write_text(json.dumps(results, indent=2))
    print(name, rc, round(time.time()-start, 2), flush=True)
    if rc:
        print((out / (name+'.log')).read_text()[-5000:], flush=True)
        raise SystemExit(rc)

if args.action in ('smoke', 'check'):
    jobs = []
    for norm in (False, True):
        for mode in (('cumulative',) if args.action == 'smoke' else ('cumulative', 'flat', 'extreme')):
            jobs.append((9, 'smoke' if args.action == 'smoke' else 'tails', mode, norm, False))
    if args.action == 'check':
        jobs += [(s, 'smoke', 'cumulative', False, False) for s in range(9)]
        jobs += [(9, 'tails', 'cumulative', n, True) for n in (False, True)]
    for stage, suite, mode, norm, perf in jobs:
        name = f'{args.tag}-{args.action}-s{stage}-{suite}-{mode}-n{int(norm)}-p{int(perf)}'
        cmd = ['python', tests+'check_stage01.py', '--library', f'{libdir}/{library_tag}-'+('perf' if perf else 'capture')+'.so',
               '--device', '3', '--stage', str(stage), '--suite', suite, '--gate-mode', mode, '--cores', '1', '--repeat', '2']
        if norm: cmd += ['--q-norm']
        if perf: cmd += ['--workspace-only']
        run(name, cmd)
else:
    for variable in (False, True):
        name = args.tag+'-profile-'+('varlen' if variable else 'fixed')
        folder = out / name
        cmd = ['msprof', '--output='+str(folder), '--ai-core=on', '--aic-metrics=PipeUtilization', '--task-time=on',
               '--ascendcl=off', 'python', tests+'bench_stage01.py', '--library', f'{libdir}/{library_tag}-perf.so',
               '--device', '3', '--stage', '9']
        if variable: cmd += ['--varlen']
        # Some msprof versions fail during automatic export; retain and export raw data.
        with (out / (name+'.log')).open('w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        paths = sorted(folder.rglob('op_summary_*.csv'))
        if not paths:
            prof = sorted(folder.glob('PROF_*'))[-1]
            with (out / (name+'-export.log')).open('w') as f:
                subprocess.run(['msprof', '--export=on', '--output='+str(prof)], stdout=f, stderr=subprocess.STDOUT, check=True)
            paths = sorted(folder.rglob('op_summary_*.csv'))
        with paths[-1].open() as f:
            rows = [r for r in csv.DictReader(f) if 'Stage01Kernel' in r.get('Op Name', '')]
        assert len(rows) == 30, (name, len(rows))
        metrics = {}
        for key in rows[0]:
            if key == 'Task Duration(us)' or key.endswith('_ratio') or key.endswith('_time(us)'):
                vals = [float(r[key]) for r in rows[-10:]]
                metrics[key] = dict(min=min(vals), median=statistics.median(vals), max=max(vals))
        results.append(dict(name=name, rc=rc, source=str(paths[-1]), samples=10, metrics=metrics, command=cmd))
        (out/(args.tag+'-profile.json')).write_text(json.dumps(results, indent=2))
        print(name, metrics['Task Duration(us)'], metrics['aiv_vec_ratio'], flush=True)
(out / (args.tag+'-'+args.action+'.status')).write_text('PASS\n')
