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
out = root / 'stage12'
tests = 'examples/fast_kernel_launch_example/tests/kda_finalize_stage01/'
p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('action', choices=['smoke', 'check', 'model', 'profile'])
p.add_argument('--library-tag')
p.add_argument('--baseline', action='store_true')
p.add_argument('--device',type=int,default=3)
args = p.parse_args()
tag = args.library_tag or args.tag
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

if args.action != 'profile':
    jobs = []
    if args.action == 'smoke':
        jobs = [(s, 'smoke', 'cumulative', n, m, False) for s in (11,12) for n in (False,True) for m in (0,1,2)]
    elif args.action == 'model':
        jobs = [(12, 'model', 'cumulative', n, 1, perf) for perf in (False,True) for n in (False,True)]
    else:
        jobs = [(12, 'tails', 'cumulative', n, m, perf) for perf in (False,True) for n in (False,True) for m in (0,1,2)]
        jobs += [(12, 'tails', g, n, 0, False) for g in ('flat','extreme') for n in (False,True)]
        jobs += [(s, 'smoke', 'cumulative', False, 0, False) for s in range(11)]
    for stage, suite, gate, norm, beta, perf in jobs:
        name = f'{args.tag}-{args.action}-s{stage}-{suite}-{gate}-n{int(norm)}-m{beta}-p{int(perf)}'
        cmd = ['python', tests+'check_stage01.py', '--library', f'stage12/{tag}-'+('perf' if perf else 'capture')+'.so',
               '--device',str(args.device),'--stage',str(stage),'--suite',suite,'--gate-mode',gate,'--beta-mode',str(beta),
               '--cores','0' if suite=='model' else '1','--repeat','1' if suite=='model' else '2',
               '--tail-golden-dir',f'stage12/{tag}-golden']
        if norm: cmd += ['--q-norm']
        if perf: cmd += ['--workspace-only']
        run(name,cmd)
else:
    for variable in (False,True):
        name=args.tag+'-profile-'+('varlen' if variable else 'fixed')
        folder=out/name
        stage='10' if args.baseline else '12'
        library='stage10/r1-perf.so' if args.baseline else f'stage12/{tag}-perf.so'
        cmd=['msprof','--output='+str(folder),'--ai-core=on','--aic-metrics=PipeUtilization','--task-time=on',
             '--ascendcl=off','python',tests+'bench_stage01.py','--library',library,'--device',str(args.device),'--stage',stage]
        if variable: cmd += ['--varlen']
        with (out/(name+'.log')).open('w') as f:
            rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
        paths=sorted(folder.rglob('op_summary_*.csv'))
        if not paths:
            prof=sorted(folder.glob('PROF_*'))[-1]
            with (out/(name+'-export.log')).open('w') as f:
                subprocess.run(['msprof','--export=on','--output='+str(prof)],stdout=f,stderr=subprocess.STDOUT,check=True)
            paths=sorted(folder.rglob('op_summary_*.csv'))
        assert paths, f'{name}: no op_summary; inspect application log before retrying'
        rows=list(csv.DictReader(paths[-1].open()))
        kernels={}
        durations=[]
        for kernel in (['Stage01Kernel'] if args.baseline else ['Stage01Kernel','Stage12Kernel']):
            selected=[r for r in rows if kernel in r.get('Op Name','')]
            assert len(selected)==30,(name,kernel,len(selected))
            metrics={}
            for key in selected[0]:
                if key=='Task Duration(us)' or key.endswith('_ratio') or key.endswith('_time(us)'):
                    try: vals=[float(r[key]) for r in selected[-10:]]
                    except ValueError: continue
                    metrics[key]=dict(min=min(vals),median=statistics.median(vals),max=max(vals))
            kernels[kernel]=metrics
            durations.append([float(r['Task Duration(us)']) for r in selected[-10:]])
        total=[sum(items) for items in zip(*durations)]
        results.append(dict(name=name,rc=rc,source=str(paths[-1]),samples=10,kernels=kernels,
                            total_us=dict(min=min(total),median=statistics.median(total),max=max(total)),command=cmd))
        (out/(args.tag+'-profile.json')).write_text(json.dumps(results,indent=2))
        print(name,results[-1]['total_us'],flush=True)
(out/(args.tag+'-'+args.action+'.status')).write_text('PASS\n')
