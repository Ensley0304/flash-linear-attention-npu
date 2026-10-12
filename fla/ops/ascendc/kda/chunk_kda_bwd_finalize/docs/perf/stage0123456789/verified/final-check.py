import json,subprocess,time
from pathlib import Path
root=Path('stage9');results=[]
test='examples/fast_kernel_launch_example/tests/kda_finalize_stage01/check_stage01.py'
while not (root/'build-r2.status').exists():time.sleep(2)
jobs=[]
for norm in (False,True):
 for mode in ('cumulative','flat','extreme'):
  jobs.append((f'r2-tails-{mode}-norm{int(norm)}',9,'tails',mode,norm,False))
for stage in range(9):jobs.append((f'r2-regression-{stage}',stage,'smoke','cumulative',False,False))
for norm in (False,True):jobs.append((f'r2-perf-tails-norm{int(norm)}',9,'tails','cumulative',norm,True))
for name,stage,suite,mode,norm,perf in jobs:
 cmd=['python',test,'--library',f'stage9/r2-{"perf" if perf else "capture"}.so','--device','3',
      '--stage',str(stage),'--suite',suite,'--gate-mode',mode,'--cores','1','--repeat','2' if stage==9 else '1']
 if norm:cmd+=['--q-norm']
 if perf:cmd+=['--workspace-only']
 with (root/(name+'.log')).open('w') as f:
  rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(name=name,rc=rc,command=cmd))
 (root/'accuracy-r2.json').write_text(json.dumps(results,indent=2))
 if rc:raise SystemExit(rc)
(root/'accuracy-r2.status').write_text('PASS\n')
