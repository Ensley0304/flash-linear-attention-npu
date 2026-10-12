import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path('/data/wys/kda-finalize-a2-stage9-20261012')
out=root/'stage10'
evidence=out/'evidence'
evidence.mkdir(exist_ok=True)
for p in out.iterdir():
 if p.suffix in ('.json','.log','.status','.py','.sh') and p.is_file():
  shutil.copy2(p,evidence/p.name)
for p in out.rglob('op_summary_*.csv'):
 if 'evidence' not in p.parts:
  kind=next(x for x in p.relative_to(out).parts if '-profile-' in x)
  shutil.copy2(p,evidence/(kind+'-op_summary.csv'))
files=[root/'fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22'/('chunk_kda_bwd_finalize_'+n+'.h') for n in ('common','cube','vector')]
files += [root/'examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp']
files += list((root/'examples/fast_kernel_launch_example/tests/kda_finalize_stage01').glob('*.py'))
files += [out/'r1-capture.so',out/'r1-perf.so',root/'bound-opt/r11-perf.so']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(evidence/'sha256.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(out/'evidence.tar.gz','w:gz') as tf:
 for p in sorted(evidence.iterdir()):tf.add(p,arcname=p.name)
print(len(list(evidence.iterdir())),(out/'evidence.tar.gz').stat().st_size)
