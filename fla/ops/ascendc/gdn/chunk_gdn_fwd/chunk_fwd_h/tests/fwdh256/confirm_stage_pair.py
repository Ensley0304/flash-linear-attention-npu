"""Adjacent probe/public measurements in one device context, with unchanged gates."""
import argparse
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser()
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
script=Path(__file__).with_name('profile_stage_pair.py')
keep_alive=[]
for label,implementation in [('probe-before','probe'),('public','ours'),('probe-after','probe')]:
    sys.argv=[str(script),'--implementation',implementation,'--inputs',str(a.inputs),
              '--output',str(a.output/label)]
    if implementation=='probe':sys.argv+=['--library',str(a.library.resolve())]
    keep_alive.append(runpy.run_path(str(script),run_name='__main__'))
