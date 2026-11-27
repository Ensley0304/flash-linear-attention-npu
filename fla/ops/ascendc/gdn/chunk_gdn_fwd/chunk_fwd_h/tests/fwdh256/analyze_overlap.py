"""Measure actual data/compute interval intersections; exclude flag/SPR waits."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();events=json.loads(a.trace.read_text())['traceEvents']
groups={}
for e in events:
    if e.get('ph')!='X':continue
    name=e['name'];pipe=e['tid'];core=e['pid']
    useful=(name=='MMAD' or name.startswith('LOAD_') or name=='FIX_L0C_TO_DST' or
            name.startswith('MOV_OUT_TO_') or name=='MOV_SRC_TO_DST_ALIGN' or
            name in ('VADDS','VMUL','VADD','VSUB','VMULS','VEXP','VBRCB'))
    if useful:groups.setdefault((core,pipe),[]).append((float(e['ts']),float(e['ts'])+float(e['dur']),name))

def overlap(left,right):
    total=0.;examples=[]
    for lo,hi,ln in left:
        for ro,rh,rn in right:
            begin,end=max(lo,ro),min(hi,rh)
            if end>begin:
                total+=end-begin
                if len(examples)<5:examples.append(dict(start_us=begin,end_us=end,left=ln,right=rn))
    return dict(intersection_us=total,examples=examples)

c='core0.cubecore0';v0='core0.veccore0';v1='core0.veccore1'
pairs=[((c,'CUBE'),(c,'MTE2')),((c,'CUBE'),(c,'MTE1')),((c,'CUBE'),(c,'FIXP')),
       ((v0,'VECTOR'),(v0,'MTE2')),((v0,'VECTOR'),(v0,'MTE3')),
       ((c,'CUBE'),(v0,'VECTOR')),((c,'CUBE'),(v1,'VECTOR'))]
out={f'{left[0]}/{left[1]} & {right[0]}/{right[1]}':overlap(groups.get(left,[]),groups.get(right,[]))
     for left,right in pairs}
a.output.write_text(json.dumps(dict(scope='simulator_instruction_intersections_not_hardware_latency',
    source=str(a.trace),instruction_counts={f'{k[0]}/{k[1]}':len(v) for k,v in groups.items()},pairs=out),indent=2))
for name,result in out.items():print(name,round(result['intersection_us'],6))
