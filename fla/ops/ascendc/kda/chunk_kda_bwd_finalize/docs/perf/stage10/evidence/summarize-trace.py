import json
from collections import Counter
from pathlib import Path
out=Path('/data/wys/kda-finalize-a2-stage9-20261012/stage10')
path=next((out/'simulator-r1').glob('*/simulator/trace.json'))
events=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('ph')=='X']
lanes={}
for pid in sorted({e['pid'] for e in events}):
 lane=[e for e in events if e['pid']==pid]
 lanes[pid]={'max_end_us':max(e['ts']+e['dur'] for e in lane),'instructions':len(lane),'mmad':sum(e['name']=='MMAD' for e in lane),'pipes':dict(Counter(e['tid'] for e in lane))}
result={'source':str(path),'lanes':lanes,'production_mmad_expected':192}
assert len(lanes)==3,result
assert sum(lane['mmad'] for lane in lanes.values())==192,result
assert sum('VECTOR' in lane['pipes'] for lane in lanes.values())==2,result
(out/'simulator-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
