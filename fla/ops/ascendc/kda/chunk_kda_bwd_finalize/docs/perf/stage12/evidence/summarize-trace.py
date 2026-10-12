import json
from collections import Counter
from pathlib import Path
out=Path('/data/wys/kda-finalize-a2-stage9-20261012/stage12')
results=[]
for name in ('main','reduce'):
    path=next((out/('simulator-main-r4' if name=='main' else 'simulator-reduce-small-r4')).glob('*/simulator/trace.json'))
    events=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('ph')=='X']
    lanes={}
    for pid in sorted({e['pid'] for e in events}):
        lane=[e for e in events if e['pid']==pid]
        lanes[pid]=dict(max_end_us=max(e['ts']+e['dur'] for e in lane),instructions=len(lane),
                        mmads=sum(e['name']=='MMAD' for e in lane),pipes=dict(Counter(e['tid'] for e in lane)))
    if name=='main':
        assert len(lanes)==3,lanes
        assert sum(x['mmads'] for x in lanes.values())==192,lanes
        assert sum('VECTOR' in x['pipes'] for x in lanes.values())==2,lanes
    else:
        assert len(lanes)==1 and all(x['mmads']==0 for x in lanes.values()),lanes
        assert all('VECTOR' in x['pipes'] for x in lanes.values()),lanes
    results.append(dict(kernel=name,source=str(path),lanes=lanes))
(out/'simulator-summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
