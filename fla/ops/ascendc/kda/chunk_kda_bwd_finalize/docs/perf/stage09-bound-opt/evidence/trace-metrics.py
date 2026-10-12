import json
from pathlib import Path
import sys

root = Path('/data/wys/kda-finalize-a2-stage9-20261012')
tag = sys.argv[1]
folder = root / ('stage9' if tag == 'r2' else 'bound-opt') / ('simulator-' + tag)
path = next(folder.glob('*/simulator/trace.json'))
events = [e for e in json.loads(path.read_text())['traceEvents'] if e.get('ph') == 'X']

def merged(events):
    intervals = []
    for lo, hi in sorted((float(e['ts']), float(e['ts']) + float(e['dur'])) for e in events):
        if intervals and lo <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], hi)
        else:
            intervals.append([lo, hi])
    return intervals

def length(intervals):
    return sum(hi - lo for lo, hi in intervals)

result = {'source': str(path), 'lanes': {}}
for pid in sorted({e['pid'] for e in events}):
    lane = [e for e in events if e['pid'] == pid]
    vectors = merged([e for e in lane if e['tid'] == 'VECTOR'])
    waits = merged([e for e in lane if 'WAIT_FLAG_DEV' in e['name']])
    covered = sum(max(0, min(a1, b1) - max(a0, b0)) for a0, a1 in waits for b0, b1 in vectors)
    result['lanes'][pid] = {
        'max_end_us': max(e['ts'] + e['dur'] for e in lane),
        'vector_union_us': length(vectors),
        'cross_wait_union_us': length(waits),
        'cross_wait_not_covered_by_vector_us': length(waits) - covered,
        'movemask_instructions': sum(e['name'] == 'MOVEMASK' for e in lane),
        'scalar_instructions': sum(e['tid'] == 'SCALAR' for e in lane),
    }
out = root / 'bound-opt' / ('trace-' + tag + '-metrics.json')
out.write_text(json.dumps(result, indent=2))
print(out.read_text())
