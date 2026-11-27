"""Check observed Host routes in a verbose public smoke/fallback log."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('log', type=Path)
p.add_argument('--fast', type=int, choices=(0, 1), required=True)
p.add_argument('--cases', type=int, required=True)
p.add_argument('--aic-cores', type=int, default=20, help='Physical AIC count of the measured device')
p.add_argument('--chain', action='store_true', help='forward/recompute trace markers; Host tiling may be cached')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
current = None
pattern = re.compile(r'FwdH route p32K256=(\d+) general=(\d+) key=(\d+) block=(\d+) B=(\d+) H=(\d+) T=(\d+) exp2=(\d+) initial=(\d+) final=(\d+)')
for line in a.log.read_text(errors='replace').splitlines():
    chain_begin = a.chain and line in ('CHAIN_FORWARD_BEGIN','CHAIN_BACKWARD_BEGIN')
    if line.startswith(('ROUTE_BEGIN ', 'GDN_ROUTE_BEGIN ')) or chain_begin:
        assert current is None, 'Previous case did not pass'
        current = dict(case=line if chain_begin else line.split()[1], routes=[], launches=[])
    match = pattern.search(line)
    if match and current is not None:
        row = dict(zip(('fast','general','key','blocks','batch','heads','tokens','exp2','initial','final'), map(int,match.groups())))
        assert row['fast'] == a.fast, row
        if a.fast:
            assert row['key'] == {(0,0):68126,(1,0):69150,(0,1):76318,(1,1):77342}[(row['exp2'],row['final'])], row
            assert row['general'] == 0 and row['blocks'] == min(row['batch']*row['heads']*4,a.aic_cores), row
        else:
            assert row['key'] < 65536, row
        current['routes'].append(row)
    if current is not None and 'StreamLaunchKernelV2:' in line:
        launched = re.search(r'kernel_name=(ChunkFwdH_[a-f0-9]+_(\d+)),.*coreDim=(\d+), taskRation=(\d+)', line)
        if launched:
            name,key,blocks,ratio = launched.groups()
            if a.fast:
                assert 'KERNEL_ATTR_TYPE_MIX(3)' in line and int(ratio) == 2, line
                assert int(key) in (68126,69150,76318,77342), line
            else:
                assert int(key) < 65536, line
            current['launches'].append(dict(kernel=name,key=int(key),blocks=int(blocks),task_ratio=int(ratio)))
    if line.startswith(('ROUTE_PASS ', 'GDN_ROUTE_PASS ')) or (a.chain and line in ('CHAIN_FORWARD_PASS','CHAIN_BACKWARD_PASS')):
        assert current and current['launches'], 'No actual kernel launch observed'
        assert a.chain or current['routes'], 'No actual Host route observed'
        assert {(r['key'],r['blocks']) for r in current['routes']} <= {
            (r['key'],r['blocks']) for r in current['launches']}, current
        rows.append(current)
        current = None
assert current is None and len(rows) == a.cases, (current,len(rows),a.cases)
a.output.write_text(json.dumps(rows,indent=2))
print('ACTUAL_ROUTES_PASSED',len(rows))
