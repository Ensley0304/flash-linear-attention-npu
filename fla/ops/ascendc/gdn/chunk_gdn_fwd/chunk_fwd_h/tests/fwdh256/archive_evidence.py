"""Archive small raw evidence without simulator device dumps or profiler databases."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();root=a.root.resolve();selected=[]
for path in (root/'logs').rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(root)
    if any(x in rel.parts for x in ('tmp_dump','device0','device_0','simulator','FRAMEWORK','sqlite')):continue
    if path.name in ('kernel_details.csv','summary.json','manifest.json','results.json') or (
            path.parent==root/'logs' and path.suffix in ('.json','.txt','.log')) or (
            path.suffix in ('.json','.txt') and 'ASCEND_PROFILER_OUTPUT' not in rel.parts):
        selected.append(path)
selected += [p for p in root.glob('*.py') if p.is_file()]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(a.output,'w:gz') as tar:
    for path in selected+[root/'evidence-manifest.json']:tar.add(path,arcname=str(path.relative_to(root)))
print(len(selected),hashlib.sha256(a.output.read_bytes()).hexdigest())
