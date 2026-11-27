"""Compare generated old/new kernel mappings, including core type and task ratio."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def read(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if '/kernel/ascend910b/chunk_fwd_h/' not in name or not name.endswith('.json'):continue
            item=json.loads(archive.read(name))
            obj=name[:-5]+'.o'
            result[Path(name).name]=dict(metadata=item,
                object_sha256=hashlib.sha256(archive.read(obj)).hexdigest())
    assert result
    return result
old,new=read(a.old),read(a.new);assert old.keys()==new.keys()
records=[]
for name in old:
    before={item['tilingKey']:item for item in old[name]['metadata']['kernelList']}
    after={item['tilingKey']:item for item in new[name]['metadata']['kernelList']}
    assert before.keys()<=after.keys()
    for key,entry in before.items():
        for field in ('kernelType','taskRation','crossCoreSync','kernelName'):
            assert entry.get(field)==after[key].get(field),(name,key,field,entry,after[key])
    added=sorted(after.keys()-before.keys());assert added==[68126,69150,76318,77342]
    for key in added:
        assert after[key]['kernelType']=='MIX_AIC' and after[key]['taskRation']=='1:2'
    records.append(dict(metadata_file=name,unchanged_old_keys=len(before),new_keys=[after[k] for k in added],
        old_object_sha256=old[name]['object_sha256'],new_object_sha256=new[name]['object_sha256']))
a.output.write_text(json.dumps(dict(status='passed',records=records),indent=2))
print(json.dumps(records,indent=2))
