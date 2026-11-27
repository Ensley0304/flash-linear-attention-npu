"""Launch a validation script and record actual loaded candidate libraries at exit."""
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

destination=Path(sys.argv[1]);script=sys.argv[2];arguments=sys.argv[3:]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def capture():
    result=dict(python=sys.executable,pid=os.getpid(),loaded_libraries={},package_files={})
    for name in ('torch','torch-npu','triton-ascend','flash-linear-attention-npu-a2'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    package=sys.modules.get('fla_npu')
    if package and getattr(package,'__file__',None):
        root=Path(package.__file__).parent;result['package_root']=str(root)
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.so','.o','.json'):
                result['package_files'][str(path.relative_to(root))]=digest(path)
    for row in Path('/proc/self/maps').read_text().splitlines():
        parts=row.split(maxsplit=5)
        if len(parts)<6:continue
        path=Path(parts[5])
        if path.is_file() and '.so' in path.name and ('fla_npu' in str(path) or 'libcust' in path.name):
            result['loaded_libraries'][str(path)]=digest(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2))
atexit.register(capture)
sys.argv=[script,*arguments]
runpy.run_path(script,run_name='__main__')
