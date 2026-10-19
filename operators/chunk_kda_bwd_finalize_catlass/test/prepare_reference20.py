"""Prepare immutable CPU reference caches; this script never runs the NPU."""
import hashlib
import json
from pathlib import Path
import sys
import torch

root = Path('/data/wys/kda-finalize-a2-cannbot-20261019')
source = root / 'source-r1/tests/atk/chunk_kda_bwd_finalize'
sys.path.insert(0, str(source))
from reference import build_inputs, reference

torch.set_num_threads(1)
matrix = json.loads((source / 'case_matrix.json').read_text())
source_hash = hashlib.sha256((source / 'reference.py').read_bytes()).hexdigest()
cache = root / 'reference-cache'
cache.mkdir(exist_ok=True)
for case_id in (0, 4, 7, 23, 39, 47, 55, 63, 71, 87, 95, 103, 119, 135, 143, 159, 175, 191, 192, 199):
    spec = matrix[case_id]
    seed = 20261013 + case_id
    key = hashlib.sha256(json.dumps([source_hash, spec, seed], sort_keys=True).encode()).hexdigest()
    path = cache / (key + '.pt')
    if not path.exists():
        print(f'case={case_id} CPU START', flush=True)
        data, cu, chunks = build_inputs(spec, seed)
        high, low = reference(data, spec, chunks, True), reference(data, spec, chunks, False)
        torch.save(dict(key=key, high=high, low=low), path)
        del data, high, low
    print(f'case={case_id} CPU CACHE READY', flush=True)
