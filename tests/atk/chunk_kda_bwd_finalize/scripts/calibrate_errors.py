"""Check that fixed input range and CT L1 reject representative wrong outputs."""
import json
import sys
from pathlib import Path

import ct
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_chunk_kda_bwd_finalize import PROFILES
from reference import NAMES, build_inputs, reference

torch.set_num_threads(1)
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
spec = PROFILES[70]
data, _, chunks = build_inputs(spec, 20261083)
gold = reference(data, spec, chunks, True)
bench = reference(data, spec, chunks, False)
records = []
for name, good, wanted in zip(NAMES, bench, gold):
    for mutation, value in [('correct', good), ('zero', torch.zeros_like(good)),
                             ('sign', -good), ('half', good * .5)]:
        result = ct.dual(value, wanted, good, level='L1', dtype=str(value.dtype).split('.')[-1],
                         out_dir=str(root / f'{name}-{mutation}'),
                         save_chart='--charts' in sys.argv[2:] and mutation in ('correct', 'zero'))
        passed = bool(result['success'])
        records.append(dict(output=name, mutation=mutation, passed=passed, ratios=result['ratios']))
        assert passed == (mutation == 'correct'), (name, mutation, result)
(root / 'calibration.json').write_text(json.dumps(records, indent=2))
print('CALIBRATION PASS: 7 correct outputs accepted; 21 wrong outputs rejected')
