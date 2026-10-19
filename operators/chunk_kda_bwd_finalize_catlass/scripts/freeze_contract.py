"""Register the previously approved interface and immutable baseline assets."""
import hashlib
import json
from pathlib import Path
import shutil

OP = Path(__file__).resolve().parents[1]
REPO = OP.parents[1]
ATK = REPO / 'tests/atk/chunk_kda_bwd_finalize'
WORKFLOW = REPO.parent / 'validation/workflow-review-20261019/cannbot/plugins-official/catlass-op-generator/catlass-cpp-generator'

for name in ('reference', 'test', 'op_host', 'op_kernel'):
    (OP / name).mkdir(exist_ok=True)
source = ATK / 'scripts/reference.py'
canonical = ATK / 'reference.py'
if not canonical.exists():
    shutil.copyfile(source, canonical)
    source.write_text('"""Compatibility import; mathematical source is ../reference.py."""\n'
                      'from pathlib import Path\nimport runpy\n'
                      'globals().update({k: v for k, v in runpy.run_path(str(Path(__file__).resolve().parents[1] / "reference.py")).items() if not k.startswith("__")})\n', encoding='utf-8')
(OP / 'reference/reference.py').write_text(
    '"""Forward to the unique repository CPU reference; no formulas here."""\n'
    'from pathlib import Path\nimport runpy\n'
    'globals().update({k: v for k, v in runpy.run_path(str(Path(__file__).resolve().parents[3] / "tests/atk/chunk_kda_bwd_finalize/reference.py")).items() if not k.startswith("__")})\n', encoding='utf-8')
for name in ('definition.template.json', 'precision-policy.json'):
    destination = OP / 'reference' / name
    if not destination.exists():
        shutil.copyfile(WORKFLOW / 'skills/catlass-cpp-reference/templates' / name, destination)
manifest = {
    'baseline_commit': '3d7f1c15',
    'baseline_tag': 'baseline/kda-finalize-a2-20261019',
    'baseline_library_sha256': '75c165084f40a2edbc4cb0f83872a7e408750015a1e178b6d79e2234bebd4748',
    'workflow_commit': '61518bc950fac84c953abe19332e43c24d4c1985',
    'repository_workflow_pr': 833,
    'repository_workflow_commit': 'c53f7811c0330fbf947a7ea14b4785c6c6b401ab',
    'catlass_commit': '0d78a73c192fc04741a4fd57c94080209424775c',
    'precision_required': 'All seven outputs: original CT L1 dual benchmark, FP64 golden and FP32 local matrix products; no threshold or seed change.',
    'seed_rounds': [0, 1, 2],
    'seed_formula': '20261013 + case_id + 10000 * seed_round',
    'acceptance_case_ids': list(range(200)),
    'target_fixed_case': {'B': 1, 'H': 96, 'K': 128, 'V': 128, 'T': [8192, 16384], 'dtype': 'bfloat16'},
    'primary_goal': 'At least 70% of historical A5 throughput at matched 8K flags; report fresh baseline A/B for all eight configurations.',
    'target_8k_no_norm_no_beta_us': 9604.066428571429,
    'assets': {},
}
for path in (canonical, ATK / 'case_matrix.json', ATK / 'atk_chunk_kda_bwd_finalize.json', ATK / 'gen_chunk_kda_bwd_finalize.py'):
    manifest['assets'][str(path.relative_to(REPO)).replace('\\', '/')] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
(OP / 'docs/baseline.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
state_path = OP / 'docs/workflow.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
assert state['stage'] == 'interface', 'Do not rewind an active workflow'
state.update(target_architecture='atlas_a2_a3', stage='reference', operator_contract='frozen')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print('Frozen interface; reference remains provisional until fresh CPU validation.')
