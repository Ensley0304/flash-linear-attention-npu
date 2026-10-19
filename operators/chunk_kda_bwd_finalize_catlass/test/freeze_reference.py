"""Freeze CPU-derived policy before any new DUT result is available."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--reports', nargs='+', required=True, type=Path)
parser.add_argument('--autograd', required=True, type=Path)
args = parser.parse_args()
op = Path(__file__).resolve().parents[1]
cases = {}
for path in args.reports:
    for case in json.loads(path.read_text())['cases']:
        assert case['case_id'] not in cases
        assert all(value['frozen_parity'] for value in case['outputs'].values())
        cases[case['case_id']] = case
required = {0, 4, 7, 23, 39, 47, 55, 63, 71, 87, 95, 103, 119, 135, 143, 159, 175, 191, 192, 199}
assert set(cases) == required, sorted(required - cases.keys())
autograd = json.loads(args.autograd.read_text())
assert len(autograd) == 24 and all(case['autograd'] == 'PASS' for case in autograd)
policy_path = op / 'reference/precision-policy.json'
policy = json.loads(policy_path.read_text())
calibration = {}
for dtype in ('bfloat16', 'float32'):
    values = [output for case in cases.values() for output in case['outputs'].values()
              if output['dtype'] == 'torch.' + dtype]
    peak_error = max(value['max_abs'] for value in values)
    peak_value = max(value['range_abs'] for value in values)
    # Two times the observed CPU quantization budget, rounded upwards to a
    # binary power. CT L1 against the unchanged FP32 benchmark is mandatory;
    # this separate gate is not allowed to replace it.
    bound = 2 ** math.ceil(math.log2(max(2 * peak_error, 1e-12)))
    config = policy['dtype'][dtype]
    config['max_abs_limit'] = max(config['atol'], bound)
    if dtype == 'float32':
        config['atol'] = max(config['atol'], bound)
    calibration[dtype] = {'peak_cpu_error': peak_error, 'peak_cpu_value': peak_value,
                          'round_up_twice_error': bound, 'policy': dict(config)}
policy['calibration'] = {'source': 'CPU FP32 benchmark versus CPU FP64 golden, before DUT',
                         'case_ids': sorted(required), 'seed_round': 0,
                         'mandatory_additional_gate': 'Original CT L1, all seven outputs',
                         'details': calibration}
policy_path.write_text(json.dumps(policy, indent=2) + '\n')
evidence = op / 'docs/evidence'
evidence.mkdir(exist_ok=True)
(evidence / 'reference20.json').write_text(json.dumps({'status': 'PASS', 'cases': list(cases.values())}, indent=2) + '\n')
(evidence / 'autograd24.json').write_text(json.dumps(autograd, indent=2) + '\n')
print(json.dumps(calibration, indent=2))
# State is deliberately not advanced here: review the generated policy and
# run the shared comparator before freezing the golden contract.
