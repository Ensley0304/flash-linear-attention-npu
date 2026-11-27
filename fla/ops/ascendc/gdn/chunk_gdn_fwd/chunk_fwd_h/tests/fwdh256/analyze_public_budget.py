"""Account for every device task; exclude only explicitly identified FwdH rows."""
import argparse
import csv
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('candidate', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
candidate = json.loads(a.candidate.read_text())
reference = json.loads(a.reference.read_text())
result = {}
for phase, summary in candidate.items():
    steps = {}
    for row in csv.DictReader(Path(summary['csv']).open()):
        group = steps.setdefault(row['Step Id'], {})
        group[row['Name']] = group.get(row['Name'], 0.) + float(row['Duration(us)'])
    assert len(steps) == 10
    names = sorted({name for group in steps.values() for name in group})
    fwdh = [name for name in names if 'chunkfwdh' in name.lower().replace('_', '')]
    if phase == 'forward':
        assert fwdh, f'FwdH identification failed: {names}'
    other = [sum(value for name, value in group.items() if name not in fwdh)
             for group in steps.values()]
    result[phase] = dict(
        public_median_us=summary['median_us'],
        reference_median_us=reference[phase]['median_us'],
        fwdh_names=fwdh,
        per_name_median_us={name: statistics.median(group.get(name, 0.) for group in steps.values())
                            for name in names},
        other_samples_us=other,
        other_median_us=statistics.median(other),
        fwdh_remaining_budget_us=(reference[phase]['median_us']-statistics.median(other)) if fwdh else None,
        public_margin_us=reference[phase]['median_us']-summary['median_us'],
        note='All CSV rows included. Per-name medians are descriptive; total uses per-step sums.')
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
