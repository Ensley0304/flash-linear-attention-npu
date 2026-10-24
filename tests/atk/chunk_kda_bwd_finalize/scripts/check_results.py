"""Require all 200 cases, seven outputs, and three execution roles per seed."""
import ast
import json
import re
import sys
from pathlib import Path

import openpyxl

root = Path(sys.argv[1])
records = []
for seed_round in range(3):
    folder = root / f'round-{seed_round}'
    reports = list(folder.glob('atk_output/*/report/*.xlsx'))
    if len(reports) != 1:
        raise RuntimeError(f'{folder}: expected exactly one completed ATK report')
    workbook = openpyxl.load_workbook(reports[0], read_only=True)
    rows = workbook['statistic'].iter_rows(values_only=True)
    header = next(rows)
    cases = [dict(zip(header, row)) for row in rows]
    assert len(cases) == 200 and {row['编号'] for row in cases} == set(range(200))
    log = (root / f'round-{seed_round}.log').read_text(errors='replace')
    roles = re.findall(r'FINALIZE_ROLE case=(\d+) seed=(\d+) backend=(\w+) golden=(\w+)', log)
    expected = {(str(i), str(20261013 + i + seed_round * 10000), backend, golden)
                for i in range(200) for backend, golden in
                [('npu', 'False'), ('cpu', 'False'), ('cpu', 'True')]}
    assert len(roles) == 600 and set(roles) == expected, 'missing or repeated execution role'
    for row in cases:
        details = ast.literal_eval(row['精度详情'])['cpu_cpu_ref']
        assert len(details) == 7
        assert {entry['filename'] for entry in details} == {f'output_{i}.pt' for i in range(7)}
        passed = row['运行结果'] == 'SUCCESS' and row['cpu_cpu_ref_精度通过'] is True
        passed = passed and all(entry['result'] is True for entry in details)
        records.append(dict(seed_round=seed_round, case_id=row['编号'], passed=passed,
                            outputs=details))
    workbook.close()
summary = dict(total=len(records), passed=sum(row['passed'] for row in records),
               failed=[dict(seed_round=row['seed_round'], case_id=row['case_id'])
                       for row in records if not row['passed']], cases=records)
(root / 'accuracy_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Accuracy: {summary['passed']}/{summary['total']}; failed={summary['failed']}")
sys.exit(0 if summary['passed'] == 600 else 1)
