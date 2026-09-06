"""Print ATK workbook rows without relying on the process exit status."""
import argparse
import ast
from collections import Counter
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    sheets = pd.read_excel(args.workbook, sheet_name=None)
    if args.summary:
        frame = sheets['statistic']
        failures = []
        output_failures = Counter()
        checked = 0
        for _, row in frame.iterrows():
            detail = ast.literal_eval(row['精度详情'])
            for backend, values in detail.items():
                for value in values:
                    checked += 1
                    if not value['result']:
                        output_failures[value['filename']] += 1
                        failures.append({'case': int(row['编号']), 'backend': backend,
                            'output': value['filename'], 'error': value['error_info'],
                            'metrics': value['new_benchmark_indicate']})
        print(json.dumps({'cases': len(frame), 'outputs_checked': checked,
            'failed_cases': sorted({v['case'] for v in failures}),
            'output_failures': output_failures, 'failures': failures}, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    for name, frame in sheets.items():
        print(name)
        print(frame.to_json(orient='records', force_ascii=False, indent=2))
