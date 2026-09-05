"""Check saved Stage0-10 outputs against independent CPU dual references.

The oracle module must export kernel_c_base_torch and kernel_c_intra_torch.
Inputs use BHTD layout; h/dh use BHNTKV. Output order is dq, dk, dv, d_beta.
This CPU-only check must not be reported as a GPU dual benchmark.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import ct
import torch


def reference(oracle, inputs, dtype):
    values = {
        name: None if value is None else value.transpose(1, 2).contiguous()
        for name, value in inputs.items()
    }
    low = dtype == torch.float32
    base = oracle.kernel_c_base_torch(
        values['q'], values['k'], values['v'], values['v_new'], values['gk'],
        values['beta'], values['akk'], values['h'], torch.zeros_like(values['v']),
        values['dh'], values['dv_scan'], values['dq_raw'], 128 ** -.5, 64,
        None, False, acc_dtype=dtype, preserve_output_dtypes=low,
        emulate_triton_casts=low)
    intra = oracle.kernel_c_intra_torch(
        values['q'], values['k'], values['gk'], values['beta'], values['d_aqk'],
        base['dAkk'], base, 64, None, torch.bfloat16 if low else None,
        acc_dtype=dtype)
    outputs = [intra['dq_hv'], intra['dk_hv'], base['dv'], intra['db']]
    for index, name in ((0, 'q'), (1, 'k')):
        if values[name + '_rstd'] is not None:
            vector = values[name].to(dtype)
            grad = outputs[index]
            outputs[index] = values[name + '_rstd'].to(dtype)[..., None] * (
                grad - vector * (grad * vector).sum(-1, keepdim=True))
    return [value.bfloat16() if low else value for value in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('kernel_ac_cpu_oracle', args.oracle)
    oracle = importlib.util.module_from_spec(spec)
    # The external oracle uses dataclasses, which resolve their owning module.
    import sys
    sys.modules[spec.name] = oracle
    spec.loader.exec_module(oracle)
    inputs = torch.load(args.inputs, map_location='cpu', weights_only=False)
    outputs = torch.load(args.outputs, map_location='cpu', weights_only=False)
    shape = inputs['q'].shape
    assert shape[-1] == 128 and shape[2] % 64 == 0, 'This checker covers aligned K=V=128 cases'
    aq = inputs['d_aqk'].reshape(*shape[:2], -1, 64, 64)
    assert torch.count_nonzero(aq.triu(1)) == 0, 'The oracle requires causal dAqk; inputs are not masked by the checker'
    golden = reference(oracle, inputs, torch.float64)
    benchmark = reference(oracle, inputs, torch.float32)
    report = {
        'benchmark_kind': 'CPU_FP32_BF16_NOT_GPU', 'level': 'L1',
        'dtype': 'bfloat16', 'shape': list(shape),
        'oracle_sha256': hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        'golden': 'FP64; no simulated Cube casts; no output downcast',
        'benchmark': 'FP32; simulated BF16 Cube casts; BF16 outputs',
        'outputs': {},
    }
    for index, name in enumerate(('dq', 'dk', 'dv', 'd_beta')):
        got = outputs[index].transpose(1, 2).contiguous()
        assert got.dtype == torch.bfloat16
        assert all(torch.isfinite(value).all() for value in (got, golden[index], benchmark[index]))
        result = ct.dual(got, golden[index], benchmark[index], level='L1', dtype='bfloat16')
        for key in ('metrics_test', 'metrics_bench'):
            result[key] = dataclasses.asdict(result[key])
        result['raw_rmse_ratio'] = result['metrics_test']['rmse'] / max(result['metrics_bench']['rmse'], 1e-300)
        report['outputs'][name] = result
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    assert all(result['success'] for result in report['outputs'].values()), 'CPU dual L1 failed; see report'


if __name__ == '__main__':
    main()
