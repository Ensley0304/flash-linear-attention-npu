"""Check saved Stage0-12 outputs against independent CPU dual references.

The oracle module must export kernel_c_base_torch and kernel_c_intra_torch.
Inputs use BHTD layout; h/dh use BHNTKV. Outputs are dq, dk, dv, d_beta,
d_g, d_a_log, d_dt_bias. Without raw_g, only the first four are checked.
Varlen dumps retain a singleton B dimension and carry CPU cu_seqlens.
This CPU-only check must not be reported as a GPU dual benchmark.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import ct
import numpy as np
import torch


def cube_intra_reference(oracle, values, base, cu):
    """CPU BF16 Cube benchmark for the factored Stage6/8 computation.

    The original intra oracle evaluates pairwise exp2(g_i-g_j) and only
    rounds dA. Cube also rounds k_neg/q_pos/bk_pos; model these documented
    operand boundaries here, without changing the independent FP64 golden.
    """
    dg = base['dg_base'].clone()
    dq = base['dq_base'].clone()
    dk = base['dk_base'].clone()
    db = base['db_base'].clone()
    for chunk in oracle.iter_chunks(dg.shape[0], dg.shape[1], 64, cu):
        b, s, end = chunk.batch, chunk.start, chunk.end
        rows = end - s
        for head in range(dg.shape[2]):
            q = values['q'][b, s:end, head].float()
            k = values['k'][b, s:end, head].float()
            e = values['gk'][b, s:end, head].float().exp2()
            beta = values['beta'][b, s:end, head].float()[:, None]
            daq = values['d_aqk'][b, s:end, head, :rows].bfloat16().float()
            dak = base['dAkk'][b, s:end, head, :rows].bfloat16().float()
            kn = (k / e).bfloat16().float()
            qp = (q * e).bfloat16().float()
            bkp = (k * e * beta).bfloat16().float()
            dq_local = (daq @ kn) * e
            dk_left = (dak @ kn) * e
            right = torch.cat((daq.T, dak.T), dim=1) @ torch.cat((qp, bkp), dim=0)
            dk_right = right / e
            dq[b, s:end, head] += dq_local
            dk[b, s:end, head] += beta * dk_left + dk_right
            db[b, s:end, head] += (dk_left * k).sum(-1)
            dg[b, s:end, head] += q * dq_local + (beta * dk_left - dk_right) * k
    return {'dq_hv': dq, 'dk_hv': dk, 'db': db, 'dg_hv': dg}


def reference(oracle, inputs, dtype, factored_intra=False):
    values = {
        name: None if value is None else value.transpose(1, 2).contiguous()
        for name, value in inputs.items() if name not in ('raw_g', 'a_log', 'dt_bias', 'cu_seqlens')
    }
    cu = inputs.get('cu_seqlens')
    low = dtype == torch.float32
    base = oracle.kernel_c_base_torch(
        values['q'], values['k'], values['v'], values['v_new'], values['gk'],
        values['beta'], values['akk'], values['h'], torch.zeros_like(values['v']),
        values['dh'], values['dv_scan'], values['dq_raw'], 128 ** -.5, 64,
        cu, False, acc_dtype=dtype, preserve_output_dtypes=low,
        emulate_triton_casts=low)
    if low and factored_intra:
        intra = cube_intra_reference(oracle, values, base, cu)
    else:
        intra = oracle.kernel_c_intra_torch(
            values['q'], values['k'], values['gk'], values['beta'], values['d_aqk'],
            base['dAkk'], base, 64, cu, torch.bfloat16 if low else None,
            acc_dtype=dtype)
    outputs = [intra['dq_hv'], intra['dk_hv'], base['dv'], intra['db']]
    for index, name in ((0, 'q'), (1, 'k')):
        if values[name + '_rstd'] is not None:
            vector = values[name].to(dtype)
            grad = outputs[index]
            outputs[index] = values[name + '_rstd'].to(dtype)[..., None] * (
                grad - vector * (grad * vector).sum(-1, keepdim=True))
    outputs = [value.bfloat16() if low else value for value in outputs]
    if 'raw_g' in inputs:
        dg_act = oracle.reverse_chunk_cumsum(intra['dg_hv'], 64, cu)
        gate = oracle.gate_backward_torch(
            inputs['raw_g'].transpose(1, 2), inputs['a_log'], inputs['dt_bias'],
            dg_act, -5.0, acc_dtype=dtype)
        outputs.extend(value.float() if low else value for value in
                       (gate['dg'], gate['dA_log'], gate['ddt_bias'].reshape(inputs['dt_bias'].shape)))
    return outputs


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
    assert shape[-1] == 128, 'This checker covers K=V=128 cases'
    for chunk in oracle.iter_chunks(shape[0], shape[2], 64, inputs.get('cu_seqlens')):
        aq = inputs['d_aqk'][chunk.batch, :, chunk.start:chunk.end]
        assert torch.count_nonzero(aq.triu(1)) == 0, 'The oracle requires causal dAqk; inputs are not masked by the checker'
    golden = reference(oracle, inputs, torch.float64)
    benchmark = reference(oracle, inputs, torch.float32)
    factored_benchmark = reference(oracle, inputs, torch.float32, factored_intra=True)
    report = {
        'benchmark_kind': 'CPU_FP32_BF16_NOT_GPU', 'level': 'L1',
        'shape': list(shape),
        'cu_seqlens': None if inputs.get('cu_seqlens') is None else inputs['cu_seqlens'].tolist(),
        'oracle_sha256': hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        'golden': 'FP64; no simulated Cube casts; no output downcast',
        'benchmark': 'Original FP32/BF16 oracle; unfactored intra quantizes dA only; BF16 dq/dk/dv/db and FP32 gate outputs',
        'factored_benchmark': 'Historical BF16 factored operand model; diagnostic only, not acceptance',
        'outputs': {},
        'factored_outputs': {},
    }
    names = ('dq', 'dk', 'dv', 'd_beta', 'd_g', 'd_a_log', 'd_dt_bias')[:len(golden)]
    for index, name in enumerate(names):
        got = outputs[index].transpose(1, 2).contiguous() if index < 5 else outputs[index]
        expected_dtype = torch.bfloat16 if index < 4 else torch.float32
        assert got.dtype == expected_dtype
        assert all(torch.isfinite(value).all() for value in (got, golden[index], benchmark[index]))
        result = ct.dual(got, golden[index], benchmark[index], level='L1',
                         dtype='bfloat16' if index < 4 else 'float32')
        for key in ('metrics_test', 'metrics_bench'):
            result[key] = dataclasses.asdict(result[key])
        result['raw_rmse_ratio'] = result['metrics_test']['rmse'] / max(result['metrics_bench']['rmse'], 1e-300)
        result['dtype'] = str(expected_dtype)
        report['outputs'][name] = result
        if index < 4:
            legacy = ct.dual(got, golden[index], factored_benchmark[index], level='L1', dtype='bfloat16')
            for key in ('metrics_test', 'metrics_bench'):
                legacy[key] = dataclasses.asdict(legacy[key])
            report['factored_outputs'][name] = legacy
    def serialize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported report value: {type(value)}')

    args.report.write_text(json.dumps(report, indent=2, default=serialize) + '\n', encoding='utf-8')
    assert all(result['success'] for result in report['outputs'].values()), 'CPU dual L1 failed; see report'


if __name__ == '__main__':
    main()
