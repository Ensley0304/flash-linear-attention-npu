"""Isolate the single-token dq path without changing either ATK reference."""
import json
import torch
from Finalize import PROFILES, build_inputs, cpu_reference, npu_call


if __name__ == '__main__':
    s = PROFILES[0]
    assert s['T'] == 1 and not s['rstd']
    x = build_inputs(s)
    low = cpu_reference(x, s, False)[0]
    high = cpu_reference(x, s, True)[0]
    got = npu_call(x, s, torch.device('npu:0'))[0].cpu()
    e = x['gk'].exp2()
    da = x['d_aqk'][..., :1].bfloat16().float()
    factored = (x['dq_raw']*e*s['scale'] + da*(x['k'].float()/e).bfloat16().float()*e).bfloat16()
    plain = (x['dq_raw']*e*s['scale'] + da*x['k'].float()).bfloat16()
    report = {'case': s, 'outputs': {}}
    for name, value in [('unfactored', low), ('factored', factored), ('single_token_plain', plain)]:
        delta = (got.float()-value.float()).abs()
        report['outputs'][name] = {'different': int((got != value).sum()),
            'max_abs': delta.max().item(), 'mean_abs': delta.mean().item()}
    report['got_fp64_max_abs'] = (got.double()-high).abs().max().item()
    print(json.dumps(report, indent=2))
