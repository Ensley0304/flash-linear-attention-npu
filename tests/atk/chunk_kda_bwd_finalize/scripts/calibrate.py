"""Cross-check full FP64 equations with an independent autograd objective."""
import itertools
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_chunk_kda_bwd_finalize import PROFILES
from reference import build_inputs, reference, round_bfloat16, NAMES

torch.set_num_threads(1)
results = []
for rows, norm, beta_mode in itertools.product((1, 7, 33, 64), (False, True), (0, 1, 2)):
    spec = dict(PROFILES[0], H=1, T=rows, B=1, beta_mode=beta_mode, norm=norm,
                output_layout='BNSD', scale=128**-.5, lower_bound=-1.)
    data, _, chunks = build_inputs(spec, 971 + rows)
    if norm:
        # Exact BF16 unit vectors isolate the normalization derivative from
        # forward quantization error. The remaining operands remain random.
        for name in ('q', 'k'):
            data[name].zero_()
            data[name][..., :4] = .5
            data[name][..., 0] = -.5 if name == 'q' else .5
            data[name + '_rstd'].fill_(2.)
    q_leaf = (data['q'][0, 0].double() / (2 if norm else 1)).requires_grad_()
    k_leaf = (data['k'][0, 0].double() / (2 if norm else 1)).requires_grad_()
    q = torch.nn.functional.normalize(q_leaf, dim=-1) if norm else q_leaf
    k = torch.nn.functional.normalize(k_leaf, dim=-1) if norm else k_leaf
    v = data['v'][0, 0].double().requires_grad_()
    saved_beta = data['beta'][0, 0].double()
    multiplier = 2 if beta_mode == 2 else 1
    beta_leaf = (torch.logit(saved_beta / multiplier) if beta_mode else saved_beta).requires_grad_()
    beta = multiplier * beta_leaf.sigmoid() if beta_mode else beta_leaf
    raw = data['raw_g'][0, 0].double().requires_grad_()
    a_log = data['a_log'][0].double().requires_grad_()
    bias = data['dt_bias'][0].double().requires_grad_()
    gate = (spec['lower_bound'] * (a_log.exp() * (raw + bias)).sigmoid()).cumsum(0) / math.log(2)
    # Saved gk is FP32; keep its value and differentiate through the raw gate.
    g = data['gk'][0, 0].double() + (gate - gate.detach())
    decay = torch.exp2(g[:, None] - g[None])
    causal = torch.ones(rows, rows).tril().double()
    s = (beta[:, None] * (k[:, None] * k[None] * decay).sum(-1)).tril(-1)
    a = torch.linalg.inv(torch.eye(rows, dtype=torch.float64) + s)
    # Use saved BF16 A values, but retain the analytic inverse differential.
    saved_a = a.detach().bfloat16().double()
    a = saved_a - saved_a @ (s - s.detach()) @ saved_a
    data['akk'][0, 0, :, :rows] = saved_a.bfloat16()
    h, dh = data['h'][0, 0, 0].double(), data['dh'][0, 0, 0].double()
    ds, vn, aq, qr = [data[n][0, 0].double() for n in ('dv_scan', 'v_new', 'dAqk', 'dq_raw')]
    loss = ((a @ (beta[:, None] * (v - (k * torch.exp2(g)) @ h))) * ds).sum()
    state = torch.exp2(g[-1])[:, None] * h + (k * torch.exp2(g[-1] - g)).T @ vn
    loss += (state * dh).sum()
    loss += (q * torch.exp2(g) * qr * spec['scale']).sum()
    loss += (aq[:, :rows] * causal * (q[:, None] * k[None] * decay).sum(-1)).sum()
    gradients = torch.autograd.grad(loss, (q_leaf, k_leaf, v, beta_leaf, raw, a_log, bias))
    expected = tuple(round_bfloat16(value) if i < 3 else value.float()
                     for i, value in enumerate(gradients))
    actual = reference(data, spec, chunks, True)
    for name, got, want in zip(NAMES, actual, expected):
        torch.testing.assert_close(got.reshape(want.shape), want, rtol=2e-4, atol=1e-6,
                                   msg=f'{name}, rows={rows}, norm={norm}, beta={beta_mode}')
    results.append(dict(rows=rows, norm=norm, beta_mode=beta_mode, autograd='PASS'))
print(json.dumps(results, indent=2))
