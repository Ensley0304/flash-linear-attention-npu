"""Design v2 P32/PM diagnostics; CPU only, not device qualification.

Independent inputs deliberately include cases outside producer-cache ranges.
Every chunk's P, Vnew, D, R and entry snapshot is compared with FP64.
PM is an execution copy only: its running state is never quantized.
"""
import argparse
import json
from pathlib import Path
import torch


def metrics(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    error = (actual - expected).abs()
    pos = int(error.flatten().argmax())
    return dict(max_abs=float(error.max()),
                rms_ratio=float(error.square().mean().sqrt() /
                                (expected.square().mean().sqrt() + 1e-8)),
                worst_flat_index=pos, actual=float(actual.flatten()[pos]),
                expected=float(expected.flatten()[pos]),
                nonfinite=int((~torch.isfinite(actual)).sum()))


def recurrence(w, u, kg, g, initial, mode):
    dtype = torch.float64 if mode == 'FP64' else torch.float32
    w, u, kg, g, state = [x.to(dtype) for x in (w, u, kg, g, initial)]
    quantize = lambda x: x.to(torch.bfloat16).float() if mode == 'PM' else x
    trace = []
    for start in range(0, len(w), 64):
        end = min(start + 64, len(w))
        snapshot = state.clone()
        p = quantize(w[start:end]) @ quantize(state)
        value = u[start:end] - p
        delta = quantize(kg[start:end]).T @ quantize(value)
        state = g[end - 1].exp()[:, None] * state + delta
        trace.append(dict(P=p, Vnew=value, D=delta, R_run=state,
                          H_snapshot=snapshot))
    trace[-1]['H_final'] = state
    return trace


def inputs(kind, tokens, seed):
    rng = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=rng, dtype=torch.float64)
    w, u, kg = rand(tokens, 256)*.01, rand(tokens, 256)*.05, rand(tokens, 256)*.01
    g, initial = torch.full_like(w, -.001), rand(256, 256)*.1
    if kind == 'small_increment':
        w.zero_(); kg.zero_(); u.zero_(); initial.fill_(1); g.zero_()
        kg[::64, 0] = 1
        u[::64] = 1/512
    elif kind == 'cancellation':
        # Nearly cancel U-WR, magnifying operand-quantization loss in PM.
        u = w @ initial + rand(tokens, 256)*1e-5
        kg *= 1e-3; g.zero_()
    elif kind == 'weak_decay':
        g.fill_(-1e-5)
    return w, u, kg, g, initial


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    rows, failures = [], []
    for kind in ('small_increment', 'weak_decay', 'cancellation', 'non_diagonal'):
        for tokens in (65, 1024, 4097):
            data = inputs(kind, tokens, 1127)
            golden = recurrence(*data, 'FP64')
            for mode in ('P32', 'PM'):
                trace = recurrence(*data, mode)
                for chunk, (actual, expected) in enumerate(zip(trace, golden)):
                    for name in expected:
                        m = metrics(actual[name], expected[name])
                        # P32 stage control criterion, not an upstream tolerance.
                        ok = m['nonfinite'] == 0 and (m['max_abs'] <= 1e-6 or m['rms_ratio'] < 1e-4)
                        rows.append(dict(kind=kind, tokens=tokens, mode=mode,
                                         chunk=chunk, tensor=name, p32_control_pass=ok, **m))
                        if mode == 'P32' and not ok:
                            failures.append(rows[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='independent_cpu_only',
        p32_pass=not failures, pm_qualified=False, records=rows), indent=2))
    for mode in ('P32', 'PM'):
        selected = [r for r in rows if r['mode'] == mode]
        print(mode, 'records', len(selected), 'max_abs', max(r['max_abs'] for r in selected),
              'max_ratio', max(r['rms_ratio'] for r in selected),
              'failed_control', sum(not r['p32_control_pass'] for r in selected), flush=True)
    assert not failures, failures[:3]


if __name__ == '__main__':
    main()
