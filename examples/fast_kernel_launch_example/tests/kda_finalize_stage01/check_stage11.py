"""FP64 final gradients, stable gate derivative, output layout and parameter sums."""
import torch

from check_stage7 import read


def make_tail(inputs, chunks, norm, device, beta_mode=0):
    shape = inputs[0].shape
    heads = shape[-3]
    variable = len(shape) == 3
    generator = torch.Generator().manual_seed(11012)
    raw = torch.randn(shape, generator=generator) * 2
    raw.flatten()[:5] = torch.tensor([-30., -8., 0., 8., 30.])
    alog = torch.linspace(-1, 1, heads)
    bias = torch.randn((heads, 128), generator=generator) * .125
    krstd = torch.rand(shape[:-1], generator=generator) + .5 if norm else None
    gate_shape = (shape[-2], heads, 128) if variable else (shape[0], shape[-2], heads, 128)
    cpu = dict(raw_g=raw, a_log=alog, dt_bias=bias, k_rstd=krstd)
    args = {key: value.to(device) if value is not None else None for key, value in cpu.items()}
    for key, dims, dtype in (
        ('dk_out', shape, torch.bfloat16), ('dg_out', gate_shape, torch.float32),
        ('db_out', gate_shape[:-1], torch.float32), ('partial_out', (heads, len(chunks), 136), torch.float32),
        ('da_log_out', (heads,), torch.float32), ('ddt_bias_out', (heads, 128), torch.float32),
    ):
        args[key] = torch.full(dims, float('nan'), dtype=dtype, device=device)
    args.update(lower_bound=-5., beta_mode=beta_mode)
    return cpu, args


def check_stage11(inputs, q, beta, daqk, chunks, variable, capture, tail, outputs, args):
    heads = inputs[0].shape[-3]
    actual = {name: value.cpu() for name, value in outputs.items() if name.endswith('_out')}
    partial = actual['partial_out'].double()
    for head in range(heads):
        for c, (sequence, _, first, rows) in enumerate(chunks):
            def token(x):
                return (x[head] if variable else x[sequence, head])[first:first + rows].double()

            key, query, gate, b, aq = (token(x) for x in (inputs[0], q, inputs[2], beta, daqk))
            record = capture[head, c]
            left = torch.empty((rows, 128), dtype=torch.float64)
            right = torch.empty_like(left)
            for start in range(0, rows, 32):
                end = min(start + 32, rows)
                exp = read(record, (424 + start // 32 * 116 + 96) * 1024, (32, 128)).double()[:end-start]
                left[start:end] = read(record, 688 * 1024, (64, 128)).double()[start:end] * exp / 65536
                right[start:end] = read(record, 784 * 1024, (64, 128)).double()[start:end] / exp / 65536
            base_k = read(record, 288 * 1024, (64, 128)).double()[:rows]
            base_g = read(record, 752 * 1024, (64, 128)).double()[:rows]
            base_b = read(record, 352 * 1024, (64,)).double()[:rows]
            diagonal = aq[torch.arange(rows), torch.arange(rows), None]
            wanted_k = base_k + b[:, None] * left + right + diagonal * query
            k_mag = base_k.abs() + (b[:, None] * left).abs() + right.abs() + (diagonal * query).abs()
            wanted_b = base_b + (left * key).sum(-1)
            b_mag = base_b.abs() + (left * key).abs().sum(-1)
            updated_g = base_g + key * (b[:, None] * left - right)
            g_mag = base_g.abs() + key.abs() * ((b[:, None] * left).abs() + right.abs())
            scan = updated_g.flip(0).cumsum(0).flip(0)
            scan_mag = g_mag.flip(0).cumsum(0).flip(0)
            if tail['k_rstd'] is not None:
                rstd = token(tail['k_rstd'])[:, None]
                wanted_k = (wanted_k - key * (wanted_k * key).sum(-1, keepdim=True)) * rstd
                k_mag = (k_mag + key.abs() * (k_mag * key.abs()).sum(-1, keepdim=True)) * rstd.abs()
            if args.beta_mode:
                factor = b * (1 - b / (2 if args.beta_mode == 2 else 1))
                wanted_b *= factor
                b_mag *= factor.abs()
            x = token(tail['raw_g']) + tail['dt_bias'][head].double()
            exp_a = tail['a_log'][head].double().exp()
            z = torch.exp(-(x * exp_a).abs())
            derivative = args.lower_bound * exp_a * z / (1 + z).square()
            wanted_g = scan * derivative
            expected = dict(dk=wanted_k, db=wanted_b, scan=scan, gate=wanted_g)
            budgets = dict(dk=3e-5*k_mag + 2e-6, db=3e-5*b_mag + 2e-38,
                           scan=3e-5*scan_mag + 2e-6, gate=4e-5*scan_mag*derivative.abs() + 2e-38)
            for name, offset, shape in (('dk', 816, (64, 128)), ('db', 848, (64,)),
                                       ('scan', 856, (64, 128)), ('gate', 888, (64, 128))):
                got = read(record, offset*1024, shape).double()[:rows]
                assert torch.isfinite(got).all(), name + ' nonfinite'
                assert ((got-expected[name]).abs() <= budgets[name]).all(), f'Stage 11 {name} head={head} chunk={c}'
                padded = (rows + 31) // 32 * 32
                assert (read(record, offset*1024, shape)[rows:padded] == 0).all(), name + ' tail'
            got_k = token(actual['dk_out'])
            torch.testing.assert_close(got_k, read(record, 816*1024, (64, 128))[:rows].bfloat16().double(), rtol=0, atol=0)
            got_g = actual['dg_out'][first:first+rows, head] if variable else actual['dg_out'][sequence, first:first+rows, head]
            got_b = actual['db_out'][first:first+rows, head] if variable else actual['db_out'][sequence, first:first+rows, head]
            torch.testing.assert_close(got_g, read(record, 888*1024, (64, 128))[:rows], rtol=0, atol=0)
            torch.testing.assert_close(got_b, read(record, 848*1024, (64,))[:rows], rtol=0, atol=0)
            bias_sum = wanted_g.sum(0)
            a_sum = (wanted_g*x).sum()
            bias_budget = budgets['gate'].sum(0) + 3e-6*wanted_g.abs().sum(0) + 2e-38
            a_budget = (budgets['gate']*x.abs()).sum() + 3e-6*(wanted_g*x).abs().sum() + 2e-38
            assert ((partial[head,c,:128]-bias_sum).abs() <= bias_budget).all(), 'bias partial'
            assert (partial[head,c,128:].sum()-a_sum).abs() <= a_budget, 'a_log partial'
            torch.testing.assert_close(partial[head,c], read(record, 920*1024, (136,)).double(), rtol=0, atol=0)
    if args.stage == 12:
        wanted_bias = partial[:,:,:128].sum(1)
        wanted_a = partial[:,:,128:].sum((1,2))
        assert torch.isfinite(actual['ddt_bias_out']).all() and torch.isfinite(actual['da_log_out']).all()
        error = (actual['ddt_bias_out'].double()-wanted_bias).abs()
        budget = 3e-5*partial[:,:,:128].abs().sum(1)+2e-38
        bad = (error > budget).nonzero()
        assert not len(bad), f'Stage 12 bias bad={bad[:8].tolist()} got={actual["ddt_bias_out"][bad[0,0],:8]} want={wanted_bias[bad[0,0],:8]}'
        assert ((actual['da_log_out'].double()-wanted_a).abs() <= 3e-5*partial[:,:,128:].abs().sum((1,2))+2e-38).all(), 'Stage 12 a_log'
