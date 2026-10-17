"""Independent full Finalize equations, with no device/capture dependencies."""
import math
import torch

NAMES = ('dq', 'dk', 'dv', 'db', 'dg', 'da_log', 'ddt_bias')


def build_inputs(spec, seed):
    gen = torch.Generator().manual_seed(seed)
    b, h, t = (spec[n] for n in ('B', 'H', 'T'))
    variable = spec['lengths'] is not None
    shape = (h, t, 128) if variable else (b, h, t, 128)

    def rand(dims, scale=.125, bf=False):
        x = torch.randn(dims, generator=gen) * scale
        return x.bfloat16() if bf else x

    data = {n: rand(shape, bf=True) for n in ('q', 'k', 'v', 'v_new', 'dv_scan')}
    for name in ('q', 'k'):
        if spec['norm']:
            x = data[name].float()
            rstd = (x.square().sum(-1) + 1e-6).rsqrt()
            data[name] = (x * rstd[..., None]).bfloat16()
            data[name + '_rstd'] = rstd
    raw = rand(shape, 1.)
    if spec['gate_mode'] == 'saturated':
        raw[..., ::4] = 30.
        raw[..., 1::4] = -30.
    data.update(raw_g=raw, a_log=rand((h,), .25), dt_bias=rand((h, 128), .125),
                dq_raw=rand(shape), dAqk=rand((*shape[:-1], 64)))
    logits = rand(shape[:-1], 2.)
    logits.reshape(-1)[:min(5, logits.numel())] = torch.tensor([-8., 0., 8., -16., 16.])[:min(5, logits.numel())]
    data['beta'] = logits.sigmoid() * (2 if spec['beta_mode'] == 2 else 1)
    cu, chunks = [0], []
    lengths = spec['lengths'] if variable else [t] * b
    for seq, length in enumerate(lengths):
        begin = cu[-1] if variable else 0
        for local, start in enumerate(range(begin, begin + length, 64)):
            chunks.append((seq, local, start, min(64, begin + length - start)))
        cu.append(cu[-1] + length)
    state_shape = (len(chunks), h, 128, 128) if variable else (b, (t+63)//64, h, 128, 128)
    data['h'], data['dh'] = rand(state_shape, bf=True), rand(state_shape, bf=True)
    data['akk'] = torch.zeros((*shape[:-1], 64), dtype=torch.bfloat16)
    data['gk'] = torch.empty(shape)
    for seq, _, first, rows in chunks:
        region = (slice(None), slice(first, first+rows)) if variable else (seq, slice(None), slice(first, first+rows))
        z = (raw[region] + data['dt_bias'][:, None, :]) * data['a_log'].exp()[:, None, None]
        data['gk'][region] = (spec['lower_bound'] * z.sigmoid()).cumsum(-2) / math.log(2)
        # A is unit lower triangular; inactive columns stay zero.
        a = torch.tril(rand((h, rows, rows), .025), -1) + torch.eye(rows)
        data['akk'][region][..., :rows] = a.bfloat16()
    return data, cu if variable else [], chunks


def round_bfloat16(value):
    """Round FP64 once, including subnormals; torch's cast can pass through FP32."""
    _, exponent = torch.frexp(value)
    step = (exponent - 8).clamp(min=-133)
    rounded = torch.ldexp(torch.ldexp(value, -step).round(), step)
    return rounded.bfloat16()


def reference(data, spec, chunks, high_precision=True):
    """FP64 mathematical golden / FP32 benchmark with documented BF16 handoffs.

    The exponential difference is factored per CPU output tile. FP64 can
    handle an entire chunk; FP32 uses eight rows to keep positive exponents
    bounded. There is no BF16 residual split or device intermediate reuse.
    """
    dtype = torch.float64 if high_precision else torch.float32
    variable = spec['lengths'] is not None
    heads = spec['H']
    shape = data['q'].shape
    # Round the FP64 golden directly to the interface dtype. An intervening
    # FP32 store can turn a near-midpoint value into a BF16 tie.
    out = {n: torch.empty(shape, dtype=torch.bfloat16) for n in ('dq', 'dk', 'dv')}
    out['dg'] = torch.empty(shape, dtype=torch.float32)
    out['db'] = torch.empty(shape[:-1])
    da = torch.zeros(heads, dtype=dtype)
    bias = torch.zeros(heads, 128, dtype=dtype)

    def handoff(x):
        return x if high_precision else x.bfloat16().float()

    for ci, (seq, local, first, rows) in enumerate(chunks):
        for head in range(heads):
            region = (head, slice(first, first+rows)) if variable else (seq, head, slice(first, first+rows))
            q, k, v, vn, ds, g, beta, aq, qr = [data[n][region].to(dtype) for n in
                ('q', 'k', 'v', 'v_new', 'dv_scan', 'gk', 'beta', 'dAqk', 'dq_raw')]
            a = data['akk'][region][:, :rows].to(dtype)
            hh = (data['h'][ci, head] if variable else data['h'][seq, local, head]).to(dtype)
            dh = (data['dh'][ci, head] if variable else data['dh'][seq, local, head]).to(dtype)
            e = torch.exp2(g)
            dw = handoff(ds @ hh.T)
            ke = handoff(k * e)
            dvb = a.T @ ds
            dkg = a.T @ dw
            zb = handoff(torch.tril((ds @ v.T - dw @ ke.T) * beta[None, :], -1))
            ak = -(a.T @ handoff(zb @ a.T)).tril(-1)
            state = (vn @ dh.T) * torch.exp2(g[-1] - g)
            dq = spec['scale'] * qr * e
            dk = state - beta[:, None] * e * dkg
            db = (v * dvb).sum(-1) - (k * e * dkg).sum(-1)
            dg = q * dq - k * state - beta[:, None] * k * e * dkg
            dg[-1] += (k * state).sum(0) + torch.exp2(g[-1]) * (hh * dh).sum(-1)
            aq_lower = aq[:, :rows].tril(-1)
            local_q, left, right = (torch.empty_like(k) for _ in range(3))
            width = rows if high_precision else 8
            for start in range(0, rows, width):
                end = min(start + width, rows)
                center = (g[start] + g[end - 1]) * .5
                factor = torch.exp2(g[start:end] - center)
                neg = k[:end] * torch.exp2(center - g[:end])
                pos = torch.exp2(g[start:] - center)
                local_q[start:end] = (aq_lower[start:end, :end] @ neg) * factor
                left[start:end] = (ak[start:end, :end] @ neg) * factor
                right[start:end] = (aq_lower[start:, start:end].T @ (q[start:] * pos) +
                                    ak[start:, start:end].T @ (beta[start:, None] * k[start:] * pos)) / factor
            diag = aq[:, :rows].diag()[:, None]
            dq = dq + local_q + diag * k
            dk = dk + beta[:, None] * left + right + diag * q
            db = db + (left * k).sum(-1)
            dg = dg + q * local_q + k * (beta[:, None] * left - right)
            scan = dg.flip(0).cumsum(0).flip(0)
            x = data['raw_g'][region].to(dtype) + data['dt_bias'][head].to(dtype)
            ea = data['a_log'][head].to(dtype).exp()
            z = (-(x * ea).abs()).exp()
            dg = scan * (spec['lower_bound'] * ea * z / (1 + z).square())
            if spec['norm']:
                dq = (dq - q * (dq * q).sum(-1, keepdim=True)) * data['q_rstd'][region].to(dtype)[:, None]
                dk = (dk - k * (dk * k).sum(-1, keepdim=True)) * data['k_rstd'][region].to(dtype)[:, None]
            if spec['beta_mode']:
                db *= beta * (1 - beta / (2 if spec['beta_mode'] == 2 else 1))
            for n, value in zip(('dq', 'dk', 'dv', 'db', 'dg'), (dq, dk, beta[:, None]*dvb, db, dg)):
                out[n][region] = (round_bfloat16(value) if high_precision and n in ('dq', 'dk', 'dv')
                                   else value.to(out[n].dtype))
            da[head] += (dg * x).sum()
            bias[head] += dg.sum(0)
    for n in ('dq', 'dk', 'dv'):
        if spec['output_layout'] in ('BSND', 'TND'):
            out[n] = out[n].transpose(-3, -2).contiguous()
    out['dg'] = out['dg'].transpose(-3, -2).contiguous()
    out['db'] = out['db'].transpose(-2, -1).contiguous()
    out['da_log'], out['ddt_bias'] = da.float(), bias.float()
    return tuple(out[n] for n in NAMES)
