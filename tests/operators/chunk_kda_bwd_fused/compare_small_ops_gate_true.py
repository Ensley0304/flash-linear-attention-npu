#!/usr/bin/env python3
"""Compare fused KDA backward with the established small-op chain.

The intended model mode is fixed deliberately:
  safe_gate=True, use_gate_in_kernel=True, lower_bound=-5.0.

The input is the CPU ``torch.save`` payload used by kda_bwd_repro.py.  By
default only the four historically failing heads are retained; KDA heads are
independent, so sequence length and the real tensor values are unchanged.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import (
    npu_chunk_gated_delta_rule_bwd_dhu,
    npu_chunk_kda_bwd,
    npu_chunk_kda_fwd,
)
from fla_npu.ops.triton import l2norm_bwd, l2norm_fwd

try:
    import fla.ops.kda.chunk_bwd as small
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Cannot import fla.ops.kda.chunk_bwd. Add the sibling GPU "
        "flash-linear-attention repository to PYTHONPATH."
    ) from exc


DEVICE = "npu:0"
CHUNK_SIZE = 64
LOWER_BOUND = -5.0


def bsnd_to_bnsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def bnsd_to_bsnd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def parse_heads(text: str, head_count: int) -> torch.Tensor | None:
    if text.strip().lower() == "all":
        return None
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values or min(values) < 0 or max(values) >= head_count:
        raise ValueError(f"--heads must be within [0,{head_count}), got {values}")
    return torch.tensor(values, dtype=torch.long)


def select_heads(x: torch.Tensor, axis: int, heads: torch.Tensor | None) -> torch.Tensor:
    return x if heads is None else x.index_select(axis, heads)


def report(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> bool:
    got32 = got.float()
    ref32 = ref.float()
    diff = (got32 - ref32).abs()
    bad = (~torch.isfinite(got32)) | (~torch.isfinite(ref32)) | (
        diff > atol + rtol * ref32.abs()
    )
    got_norm = torch.linalg.vector_norm(got32)
    ref_norm = torch.linalg.vector_norm(ref32)
    rel_l2 = torch.linalg.vector_norm(got32 - ref32) / ref_norm.clamp_min(1e-30)
    cosine = ((got32.flatten() * ref32.flatten()).sum() /
              (got_norm * ref_norm).clamp_min(1e-30))
    bad_count = int(bad.sum().cpu())
    print(
        f"{'PASS' if bad_count == 0 else 'FAIL'} {name:<9} "
        f"shape={tuple(got.shape)} dtype={got.dtype}/{ref.dtype} "
        f"nan={int(torch.isnan(got32).sum().cpu())}/{int(torch.isnan(ref32).sum().cpu())} "
        f"inf={int(torch.isinf(got32).sum().cpu())}/{int(torch.isinf(ref32).sum().cpu())} "
        f"max_abs={float(diff.max().cpu()):.6e} "
        f"mean_abs={float(diff.mean().cpu()):.6e} "
        f"rel_l2={float(rel_l2.cpu()):.6e} cosine={float(cosine.cpu()):.8f} "
        f"norm={float(got_norm.cpu()):.6e}/{float(ref_norm.cpu()):.6e} "
        f"bad={bad_count}/{bad.numel()}",
        flush=True,
    )
    if bad_count:
        first = tuple(int(v) for v in bad.nonzero(as_tuple=False)[0].cpu())
        print(
            f"  first_bad={first} got={float(got32[first].cpu()):.8e} "
            f"ref={float(ref32[first].cpu()):.8e}",
            flush=True,
        )
    return bad_count == 0


def raw_gate_backward(
    upstream: torch.Tensor,
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Safe-gate chain rule after chunk-local reverse cumsum."""
    heads = raw_g.shape[2]
    key_dim = raw_g.shape[3]
    rate = a_log.float().exp().view(1, 1, heads, 1)
    x = raw_g.float()
    if dt_bias is not None:
        x = x + dt_bias.float().view(1, 1, heads, key_dim)
    sigmoid = torch.sigmoid(rate * x)
    dg_raw = upstream.float() * LOWER_BOUND * rate * sigmoid * (1.0 - sigmoid)
    d_a = (dg_raw * x).sum(dim=(0, 1, 3))
    d_bias = dg_raw.sum(dim=(0, 1)) if dt_bias is not None else None
    return dg_raw.contiguous(), d_a.contiguous(), (
        None if d_bias is None else d_bias.contiguous()
    )


def load_case(path: str, head_spec: str):
    data = torch.load(path, map_location="cpu")
    required = ("q", "k", "v", "beta", "g_raw", "A_log")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"input payload is missing {missing}")

    q_cpu = data["q"]
    if q_cpu.ndim != 4:
        raise ValueError(f"this script expects dense BSND q, got {tuple(q_cpu.shape)}")
    heads = parse_heads(head_spec, q_cpu.shape[2])
    q0 = select_heads(data["q"], 2, heads).to(DEVICE)
    k0 = select_heads(data["k"], 2, heads).to(DEVICE)
    v = select_heads(data["v"], 2, heads).to(DEVICE)
    beta = select_heads(data["beta"], 2, heads).to(DEVICE)
    raw_g = select_heads(data["g_raw"], 2, heads).to(DEVICE).contiguous()
    a_log = select_heads(data["A_log"].float(), 0, heads).to(DEVICE).contiguous()

    key_dim = q0.shape[-1]
    dt_bias = data.get("dt_bias")
    if dt_bias is not None:
        dt_bias = dt_bias.float().reshape(-1, key_dim)
        dt_bias = select_heads(dt_bias, 0, heads).to(DEVICE).contiguous()

    d_o = data.get("d_o")
    if d_o is None:
        d_o = torch.randn(v.shape, generator=torch.Generator().manual_seed(7), dtype=v.dtype)
    else:
        d_o = select_heads(d_o, 2, heads)
    d_o = d_o.to(DEVICE).to(v.dtype).contiguous()
    return data, q0, k0, v, beta, raw_g, a_log, dt_bias, d_o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CPU torch.save payload from the model")
    parser.add_argument(
        "--heads", default="1,15,19,33",
        help="comma-separated original head ids, or 'all' (default: 1,15,19,33)",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    data, q0, k0, v, beta, raw_g, a_log, dt_bias, d_o = load_case(
        args.input, args.heads
    )
    if q0.shape[1] % CHUNK_SIZE:
        raise ValueError("dense comparison currently requires S divisible by 64")
    scale = float(data.get("scale", q0.shape[-1] ** -0.5))
    input_lower_bound = float(data.get("lower_bound", LOWER_BOUND))
    if input_lower_bound != LOWER_BOUND:
        raise ValueError(
            f"model payload lower_bound={input_lower_bound}, expected {LOWER_BOUND}"
        )

    use_l2norm = str(data.get("use_qk_l2norm_in_kernel", "true")).lower() == "true"
    if use_l2norm:
        q, q_rstd = l2norm_fwd(q0)
        k, k_rstd = l2norm_fwd(k0)
    else:
        q, k = q0, k0
        q_rstd = k_rstd = None

    h_count = q.shape[2]
    rate = a_log.exp().view(1, 1, h_count, 1)
    gate_x = raw_g.float()
    if dt_bias is not None:
        gate_x = gate_x + dt_bias.view(1, 1, h_count, q.shape[-1])
    gate = (LOWER_BOUND * torch.sigmoid(rate * gate_x)).contiguous()

    print(
        f"CASE B={q.shape[0]} S={q.shape[1]} H={q.shape[2]} "
        f"K={q.shape[3]} V={v.shape[3]} heads={args.heads} "
        f"safe_gate=True use_gate_in_kernel=True lower_bound={LOWER_BOUND} "
        f"l2norm={use_l2norm}",
        flush=True,
    )

    fwd = npu_chunk_kda_fwd(
        q, k, v, raw_g, beta, scale,
        chunk_size=CHUNK_SIZE, layout="BSND", initial_state=None,
        output_final_state=False, cu_seqlens=None, chunk_indices=None,
        safe_gate=True, lower_bound=LOWER_BOUND, use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=None if dt_bias is None else dt_bias.reshape(-1),
        disable_recompute=True,
        return_intermediate_states=False, state_v_first=False,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = fwd

    got = npu_chunk_kda_bwd(
        bsnd_to_bnsd(q), bsnd_to_bnsd(k), bsnd_to_bnsd(v),
        beta.permute(0, 2, 1).contiguous(), gk, aqk, akk, w, qg, kg,
        v_new, h, bsnd_to_bnsd(d_o), scale,
        raw_g=bsnd_to_bnsd(raw_g), A_log=a_log, dt_bias=dt_bias,
        initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None,
        chunk_size=CHUNK_SIZE, safe_gate=True, lower_bound=LOWER_BOUND,
        use_gate_in_kernel=True, disable_recompute=True, use_exp2=True,
        state_v_first=False,
    )
    torch.npu.synchronize()

    print("SMALL_STAGE dAv", flush=True)
    d_aqk, dv0 = small.chunk_kda_bwd_dAv(
        q=q, k=k, v=bnsd_to_bsnd(v_new), do=d_o,
        A=bnsd_to_bsnd(aqk), scale=scale, cu_seqlens=None,
        chunk_size=CHUNK_SIZE, chunk_indices=None,
    )
    torch.npu.synchronize()

    print("SMALL_STAGE dhu (mature standalone AscendC)", flush=True)
    dh_n, _, dv_scan_n = npu_chunk_gated_delta_rule_bwd_dhu(
        qg, kg, w, bsnd_to_bnsd(d_o), bsnd_to_bnsd(dv0), scale,
        CHUNK_SIZE, gK=gk, h0=None, dht=None, cu_seqlens=None,
        chunk_indices=None, use_exp2=True, transpose_state_layout=False,
    )
    torch.npu.synchronize()
    dh = dh_n.permute(0, 2, 1, 3, 4).contiguous()
    dv_scan = bnsd_to_bsnd(dv_scan_n)

    print("SMALL_STAGE wy", flush=True)
    rdq, rdk, rdv, rdb, rdg, d_akk = small.chunk_kda_bwd_wy_dqkg_fused(
        q=q, k=k, v=v, v_new=bnsd_to_bsnd(v_new),
        g=bnsd_to_bsnd(gk), beta=beta,
        A=bnsd_to_bsnd(akk), h=h, do=d_o, dh=dh, dv=dv_scan,
        scale=scale, cu_seqlens=None, chunk_size=CHUNK_SIZE,
        chunk_indices=None, state_v_first=False,
    )
    torch.npu.synchronize()

    print("SMALL_STAGE intra", flush=True)
    rdq, rdk, rdb, rdg = small.chunk_kda_bwd_intra(
        q=q, k=k, g=bnsd_to_bsnd(gk), beta=beta,
        dAqk=d_aqk, dAkk=d_akk,
        dq=rdq, dk=rdk, db=rdb, dg=rdg, cu_seqlens=None,
        chunk_size=CHUNK_SIZE, chunk_indices=None, safe_gate=False,
    )
    torch.npu.synchronize()

    print("SMALL_STAGE reverse_cumsum + raw_gate_chain", flush=True)
    rdg = small.chunk_local_cumsum(
        rdg, chunk_size=CHUNK_SIZE, reverse=True,
        cu_seqlens=None, chunk_indices=None,
    )
    rdg, rd_a, rd_bias = raw_gate_backward(rdg, raw_g, a_log, dt_bias)
    torch.npu.synchronize()

    got_bsnd = (
        bnsd_to_bsnd(got[0]),
        bnsd_to_bsnd(got[1]),
        bnsd_to_bsnd(got[2]),
        got[3].permute(0, 2, 1).contiguous(),
        bnsd_to_bsnd(got[4]),
        got[5],
        got[6],
    )
    refs = (rdq, rdk, rdv, rdb, rdg, rd_a, rd_bias)
    names = ("dq", "dk", "dv", "db", "dg", "dA", "dbias")
    passed = True
    for name, actual, expected in zip(names, got_bsnd, refs):
        if actual is None or expected is None:
            ok = actual is expected
            print(f"{'PASS' if ok else 'FAIL'} {name}: None={actual is None}/{expected is None}")
        else:
            atol = 1e-2 if name == "dv" else args.atol
            ok = report(name, actual, expected, atol, args.rtol)
        passed &= ok

    if use_l2norm:
        passed &= report(
            "dq_norm", l2norm_bwd(q, q_rstd, got_bsnd[0]),
            l2norm_bwd(q, q_rstd, rdq), args.atol, args.rtol,
        )
        passed &= report(
            "dk_norm", l2norm_bwd(k, k_rstd, got_bsnd[1]),
            l2norm_bwd(k, k_rstd, rdk), args.atol, args.rtol,
        )

    print("RESULT:", "PASS" if passed else "FAIL", flush=True)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
