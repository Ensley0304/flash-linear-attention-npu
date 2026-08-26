#!/usr/bin/env python3
"""Model-boundary accuracy and repeatability probe for fused KDA backward.

The FLAN reference and AscendC candidate both run the complete model-visible
chain: Q/K L2Norm forward, KDA forward/backward, raw-gate chain rule, and
Q/K L2Norm backward.  Therefore dq/dk below are gradients of the original
model inputs, not raw gradients of normalized Q/K.

Usage requires the FLAN reference source on ``PYTHONPATH`` before launch,
for example::

  PYTHONPATH=/path/to/gpu_reference:$PYTHONPATH python \
    tests/operators/chunk_kda_bwd_fused/compare_model_boundary_repeat.py \
    /path/to/kda_real_inputs.pt --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Iterable

import torch
import torch_npu  # noqa: F401

try:
    from fla.ops.kda.chunk import chunk_kda as flan_chunk_kda
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Cannot import FLAN reference. Put the repository containing "
        "fla/ops/kda/chunk.py on PYTHONPATH."
    ) from exc

from fla_npu.ops.ascendc import npu_chunk_kda_bwd, npu_chunk_kda_fwd
from fla_npu.ops.triton import l2norm_bwd, l2norm_fwd


DEVICE = "npu:0"
CHUNK_SIZE = 64
GRAD_NAMES = ("dq", "dk", "dv", "dg_raw", "dbeta", "dA_log", "ddt_bias")
RESULT_NAMES = ("o",) + GRAD_NAMES


def bsnd_to_bnsd(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.permute(0, 2, 1, 3).contiguous()


def bnsd_to_bsnd(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.permute(0, 2, 1, 3).contiguous()


def bsh_to_bhs(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.permute(0, 2, 1).contiguous()


def bhs_to_bsh(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.permute(0, 2, 1).contiguous()


class AscendCModelBoundary(torch.autograd.Function):
    """The mathematical boundary implemented by model chunk_kda_wrapper.py."""

    @staticmethod
    def forward(ctx, q, k, v, raw_g, beta, a_log, dt_bias, scale, lower_bound):
        q_norm, q_rstd = l2norm_fwd(q)
        k_norm, k_rstd = l2norm_fwd(k)
        (o, final_state, gk, aqk, akk, w, _u, qg, kg, v_new, h,
         _initial_state) = npu_chunk_kda_fwd(
            q_norm, k_norm, v, raw_g, beta, float(scale), CHUNK_SIZE,
            layout="BSND", initial_state=None, output_final_state=False,
            cu_seqlens=None, chunk_indices=None, safe_gate=True,
            lower_bound=float(lower_bound), use_gate_in_kernel=True,
            A_log=a_log, dt_bias=dt_bias, disable_recompute=True,
            return_intermediate_states=False, state_v_first=False,
        )
        if final_state is not None:
            raise RuntimeError("unexpected final_state from dense model boundary")
        intermediates = {
            "gk": gk, "aqk": aqk, "akk": akk, "w": w, "qg": qg,
            "kg": kg, "v_new": v_new, "h": h,
        }
        missing = [name for name, value in intermediates.items() if value is None]
        if missing:
            raise RuntimeError(f"forward omitted backward intermediates: {missing}")
        ctx.save_for_backward(
            q_norm, q_rstd, k_norm, k_rstd, v, raw_g, beta, a_log, dt_bias,
            gk, aqk, akk, w, qg, kg, v_new, h,
        )
        ctx.scale = float(scale)
        ctx.lower_bound = float(lower_bound)
        return o.type_as(q)

    @staticmethod
    def backward(ctx, d_o):
        (q, q_rstd, k, k_rstd, v, raw_g, beta, a_log, dt_bias, gk, aqk,
         akk, w, qg, kg, v_new, h) = ctx.saved_tensors
        heads, key_dim = q.shape[2:]
        dq_h, dk_h, dv_h, db_h, dg_h, d_a, d_bias = npu_chunk_kda_bwd(
            bsnd_to_bnsd(q), bsnd_to_bnsd(k), bsnd_to_bnsd(v),
            bsh_to_bhs(beta), gk, aqk, akk, w, qg, kg, v_new, h,
            bsnd_to_bnsd(d_o), ctx.scale, raw_g=bsnd_to_bnsd(raw_g),
            A_log=a_log, dt_bias=dt_bias.reshape(heads, key_dim).contiguous(),
            initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None,
            chunk_size=CHUNK_SIZE, safe_gate=True,
            lower_bound=ctx.lower_bound, use_gate_in_kernel=True,
            disable_recompute=True, use_exp2=True, state_v_first=False,
        )
        dq = l2norm_bwd(q, q_rstd, bnsd_to_bsnd(dq_h))
        dk = l2norm_bwd(k, k_rstd, bnsd_to_bsnd(dk_h))
        d_bias = d_bias.reshape(dt_bias.shape)
        return (
            dq, dk, bnsd_to_bsnd(dv_h), bnsd_to_bsnd(dg_h), bhs_to_bsh(db_h),
            d_a, d_bias, None, None,
        )


def parse_heads(text: str, head_count: int) -> torch.Tensor | None:
    if text.strip().lower() == "all":
        return None
    heads = [int(value) for value in text.split(",") if value.strip()]
    if not heads or min(heads) < 0 or max(heads) >= head_count:
        raise ValueError(f"invalid --heads={text!r} for H={head_count}")
    return torch.tensor(heads, dtype=torch.long)


def select_heads(x: torch.Tensor, axis: int, heads: torch.Tensor | None) -> torch.Tensor:
    return x if heads is None else x.index_select(axis, heads)


def parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_case(path: str, head_text: str):
    payload = torch.load(path, map_location="cpu")
    needed = ("q", "k", "v", "g_raw", "beta", "A_log", "dt_bias")
    missing = [name for name in needed if name not in payload]
    if missing:
        raise KeyError(f"input payload misses {missing}")
    if payload["q"].ndim != 4:
        raise ValueError(f"only dense BSND inputs are supported, got {tuple(payload['q'].shape)}")
    if not parse_bool(payload.get("use_gate_in_kernel"), True):
        raise ValueError("this model-boundary probe requires use_gate_in_kernel=True")
    if not parse_bool(payload.get("use_qk_l2norm_in_kernel"), True):
        raise ValueError("this model-boundary probe requires use_qk_l2norm_in_kernel=True")
    if not parse_bool(payload.get("safe_gate"), True):
        raise ValueError("this model-boundary probe requires safe_gate=True")

    heads = parse_heads(head_text, payload["q"].shape[2])
    key_dim = payload["q"].shape[-1]
    tensors = {
        "q": select_heads(payload["q"], 2, heads).contiguous(),
        "k": select_heads(payload["k"], 2, heads).contiguous(),
        "v": select_heads(payload["v"], 2, heads).contiguous(),
        "g_raw": select_heads(payload["g_raw"], 2, heads).contiguous(),
        "beta": select_heads(payload["beta"], 2, heads).contiguous(),
        "A_log": select_heads(payload["A_log"].float(), 0, heads).contiguous(),
        "dt_bias": select_heads(payload["dt_bias"].float().reshape(-1, key_dim), 0, heads)
                   .reshape(-1).contiguous(),
    }
    d_o = payload.get("d_o")
    if d_o is None:
        generator = torch.Generator().manual_seed(20260826)
        d_o = torch.randn(tensors["v"].shape, generator=generator,
                          dtype=tensors["v"].dtype)
    else:
        d_o = select_heads(d_o, 2, heads).contiguous()
    scale = float(payload.get("scale", key_dim ** -0.5))
    lower_bound = float(payload.get("lower_bound", -5.0))
    if lower_bound < -5.0 or lower_bound >= 0.0:
        raise ValueError(f"safe_gate lower_bound must be in [-5,0), got {lower_bound}")
    return tensors, d_o, scale, lower_bound


def make_leaves(cpu_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.to(DEVICE).detach().requires_grad_(True)
            for name, value in cpu_tensors.items()}


def to_cpu_result(o: torch.Tensor, grads: Iterable[torch.Tensor]):
    torch.npu.synchronize()
    return (o.detach().cpu(),) + tuple(grad.detach().cpu() for grad in grads)


def run_flan(cpu_tensors, d_o_cpu, scale, lower_bound):
    x = make_leaves(cpu_tensors)
    o, final_state = flan_chunk_kda(
        q=x["q"], k=x["k"], v=x["v"], g=x["g_raw"], beta=x["beta"],
        scale=scale, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=True, lower_bound=lower_bound, disable_recompute=True,
        A_log=x["A_log"], dt_bias=x["dt_bias"],
    )
    if final_state is not None:
        raise RuntimeError("FLAN reference unexpectedly returned final_state")
    ordered = tuple(x[name] for name in ("q", "k", "v", "g_raw", "beta", "A_log", "dt_bias"))
    return to_cpu_result(o, torch.autograd.grad(o, ordered, d_o_cpu.to(DEVICE)))


def run_ascendc(cpu_tensors, d_o_cpu, scale, lower_bound):
    x = make_leaves(cpu_tensors)
    o = AscendCModelBoundary.apply(
        x["q"], x["k"], x["v"], x["g_raw"], x["beta"],
        x["A_log"], x["dt_bias"], scale, lower_bound,
    )
    ordered = tuple(x[name] for name in ("q", "k", "v", "g_raw", "beta", "A_log", "dt_bias"))
    return to_cpu_result(o, torch.autograd.grad(o, ordered, d_o_cpu.to(DEVICE)))


def report_accuracy(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> bool:
    a, b = got.float(), ref.float()
    diff = (a - b).abs()
    bad = (~torch.isfinite(a)) | (~torch.isfinite(b)) | (diff > atol + rtol * b.abs())
    a_norm, b_norm = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    rel_l2 = torch.linalg.vector_norm(a - b) / b_norm.clamp_min(1e-30)
    cosine = (a.flatten() * b.flatten()).sum() / (a_norm * b_norm).clamp_min(1e-30)
    bad_count = int(bad.sum())
    print(
        f"{'PASS' if bad_count == 0 else 'FAIL'} accuracy {name:<9} "
        f"nan={int(torch.isnan(a).sum())}/{int(torch.isnan(b).sum())} "
        f"inf={int(torch.isinf(a).sum())}/{int(torch.isinf(b).sum())} "
        f"max_abs={float(diff.max()):.6e} mean_abs={float(diff.mean()):.6e} "
        f"rel_l2={float(rel_l2):.6e} cos={float(cosine):.8f} "
        f"norm={float(a_norm):.6e}/{float(b_norm):.6e} bad={bad_count}/{bad.numel()}",
        flush=True,
    )
    if bad_count:
        first = tuple(int(value) for value in bad.nonzero(as_tuple=False)[0])
        print(f"  first_bad={first} got={float(a[first]):.8e} ref={float(b[first]):.8e}", flush=True)
    return bad_count == 0


def report_repeat(name: str, current: torch.Tensor, first: torch.Tensor) -> bool:
    finite = torch.isfinite(current) & torch.isfinite(first)
    nan_pattern_changed = torch.isnan(current) ^ torch.isnan(first)
    inf_pattern_changed = torch.isinf(current) ^ torch.isinf(first)
    value_changed = finite & (current != first)
    changed = nan_pattern_changed | inf_pattern_changed | value_changed
    changed_count = int(changed.sum())
    finite_diff = (current.float() - first.float()).abs()[finite]
    max_abs = float(finite_diff.max()) if finite_diff.numel() else float("nan")
    print(
        f"{'STABLE' if changed_count == 0 else 'UNSTABLE'} repeat   {name:<9} "
        f"changed={changed_count}/{changed.numel()} "
        f"nan={int(torch.isnan(current).sum())}/{int(torch.isnan(first).sum())} "
        f"max_abs_finite={max_abs:.6e}",
        flush=True,
    )
    return changed_count == 0


def clear_npu_cache():
    gc.collect()
    torch.npu.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="torch.save payload produced by the model")
    parser.add_argument("--heads", default="all", help="original head ids, e.g. 1,15,19,33, or all")
    parser.add_argument("--repeats", type=int, default=2, help="AscendC launches; >=2 also checks stability")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    cpu_tensors, d_o, scale, lower_bound = load_case(args.input, args.heads)
    q, v = cpu_tensors["q"], cpu_tensors["v"]
    print(
        f"CASE B={q.shape[0]} S={q.shape[1]} H={q.shape[2]} K={q.shape[3]} V={v.shape[3]} "
        f"heads={args.heads} scale={scale} safe_gate=True use_gate_in_kernel=True "
        f"lower_bound={lower_bound} use_qk_l2norm_in_kernel=True repeats={args.repeats}",
        flush=True,
    )

    print("RUN FLAN_SMALL_MODEL_BOUNDARY", flush=True)
    reference = run_flan(cpu_tensors, d_o, scale, lower_bound)
    clear_npu_cache()

    baseline = None
    passed = True
    for index in range(args.repeats):
        print(f"RUN ASCENDC_MODEL_BOUNDARY {index + 1}/{args.repeats}", flush=True)
        current = run_ascendc(cpu_tensors, d_o, scale, lower_bound)
        for name, got, ref in zip(RESULT_NAMES, current, reference):
            atol = 1e-2 if name in {"o", "dv"} else args.atol
            passed &= report_accuracy(name, got, ref, atol, args.rtol)
        if baseline is None:
            baseline = current
        else:
            for name, got, first in zip(RESULT_NAMES, current, baseline):
                passed &= report_repeat(name, got, first)
        del current
        clear_npu_cache()

    print("RESULT:", "PASS" if passed else "FAIL", flush=True)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
