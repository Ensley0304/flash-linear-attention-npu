#!/usr/bin/env python3
"""Repeatability probe for an exported KDA backward input.

The payload is a CPU ``torch.save`` mapping with BSND ``q/k/v/g_raw`` and
BSN ``beta``.  ``npu:0`` deliberately refers to the device selected by
``ASCEND_RT_VISIBLE_DEVICES``; select a physical card outside this script.
"""

import argparse
import gc

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import npu_chunk_kda_bwd, npu_chunk_kda_fwd
from fla_npu.ops.triton import l2norm_fwd


DEVICE = "npu:0"
CHUNK_SIZE = 64
OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias")


def is_true(value):
    return str(value).lower() == "true"


def bsnd_to_bnsd(tensor):
    return tensor.permute(0, 2, 1, 3).contiguous()


def load_input(path):
    data = torch.load(path, map_location="cpu")
    q, k, v = (data[name].to(DEVICE) for name in ("q", "k", "v"))
    beta = data["beta"].to(DEVICE)
    if is_true(data.get("use_qk_l2norm_in_kernel", False)):
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    raw_g = data["g_raw"].to(DEVICE)
    heads, key_dim = q.shape[2:]
    a_log = data["A_log"].to(DEVICE).float()
    dt_bias = data["dt_bias"].to(DEVICE).float()
    lower_bound = float(data["lower_bound"])
    gate = raw_g.float()
    if is_true(data.get("use_gate_in_kernel", False)):
        rate = a_log.exp().view(1, 1, heads, 1)
        bias = dt_bias.view(1, 1, heads, key_dim)
        gate = lower_bound * torch.sigmoid(rate * (gate + bias))
    return q, k, v, beta, gate.contiguous(), raw_g.contiguous(), a_log, dt_bias, \
        float(data["scale"]), lower_bound, data.get("d_o")


def launch(q, k, v, beta, gate, raw_g, a_log, dt_bias, d_o, scale,
           lower_bound, gate_in_kernel):
    fwd = npu_chunk_kda_fwd(
        q, k, v, raw_g if gate_in_kernel else gate, beta, scale,
        chunk_size=CHUNK_SIZE, layout="BSND", initial_state=None,
        output_final_state=False, cu_seqlens=None, chunk_indices=None,
        safe_gate=gate_in_kernel, lower_bound=lower_bound,
        use_gate_in_kernel=gate_in_kernel,
        A_log=a_log if gate_in_kernel else None,
        dt_bias=dt_bias if gate_in_kernel else None,
        disable_recompute=True, return_intermediate_states=False,
        state_v_first=False,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = fwd
    return npu_chunk_kda_bwd(
        bsnd_to_bnsd(q), bsnd_to_bnsd(k), bsnd_to_bnsd(v),
        beta.permute(0, 2, 1).contiguous(), gk, aqk, akk, w, qg, kg,
        v_new, h, bsnd_to_bnsd(d_o), scale,
        raw_g=bsnd_to_bnsd(raw_g) if gate_in_kernel else None,
        A_log=a_log if gate_in_kernel else None,
        dt_bias=dt_bias.view(q.shape[2], q.shape[3]) if gate_in_kernel else None,
        initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None,
        chunk_size=CHUNK_SIZE, safe_gate=gate_in_kernel,
        lower_bound=lower_bound, use_gate_in_kernel=gate_in_kernel,
        disable_recompute=True, use_exp2=True, state_v_first=False,
    )


def churn_allocator():
    scratch = torch.full((128, 1024, 1024), 1.2345678e30, device=DEVICE)
    del scratch
    gc.collect()
    torch.npu.synchronize()


def report(name, first, second):
    changed = first != second
    count = int(changed.sum())
    nan_first = int(first.float().isnan().sum())
    nan_second = int(second.float().isnan().sum())
    verdict = "OK" if count == 0 else "DIFF"
    print(
        f"{verdict} {name:<6} changed={count:,} "
        f"({count / first.numel() * 100:.5f}%) nan={nan_first}/{nan_second} "
        f"max={first.float().abs().max():.5e}/{second.float().abs().max():.5e}")
    if count and changed.ndim >= 3:
        indices = changed.nonzero(as_tuple=False)
        print("  heads:", sorted(indices[:, 1].unique().tolist())[:12])
        print("  chunks:", sorted((indices[:, 2] // CHUNK_SIZE).unique().tolist())[:12])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CPU torch.save payload")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--churn", action="store_true")
    parser.add_argument("--in-kernel-gate", action="store_true")
    args = parser.parse_args()

    q, k, v, beta, gate, raw_g, a_log, dt_bias, scale, lower_bound, d_o = load_input(args.input)
    if d_o is None:
        d_o = torch.randn(v.shape, generator=torch.Generator().manual_seed(7))
    d_o = d_o.to(DEVICE).to(v.dtype)
    print(f"replay: B={q.shape[0]} S={q.shape[1]} H={q.shape[2]} K={q.shape[3]} "
          f"V={v.shape[-1]} C={CHUNK_SIZE} in_kernel_gate={args.in_kernel_gate}")
    for iteration in range(args.repeats):
        first = [None if value is None else value.clone() for value in launch(
            q, k, v, beta, gate, raw_g, a_log, dt_bias, d_o, scale,
            lower_bound, args.in_kernel_gate)]
        if args.churn:
            churn_allocator()
        second = launch(q, k, v, beta, gate, raw_g, a_log, dt_bias, d_o,
                        scale, lower_bound, args.in_kernel_gate)
        print(f"--- repeat {iteration + 1}/{args.repeats} ---")
        for name, value_a, value_b in zip(OUTPUT_NAMES, first, second):
            if value_a is not None:
                report(name, value_a, value_b)


if __name__ == "__main__":
    main()
