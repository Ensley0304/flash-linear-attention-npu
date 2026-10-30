#!/usr/bin/env python3
"""KDA forward/backward: no CP, AllGather, or PTP (recursive doubling).

A2/A3, B=1, Hk=Hv, K=V=128, BF16 QKV, FP32 gate/beta, chunk_size=64.
Contiguous equal token shards; packed sequences reset at global boundaries.
QK L2Norm, beta sigmoid and gate are fused; intermediates are saved (no recompute).
Explicit backward calls, no autograd wrapper or external state/state loss.

python examples/cp/chunk_kda_cp.py --communication none --accuracy-check
torchrun --standalone --nproc_per_node=4 examples/cp/chunk_kda_cp.py \
    --communication allgather --accuracy-check
Replace allgather with ptp to run the ordered P/E scan from MindSpeed issue #4.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--communication", choices=("none", "allgather", "ptp"), default="allgather")
    parser.add_argument("--tokens", type=int, default=256, help="global token count")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--cu-seqlens", help="global packed offsets, e.g. 0,130,256")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", type=int)
    parser.add_argument("--accuracy-check", action="store_true", help="compare with CPU recurrence (small inputs)")
    parser.add_argument("--cpu-reference-only", action="store_true")
    parser.add_argument("--reference", type=Path, help="compare with a saved no-CP NPU result")
    parser.add_argument("--save-result", type=Path, help="save outputs/gradients from a single-rank run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="save CANN op_summary and timeline after timing")
    parser.add_argument("--output-tol", type=float, default=0.005)
    parser.add_argument("--grad-tol", type=float, default=0.008)
    parser.add_argument("--gate-tol", type=float, default=0.02)
    parser.add_argument("--boundary-tol", type=float, default=0.02)
    args = parser.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.tokens <= 0 or args.tokens % world:
        parser.error("tokens must be positive and divisible by WORLD_SIZE")
    if not 1 <= args.heads <= 128 or args.warmup < 0 or args.iterations < 1:
        parser.error("heads must be in [1,128], warmup >= 0, iterations >= 1")
    if min(args.output_tol, args.grad_tol, args.gate_tol, args.boundary_tol) <= 0:
        parser.error("tolerances must be positive")
    if world > 1 and (args.communication == "none" or args.device is not None
                      or args.cpu_reference_only or args.save_result is not None):
        parser.error("none/device/cpu-reference-only/save-result require a single process")
    if args.profile and args.output_dir is None:
        parser.error("--profile requires --output-dir")
    if (args.accuracy_check or args.cpu_reference_only) and args.tokens * args.heads > 8192:
        parser.error("CPU autograd reference is limited to tokens*heads <= 8192; use --reference for large cases")
    try:
        args.cu = ([int(s) for s in args.cu_seqlens.split(",")]
                   if args.cu_seqlens else [0, args.tokens])
    except ValueError:
        parser.error("cu-seqlens must contain comma-separated integers")
    if (len(args.cu) < 2 or args.cu[0] != 0 or args.cu[-1] != args.tokens
            or any(b < a for a, b in zip(args.cu, args.cu[1:]))):
        parser.error("cu-seqlens must be nondecreasing from 0 to tokens")
    return args


def make_inputs(tokens, heads, seed, lo=0, hi=None, device="cpu"):
    """Generate identical global inputs, moving only this rank's shard to NPU."""
    gen = torch.Generator().manual_seed(seed)
    shape = (1, heads, tokens, 128)
    result = {}
    for name in ("q", "k", "v", "g", "do"):
        value = torch.randn(shape, generator=gen)
        if name == "g":
            value = value * 0.1 - 5.0  # slow decay keeps cross-rank dependence visible
        else:
            if name == "v":
                value *= 0.5
            value = value.bfloat16()
        result[name] = value[:, :, lo:hi].contiguous().to(device)
    beta = torch.randn(shape[:-1], generator=gen)
    result["beta"] = beta[:, :, lo:hi].contiguous().to(device)
    result["A_log"] = torch.zeros(heads, device=device)
    result["dt_bias"] = torch.zeros(heads, 128, device=device)
    return result


@torch.enable_grad()
def recurrent_reference(inputs, cu, local_tokens):
    """Independent FP32 recurrence; boundaries retain only future-loss gradients."""
    names = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
    leaves = {n: inputs[n].float().detach().requires_grad_() for n in names}
    q, k = (leaves[n] * torch.rsqrt(leaves[n].square().sum(-1, keepdim=True) + 1e-6)
            for n in ("q", "k"))
    beta = leaves["beta"].sigmoid()
    gate = -5.0 * torch.sigmoid(leaves["A_log"].exp()[None, :, None, None]
                               * (leaves["g"] + leaves["dt_bias"][None, :, None, :]))
    state = torch.zeros(1, q.shape[1], 128, 128)
    boundaries, outputs = [state], []
    for t in range(q.shape[2]):
        if t in cu:
            state = torch.zeros_like(state)
        state = state * gate[:, :, t].exp().unsqueeze(-1)
        key = k[:, :, t]
        delta = (leaves["v"][:, :, t] - (key.unsqueeze(-1) * state).sum(-2)) * beta[:, :, t, None]
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((q[:, :, t, :, None] * state).sum(-2) * 128 ** -0.5)
        if (t + 1) % local_tokens == 0:
            state = state + 0.0
            state.retain_grad()
            boundaries.append(state)
    output = torch.stack(outputs, dim=2)
    (output * inputs["do"].float()).sum().backward()
    result = {"o": output.detach().transpose(1, 2).contiguous()}
    result.update({"d" + n: (leaf.grad.transpose(1, 2).contiguous()
                             if n not in ("A_log", "dt_bias") else leaf.grad)
                   for n, leaf in leaves.items()})
    dstates = [torch.zeros_like(state) if s.grad is None else s.grad.detach() for s in boundaries]
    return result, [s.detach() for s in boundaries], dstates


def exchange_boundary(summary, plan, *, forward, communication):
    """Summary [E | M] maps H -> M @ H + E. Return only same-sequence state.

    PTP follows issue #4 section 4.6: inclusive P is sent, exclusive E is local.
    Every round sends the OLD P; compositions use fresh buffers. Reverse backward
    rank order, but keep the same matrix multiplication order. A packed reset
    zeroes outgoing M, so earlier sequences cannot leak through the scan.
    """
    rank, world = plan["rank"], plan["world"]
    count = plan["prefix" if forward else "suffix"]
    states = summary.new_zeros((len(plan["cu"]) - 1, summary.shape[0], 128, 128))
    if communication == "allgather":
        from fla_npu.ops.ascendc import merge_fwd_bwd_kernel
        gathered = [torch.empty_like(summary) for _ in range(world)]
        dist.all_gather(gathered, summary.contiguous())
        if count:
            merge_fwd_bwd_kernel(
                states[0 if forward else -1], torch.stack(gathered), count, rank,
                forward=forward, state_v_first=False,
            )
    elif communication == "ptp":
        logical_rank = rank if forward else world - 1 - rank
        inclusive = summary.contiguous()
        if len(plan["cu"]) > 2 or count == 0:
            inclusive = inclusive.clone()
            inclusive[..., 128:].zero_()
        exclusive = None  # identity map; first received prefix can be used directly
        distance = 1
        while distance < world:
            received = torch.empty_like(inclusive) if logical_rank >= distance else None
            operations = []
            if logical_rank + distance < world:
                peer = rank + distance if forward else rank - distance
                operations.append(dist.P2POp(dist.isend, inclusive, peer))
            if received is not None:
                peer = rank - distance if forward else rank + distance
                operations.append(dist.P2POp(dist.irecv, received, peer))
            if operations:
                for request in dist.batch_isend_irecv(operations):
                    request.wait()
            if received is not None:
                # compose(earlier, later) = [M_later E_earlier + E_later | M_later M_earlier]
                # Final inclusive maps with no subsequent sender are unused.
                if logical_rank + 2 * distance < world:
                    merged = torch.bmm(inclusive[..., 128:].contiguous(), received)
                    merged[..., :128].add_(inclusive[..., :128])
                    inclusive = merged
                if exclusive is None:
                    exclusive = received
                else:
                    merged = torch.bmm(exclusive[..., 128:].contiguous(), received)
                    merged[..., :128].add_(exclusive[..., :128])
                    exclusive = merged
            distance *= 2
        if count:
            states[0 if forward else -1].copy_(exclusive[..., :128])
    else:
        raise ValueError("boundary exchange requires allgather or ptp")
    # FwdH selects recurrent-state precision from h0: preserve FP32.
    # Dhu's terminal gradient ABI instead requires BF16.
    return states if forward else states.bfloat16()


def run_kda(x, plan, communication):
    """Keep the forward/backward operator chain visible in one place."""
    from fla_npu.ops import ascendc as op
    q, k, v, g, beta, do = (x[n] for n in ("q", "k", "v", "g", "beta", "do"))
    cu, scale = plan["cu"], 128 ** -0.5
    gate_options = dict(safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
                        A_log=x["A_log"], dt_bias=x["dt_bias"].flatten(), use_exp2=True,
                        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    if communication == "none":
        qhat, khat = torch.empty_like(q), torch.empty_like(k)
        qr, kr, beta_eff = (torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
                            for _ in range(3))
        o, _, gk, aqk, akk, w, u, qg, kg, vn, h, _ = op.chunk_kda_fwd(
            q, k, v, g, beta, scale, layout="BNSD", cu_seqlens=cu,
            disable_recompute=True, q_hat_out=qhat, k_hat_out=khat,
            q_rstd_out=qr, k_rstd_out=kr, beta_eff_out=beta_eff, **gate_options,
        )
        # PR #993: qhat/khat and saved caches stay head-major; layout controls v/dq/dk/dv.
        # do/raw_g/original beta are ALWAYS token-major; beta_eff is FP32 head-major.
        dq, dk, dv, db, dg, _, dA, dbias = op.chunk_kda_bwd(
            qhat, khat, v.transpose(1, 2).contiguous(), beta.transpose(1, 2).contiguous(),
            gk, aqk, akk, w, qg, kg, vn, h, do.transpose(1, 2).contiguous(), scale,
            raw_g=g.transpose(1, 2).contiguous(), A_log=x["A_log"], dt_bias=x["dt_bias"],
            q_rstd=qr, k_rstd=kr, beta_eff=beta_eff, layout="BSND", cu_seqlens=cu,
            safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, disable_recompute=True, use_exp2=True,
        )
        h0 = dht = None
    else:
        saved = op.chunk_kda_fwd_prepare(
            q, k, v, g, beta, scale, layout="BNSD", cu_seqlens=cu,
            backward_mode="save", **gate_options,
        )
        gk, aqk, akk, w, u, qg, kg, qgs, qhat, khat, qr, kr, beta_eff = saved
        summary = q.new_zeros((q.shape[1], 128, 256), dtype=torch.float32)
        if plan["suffix"]:
            summary = op.pre_process_fwd_kernel_merged(
                kg, w, u, gk=gk, cu_seqlens=cu[-2:], chunk_size=64,
            )[0]
        h0 = exchange_boundary(summary, plan, forward=True, communication=communication)
        h, vn, _ = op.chunk_fwd_h(
            kg, w, u, gk=gk, initial_state=h0, cu_seqlens=cu,
            save_new_value=True, output_final_state=False, use_exp2=True,
        )
        o = op.chunk_kda_fwd_finalize(qgs, aqk, vn, h, cu_seqlens=cu, output_layout="BSND")
        # Small Prepare's layout applies to ALL three token inputs, not only do.
        da, dv_local, dq_raw = op.chunk_kda_bwd_prepare(
            aqk, vn, do, h, scale=scale, layout="BNSD", cu_seqlens=cu,
        )
        summary = q.new_zeros((q.shape[1], 128, 256), dtype=torch.float32)
        if plan["prefix"]:
            summary = op.pre_process_bwd_kernel_merged(
                qg, kg, w, do, dv_local, scale, 64,
                gk=gk.to(q.dtype), cu_seqlens=cu[:2],
            )
        dht = exchange_boundary(summary, plan, forward=False, communication=communication)
        dh, _, dv_scan = op.chunk_gated_delta_rule_bwd_dhu(
            qg, kg, w, do, dv_local, scale, 64, gK=gk, dht=dht,
            cu_seqlens=cu, chunk_indices=plan["indices"], use_exp2=True,
        )
        # Unlike total backward, small Finalize takes head-major beta_eff directly.
        # output_layout now controls ALL five token gradients, including db/dg.
        dq, dk, dv, db, dg, dA, dbias = op.chunk_kda_bwd_finalize(
            qhat, khat, v, gk, g, beta_eff, x["A_log"], x["dt_bias"],
            akk, vn, h, dh, dv_scan, da, dq_raw, scale=scale,
            q_rstd=qr, k_rstd=kr, cu_seqlens=cu, output_layout="BSND",
            safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, use_exp2=True,
        )
        if plan["world"] > 1:
            dist.all_reduce(dA)  # shared parameters: sum, never average
            dist.all_reduce(dbias)
    result = dict(o=o, dq=dq, dk=dk, dv=dv, dbeta=db, dg=dg, dA_log=dA, ddt_bias=dbias)
    if h0 is not None:
        result.update(h0=h0, dht=dht)
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_tokens = args.tokens // world
    lo, hi = rank * local_tokens, (rank + 1) * local_tokens
    segments = [(a, b) for a, b in zip(args.cu, args.cu[1:]) if a < hi and b > lo and b > a]
    plan = dict(rank=rank, world=world, cu=[0] + [min(b, hi) - lo for a, b in segments],
                prefix=rank - segments[0][0] // local_tokens,
                suffix=(segments[-1][1] - 1) // local_tokens - rank)
    plan["indices"] = [value for seq, (a, b) in enumerate(zip(plan["cu"], plan["cu"][1:]))
                       for chunk in range((b - a + 63) // 64) for value in (seq, chunk)]
    if args.cpu_reference_only:
        result, _, _ = recurrent_reference(make_inputs(args.tokens, args.heads, args.seed), args.cu, local_tokens)
        assert all(torch.isfinite(t).all() for t in result.values())
        print("CPU reference PASS; no NPU operators executed")
        return 0
    import fla_npu
    import torch_npu
    device_id = int(os.environ.get("LOCAL_RANK", "0")) if args.device is None else args.device
    torch.npu.set_device(device_id)
    device_name = torch.npu.get_device_name(device_id)
    if "Ascend910B" not in device_name and "Ascend910_93" not in device_name:
        raise RuntimeError("Fused beta sigmoid backward example requires A2/A3; A5 is reserved")
    if args.communication != "none":
        # Establish a collective before the first P2P round (HCCL initialization).
        if world == 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("hccl", rank=rank, world_size=world, timeout=timedelta(seconds=180))
        dist.barrier()
    output_dir = args.output_dir / f"rank{rank}" if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        x = make_inputs(args.tokens, args.heads, args.seed, lo, hi, f"npu:{device_id}")
        metadata = dict(communication=args.communication, rank=rank, world=world, tokens=args.tokens,
                        heads=args.heads, cu_seqlens=args.cu, seed=args.seed, disable_recompute=True,
                        device=device_name, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                        package=fla_npu.__file__, local_plan={k: v for k, v in plan.items() if k != "indices"})
        print(json.dumps(metadata), flush=True)
        with torch.no_grad():
            actual = run_kda(x, plan, args.communication)
            torch.npu.synchronize()
            passed = all(bool(torch.isfinite(t).all()) for t in actual.values())
            checks = {}
            reference = None
            if args.accuracy_check:
                reference, states, dstates = recurrent_reference(
                    make_inputs(args.tokens, args.heads, args.seed), args.cu, local_tokens,
                )
            elif args.reference:
                payload = torch.load(args.reference, map_location="cpu", weights_only=True, mmap=True)
                for name in ("tokens", "heads", "cu_seqlens", "seed"):
                    if payload["metadata"][name] != metadata[name]:
                        raise ValueError(f"reference input mismatch: {name}")
                reference = payload["outputs"]
            if reference is not None:
                reference = {n: t if n in ("dA_log", "ddt_bias") else t[:, lo:hi]
                             for n, t in reference.items() if n not in ("h0", "dht")}
                if args.accuracy_check and args.communication != "none":
                    boundary = torch.zeros_like(actual["h0"], device="cpu")
                    terminal = torch.zeros_like(actual["dht"], device="cpu")
                    if plan["prefix"]:
                        boundary[0] = states[rank][0]
                    if plan["suffix"]:
                        terminal[-1] = dstates[rank + 1][0]
                    reference.update(h0=boundary, dht=terminal)
                for name, expected in reference.items():
                    value, expected = actual[name].float().cpu(), expected.float().contiguous()
                    if value.shape != expected.shape:
                        raise RuntimeError(f"{name}: {value.shape} != {expected.shape}")
                    tol = (args.boundary_tol if name in ("h0", "dht") else args.output_tol if name == "o"
                           else args.gate_tol if name in ("dbeta", "dg", "dA_log", "ddt_bias") else args.grad_tol)
                    delta = (value - expected).abs()
                    bad = int((delta > tol + tol * expected.abs()).sum())
                    finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
                    # Long model vectors need FP64 accumulation: FP32 norm/dot
                    # reductions can even yield a spurious cosine above one.
                    actual_norm = torch.linalg.vector_norm(value, dtype=torch.float64)
                    expected_norm = torch.linalg.vector_norm(expected, dtype=torch.float64)
                    rel_l2 = float(torch.linalg.vector_norm(delta, dtype=torch.float64)
                                   / expected_norm.clamp_min(1e-8))
                    cosine = (float(((value * expected).sum(dtype=torch.float64)
                                     / (actual_norm * expected_norm).clamp_min(1e-16)).clamp(-1, 1))
                              if expected_norm > 0 else float(bad == 0))
                    cos_min = 0.99 if name in ("dbeta", "dg", "dA_log", "ddt_bias") else 0.999
                    ok = finite and bad == 0 and cosine >= cos_min
                    checks[name] = dict(passed=ok, mismatches=bad, rel_l2=rel_l2,
                                        max_abs=float(delta.max()), cosine=cosine, tol=tol, cos_min=cos_min)
                    passed &= ok
                    print(f"rank={rank} {name}: {json.dumps(checks[name])}", flush=True)
            if args.save_result:
                args.save_result.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(metadata=metadata, outputs={n: t.cpu() for n, t in actual.items()}), args.save_result)
            del actual, reference
            flag = torch.tensor([int(passed)], dtype=torch.int32, device=f"npu:{device_id}")
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            passed = bool(flag.item())
            # Validation and CPU copies are excluded from timing/profiling.
            for _ in range(args.warmup):
                run_kda(x, plan, args.communication)
            torch.npu.synchronize()
            if dist.is_initialized():
                dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                run_kda(x, plan, args.communication)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / args.iterations
            times = torch.tensor([elapsed], device=f"npu:{device_id}", dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir / "profiling")),
                    record_shapes=True,
                ) as prof:
                    run_kda(x, plan, args.communication)
                    torch.npu.synchronize()
                    prof.step()
            metadata.update(passed=passed, accuracy_checked=bool(checks), checks=checks,
                            warmup=args.warmup, iterations=args.iterations,
                            rank_ms=elapsed, max_rank_ms=float(times.item()))
            if output_dir:
                (output_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"RESULT {json.dumps(metadata)}", flush=True)
            return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
