"""Run the A2 split/merged finalize schedules with identical saved tensors."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import _aclnn_ctypes as aclnn


OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dA", "dBias")


def _full(shape, value, *, dtype, device):
    return torch.full(shape, value, dtype=dtype, device=device)


def make_inputs(
    batch: int, heads: int, tokens: int, device: torch.device,
    with_dt_bias: bool = True,
    nonzero_gate: bool = False,
):
    if tokens % 64:
        raise ValueError("This A/B harness requires T to be divisible by 64.")
    token_shape = (batch, heads, tokens, 128)
    scalar_shape = (batch, heads, tokens)
    chunks = tokens // 64
    bf16 = torch.bfloat16

    eye = torch.full((64, 64), 1.0 / 64.0, dtype=bf16, device=device)
    akk = (
        eye.view(1, 1, 1, 64, 64)
        .expand(batch, heads, chunks, 64, 64)
        .reshape(batch, heads, tokens, 64)
        .contiguous()
    )
    inputs = {
        "q": _full(token_shape, 0.03125, dtype=bf16, device=device),
        "k": _full(token_shape, 0.0234375, dtype=bf16, device=device),
        "v": _full(token_shape, 0.015625, dtype=bf16, device=device),
        "beta": _full(scalar_shape, 0.5, dtype=bf16, device=device),
        "gk": torch.zeros(token_shape, dtype=torch.float32, device=device),
        "Aqk": torch.zeros(
            (batch, heads, tokens, 64), dtype=bf16, device=device
        ),
        "Akk": akk,
        "v_new": _full(token_shape, 0.013671875, dtype=bf16, device=device),
        "h": torch.zeros(
            (batch, chunks, heads, 128, 128), dtype=bf16, device=device
        ),
        "dh": _full(
            (batch, chunks, heads, 128, 128),
            0.00390625,
            dtype=bf16,
            device=device,
        ),
        "dv_scan": _full(token_shape, 0.017578125, dtype=bf16, device=device),
        "d_aqk": _full(
            (batch, heads, tokens, 64),
            0.005859375,
            dtype=torch.float32,
            device=device,
        ),
        "dq_raw": _full(
            token_shape, 0.009765625, dtype=torch.float32, device=device
        ),
        "raw_g": torch.zeros(token_shape, dtype=torch.float32, device=device),
        "A_log": torch.zeros((heads,), dtype=torch.float32, device=device),
        "dt_bias": (
            torch.zeros((heads, 128), dtype=torch.float32, device=device)
            if with_dt_bias else None
        ),
    }
    if nonzero_gate:
        inputs["raw_g"].fill_(0.125)
        inputs["A_log"].fill_(-0.25)
        if inputs["dt_bias"] is not None:
            inputs["dt_bias"].fill_(0.0625)
    return inputs


def launch(inputs, use_gate: bool = True):
    outputs = (
        torch.full_like(inputs["q"], -7.0),
        torch.full_like(inputs["k"], -7.0),
        torch.full_like(inputs["v"], -7.0),
        torch.full_like(inputs["beta"], -7.0),
        torch.full_like(inputs["gk"], -7.0),
        torch.full_like(inputs["A_log"], -7.0),
        (
            torch.full_like(inputs["dt_bias"], -7.0)
            if inputs["dt_bias"] is not None else None
        ),
    )

    def tensor(ctx, value, name):
        return ctx.tensor(
            value,
            name,
            storage_shape_override=None if value is None else tuple(value.shape),
        )

    args = (
        "q", "k", "v", "gk", "raw_g", "beta", "A_log", "dt_bias",
        "Akk", "v_new", "h", "dh", "dv_scan", "d_aqk", "dq_raw",
    )
    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdFinalize",
        lambda ctx: [
            *[tensor(ctx, inputs[name], name) for name in args],
            tensor(ctx, None, "q_rstd"),
            tensor(ctx, None, "k_rstd"),
            ctx.int_array(None),
            ctx.int_array(None),
            ctypes.c_double(128**-0.5),
            ctypes.c_double(-5.0),
            ctypes.c_int64(64),
            ctypes.c_bool(True),
            ctypes.c_bool(use_gate),
            ctypes.c_bool(True),
            ctypes.c_bool(False),
            *[
                tensor(ctx, output, name)
                for output, name in zip(outputs, OUTPUT_NAMES)
            ],
        ],
        outputs,
    )


def tensor_summary(name, value):
    if value is None:
        return {"name": name, "none": True}
    fp32 = value.float()
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(fp32).all().item()),
        "sum": float(fp32.sum().item()),
        "abs_max": float(fp32.abs().max().item()),
    }


def run(args):
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    inputs = make_inputs(
        args.batch, args.heads, args.tokens, device,
        args.with_dt_bias, args.nonzero_gate,
    )
    for _ in range(args.warmup):
        outputs = launch(inputs, args.use_gate)
    torch.npu.synchronize()

    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = launch(inputs, args.use_gate)
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    summaries = [
        tensor_summary(name, output)
        for name, output in zip(OUTPUT_NAMES, outputs)
    ]
    if not all(item.get("none") or item["finite"] for item in summaries):
        raise RuntimeError("non-finite output detected")
    if args.save:
        saved = {
            name: None if output is None else output.detach().cpu()
            for name, output in zip(OUTPUT_NAMES, outputs)
        }
        torch.save(saved, args.save)

    ordered = sorted(samples_ms)
    result = {
        "case": {
            "batch": args.batch,
            "heads": args.heads,
            "tokens": args.tokens,
            "chunk": 64,
        },
        "device": torch.npu.get_device_name(args.device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_ms": {
            "median": statistics.median(samples_ms),
            "min": ordered[0],
            "max": ordered[-1],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        },
        "outputs": summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def compare(first: Path, second: Path):
    lhs = torch.load(first, map_location="cpu", weights_only=True)
    rhs = torch.load(second, map_location="cpu", weights_only=True)
    result = {}
    for name in OUTPUT_NAMES:
        a, b = lhs[name], rhs[name]
        if a is None or b is None:
            result[name] = {"same_none": a is None and b is None}
            continue
        diff = (a.float() - b.float()).abs()
        flat_diff = diff.reshape(-1)
        max_index = int(flat_diff.argmax().item())
        nonzero = int(torch.count_nonzero(flat_diff).item())
        first_different = torch.nonzero(flat_diff, as_tuple=False)[:16]
        result[name] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "equal": bool(torch.equal(a, b)),
            "different_elements": nonzero,
            "first_max_flat_index": max_index,
            "lhs_at_max": float(a.reshape(-1)[max_index].float().item()),
            "rhs_at_max": float(b.reshape(-1)[max_index].float().item()),
            "first_different_flat_indices": [
                int(index.item()) for index in first_different
            ],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--nonzero-gate", action="store_true")
    parser.add_argument(
        "--use-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--with-dt-bias", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compare", nargs=2, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.compare:
        compare(*parsed.compare)
    else:
        run(parsed)
