#!/usr/bin/env python3
"""Benchmark the installed ChunkKdaBwdIntra AscendC operator."""

import argparse
import gc
import statistics
import time

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Logical NPU device ID")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64, choices=(64, 128))
    parser.add_argument("--gate-scale", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--unsafe-gate", action="store_true")
    return parser.parse_args()


def make_inputs(args):
    torch.manual_seed(args.seed)
    feature_shape = (args.batch, args.tokens, args.heads, args.head_dim)
    gate_shape = (args.batch, args.tokens, args.value_heads, args.head_dim)
    scalar_shape = (args.batch, args.tokens, args.value_heads)
    da_shape = (args.batch, args.tokens, args.value_heads, args.chunk_size)

    q = torch.randn(feature_shape, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    g = -torch.rand(gate_shape, dtype=torch.float32).cumsum(dim=1) * args.gate_scale
    beta = torch.sigmoid(torch.randn(scalar_shape, dtype=torch.float32))
    d_a_qk = torch.randn(da_shape, dtype=torch.float32) * 0.02
    d_a_kk = torch.randn_like(d_a_qk) * 0.02
    dq = torch.randn(gate_shape, dtype=torch.float32) * 0.01
    dk = torch.randn_like(dq) * 0.01
    db = torch.randn(scalar_shape, dtype=torch.float32) * 0.01
    dg = torch.randn_like(dq) * 0.01
    return q, k, g, beta, d_a_qk, d_a_kk, dq, dk, db, dg


def main():
    args = parse_args()
    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("--repeat must be positive and --warmup must be non-negative")
    if args.value_heads < args.heads or args.value_heads % args.heads != 0:
        raise ValueError("--value-heads must be divisible by --heads")

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    print(
        "shape="
        f"B{args.batch}_T{args.tokens}_H{args.heads}_HV{args.value_heads}_"
        f"K{args.head_dim}_BT{args.chunk_size}_BF16_"
        f"{'unsafe' if args.unsafe_gate else 'safe'}",
        flush=True,
    )
    print("creating CPU inputs", flush=True)
    cpu_inputs = make_inputs(args)
    print(f"copying inputs to {device}", flush=True)
    device_inputs = tuple(tensor.to(device) for tensor in cpu_inputs)
    del cpu_inputs
    gc.collect()
    torch.npu.synchronize()

    def launch():
        return ascendc.chunk_kda_bwd_intra(
            *device_inputs,
            chunk_size=args.chunk_size,
            safe_gate=not args.unsafe_gate,
        )

    print(f"warmup={args.warmup}", flush=True)
    for _ in range(args.warmup):
        outputs = launch()
        torch.npu.synchronize()
        del outputs

    samples = []
    print(f"repeat={args.repeat}", flush=True)
    for index in range(args.repeat):
        torch.npu.synchronize()
        begin = time.perf_counter()
        outputs = launch()
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - begin) * 1000.0
        samples.append(elapsed_ms)
        del outputs
        print(f"sample[{index}]={elapsed_ms:.3f} ms", flush=True)

    print(
        f"e2e_ms: median={statistics.median(samples):.3f}, "
        f"mean={statistics.mean(samples):.3f}, "
        f"min={min(samples):.3f}, max={max(samples):.3f}",
        flush=True,
    )
    print("samples_ms:", ",".join(f"{sample:.3f}" for sample in samples), flush=True)


if __name__ == "__main__":
    main()
