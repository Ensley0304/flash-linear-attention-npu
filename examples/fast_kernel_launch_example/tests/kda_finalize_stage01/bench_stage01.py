#!/usr/bin/env python3
"""Profile a capture-disabled Stage 0-5 build; report latency from msprof CSVs."""
import argparse
import os
import signal
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--cores", type=int, default=0)
    parser.add_argument("--stage", choices=("0", "1", "2", "3", "4", "5", "both"), default="both")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    simulator = os.environ.get("IS_SIMULATOR_ENV", "").lower() == "true"
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    cu = []
    chunks = (args.tokens + 63) // 64
    if args.varlen:
        assert args.tokens == 8192, "the fixed comparison case uses 8192 packed tokens"
        lengths = [0, 1, 31, 64, 65, 4096, 3935, 0]
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        chunks = sum((length + 63) // 64 for length in lengths)
    shape = (args.heads, args.tokens, 128) if args.varlen else (1, args.heads, args.tokens, 128)
    state = (chunks, args.heads, 128, 128) if args.varlen else (1, chunks, args.heads, 128, 128)
    device = f"npu:{args.device}"
    source_device = "cpu" if simulator else device
    # CPU initialization avoids unrelated random-number kernels in simulator traces.
    def rand(dims):
        return (torch.randn(dims, device=source_device) * 0.125).bfloat16().to(device)

    k, v, v_new, dv_scan = [rand(shape) for _ in range(4)]
    gk = (-torch.rand(shape, device=source_device) * 32).to(device)
    akk = rand((*shape[:-1], 64))
    h, dh = rand(state), rand(state)
    beta = torch.rand(shape[:-1], device=source_device).to(device)
    dv = torch.empty_like(k)
    q = rand(shape)
    dq_raw = (torch.randn(shape, device=source_device) * .125).to(device)
    torch.npu.synchronize()
    stages = (0, 1) if args.stage == "both" else (int(args.stage),)
    for stage in stages:
        for _ in range(args.warmup + args.repeat):
            if not simulator:
                signal.alarm(60)
            capture, _, _ = torch.ops.ascend_ops.kda_finalize_stage01(
                k, v, gk, akk, v_new, h, dh, dv_scan, cu, args.cores, stage, beta, dv, q, dq_raw, 128 ** -.5)
            torch.npu.synchronize()
            signal.alarm(0)
            assert capture.numel() == 1, "rebuild with KDA_FINALIZE_STAGE01_CAPTURE=0"
        print(f"DONE stage={stage} H={args.heads} T={args.tokens} varlen={args.varlen} "
              f"warmup={args.warmup} samples={args.repeat}", flush=True)


if __name__ == "__main__":
    main()
