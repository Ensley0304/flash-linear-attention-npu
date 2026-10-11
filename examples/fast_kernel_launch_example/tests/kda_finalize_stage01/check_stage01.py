#!/usr/bin/env python3
"""Private Stage 0/1 checks; not an end-to-end Finalize accuracy claim."""
import argparse
import os
import signal
from pathlib import Path

import torch
import torch_npu  # noqa: F401

CAPTURE_BYTES = 160 * 1024
OFFSETS = {
    "dk_state_raw": (0, torch.float32, (64, 128)),
    "dv_b": (32 * 1024, torch.float32, (64, 128)),
    "zV": (64 * 1024, torch.float32, (64, 64)),
    "dW": (80 * 1024, torch.bfloat16, (64, 128)),
    "exp2_gk": (96 * 1024, torch.float32, (64, 128)),
    "kE": (128 * 1024, torch.bfloat16, (64, 128)),
    "rH": (144 * 1024, torch.float32, (128,)),
    "gk_last": (144 * 1024 + 512, torch.float32, (128,)),
}


def make_case(batch, heads, tokens, lengths=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    variable = lengths is not None
    shape = (heads, tokens, 128) if variable else (batch, heads, tokens, 128)

    def rand(dims):
        return (torch.randn(dims, generator=generator) * 0.125).to(torch.bfloat16)

    k, v, v_new, dv_scan = [rand(shape) for _ in range(4)]
    akk = rand((*shape[:-1], 64))
    gk = torch.empty(shape, dtype=torch.float32)
    cu = [0]
    if variable:
        for length in lengths:
            cu.append(cu[-1] + length)
    chunks = []
    for sequence in range(len(lengths) if variable else batch):
        begin, end = (cu[sequence], cu[sequence + 1]) if variable else (0, tokens)
        for local, start in enumerate(range(begin, end, 64)):
            rows = min(64, end - start)
            chunks.append((sequence, local, start, rows))
            gate = -torch.rand((heads, rows, 128), generator=generator) * 5.0
            # FP32 cumsum is local to a chunk and is stored in base-2 units.
            gate = gate.cumsum(1) / torch.log(torch.tensor(2.0))
            if variable:
                gk[:, start:start + rows] = gate
            else:
                gk[sequence, :, start:start + rows] = gate
    state_shape = (len(chunks), heads, 128, 128) if variable else (
        batch, (tokens + 63) // 64, heads, 128, 128)
    h, dh = rand(state_shape), rand(state_shape)
    return (k, v, gk, akk, v_new, h, dh, dv_scan), cu if variable else [], chunks


def reference(inputs, chunks, head, index, variable):
    k, v, gk, akk, v_new, h, dh, dv_scan = inputs
    sequence, local, start, rows = chunks[index]

    def token(x, width):
        out = torch.zeros((64, width), dtype=torch.float64)
        src = x[head] if variable else x[sequence, head]
        out[:rows] = src[start:start + rows].double()
        return out

    kk, vv, new, scan = (token(x, 128) for x in (k, v, v_new, dv_scan))
    a = token(akk, 64)
    a[:, rows:] = 0
    hh = (h[index, head] if variable else h[sequence, local, head]).double()
    dd = (dh[index, head] if variable else dh[sequence, local, head]).double()
    gate = token(gk, 128).float()
    exponent = torch.zeros_like(gate)
    exponent[:rows] = torch.exp2(gate[:rows])
    return {
        "dk_state_raw": (new @ dd.T).float(),
        "dv_b": (a.T @ scan).float(),
        "dW": (scan @ hh.T).float().bfloat16(),
        "zV": (scan @ vv.T).float(),
        "rH": (hh * dd).sum(-1).float(),
        "gk_last": gate[rows - 1],
        "exp2_gk": exponent,
        "kE": (kk.float() * exponent).bfloat16(),
    }


def decode(record, name):
    offset, dtype, shape = OFFSETS[name]
    elements = 1
    for size in shape:
        elements *= size
    size = elements * torch.empty((), dtype=dtype).element_size()
    return record[offset:offset + size].clone().view(dtype).reshape(shape)


def check_case(args, batch, heads, tokens, lengths=None):
    inputs, cu, chunks = make_case(batch, heads, tokens, lengths)
    expected = None if args.workspace_only else [
        [reference(inputs, chunks, head, c, lengths is not None) for c in range(len(chunks))]
        for head in range(heads)]
    device_inputs = [x.to(f"npu:{args.device}") for x in inputs]
    torch.npu.synchronize()
    for iteration in range(args.repeat):
        # The alarm covers kernel execution and the following device synchronize.
        signal.alarm(60)
        try:
            capture, trace, workspace = torch.ops.ascend_ops.kda_finalize_stage01(
                *device_inputs, cu, args.cores, args.stage)
            torch.npu.synchronize()
        finally:
            signal.alarm(0)
        capture, trace, workspace = capture.cpu(), trace.cpu(), workspace.cpu()
        if args.workspace_only:
            assert capture.numel() == 1, "workspace-only requires a capture-disabled build"
            used_cores = workspace.numel() // (1024 * 1024)
            works = ((heads + 1) // 2) * len(chunks)
            for core in range(used_cores):
                work = core + ((works - 1 - core) // used_cores) * used_cores
                c = work % len(chunks)
                for owner in range(2):
                    head = (work // len(chunks)) * 2 + owner
                    if head >= heads:
                        continue
                    wanted = reference(inputs, chunks, head, c, lengths is not None)
                    base = (2 * core + owner) * 512 * 1024
                    record = workspace[base:base + 512 * 1024]
                    for name in ("dk_state_raw", "dv_b", "zV"):
                        actual = decode(record, name)
                        assert torch.isfinite(actual).all(), name
                        torch.testing.assert_close(actual, wanted[name], rtol=3e-4, atol=2e-6)
                    if args.stage == 1:
                        actual = record[144 * 1024:160 * 1024].clone().view(torch.bfloat16).reshape(64, 128)
                        assert torch.isfinite(actual).all(), "kE"
                        torch.testing.assert_close(actual.float(), wanted["kE"].float(), rtol=8e-3, atol=2e-5)
            print(f"PASS workspace stage={args.stage} B={batch} H={heads} T={tokens} "
                  f"varlen={lengths is not None} cores={used_cores} repeat={iteration + 1}", flush=True)
            continue
        names = list(OFFSETS)[:4] if args.stage == 0 else list(OFFSETS)
        # Batch comparisons keep the full 8K check practical without sampling or
        # changing the per-element tolerance / independent CPU reference.
        for name in names:
            offset, dtype, shape = OFFSETS[name]
            elements = 1
            for size in shape:
                elements *= size
            size = elements * torch.empty((), dtype=dtype).element_size()
            actual = capture[:, :, offset:offset + size].contiguous().view(dtype).reshape(
                heads, len(chunks), *shape)
            wanted = torch.stack([torch.stack([expected[head][c][name] for c in range(len(chunks))])
                                  for head in range(heads)])
            assert torch.isfinite(actual).all(), name
            if name in ("dW", "kE"):
                rtol, atol = 8e-3, 2e-5
            elif name in ("exp2_gk", "gk_last"):
                rtol, atol = 3e-5, 1e-7
            else:
                rtol, atol = 3e-4, 2e-6
            torch.testing.assert_close(
                actual.float(), wanted.float(), rtol=rtol, atol=atol,
                msg=lambda msg: f"{name}, indices=(head,chunk,...): {msg}")
        assert (capture[:, :, 145 * 1024:] == 0xA5).all(), "capture guard overwritten"
        works = ((heads + 1) // 2) * len(chunks)
        assert trace.shape == (works * 2, 16)
        used_cores = int(trace[0, 6])
        for work in range(works):
            for owner in range(2):
                record = trace[2 * work + owner].tolist()
                head = (work // len(chunks)) * 2 + owner
                assert record[:7] == [
                    work, work % used_cores, owner, head, work % len(chunks),
                    work // used_cores + 1, used_cores,
                ], record
                assert record[7:11] == [
                    chunks[work % len(chunks)][3], int(head < heads), args.stage, 7,
                ], record
                assert record[11] in (used_cores, 2 * used_cores), record
                assert record[12:15] == [
                    2, 2 * (work % used_cores) + owner,
                    (2 * (work % used_cores) + owner) * 512 * 1024,
                ], record
        # Read the actual reusable slots, not just the extra capture writes.
        for core in range(used_cores):
            work = core + ((works - 1 - core) // used_cores) * used_cores
            c = work % len(chunks)
            for owner in range(2):
                head = (work // len(chunks)) * 2 + owner
                if head >= heads:
                    continue
                base = (2 * core + owner) * 512 * 1024
                record = workspace[base:base + 512 * 1024]
                for name in ("dk_state_raw", "dv_b", "zV"):
                    torch.testing.assert_close(decode(record, name), decode(capture[head, c], name),
                                               rtol=0, atol=0)
                if args.stage == 1:
                    actual_ke = record[144 * 1024:160 * 1024].clone().view(torch.bfloat16).reshape(64, 128)
                    torch.testing.assert_close(actual_ke, decode(capture[head, c], "kE"), rtol=0, atol=0)
                assert (record[80 * 1024:144 * 1024] == 0x5A).all(), "workspace guard overwritten"
                assert (record[160 * 1024:] == 0x5A).all(), "workspace guard overwritten"
        if args.cores == 1 and works > 1:
            assert trace[-1, 5] > 1, "workspace reuse was not exercised"
        print(f"PASS stage={args.stage} B={batch} H={heads} T={tokens} "
              f"varlen={lengths is not None} cores={used_cores} repeat={iteration + 1}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--suite", choices=("smoke", "tails", "model"), default="smoke")
    parser.add_argument("--workspace-only", action="store_true")
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    cases = [(1, 3, 65, None), (1, 3, 99, [0, 1, 0, 33, 65, 0])]
    if args.suite == "tails":
        cases = [(1, 3, t, None) for t in (1, 16, 31, 32, 33, 63, 64, 129)]
        cases += [(2, 3, 65, None), (1, 96, 65, None), (1, 3, 289, [0, 1, 31, 32, 33, 63, 64, 65, 0])]
    elif args.suite == "model":
        cases = [(1, 96, 8192, None), (1, 96, 8192, [0, 1, 31, 64, 65, 4096, 3935, 0])]
    for case in cases:
        check_case(args, *case)


if __name__ == "__main__":
    main()
