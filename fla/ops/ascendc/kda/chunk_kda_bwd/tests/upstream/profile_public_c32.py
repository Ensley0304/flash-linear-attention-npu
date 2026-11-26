"""Capture complete public C32/C64 calls; no numerical reference in measured ranges.

Two warmups and configurable measured calls per mode. Inspect every AI Core/Vector
task in each exported kernel_details.csv, including transforms and fills.
This checks launch count, not a speedup target or production throughput.
"""
import argparse
from pathlib import Path

import torch
import torch_npu
from fla_npu.ops.ascendc import npu_chunk_kda_fwd, npu_chunk_kda_bwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, choices=(32, 64), default=32)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--tokens', type=int, default=65)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--keys', type=int, default=64)
    parser.add_argument('--values', type=int, default=32)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='float16')
    parser.add_argument('--backward-only', action='store_true')
    parser.add_argument('--no-state', action='store_true')
    parser.add_argument('--raw-safe', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(21)
    device = 'npu'
    dtype = getattr(torch, args.dtype)
    q = torch.nn.functional.normalize(torch.randn(1, args.tokens, args.heads, args.keys), dim=-1).to(device, dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q.cpu()), dim=-1).to(device, dtype)
    v = torch.randn(1, args.tokens, args.heads, args.values, device=device, dtype=dtype)
    g = torch.full(q.shape, -.1, device=device, dtype=torch.float32)
    beta = torch.full(q.shape[:-1], .5, device=device, dtype=dtype)
    h0 = torch.randn(1, args.heads, args.keys, args.values, device=device, dtype=torch.float32) * .1
    dht = torch.randn_like(h0)
    if args.no_state:
        h0 = None
        dht = None
    do = torch.randn_like(v)
    gate_options = dict(use_gate_in_kernel=False, safe_gate=False)
    if args.raw_safe:
        gate_options = dict(use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.,
                            A_log=torch.zeros(args.heads, device=device, dtype=torch.float32),
                            dt_bias=torch.zeros(args.heads * args.keys, device=device, dtype=torch.float32))
    def forward(save):
        return npu_chunk_kda_fwd(q, k, v, g, beta, .37, chunk_size=args.chunk_size,
            initial_state=h0, output_final_state=True, disable_recompute=save, **gate_options)

    saved = forward(True)
    recomputed = forward(False)
    def backward(result, save):
        _, _, G, aqk, akk, w, _, qg, kg, vn, h, _ = result
        return npu_chunk_kda_bwd(q, k, v, beta, G, aqk, akk, w, qg, kg, vn, h, do,
            .37, raw_g=g, initial_state=h0, dht=dht, chunk_size=args.chunk_size,
            layout='BSND', disable_recompute=save, **gate_options)

    modes = {
        'forward_save': lambda: forward(True),
        'forward_recompute': lambda: forward(False),
        'backward_save': lambda: backward(saved, True),
        'backward_recompute': lambda: backward(recomputed, False),
    }
    for name, call in modes.items():
        if args.backward_only and name.startswith('forward'):
            continue
        for _ in range(2):
            call()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=args.samples, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / name)),
        ) as prof:
            for _ in range(args.samples):
                with torch.autograd.profiler.record_function(name):
                    call()
                torch.npu.synchronize()
                prof.step()
        print(name, 'chunk_size', args.chunk_size, 'captured', args.samples, 'public calls', flush=True)


if __name__ == '__main__':
    main()
