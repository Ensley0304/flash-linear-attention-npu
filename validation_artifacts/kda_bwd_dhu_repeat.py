import math
import os

import torch
import torch_npu

from fla_npu.ops import ascendc


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    shape = (1, int(os.getenv("KDA_BWD_H", "2")), 64, 128)

    def bf16(factor=0.01):
        return (torch.randn(shape, dtype=torch.bfloat16) * factor).npu()

    q, k, w, d_o, dv = (bf16() for _ in range(5))
    gk = (-torch.rand(shape, dtype=torch.float32) * 0.02).npu()

    def run():
        return ascendc.chunk_gated_delta_rule_bwd_dhu(
            q,
            k,
            w,
            d_o,
            dv,
            1.0 / math.sqrt(128),
            64,
            g=None,
            gK=gk,
            h0=None,
            dht=None,
            cu_seqlens=None,
            chunk_indices=None,
        )

    first = run()
    second = run()
    torch.npu.synchronize()
    for name, lhs, rhs in zip(("dh", "dh0", "dv_scan"), first, second):
        if lhs is None:
            continue
        diff = (lhs.float() - rhs.float()).abs().max().cpu().item()
        print(f"K2 repeat {name} max_diff={diff:.6g}", flush=True)


if __name__ == "__main__":
    main()
