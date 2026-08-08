import os

import torch
import torch_npu

from fla_npu.ops import ascendc


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    batch = 1
    heads = int(os.getenv("KDA_BWD_H", "2"))
    seqlen = int(os.getenv("KDA_BWD_T", "64"))
    shape = (batch, heads, seqlen, 128)
    matrix_shape = (batch, heads, seqlen, 64)
    scalar_shape = (batch, heads, seqlen)

    def bf16(current_shape):
        return (torch.randn(current_shape, dtype=torch.bfloat16) * 0.01).npu()

    q = bf16(shape)
    k = bf16(shape)
    gk = (-torch.rand(shape, dtype=torch.float32) * 0.02).npu()
    beta = (torch.rand(scalar_shape, dtype=torch.bfloat16) * 0.1).npu()
    d_aqk = (torch.randn(matrix_shape, dtype=torch.float32) * 1e-5).npu()
    d_akk = (torch.randn(matrix_shape, dtype=torch.float32) * 1e-5).npu()
    dq = (torch.randn(shape, dtype=torch.float32) * 1e-5).npu()
    dk = (torch.randn(shape, dtype=torch.float32) * 1e-5).npu()
    db = (torch.randn(scalar_shape, dtype=torch.float32) * 1e-5).npu()
    dg = (torch.randn(shape, dtype=torch.float32) * 1e-5).npu()

    def run():
        return ascendc.chunk_kda_bwd_intra(
            q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg,
            cu_seqlens=None, chunk_indices=None, chunk_size=64,
            safe_gate=True, layout="BNSD",
        )

    first = run()
    torch.npu.synchronize()
    second = run()
    torch.npu.synchronize()
    for name, lhs, rhs in zip(("dq", "dk", "db", "dg"), first, second):
        finite = int(torch.isfinite(lhs).sum().cpu())
        diff = (lhs.float() - rhs.float()).abs().max().cpu().item()
        print(
            f"K4 {name} finite={finite}/{lhs.numel()} "
            f"abs_max={lhs.float().abs().max().cpu().item():.6g} "
            f"repeat_diff={diff:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
