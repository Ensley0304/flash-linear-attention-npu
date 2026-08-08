import ctypes
import math
import os

import torch
import torch_npu

from fla_npu.ops.ascendc import _aclnn_ctypes as aclnn


SCALE = 1.0 / math.sqrt(128)


def run_dav(aqk, v_new, d_o):
    outputs = (
        torch.empty_like(aqk, dtype=torch.float32),
        torch.empty_like(v_new),
    )

    def nd_tensor(ctx, tensor, name):
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=aclnn.ACL_FORMAT_ND,
            storage_shape_override=tuple(tensor.shape),
        )

    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdDav",
        lambda ctx: [
            nd_tensor(ctx, aqk, "Aqk"),
            nd_tensor(ctx, v_new, "v_new"),
            nd_tensor(ctx, d_o, "do"),
            ctypes.c_float(SCALE),
            ctypes.c_int64(64),
            nd_tensor(ctx, outputs[0], "dAqk"),
            nd_tensor(ctx, outputs[1], "dv0"),
        ],
        outputs,
    )


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    heads = int(os.getenv("KDA_BWD_H", "2"))
    seqlen = int(os.getenv("KDA_BWD_T", "64"))
    aqk_cpu = torch.randn((1, heads, seqlen, 64), dtype=torch.bfloat16) * 0.01
    aqk_mask = torch.zeros((seqlen, 64), dtype=torch.bfloat16)
    for begin in range(0, seqlen, 64):
        valid = min(64, seqlen - begin)
        aqk_mask[begin : begin + valid, :valid] = torch.tril(
            torch.ones((valid, valid), dtype=torch.bfloat16)
        )
    aqk_cpu *= aqk_mask
    v_new_cpu = torch.randn((1, heads, seqlen, 128), dtype=torch.bfloat16) * 0.01
    d_o_cpu = torch.randn((1, heads, seqlen, 128), dtype=torch.bfloat16) * 0.01
    aqk = aqk_cpu.npu()
    v_new = v_new_cpu.npu()
    d_o = d_o_cpu.npu()
    first = run_dav(aqk, v_new, d_o)
    torch.npu.synchronize()
    second = run_dav(aqk, v_new, d_o)
    torch.npu.synchronize()
    for name, lhs, rhs in zip(("dAqk", "dv0"), first, second):
        delta = (lhs.float() - rhs.float()).abs()
        diff = delta.max().cpu().item()
        changed = (delta != 0).nonzero().cpu()
        heads_changed = sorted(set(changed[:, 1].tolist())) if changed.numel() else []
        print(
            f"K1 repeat {name} max_diff={diff:.6g} "
            f"changed={changed.shape[0]} heads={heads_changed}",
            flush=True,
        )

    golden_daqk = torch.zeros_like(aqk_cpu, dtype=torch.float32)
    golden_dv0 = torch.zeros_like(v_new_cpu)
    for begin in range(0, seqlen, 64):
        valid = min(64, seqlen - begin)
        token_slice = slice(begin, begin + valid)
        d_o_chunk = d_o_cpu[:, :, token_slice].float()
        v_new_chunk = v_new_cpu[:, :, token_slice].float()
        aqk_chunk = aqk_cpu[:, :, token_slice, :valid].float()
        golden_daqk[:, :, token_slice, :valid] = torch.tril(
            torch.matmul(d_o_chunk, v_new_chunk.transpose(-1, -2))
        ) * SCALE
        golden_dv0[:, :, token_slice] = torch.matmul(
            aqk_chunk.transpose(-1, -2), d_o_chunk
        ).to(torch.bfloat16)
    for name, got, want, rtol, atol in (
        ("dAqk", first[0], golden_daqk, 5e-3, 2e-5),
        ("dv0", first[1], golden_dv0, 5e-3, 2e-4),
    ):
        got_cpu = got.float().cpu()
        want_cpu = want.float()
        diff = (got_cpu - want_cpu).abs()
        print(
            f"K1 golden {name} max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g}",
            flush=True,
        )
        torch.testing.assert_close(got_cpu, want_cpu, rtol=rtol, atol=atol)


if __name__ == "__main__":
    main()
