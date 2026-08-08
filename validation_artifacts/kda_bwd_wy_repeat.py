import ctypes
import math
import os

import torch
import torch_npu

from fla_npu.ops.ascendc import _aclnn_ctypes as aclnn


def run_wy(inputs, scale):
    q, k, v, v_new, gk, beta, akk, h, d_o, dh, dv_scan = inputs
    outputs = (
        torch.empty_like(q, dtype=torch.float32),
        torch.empty_like(k, dtype=torch.float32),
        torch.empty_like(v),
        torch.empty_like(beta, dtype=torch.float32),
        torch.empty_like(gk),
        torch.empty_like(akk, dtype=torch.float32),
    )

    def nd_tensor(ctx, tensor, name):
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=aclnn.ACL_FORMAT_ND,
            storage_shape_override=tuple(tensor.shape),
        )

    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdWy",
        lambda ctx: [
            *(nd_tensor(ctx, tensor, name) for tensor, name in zip(
                inputs,
                ("q", "k", "v", "v_new", "gk", "beta", "Akk", "h", "do", "dh", "dv_scan"),
            )),
            ctypes.c_float(scale),
            ctypes.c_int64(64),
            *(nd_tensor(ctx, tensor, name) for tensor, name in zip(
                outputs, ("dq", "dk", "dv", "db", "dg", "dAkk")
            )),
        ],
        outputs,
    )


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    batch = 1
    heads = int(os.getenv("KDA_BWD_H", "2"))
    seqlen = int(os.getenv("KDA_BWD_T", "64"))
    dim = 128
    chunks = (seqlen + 63) // 64
    shape = (batch, heads, seqlen, dim)

    def bf16(shape, factor=0.01):
        return (torch.randn(shape, dtype=torch.bfloat16) * factor).npu()

    inputs = (
        bf16(shape),
        bf16(shape),
        bf16(shape),
        bf16(shape),
        (-torch.rand(shape, dtype=torch.float32) * 0.02).npu(),
        (torch.rand((batch, heads, seqlen), dtype=torch.bfloat16) * 0.1).npu(),
        bf16((batch, heads, seqlen, 64)),
        bf16((batch, chunks, heads, dim, dim)),
        bf16(shape),
        bf16((batch, heads, chunks, dim, dim)),
        bf16(shape),
    )
    repeated_inputs = tuple(tensor.clone() for tensor in inputs)
    first = run_wy(inputs, 1.0 / math.sqrt(dim))
    torch.npu.synchronize()
    for name, value in zip(("dq", "dk", "dv", "db", "dg", "dAkk"), first):
        print(
            f"K3 first {name} finite={int(torch.isfinite(value).sum().cpu())}/"
            f"{value.numel()} abs_max={value.float().abs().max().cpu().item():.6g}",
            flush=True,
        )
    second = run_wy(repeated_inputs, 1.0 / math.sqrt(dim))
    torch.npu.synchronize()
    dv_scan_delta = (inputs[-1].float() - repeated_inputs[-1].float()).abs()
    print(
        "K3 repeat dv_scan_scratch max_diff="
        f"{dv_scan_delta.max().cpu().item():.6g} "
        f"changed={int((dv_scan_delta != 0).sum().cpu())}",
        flush=True,
    )
    for name, lhs, rhs in zip(("dq", "dk", "dv", "db", "dg", "dAkk"), first, second):
        delta = (lhs.float() - rhs.float()).abs()
        flat = delta.reshape(-1)
        max_value, max_offset = flat.max(dim=0)
        max_offset = int(max_offset.cpu())
        index = []
        for size in reversed(delta.shape):
            index.append(max_offset % size)
            max_offset //= size
        index.reverse()
        changed = int((delta != 0).sum().cpu())
        gt_1e7 = int((delta > 1e-7).sum().cpu())
        print(
            f"K3 repeat {name} max_diff={max_value.cpu().item():.6g} "
            f"index={tuple(index)} changed={changed} gt_1e-7={gt_1e7}",
            flush=True,
        )
        if name == "dAkk" and changed:
            changed_idx = (delta != 0).nonzero()
            head_counts = torch.bincount(
                changed_idx[:, 1].cpu(), minlength=delta.shape[1]
            )
            row_counts = torch.bincount(
                changed_idx[:, 2].cpu(), minlength=delta.shape[2]
            )
            print(
                "K3 repeat dAkk changed_heads="
                f"{[(i, int(v)) for i, v in enumerate(head_counts) if v]} "
                "changed_rows="
                f"{[(i, int(v)) for i, v in enumerate(row_counts) if v]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
