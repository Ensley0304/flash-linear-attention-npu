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
            *(
                nd_tensor(ctx, tensor, name)
                for tensor, name in zip(
                    inputs,
                    (
                        "q",
                        "k",
                        "v",
                        "v_new",
                        "gk",
                        "beta",
                        "Akk",
                        "h",
                        "do",
                        "dh",
                        "dv_scan",
                    ),
                )
            ),
            ctypes.c_float(scale),
            ctypes.c_int64(64),
            *(
                nd_tensor(ctx, tensor, name)
                for tensor, name in zip(
                    outputs, ("dq", "dk", "dv", "db", "dg", "dAkk")
                )
            ),
        ],
        outputs,
    )


def main():
    torch.npu.set_device(0)
    batch = 1
    heads = int(os.getenv("KDA_BWD_H", "96"))
    seqlen = int(os.getenv("KDA_BWD_T", "18432"))
    dim = 128
    chunks = (seqlen + 63) // 64
    device = "npu:0"
    shape = (batch, heads, seqlen, dim)

    def bf16(shape):
        return torch.zeros(shape, dtype=torch.bfloat16, device=device)

    inputs = (
        bf16(shape),
        bf16(shape),
        bf16(shape),
        bf16(shape),
        torch.zeros(shape, dtype=torch.float32, device=device),
        bf16((batch, heads, seqlen)),
        bf16((batch, heads, seqlen, 64)),
        bf16((batch, chunks, heads, dim, dim)),
        bf16(shape),
        bf16((batch, heads, chunks, dim, dim)),
        bf16(shape),
    )

    mode = "split" if os.getenv("KDA_BWD_WY_SPLIT_FALLBACK", "0") != "0" else "fused"
    stage_count = os.getenv("KDA_BWD_WY_STAGE_COUNT", "all")
    # Exclude first-use runtime/kernel-loading overhead, then use the same
    # device-event sampling protocol as the other KDA L0 timing harness.
    run_wy(inputs, 1.0 / math.sqrt(dim))
    torch.npu.synchronize()
    repeats = int(os.getenv("KDA_BWD_TIMING_REPEATS", "1"))
    for index in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run_wy(inputs, 1.0 / math.sqrt(dim))
        end.record()
        torch.npu.synchronize()
        print(
            f"K3[{index}] mode={mode} stage_count={stage_count} "
            f"device_elapsed_ms={start.elapsed_time(end):.6f}",
            flush=True,
        )
    print("K3_TARGET_PASS", flush=True)


if __name__ == "__main__":
    main()
