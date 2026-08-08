import ctypes
import os

import torch
import torch_npu

from fla_npu.ops.ascendc import _aclnn_ctypes as aclnn


def run_gate_post(value):
    output = torch.empty_like(value)
    shape = tuple(value.shape)

    def nd_tensor(ctx, tensor, name):
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=aclnn.ACL_FORMAT_ND,
            storage_shape_override=shape,
        )

    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdGatePost",
        lambda ctx: [
            nd_tensor(ctx, value, "dg_hv"),
            ctypes.c_int64(64),
            nd_tensor(ctx, output, "dg"),
        ],
        output,
    )


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    heads = int(os.getenv("KDA_BWD_H", "2"))
    value = torch.randn((1, heads, 64, 128), dtype=torch.float32).npu() * 0.01
    expected = torch.flip(
        torch.cumsum(torch.flip(value.cpu(), dims=(2,)), dim=2), dims=(2,)
    )
    first = run_gate_post(value)
    second = run_gate_post(value)
    torch.npu.synchronize()
    for index, result in enumerate((first, second)):
        diff = (result.cpu() - expected).abs().max().item()
        print(f"K6 run={index} max_diff={diff:.6g}", flush=True)
    repeat_diff = (first - second).abs().max().cpu().item()
    print(f"K6 repeat max_diff={repeat_diff:.6g}", flush=True)


if __name__ == "__main__":
    main()
