import math
import os

import torch
import torch_npu
import fla_npu

from kda_bwd_wy_repeat import run_wy


def main():
    fla_npu.load_ascendc_opapi_libraries()
    torch.npu.set_device(0)
    torch.manual_seed(20260806)
    batch = 1
    heads = int(os.getenv("CASE_HEADS", "2"))
    seqlen = int(os.getenv("CASE_SEQLEN", "64"))
    dim = 128
    chunks = (seqlen + 63) // 64
    shape = (batch, heads, seqlen, dim)

    def bf16(tensor_shape, factor=0.01):
        return (torch.randn(tensor_shape, dtype=torch.bfloat16) * factor).npu()

    source = (
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
    fused_inputs = tuple(tensor.clone() for tensor in source)
    split_inputs = tuple(tensor.clone() for tensor in source)
    scale = 1.0 / math.sqrt(dim)

    os.environ.pop("KDA_BWD_WY_SPLIT_FALLBACK", None)
    fused = run_wy(fused_inputs, scale)
    torch.npu.synchronize()
    os.environ["KDA_BWD_WY_SPLIT_FALLBACK"] = "1"
    split = run_wy(split_inputs, scale)
    torch.npu.synchronize()

    names = ("dq", "dk", "dv", "db", "dg", "dAkk")
    failed = []
    for name, actual, expected in zip(names, fused, split):
        delta = (actual.float() - expected.float()).abs()
        max_abs = delta.max().cpu().item()
        changed = int((delta != 0).sum().cpu())
        try:
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=3e-2, atol=3e-2,
                equal_nan=False,
            )
            status = "PASS"
        except AssertionError:
            status = "FAIL"
            failed.append(name)
        print(
            f"{status} fused_vs_split {name} max_abs={max_abs:.6g} "
            f"changed={changed}/{delta.numel()}",
            flush=True,
        )
    if failed:
        raise AssertionError(f"fused/split mismatch: {', '.join(failed)}")


if __name__ == "__main__":
    main()
