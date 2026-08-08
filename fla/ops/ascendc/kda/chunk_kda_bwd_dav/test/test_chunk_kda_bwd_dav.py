import math
import os

import torch
import torch_npu


def reference(aqk: torch.Tensor, v_new: torch.Tensor, d_o: torch.Tensor, scale: float):
    batch, heads, seqlen, chunk_size = aqk.shape
    d_aqk = torch.empty_like(aqk, dtype=torch.float32)
    dv = torch.empty_like(v_new)
    for begin in range(0, seqlen, chunk_size):
        end = min(begin + chunk_size, seqlen)
        valid = end - begin
        do_chunk = d_o[:, :, begin:end].float()
        v_chunk = v_new[:, :, begin:end].float()
        score = torch.matmul(do_chunk, v_chunk.transpose(-1, -2)) * scale
        d_aqk[:, :, begin:end] = 0
        d_aqk[:, :, begin:end, :valid] = torch.tril(score)
        dv[:, :, begin:end] = torch.matmul(
            aqk[:, :, begin:end, :valid].float(), do_chunk
        ).to(dv.dtype)
    return d_aqk, dv


def run_case(seqlen: int) -> None:
    torch.manual_seed(20260805)
    shape_v = (1, 4, seqlen, 128)
    aqk = torch.randn((1, 4, seqlen, 64), dtype=torch.bfloat16) * 0.02
    for begin in range(0, seqlen, 64):
        end = min(begin + 64, seqlen)
        valid = end - begin
        aqk[:, :, begin:end, :valid] = torch.tril(aqk[:, :, begin:end, :valid])
        if valid < 64:
            aqk[:, :, begin:end, valid:] = 0
    v_new = torch.randn(shape_v, dtype=torch.bfloat16) * 0.02
    d_o = torch.randn(shape_v, dtype=torch.bfloat16) * 0.02
    scale = 1.0 / math.sqrt(128)
    expected_da, expected_dv = reference(aqk, v_new, d_o, scale)
    d_aqk, dv = torch.ops.npu.npu_chunk_kda_bwd_dav(
        aqk.npu(), v_new.npu(), d_o.npu(), scale=scale, chunk_size=64
    )
    torch.testing.assert_close(d_aqk.cpu(), expected_da, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv.cpu(), expected_dv, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    run_case(128)
    run_case(145)
