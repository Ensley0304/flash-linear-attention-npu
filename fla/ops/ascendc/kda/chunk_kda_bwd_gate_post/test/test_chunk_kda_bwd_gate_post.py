import os

import torch
import torch_npu


def reference(dg_hv: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty_like(dg_hv)
    for begin in range(0, dg_hv.shape[2], chunk_size):
        end = min(begin + chunk_size, dg_hv.shape[2])
        out[:, :, begin:end] = torch.flip(
            torch.cumsum(torch.flip(dg_hv[:, :, begin:end], dims=[2]), dim=2), dims=[2]
        )
    return out


def run_case(shape: tuple[int, int, int, int], chunk_size: int = 64) -> None:
    torch.manual_seed(20260805)
    dg_hv = torch.randn(shape, dtype=torch.float32) * 0.01
    expected = reference(dg_hv, chunk_size)
    actual = torch.ops.npu.npu_chunk_kda_bwd_gate_post(
        dg_hv.npu(), chunk_size=chunk_size
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    run_case((1, 4, 128, 128))
    run_case((1, 3, 145, 128))
