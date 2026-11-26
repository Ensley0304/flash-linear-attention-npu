"""Exercise the unchanged fla-org test_chunk body on the C64 pipeline matrix."""
import pytest
import torch
from public_backward_plugin import public_chunk
from test_public_backward_expanded import upstream


@pytest.mark.parametrize("D", [60, 64, 100, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("raw_safe", [False, True])
def test_c64_pipeline_upstream(D, dtype, save, norm, raw_safe, monkeypatch):
    monkeypatch.setattr(upstream, "chunk_kda", public_chunk)
    upstream.test_chunk(
        B=2, T=65, H=2, HV=2, D=D, scale=.37,
        gate_logit_normalizer=10, mask_p=0,
        use_qk_l2norm_in_kernel=norm, use_gate_in_kernel=raw_safe,
        has_a_log=raw_safe, dtype=dtype, safe_gate=raw_safe,
        disable_recompute=save, chunk_size=64,
    )


@pytest.mark.parametrize("T", [63, 95, 129, 1024])
@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c64_odd_head_reuse(T, save, dtype, monkeypatch):
    monkeypatch.setattr(upstream, "chunk_kda", public_chunk)
    upstream.test_chunk(
        B=2, T=T, H=3, HV=3, D=256, scale=.37,
        gate_logit_normalizer=10, mask_p=0,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
        has_a_log=True, dtype=dtype, safe_gate=True,
        disable_recompute=save, chunk_size=64,
    )
