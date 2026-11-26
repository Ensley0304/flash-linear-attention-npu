"""Exercise mixed-owner reuse through the unchanged fla-org training test."""
from functools import partial
import pytest
import torch
from test_public_backward_expanded import upstream
from public_backward_plugin import public_chunk


@pytest.mark.parametrize('D', [64, 256])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('save', [True, False])
def test_mixed_owner_reuse(D, dtype, save, monkeypatch):
    monkeypatch.setattr(upstream, 'chunk_kda', partial(public_chunk, chunk_size=32))
    # More heads than AIC owners on the validation device, including empty and
    # one-token sequences; each owner must retire all service commands exactly.
    upstream.test_chunk_varlen_fused_training_contract(
        65, D, [0, 0, 1, 34, 65], save, dtype)
