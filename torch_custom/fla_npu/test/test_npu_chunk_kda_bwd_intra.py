# Copyright (c) 2026 Tianjin University, Ltd.

import inspect
import os
import pathlib
import sys

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

from fla_npu.ops import ascendc as fla_ascendc

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.reference.chunk_kda_reference import chunk_kda_backward_intra_reference  # noqa: E402


def _device():
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
        return torch.device(f"npu:{device_id}")
    pytest.skip("Ascend NPU is required for ChunkKdaBwdIntra runtime validation")


def _case(
    b=1,
    t=40,
    h=1,
    hv=2,
    kdim=128,
    chunk_size=64,
    dtype=torch.float16,
    gate_dtype=torch.float32,
    gate_scale=0.05,
):
    torch.manual_seed(20260721)
    q = torch.randn(b, t, h, kdim, dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    g = (-torch.rand(b, t, hv, kdim, dtype=torch.float32).cumsum(dim=1) * gate_scale).to(gate_dtype)
    beta = torch.sigmoid(torch.randn(b, t, hv, dtype=torch.float32)).to(gate_dtype)
    dAqk = torch.randn(b, t, hv, chunk_size, dtype=torch.float32) * 0.02
    dAkk = torch.randn_like(dAqk) * 0.02
    dq = torch.randn(b, t, hv, kdim, dtype=torch.float32) * 0.01
    dk = torch.randn_like(dq) * 0.01
    db = torch.randn(b, t, hv, dtype=torch.float32) * 0.01
    dg = torch.randn_like(dq) * 0.01
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def _assert_outputs(got, ref, rtol=3e-3, atol=3e-3):
    for name, actual, expected in zip(("dq", "dk", "db", "dg"), got, (ref.dq, ref.dk, ref.db, ref.dg)):
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=rtol, atol=atol,
                                   msg=f"{name} does not match the FP64 CPU golden")


def _golden(*inputs, chunk_size=64, cu_seqlens=None, safe_gate=True):
    return chunk_kda_backward_intra_reference(
        *inputs,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        acc_dtype=torch.float64,
    )


def test_chunk_kda_bwd_intra_reference_safe_and_unsafe_fp64_agree():
    inputs = _case(t=23, h=2, hv=4, kdim=32, gate_scale=0.1)
    safe = _golden(*inputs, chunk_size=64, safe_gate=True)
    unsafe = _golden(*inputs, chunk_size=64, safe_gate=False)
    for name in ("dq", "dk", "db", "dg"):
        torch.testing.assert_close(
            getattr(safe, name),
            getattr(unsafe, name),
            rtol=1e-10,
            atol=1e-10,
            msg=f"safe/unsafe FP64 references disagree for {name}",
        )


def test_chunk_kda_bwd_intra_default_keeps_upstream_unsafe_contract():
    signature = inspect.signature(fla_ascendc.chunk_kda_bwd_intra)
    assert signature.parameters["safe_gate"].default is False


@pytest.mark.parametrize("kdim", [15, 17, 257])
def test_chunk_kda_bwd_intra_rejects_unsupported_k(kdim):
    inputs = _case(t=3, h=1, hv=1, kdim=kdim)
    with pytest.raises(RuntimeError, match=r"multiple of 16 in \[16, 256\]"):
        fla_ascendc.chunk_kda_bwd_intra(*inputs, chunk_size=64, safe_gate=True)


def test_chunk_kda_bwd_intra_safe_gate_large_negative_tail():
    device = _device()
    inputs = _case(t=40, kdim=48, dtype=torch.bfloat16, gate_dtype=torch.bfloat16, gate_scale=4.0)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


def test_chunk_kda_bwd_intra_unsafe_branch():
    device = _device()
    inputs = _case(t=31, kdim=48, gate_scale=0.01)
    ref = _golden(*inputs, chunk_size=64, safe_gate=False)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=False
    )
    _assert_outputs(got, ref)


def test_chunk_kda_bwd_intra_bnsd_gva():
    device = _device()
    inputs = _case(t=23, h=1, hv=4, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = inputs
    to_bnsd = lambda x: x.permute(0, 2, 1, 3).contiguous().to(device)
    to_bns = lambda x: x.permute(0, 2, 1).contiguous().to(device)
    got = fla_ascendc.chunk_kda_bwd_intra(
        to_bnsd(q), to_bnsd(k), to_bnsd(g), to_bns(beta), to_bnsd(dAqk), to_bnsd(dAkk),
        to_bnsd(dq), to_bnsd(dk), to_bns(db), to_bnsd(dg), 64, layout="BNSD", safe_gate=True,
    )
    got_bsnd = (got[0].permute(0, 2, 1, 3), got[1].permute(0, 2, 1, 3),
                got[2].permute(0, 2, 1), got[3].permute(0, 2, 1, 3))
    _assert_outputs(got_bsnd, ref)


def test_chunk_kda_bwd_intra_varlen():
    device = _device()
    inputs = _case(t=90, gate_scale=0.2)
    cu = torch.tensor([0, 17, 90], dtype=torch.int64)
    ref = _golden(*inputs, chunk_size=64, cu_seqlens=cu, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs),
        chunk_size=64,
        cu_seqlens=cu.tolist(),
        chunk_indices=(0, 0, 1, 0, 1, 1),
        safe_gate=True,
    )
    _assert_outputs(got, ref)


def test_chunk_kda_bwd_intra_ntd_chunk128():
    device = _device()
    inputs = _case(t=65, h=2, hv=2, chunk_size=128, gate_scale=0.1)
    ref = _golden(*inputs, chunk_size=128, safe_gate=True)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = inputs
    to_ntd = lambda x: x.squeeze(0).permute(1, 0, 2).contiguous().to(device)
    to_nt = lambda x: x.squeeze(0).permute(1, 0).contiguous().to(device)
    got = fla_ascendc.chunk_kda_bwd_intra(
        to_ntd(q), to_ntd(k), to_ntd(g), to_nt(beta), to_ntd(dAqk), to_ntd(dAkk),
        to_ntd(dq), to_ntd(dk), to_nt(db), to_ntd(dg), 128, layout="NTD", safe_gate=True,
    )
    got_bsnd = (got[0].permute(1, 0, 2).unsqueeze(0), got[1].permute(1, 0, 2).unsqueeze(0),
                got[2].permute(1, 0).unsqueeze(0), got[3].permute(1, 0, 2).unsqueeze(0))
    _assert_outputs(got_bsnd, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("kdim", [16, 48, 96, 256])
def test_chunk_kda_bwd_intra_safe_gate_k_boundaries(kdim):
    device = _device()
    inputs = _case(t=19, h=1, hv=2, kdim=kdim, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


def test_chunk_kda_bwd_intra_safe_gate_dense_multibatch_multichunk_gva():
    device = _device()
    inputs = _case(b=2, t=70, h=2, hv=4, kdim=96, gate_scale=0.1)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


def test_chunk_kda_bwd_intra_tnd_safe_gate():
    device = _device()
    inputs = _case(t=33, h=2, hv=4, kdim=64, gate_scale=0.1)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg = inputs
    to_tnd = lambda x: x.squeeze(0).contiguous().to(device)
    got = fla_ascendc.chunk_kda_bwd_intra(
        to_tnd(q), to_tnd(k), to_tnd(g), to_tnd(beta), to_tnd(dAqk), to_tnd(dAkk),
        to_tnd(dq), to_tnd(dk), to_tnd(db), to_tnd(dg), 64, layout="TND", safe_gate=True,
    )
    got_bsnd = tuple(tensor.unsqueeze(0) for tensor in got)
    _assert_outputs(got_bsnd, ref, rtol=5e-3, atol=5e-3)
