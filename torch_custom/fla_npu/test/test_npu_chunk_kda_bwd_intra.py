# Copyright (c) 2026 Tianjin University, Ltd.

import inspect
import os
import pathlib
import re
import sys

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

from fla_npu.ops import ascendc as fla_ascendc
from fla_npu.ops.ascendc import _aclnn_ctypes

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
    failures = []
    for name, actual, expected in zip(("dq", "dk", "db", "dg"), got, (ref.dq, ref.dk, ref.db, ref.dg)):
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        try:
            torch.testing.assert_close(
                actual_cpu,
                expected_cpu,
                rtol=rtol,
                atol=atol,
                check_dtype=False,
            )
        except AssertionError as error:
            failures.append(f"{name} does not match the FP64 CPU golden:\n{error}")
    if failures:
        raise AssertionError("\n\n".join(failures))


def _golden(*inputs, chunk_size=64, cu_seqlens=None, safe_gate=True):
    return chunk_kda_backward_intra_reference(
        *inputs,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        acc_dtype=torch.float64,
    )


def _safe_gate_endpoint_reassociation_case():
    """Keep a finite cross-block signal with legal large BF16 features."""
    b, t, h, kdim, chunk_size = 1, 64, 1, 128, 64
    q = torch.zeros(b, t, h, kdim, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    g = torch.zeros(b, t, h, kdim, dtype=torch.float32)
    beta = torch.ones(b, t, h, dtype=torch.float32)
    dAqk = torch.zeros(b, t, h, chunk_size, dtype=torch.float32)
    dAkk = torch.zeros_like(dAqk)
    dq = torch.zeros(b, t, h, kdim, dtype=torch.float32)
    dk = torch.zeros_like(dq)
    db = torch.zeros(b, t, h, dtype=torch.float32)
    dg = torch.zeros_like(dq)

    step = 125.5 / 18.0
    g[:] = (-torch.arange(t, dtype=torch.float32) * step).reshape(1, t, 1, 1)
    q[0, 33, 0, 0] = 2.0 ** 120
    k[0, 14, 0, 0] = 2.0 ** 120
    dAqk[0, 33, 0, 14] = 1.0
    expected = 2.0 ** (120.0 + float(g[0, 33, 0, 0] - g[0, 14, 0, 0]))
    return (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg), expected


def _one_hot_da_case(source):
    """Isolate row/column dA paths with a single non-zero matrix entry."""
    t, kdim, chunk_size = 19, 16, 64
    q = torch.zeros(1, t, 1, kdim, dtype=torch.float16)
    k = torch.zeros_like(q)
    q[0, 18, 0] = torch.arange(1, kdim + 1, dtype=torch.float16) / 32
    k[0, 2, 0] = torch.arange(kdim, 0, -1, dtype=torch.float16) / 32
    k[0, 18, 0] = torch.arange(1, kdim + 1, dtype=torch.float16) / 64
    g = torch.zeros(1, t, 1, kdim, dtype=torch.float32)
    beta = torch.linspace(0.25, 0.75, t, dtype=torch.float32).reshape(1, t, 1)
    dAqk = torch.zeros(1, t, 1, chunk_size, dtype=torch.float32)
    dAkk = torch.zeros_like(dAqk)
    if source == "dAqk":
        dAqk[0, 18, 0, 2] = 0.5
    else:
        dAkk[0, 18, 0, 2] = 0.5
    dq = torch.zeros(1, t, 1, kdim, dtype=torch.float32)
    dk = torch.zeros_like(dq)
    db = torch.zeros(1, t, 1, dtype=torch.float32)
    dg = torch.zeros_like(dq)
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def _rowblock3_off_left_cube_canary_case(source):
    """Isolate the BF16 safe rowBlock3 x source[0:48] contraction."""
    b, t, h, kdim, chunk_size = 1, 64, 1, 128, 64
    target, source_token = 55, 9
    q = torch.zeros(b, t, h, kdim, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    k[0, source_token, 0] = (
        torch.arange(1, kdim + 1, dtype=torch.float32) / 256
    ).to(torch.bfloat16)

    feature_slope = 0.002 + torch.arange(kdim, dtype=torch.float32) * 1.0e-5
    g = -torch.arange(t, dtype=torch.float32).reshape(1, t, 1, 1) * feature_slope.reshape(1, 1, 1, kdim)
    beta = torch.linspace(0.25, 0.75, t, dtype=torch.float32).reshape(1, t, 1)
    dAqk = torch.zeros(b, t, h, chunk_size, dtype=torch.float32)
    dAkk = torch.zeros_like(dAqk)
    if source == "dAqk":
        dAqk[0, target, 0, source_token] = 0.75
    else:
        dAkk[0, target, 0, source_token] = 0.75

    dq = torch.zeros(b, t, h, kdim, dtype=torch.float32)
    dk = torch.zeros_like(dq)
    db = torch.zeros(b, t, h, dtype=torch.float32)
    dg = torch.zeros_like(dq)
    return (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg), target


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


def test_chunk_kda_bwd_intra_safe_gate_endpoint_reassociation_guard():
    inputs, expected = _safe_gate_endpoint_reassociation_case()
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    device = _device()
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=0.0)
    for output, name, token in ((got[0], "dq", 33), (got[1], "dk", 14)):
        signal = output.detach().cpu()[0, token, 0, 0]
        assert torch.isfinite(signal) and signal != 0, (
            f"NPU {name}[{token},0] endpoint-reassociated signal was lost"
        )
        torch.testing.assert_close(
            signal,
            torch.tensor(expected, dtype=signal.dtype),
            rtol=5e-3,
            atol=0.0,
        )


def test_chunk_kda_bwd_intra_default_keeps_upstream_unsafe_contract():
    signature = inspect.signature(_aclnn_ctypes.npu_chunk_kda_bwd_intra)
    assert signature.parameters["safe_gate"].default is False


def test_chunk_kda_bwd_intra_zero_da_preserves_accumulators():
    device = _device()
    inputs = list(_case(t=19, h=1, hv=1, kdim=16, gate_scale=0.2))
    inputs[4].zero_()
    inputs[5].zero_()
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("source", ["dAqk", "dAkk"])
@pytest.mark.parametrize("safe_gate", [True, False])
def test_chunk_kda_bwd_intra_one_hot_da_paths(source, safe_gate):
    device = _device()
    inputs = _one_hot_da_case(source)
    ref = _golden(*inputs, chunk_size=64, safe_gate=safe_gate)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=safe_gate
    )
    _assert_outputs(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("source", ["dAqk", "dAkk"])
def test_chunk_kda_bwd_intra_safe_gate_rowblock3_off_left_cube_canary(source):
    """Protect the first Cube slice without mixing diagonal or right terms."""
    device = _device()
    inputs, target = _rowblock3_off_left_cube_canary_case(source)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref)

    output_index = 0 if source == "dAqk" else 1
    signal = got[output_index].detach().cpu()[0, target, 0]
    assert torch.isfinite(signal).all()
    assert torch.count_nonzero(signal) == signal.numel(), (
        f"rowBlock3 off-left {source} signal was not propagated across all K features"
    )


def test_chunk_kda_bwd_intra_safe_gate_rowblock3_cube_dense_random():
    """Exercise all outputs on the exact BF16/safe Cube eligibility."""
    device = _device()
    inputs = _case(t=64, h=2, hv=2, kdim=128, dtype=torch.bfloat16, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


def test_chunk_kda_bwd_intra_safe_gate_rowblock3_cube_repeated_launch():
    """The serial three-stage fastpath must not retain transient event state."""
    device = _device()
    inputs = _case(t=64, h=1, hv=1, kdim=128, dtype=torch.bfloat16, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)
    for _ in range(100):
        got = fla_ascendc.chunk_kda_bwd_intra(
            *device_inputs, chunk_size=64, safe_gate=True
        )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


def test_chunk_kda_bwd_intra_rowblock3_cube_source_contract():
    """Keep the experimental Cube stages isolated from cross-core handshakes."""
    op_root = ROOT / "fla" / "ops" / "ascendc" / "kda" / "chunk_kda_bwd_intra"
    kernel_source = (op_root / "op_kernel" / "chunk_kda_bwd_intra.cpp").read_text(
        encoding="utf-8"
    )
    source_files = sorted(
        path for path in op_root.rglob("*") if path.suffix in {".cpp", ".h", ".hpp"}
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    key_names = (
        "KDA_ROW3_PREP_TILING_KEY",
        "KDA_ROW3_CUBE_TILING_KEY",
        "KDA_ROW3_CONSUME_TILING_KEY",
    )
    key_values = []
    for key_name in key_names:
        match = re.search(rf"\b{key_name}\b\s*=\s*(\d+)", source)
        assert match is not None, f"missing independent rowBlock3 stage key: {key_name}"
        key_values.append(int(match.group(1)))
    assert len(set(key_values)) == len(key_values), "rowBlock3 prep/Cube/consume keys must be distinct"

    cube_branch = re.search(
        r"else if \(TILING_KEY_IS\(10\)\) \{(?P<body>.*?)"
        r"\n\s*\} else if \(TILING_KEY_IS\(11\)\)",
        kernel_source,
        re.DOTALL,
    )
    assert cube_branch is not None, "missing standalone rowBlock3 Cube branch"
    cube_body = cube_branch.group("body")
    assert "KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIC_ONLY)" in cube_body
    assert "KERNEL_TYPE_MIX" not in cube_body
    assert "ASCEND_IS_AIC" not in cube_body
    assert '#include "lib/matmul_intf.h"' not in kernel_source
    assert "CalcTschBlockDim" not in source
    assert "BlockMmadTla" in source
    for element in ("ElementA", "ElementB", "ElementC"):
        assert re.search(rf"\busing\s+{element}\s*=\s*float\s*;", source), (
            f"rowBlock3 Cube {element} must remain FP32"
        )
    assert re.search(
        r"(?:USE_HF32[A-Z0-9_]*\s*=\s*false|static_assert\s*\(\s*!\s*[^;\n]*USE_HF32_MODE)",
        source,
    ), "rowBlock3 Cube must explicitly disable HF32"

    for forbidden in ("CrossCoreSetFlag", "CrossCoreWaitFlag", "CrossCoreFlagWithReverse"):
        assert forbidden not in source, f"rowBlock3 serial stages must not use {forbidden}"

    assert "if ((usedCoreNum_ % nc) == 0)" in kernel_source
    assert "const uint64_t rotatedRowBlock = (taskLane + iteration) % nc;" in kernel_source
    assert "const uint64_t task = taskGroup * nc + rotatedRowBlock;" in kernel_source

    # Model the target-shape mapping and prove that rotation neither drops nor
    # duplicates work while distributing the cheaper rowBlock3 across cores.
    nc = 4
    used_core_num = 40
    task_count = 128 * 32 * nc
    scheduled = []
    row_blocks_per_core = []
    for core_idx in range(used_core_num):
        core_tasks = []
        iteration = 0
        linear_task = core_idx
        while linear_task < task_count:
            task_group, task_lane = divmod(linear_task, nc)
            rotated_row_block = (task_lane + iteration) % nc
            task = task_group * nc + rotated_row_block
            scheduled.append(task)
            core_tasks.append(rotated_row_block)
            linear_task += used_core_num
            iteration += 1
        row_blocks_per_core.append(core_tasks)

    assert sorted(scheduled) == list(range(task_count))
    for row_block in range(nc):
        counts = [core_tasks.count(row_block) for core_tasks in row_blocks_per_core]
        assert max(counts) - min(counts) <= 1


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


def test_chunk_kda_bwd_intra_safe_gate_repeated_launch():
    """A second launch must not reuse transient pipeline event state."""
    device = _device()
    inputs = _case(t=19, h=1, hv=2, kdim=256, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)

    def run_once():
        return fla_ascendc.chunk_kda_bwd_intra(
            *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
        )

    first = run_once()
    _assert_outputs(first, ref, rtol=5e-3, atol=5e-3)
    second = run_once()
    _assert_outputs(second, ref, rtol=5e-3, atol=5e-3)


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
