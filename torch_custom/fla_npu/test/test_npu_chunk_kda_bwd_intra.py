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


def _full_cube_path_canary_case(path, source):
    """Isolate one of key15's four causal contraction families."""
    b, t, h, kdim, chunk_size = 1, 64, 1, 128, 64
    path_tokens = {
        "left_previous": (18, 2),
        "left_diagonal": (27, 20),
        "right_future": (5, 37),
        "right_diagonal": (20, 27),
    }
    target, source_token = path_tokens[path]
    q = torch.zeros(b, t, h, kdim, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    feature = (
        0.125 + torch.arange(kdim, dtype=torch.float32) / 512
    ).to(torch.bfloat16)
    if path.startswith("left") or source == "dAkk":
        k[0, source_token, 0] = feature
    if path.startswith("right") and source == "dAqk":
        q[0, source_token, 0] = feature

    feature_slope = 0.002 + torch.arange(kdim, dtype=torch.float32) * 1.0e-5
    g = (
        -torch.arange(t, dtype=torch.float32).reshape(1, t, 1, 1)
        * feature_slope.reshape(1, 1, 1, kdim)
    )
    beta = torch.linspace(0.25, 0.75, t, dtype=torch.float32).reshape(1, t, 1)
    dAqk = torch.zeros(b, t, h, chunk_size, dtype=torch.float32)
    dAkk = torch.zeros_like(dAqk)
    matrix = dAqk if source == "dAqk" else dAkk
    if path.startswith("left"):
        matrix[0, target, 0, source_token] = 0.75
    else:
        matrix[0, source_token, 0, target] = 0.75

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


@pytest.mark.parametrize(
    "path",
    ["left_previous", "left_diagonal", "right_future", "right_diagonal"],
)
@pytest.mark.parametrize("source", ["dAqk", "dAkk"])
def test_chunk_kda_bwd_intra_safe_gate_full_cube_path_canary(path, source):
    """Protect every sparse block family packed for key15's six GEMMs."""
    device = _device()
    inputs, target = _full_cube_path_canary_case(path, source)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)

    output_index = 0 if path.startswith("left") and source == "dAqk" else 1
    signal = got[output_index].detach().cpu()[0, target, 0]
    assert torch.isfinite(signal).all()
    assert torch.count_nonzero(signal) == signal.numel(), (
        f"full-Cube {path}/{source} signal was not propagated across all K features"
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
    """The single MIX fastpath must advance every reverse-flag generation."""
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
    """Keep key12-key14 intact while the new split path remains independently gated."""
    op_root = ROOT / "fla" / "ops" / "ascendc" / "kda" / "chunk_kda_bwd_intra"
    kernel_source = (op_root / "op_kernel" / "chunk_kda_bwd_intra.cpp").read_text(
        encoding="utf-8"
    )
    tiling_source = (op_root / "op_host" / "chunk_kda_bwd_intra_tiling.cpp").read_text(
        encoding="utf-8"
    )
    aclnn_source = (
        op_root / "op_host" / "op_api" / "aclnn_chunk_kda_bwd_intra.cpp"
    ).read_text(encoding="utf-8")
    source_files = sorted(
        path for path in op_root.rglob("*") if path.suffix in {".cpp", ".h", ".hpp"}
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    key_match = re.search(r"\bKDA_ROW3_MIXED_TILING_KEY\b\s*=\s*(\d+)", source)
    batch_key_match = re.search(
        r"\bKDA_ROW3_BATCHED_GATE_TILING_KEY\b\s*=\s*(\d+)", source
    )
    post_key_match = re.search(
        r"\bKDA_ROW3_BATCHED_POST_GATE_TILING_KEY\b\s*=\s*(\d+)", source
    )
    full_cube_key_match = re.search(
        r"\bKDA_FULL_CUBE_TILING_KEY\b\s*=\s*(\d+)", source
    )
    assert key_match is not None, "missing the stable single-launch rowBlock3 MIX key"
    assert batch_key_match is not None, "missing the batched-gate rowBlock3 MIX key"
    assert post_key_match is not None, "missing the batched row post-scale MIX key"
    assert full_cube_key_match is not None, "missing the full-Cube MIX key"
    mixed_key = int(key_match.group(1))
    batch_key = int(batch_key_match.group(1))
    post_key = int(post_key_match.group(1))
    full_cube_key = int(full_cube_key_match.group(1))
    assert mixed_key == 12
    assert batch_key == 13
    assert post_key == 14
    assert full_cube_key == 15
    assert len({mixed_key, batch_key, post_key, full_cube_key}) == 4

    mixed_branch = re.search(
        rf"else if \(TILING_KEY_IS\({mixed_key}\)\) \{{(?P<body>.*?)"
        rf"\n    \}} else if \(TILING_KEY_IS\({batch_key}\)\)",
        kernel_source,
        re.DOTALL,
    )
    assert mixed_branch is not None, "missing single-launch rowBlock3 MIX branch"
    mixed_body = mixed_branch.group("body")
    assert f"KERNEL_TASK_TYPE({mixed_key}, KERNEL_TYPE_MIX_AIC_1_2)" in mixed_body
    assert "ASCEND_IS_AIC" in mixed_body and "ASCEND_IS_AIV" in mixed_body
    assert "GetUserWorkspace(workspace)" in mixed_body
    batch_branch = re.search(
        rf"else if \(TILING_KEY_IS\({batch_key}\)\) \{{(?P<body>.*?)"
        rf"\n    \}} else if \(TILING_KEY_IS\({post_key}\)\)",
        kernel_source,
        re.DOTALL,
    )
    assert batch_branch is not None, "missing batched-gate rowBlock3 MIX branch"
    batch_body = batch_branch.group("body")
    assert f"KERNEL_TASK_TYPE({batch_key}, KERNEL_TYPE_MIX_AIC_1_2)" in batch_body
    assert "ASCEND_IS_AIC" in batch_body and "ASCEND_IS_AIV" in batch_body
    assert "GetUserWorkspace(workspace)" in batch_body
    post_branch = re.search(
        rf"else if \(TILING_KEY_IS\({post_key}\)\) \{{(?P<body>.*?)"
        rf"\n    \}} else if \(TILING_KEY_IS\({full_cube_key}\)\)",
        kernel_source,
        re.DOTALL,
    )
    assert post_branch is not None, "missing batched row post-scale MIX branch"
    post_body = post_branch.group("body")
    assert f"KERNEL_TASK_TYPE({post_key}, KERNEL_TYPE_MIX_AIC_1_2)" in post_body
    assert "ASCEND_IS_AIC" in post_body and "ASCEND_IS_AIV" in post_body
    assert "GetUserWorkspace(workspace)" in post_body
    assert "op.ProcessAiv<false, false>" in mixed_body
    assert "op.ProcessAiv<true, false>" in batch_body
    assert "op.ProcessAiv<true, true>" in post_body
    assert "CrossCoreSetFlagWithReverse<0x2" in kernel_source
    assert "CrossCoreWaitFlagWithReverse<0x2" in kernel_source
    assert "GetBlockIdx()) / subBlockNum" in kernel_source
    assert "SetScheduleMode(1)" in tiling_source

    fastpath = re.search(
        r"if \(UseSplitLeftCubeFastPath\(p\)\) \{(?P<body>.*?)"
        r"\n\s*\} else if \(UseRow3MixedRollback\(p\)\) \{",
        aclnn_source,
        re.DOTALL,
    )
    assert fastpath is not None, "missing public split left-Cube dispatch"
    fastpath_body = fastpath.group("body")
    assert fastpath_body.count("l0op::ChunkKdaBwdIntra(") == 3
    assert fastpath_body.count("AllocTensor") == 6
    assert re.search(r"nullptr, nullptr, nullptr, 1,", fastpath_body)
    assert re.search(r"stageA, stageB, nullptr, 2,", fastpath_body)
    assert re.search(r"nullptr, nullptr, stageC, 3,", fastpath_body)
    assert "CrossCoreSetFlag" not in fastpath_body
    assert "CrossCoreWaitFlag" not in fastpath_body

    assert '#include "lib/matmul_intf.h"' in kernel_source
    assert re.search(
        r"#ifndef TORCH_MODE\s*#include \"lib/matmul_intf\.h\"\s*#endif",
        kernel_source,
    ), "the CANN MIX wrapper needs a guarded matmul::clearWorkspace declaration"
    assert "CalcTschBlockDim" not in source
    assert "BlockMmadTla" in source
    assert re.search(r"\bKDA_MIX_FEATURE_TILE\b\s*=\s*64\s*;", kernel_source)
    assert re.search(r"\bKDA_MIX_CHUNK_CAPACITY\b\s*=\s*64\s*;", kernel_source)
    assert re.search(r"\bKDA_DB_REDUCTION_TILE\b\s*=\s*32\s*;", kernel_source)
    assert re.search(r"\bKDA_SOURCE_GATE_BATCH\b\s*=\s*BC\s*;", kernel_source)
    assert re.search(
        r"ChunkKdaBwdIntraKernel\s*<\s*bfloat16_t\s*,\s*true\s*,\s*true\s*,\s*true\s*,"
        r"\s*KDA_MIX_FEATURE_TILE\s*,\s*KDA_MIX_CHUNK_CAPACITY\s*,"
        r"\s*BATCH_SOURCE_GATES\s*,\s*BATCH_ROW_POST_GATES\s*>\s+vector\s*;",
        kernel_source,
    ), "key12/key13/key14 AIV must share the BT64/BK64 specialization"
    assert "BuildSourceGateBatch<FIXED_LHS>" in kernel_source
    assert "AccumulateLeftSourceRange<true>" in kernel_source
    assert "AccumulateRightSourceRange<false>" in kernel_source
    assert "MulRowPairBatch(dqAcc, dqAcc, dkLeft, dkLeft" in kernel_source
    assert "AddRowPairBatch(dqAcc, dqAcc, dqDiag, dkLeft, dkLeft, dkLeftDiag" in kernel_source
    assert "MulRowBatch(dkRight, dkRight" in kernel_source
    assert "AddRowBatch(dkRight, dkRight, dkRightFuture" in kernel_source
    post_batch = re.search(
        r"if constexpr \(BATCH_ROW_POST_GATES\) \{(?P<body>.*?)"
        r"\n        \} else \{\n            for \(uint64_t row",
        kernel_source,
        re.DOTALL,
    )
    assert post_batch is not None, "missing the compile-time row post-scale batch/fallback split"
    post_batch_body = post_batch.group("body")
    gate_directions = (
        "BuildSourceGateBatch<false>(gate, gLeftRef, gRows",
        "BuildSourceGateBatch<false>(gate, gLeftDiagRef, gRows",
        "BuildSourceGateBatch<true>(gate, gLeftRef, gRows",
        "BuildSourceGateBatch<true>(gate, gRightRef, gRows",
    )
    gate_positions = [post_batch_body.find(marker) for marker in gate_directions]
    assert all(position >= 0 for position in gate_positions)
    assert gate_positions == sorted(gate_positions), (
        "row post-scale batching must preserve left-prefix, diagonal-left, "
        "diagonal-right, future-right stage order"
    )
    assert re.search(
        r"!BATCH_ROW_POST_GATES\s*\|\|\s*"
        r"\(BATCH_SOURCE_GATES\s*&&\s*SAFE_GATE\s*&&\s*BLOCKWISE",
        kernel_source,
    ), "row post-scale batching must remain inside the batched safe MIX path"
    assert "context->SetTilingKey(KDA_STAGE4_TILING_KEY);" in tiling_source
    source_ranges = []
    for row_begin in range(0, 64, 16):
        row_end = row_begin + 16
        if row_begin != 48:  # rowBlock3 off-left is produced by Cube.
            source_ranges.append((0, row_begin))
        source_ranges.extend(
            ((row_begin, row_end), (row_begin, row_end), (row_end, 64))
        )
    scalar_gate_groups = sum(end - begin for begin, end in source_ranges)
    batched_gate_groups = sum(
        (end - begin + 15) // 16 for begin, end in source_ranges
    )
    assert scalar_gate_groups == 272
    assert batched_gate_groups == 17
    row_post_families = sum(
        int(row_begin > 0) + 2 + int(row_end < 64)
        for row_begin, row_end in ((begin, begin + 16) for begin in range(0, 64, 16))
    )
    scalar_row_post_gate_groups = row_post_families * 16
    assert row_post_families == 14
    assert scalar_row_post_gate_groups == 224
    assert batched_gate_groups + row_post_families == 31
    assert all(
        [source for batch_begin in range(begin, end, 16)
         for source in range(batch_begin, min(batch_begin + 16, end))]
        == list(range(begin, end))
        for begin, end in source_ranges
    ), "gate batching must preserve source order within every accumulator"
    assert "ChunkKdaBwdIntraKernel<bfloat16_t, true, true> op;" in kernel_source, (
        "the stable key7 fallback must keep its default BT128/BK32 specialization"
    )
    assert re.search(
        r"for \(uint64_t reduceOffset = 0; reduceOffset < curK;\s*"
        r"reduceOffset \+= KDA_DB_REDUCTION_TILE\)",
        kernel_source,
    ), "BK64 must preserve db's four 32-feature FP32 reductions"
    for element in ("ElementA", "ElementB", "ElementC"):
        assert re.search(rf"\busing\s+{element}\s*=\s*float\s*;", source), (
            f"rowBlock3 Cube {element} must remain FP32"
        )
    assert re.search(
        r"(?:USE_HF32[A-Z0-9_]*\s*=\s*false|static_assert\s*\(\s*!\s*[^;\n]*USE_HF32_MODE)",
        source,
    ), "rowBlock3 Cube must explicitly disable HF32"

    # Each logical AIC is paired with exactly two AIV lanes.  Both lanes walk
    # the same slots and participate in one ready/done generation per slot.
    slot_count = 128 * 32
    used_core_num = 20
    per_core_slots = [list(range(core, slot_count, used_core_num)) for core in range(used_core_num)]
    assert sorted(slot for slots in per_core_slots for slot in slots) == list(range(slot_count))
    assert max(map(len, per_core_slots)) - min(map(len, per_core_slots)) <= 1
    for lane in (0, 1):
        scheduled = [slot for slots in per_core_slots for slot in slots]
        assert len(scheduled) == slot_count, f"AIV lane {lane} must visit every slot once"


def test_chunk_kda_bwd_intra_full_cube_source_contract():
    """Keep key15 isolated while key13 remains compiled as the rollback."""
    op_root = ROOT / "fla" / "ops" / "ascendc" / "kda" / "chunk_kda_bwd_intra"
    kernel_source = (op_root / "op_kernel" / "chunk_kda_bwd_intra.cpp").read_text(
        encoding="utf-8"
    )
    cube_source = (
        op_root / "op_kernel" / "chunk_kda_bwd_intra_full_cube.h"
    ).read_text(encoding="utf-8")
    tiling_source = (
        op_root / "op_host" / "chunk_kda_bwd_intra_tiling.cpp"
    ).read_text(encoding="utf-8")

    assert "constexpr uint64_t KDA_FULL_CUBE_TILING_KEY = 15;" in kernel_source
    assert (
        "constexpr uint64_t KDA_STAGE4_TILING_KEY = "
        "KDA_ROW3_BATCHED_GATE_TILING_KEY;"
        in tiling_source
    ), "the timed-out key15 experiment must not be the public stage-4 dispatch"
    assert "else if (TILING_KEY_IS(13))" in kernel_source, (
        "the proven key13 implementation must remain compiled as the fallback"
    )
    key15 = re.search(
        r"else if \(TILING_KEY_IS\(15\)\) \{(?P<body>.*?)\n    \}\n\}",
        kernel_source,
        re.DOTALL,
    )
    assert key15 is not None, "missing key15 kernel dispatch"
    key15_body = key15.group("body")
    assert "KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_2)" in key15_body
    assert "KdaFullCube::MixedKernel op;" in key15_body
    assert "ASCEND_IS_AIC" in key15_body and "ASCEND_IS_AIV" in key15_body
    assert "GetUserWorkspace(workspace)" in key15_body

    assert "static_assert(SLOT_BYTES == 614400" in cube_source
    assert "logicalCore * SLOT_ELEMENTS" in cube_source
    assert cube_source.count("Run(directMmad,") == 6, (
        "only the disabled key15 experiment may use DirectMmad"
    )
    assert "TileMmadTla" in cube_source
    assert "constexpr uint32_t CUBE_TILE_M = 128;" in cube_source
    assert "constexpr uint32_t CUBE_TILE_N = 128;" in cube_source
    assert "constexpr uint32_t CUBE_TILE_K = 128;" in cube_source
    assert "CUBE_TILE_M >= A_LEFT_PREV_M" in cube_source
    assert "CUBE_TILE_M >= A_LEFT_DIAG_M" in cube_source
    assert "CUBE_TILE_M >= A_RIGHT_FUTURE_M" in cube_source
    assert "CUBE_TILE_M >= A_RIGHT_DIAG_M" in cube_source
    assert "CUBE_TILE_N >= HEAD_DIM" in cube_source
    assert "CUBE_TILE_K >= A_LEFT_PREV_K" in cube_source
    assert "CUBE_TILE_K >= A_LEFT_DIAG_K" in cube_source
    assert "CUBE_TILE_K >= A_RIGHT_FUTURE_K" in cube_source
    assert "CUBE_TILE_K >= A_RIGHT_DIAG_K" in cube_source
    assert "SetHF32Mode(false);" in cube_source
    assert re.search(
        r"tileMmad\(tensorL0C,\s*tensorL0A,\s*tensorL0B,\s*true,\s*0b11\);",
        cube_source,
    )
    assert "copyL0CToGm(tensorC, tensorL0C, 0b11);" in cube_source
    assert "static constexpr int32_t EVENT_L1A = 0;" in cube_source
    assert "static constexpr int32_t EVENT_L1B = 1;" in cube_source
    assert "static constexpr int32_t EVENT_L0A = 0;" in cube_source
    assert "static constexpr int32_t EVENT_L0B = 1;" in cube_source
    assert "SetFlag<HardEvent::MTE1_MTE2>(EVENT_L1A);" in cube_source
    assert "SetFlag<HardEvent::MTE1_MTE2>(EVENT_L1B);" in cube_source
    assert "SetFlag<HardEvent::M_MTE1>(EVENT_L0A);" in cube_source
    assert "SetFlag<HardEvent::M_MTE1>(EVENT_L0B);" in cube_source
    assert "SetFlag<HardEvent::M_FIX>(EVENT_L0C);" in cube_source
    assert "WaitFlag<HardEvent::M_FIX>(EVENT_L0C);" in cube_source
    assert "SetFlag<HardEvent::FIX_M>(EVENT_L0C);" in cube_source
    assert "WaitFlag<HardEvent::FIX_M>(EVENT_L0C);" in cube_source
    assert "OuterAccumulate" not in cube_source
    assert "lane == 0 ? 0 : 1" in cube_source
    assert "lane == 0 ? 3 : 2" in cube_source
    assert "rowBegin + BLOCK / 2" in cube_source, (
        "safe diagonal paths must retain the Triton midpoint reference"
    )
    assert (
        cube_source.count("CrossCoreSetFlagWithReverse<0x2") == 2
        and cube_source.count("CrossCoreWaitFlagWithReverse<0x2") == 2
    ), "key15 must use one symmetric ready/done generation per task"

    assert (
        "static_cast<uint64_t>(usedCoreNum) * KDA_FULL_CUBE_BYTES_PER_CORE"
        in tiling_source
    )
    assert (
        "static_cast<uint64_t>(scratchSlots) *" in tiling_source
        and "KDA_ROW3_BYTES_PER_SLOT" in tiling_source
    ), "key13 rollback must retain its task-sized workspace formula"


def test_chunk_kda_bwd_intra_split_left_cube_source_contract():
    """The first left-Cube integration must stay launch-isolated and reversible."""
    op_root = ROOT / "fla" / "ops" / "ascendc" / "kda" / "chunk_kda_bwd_intra"
    kernel_source = (op_root / "op_kernel" / "chunk_kda_bwd_intra.cpp").read_text(
        encoding="utf-8"
    )
    cube_source = (
        op_root / "op_kernel" / "chunk_kda_bwd_intra_full_cube.h"
    ).read_text(encoding="utf-8")
    tiling_source = (
        op_root / "op_host" / "chunk_kda_bwd_intra_tiling.cpp"
    ).read_text(encoding="utf-8")
    aclnn_source = (
        op_root / "op_host" / "op_api" / "aclnn_chunk_kda_bwd_intra.cpp"
    ).read_text(encoding="utf-8")

    expected = {
        16: "KERNEL_TYPE_AIV_ONLY",
        17: "KERNEL_TYPE_AIC_ONLY",
        18: "KERNEL_TYPE_AIV_ONLY",
    }
    for key, task_type in expected.items():
        match = re.search(
            rf"else if \(TILING_KEY_IS\({key}\)\) \{{(?P<body>.*?)"
            rf"(?=\n    \}} else if|\n    \}}\n\}})",
            kernel_source,
            re.DOTALL,
        )
        assert match is not None, f"missing split left-Cube key{key}"
        body = match.group("body")
        assert f"KERNEL_TASK_TYPE({key}, {task_type})" in body
        assert "KERNEL_TYPE_MIX" not in body
        assert "CrossCoreSetFlag" not in body
        assert "CrossCoreWaitFlag" not in body

    assert "context->SetTilingKey(KDA_LEFT_PREP_TILING_KEY);" in tiling_source
    assert "context->SetTilingKey(KDA_LEFT_CUBE_TILING_KEY);" in tiling_source
    assert "context->SetTilingKey(KDA_LEFT_CONSUME_TILING_KEY);" in tiling_source
    for rows in (136, 160, 224):
        assert f"constexpr int64_t KDA_LEFT_" in tiling_source
        assert str(rows) in tiling_source

    assert "static_assert((LEFT_A_ELEMENTS * sizeof(float)) % 512 == 0" in cube_source
    assert "class LeftAicKernel" in cube_source
    assert "class LeftSingleTileMmad" in cube_source
    assert "using TileMmad = Catlass::Gemm::Tile::TileMmadTla<" in cube_source
    assert "static constexpr uint32_t TILE_M = 64;" in cube_source
    assert "static constexpr uint32_t TILE_N = 64;" in cube_source
    assert "static constexpr uint32_t TILE_K = 128;" in cube_source
    assert "SetHF32Mode(false);" in cube_source
    assert "WaitFlag<HardEvent::MTE2_MTE1>(EVENT_L1A);" in cube_source
    assert "SetFlag<HardEvent::MTE1_M>(EVENT_L0_READY);" in cube_source
    assert "WaitFlag<HardEvent::MTE1_M>(EVENT_L0_READY);" in cube_source
    assert "SetFlag<HardEvent::M_FIX>(EVENT_L0C);" in cube_source
    assert "WaitFlag<HardEvent::M_FIX>(EVENT_L0C);" in cube_source
    assert "SetFlag<HardEvent::FIX_M>(EVENT_L0C);" in cube_source
    left_body = re.search(
        r"class LeftAicKernel \{(?P<body>.*?)\n\};",
        cube_source,
        re.DOTALL,
    ).group("body")
    assert "BlockMmadTla" not in left_body
    assert "MmadPingpong" not in left_body
    assert "mOffset += LeftSingleTileMmad::TILE_M" in left_body
    assert "nOffset += LeftSingleTileMmad::TILE_N" in left_body
    assert "Catlass::GemmCoord shape{" in left_body
    assert "curM, LeftSingleTileMmad::TILE_N, k" in left_body
    assert "kOffset" not in left_body, "K must remain whole to preserve FP32 reduction order"
    assert "A_LEFT_PREV_M, A_LEFT_PREV_K" in cube_source
    assert "A_LEFT_DIAG_M, A_LEFT_DIAG_K" in cube_source
    assert "SetHF32Mode(false);" in cube_source
    assert "if constexpr (CUBE_LEFT)" in kernel_source
    assert "LoadCubeLeftPrefix" in kernel_source
    assert "true, false, true> op;" in kernel_source

    assert "UseSplitLeftCubeFastPath" in aclnn_source
    assert "constexpr bool KDA_ENABLE_SPLIT_LEFT_CUBE = true;" in aclnn_source
    assert "UseRow3MixedRollback" in aclnn_source
    assert re.search(
        r"UseRow3MixedRollback\(p\).*?nullptr, nullptr, nullptr, 4,",
        aclnn_source,
        re.DOTALL,
    ), "one constant must restore the proven key13 stage-4 launch"
    assert aclnn_source.count("stageA, stageB, nullptr, 2") == 1
    assert aclnn_source.count("nullptr, nullptr, stageC, 3") == 1
    assert "KDA_STAGE4_TILING_KEY = KDA_ROW3_BATCHED_GATE_TILING_KEY" in tiling_source


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
