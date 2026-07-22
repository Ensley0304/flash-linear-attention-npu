# Copyright (c) 2026 Tianjin University, Ltd.

import inspect
import math
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


def _empty_grouped_precision_case():
    """Build an exact sparse input selecting grouped key 23 on DAV_2201."""
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
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def _set_legal_extreme_gate(g):
    """Use the safe-gate lower-bound step, represented in base-2 log space."""
    step = -5.0 / math.log(2.0)
    gate = torch.arange(g.shape[1], dtype=torch.float32) * step
    g[:] = gate.reshape(1, -1, 1, 1)


def _right_diag_ftz_case():
    """A first-reference right diagonal product FTZs before its outer restore."""
    inputs = list(_empty_grouped_precision_case())
    q, _, g, _, dAqk, _, _, _, _, _ = inputs
    _set_legal_extreme_gate(g)
    q[0, 15, 0, 0] = 2.0 ** -9
    dAqk[0, 15, 0, 15] = 2.0 ** -9
    return tuple(inputs)


def _left_diag_overflow_case():
    """A first-reference left diagonal product overflows before downscaling."""
    inputs = list(_empty_grouped_precision_case())
    _, k, g, _, dAqk, _, _, _, _, _ = inputs
    _set_legal_extreme_gate(g)
    k[0, 15, 0, 0] = 1.0
    dAqk[0, 15, 0, 15] = 2.0 ** 20
    return tuple(inputs)


def _off_right_cross_block_ftz_case():
    """Keep a right-path signal that a distant common reference would flush."""
    inputs = list(_empty_grouped_precision_case())
    q, _, g, _, dAqk, _, _, _, _, _ = inputs
    _set_legal_extreme_gate(g)
    q[0, 32, 0, 0] = 1.0
    dAqk[0, 32, 0, 0] = 2.0 ** 127
    gate_delta = float(g[0, 32, 0, 0] - g[0, 0, 0, 0])
    return tuple(inputs), 2.0 ** (127.0 + gate_delta)


def _pair_bridge_reassociation_case():
    """Exercise non-unit outer/bridge factors just above the FP32 FTZ edge."""
    inputs = list(_empty_grouped_precision_case())
    q, k, g, _, dAqk, _, _, _, _, _ = inputs
    step = 125.5 / 18.0
    gate = -torch.arange(g.shape[1], dtype=torch.float32) * step
    g[:] = gate.reshape(1, -1, 1, 1)
    q[0, 33, 0, 0] = 2.0 ** 120
    k[0, 14, 0, 0] = 2.0 ** 120
    dAqk[0, 33, 0, 14] = 1.0
    exponent = 120.0 + float(g[0, 33, 0, 0] - g[0, 14, 0, 0])
    expected = 2.0 ** exponent
    return tuple(inputs), expected


def _grouped_cancellation_case():
    """Keep a unit residual after two off-diagonal FP32 contributions cancel."""
    inputs = list(_empty_grouped_precision_case())
    _, k, _, _, dAqk, _, _, _, _, _ = inputs
    target = 48
    # Existing pair order is early0, early1, early2, diagonal. Keeping the
    # diagonal separate is insufficient if the three off-diagonal blocks are
    # reassociated in a way that loses the final unit residual.
    for source, value in ((0, 2.0 ** 24), (16, -(2.0 ** 24)), (32, 1.0)):
        k[0, source, 0, 0] = 1.0
        dAqk[0, target, 0, source] = value
    return tuple(inputs)


def _assert_sparse_signal(ref, name, token, expected, *, rtol):
    signal = getattr(ref, name)[0, token, 0, 0]
    assert torch.isfinite(signal), f"{name}[{token},0] must stay finite"
    torch.testing.assert_close(
        signal,
        torch.tensor(expected, dtype=signal.dtype),
        rtol=rtol,
        atol=0.0,
        msg=f"{name}[{token},0] lost the safe-gate sparse signal",
    )


def _run_grouped_precision_guard(inputs, name, token, expected):
    device = _device()
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    # Keep the established mixed-path relative tolerance, but use zero absolute
    # tolerance so an FTZ-to-zero result cannot pass because the signal is tiny.
    _assert_outputs(got, ref, rtol=5e-3, atol=0.0)
    output_index = {"dq": 0, "dk": 1, "db": 2, "dg": 3}[name]
    actual = got[output_index].detach().cpu()[0, token, 0, 0]
    assert torch.isfinite(actual), f"NPU {name}[{token},0] must stay finite"
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=actual.dtype),
        rtol=5e-3,
        atol=0.0,
        msg=f"NPU {name}[{token},0] lost the safe-gate sparse signal",
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


def test_chunk_kda_bwd_intra_reference_right_diag_ftz_guard():
    ref = _golden(*_right_diag_ftz_case(), chunk_size=64, safe_gate=True)
    _assert_sparse_signal(ref, "dk", 15, 2.0 ** -18, rtol=1e-12)


def test_chunk_kda_bwd_intra_reference_left_diag_overflow_guard():
    ref = _golden(*_left_diag_overflow_case(), chunk_size=64, safe_gate=True)
    _assert_sparse_signal(ref, "dq", 15, 2.0 ** 20, rtol=1e-12)


def test_chunk_kda_bwd_intra_reference_off_right_cross_block_ftz_guard():
    inputs, expected = _off_right_cross_block_ftz_case()
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    _assert_sparse_signal(ref, "dk", 0, expected, rtol=1e-12)


def test_chunk_kda_bwd_intra_reference_grouped_cancellation_guard():
    ref = _golden(*_grouped_cancellation_case(), chunk_size=64, safe_gate=True)
    _assert_sparse_signal(ref, "dq", 48, 1.0, rtol=0.0)


def test_chunk_kda_bwd_intra_default_keeps_upstream_unsafe_contract():
    signature = inspect.signature(_aclnn_ctypes.npu_chunk_kda_bwd_intra)
    assert signature.parameters["safe_gate"].default is False


def test_chunk_kda_bwd_intra_grouped_dispatch_source_contract():
    """Pin the delivery default to the last device-proven AIV key 7."""
    op_root = ROOT / "fla" / "ops" / "ascendc" / "kda" / "chunk_kda_bwd_intra"
    host = (op_root / "op_host" / "chunk_kda_bwd_intra_tiling.cpp").read_text(encoding="utf-8")
    kernel = (op_root / "op_kernel" / "chunk_kda_bwd_intra.cpp").read_text(encoding="utf-8")
    grouped = (op_root / "op_kernel" / "chunk_kda_bwd_intra_grouped.hpp").read_text(
        encoding="utf-8"
    )
    multi_mmad = (
        ROOT
        / "fla"
        / "ops"
        / "ascendc"
        / "common"
        / "kernel_utils"
        / "block"
        / "block_mmad_pingpong_tla_multi.hpp"
    ).read_text(encoding="utf-8")
    profile_analyzer = (
        ROOT / "scripts" / "analyze_chunk_kda_bwd_intra_profile.py"
    ).read_text(encoding="utf-8")
    profile_comparator = (
        ROOT / "scripts" / "compare_chunk_kda_bwd_intra_profiles.py"
    ).read_text(encoding="utf-8")
    validation_runner = (
        ROOT / "scripts" / "run_chunk_kda_bwd_intra_grouped_validation.sh"
    ).read_text(encoding="utf-8")

    assert "constexpr bool ENABLE_MIXED_SAFE = false;" in host
    assert "constexpr bool ENABLE_GROUPED_SAFE = false;" in host
    assert "constexpr uint64_t GROUPED_TILING_KEY = 23;" in host
    assert "constexpr uint64_t GROUPED_SLOT_ELEMENTS = 49152;" in host
    host_flat = " ".join(host.split())
    assert (
        "const bool useFastDomain = safeGate && "
        "qDesc->GetDataType() == ge::DT_BF16 && "
        "chunkSize == MIXED_CHUNK_SIZE && k == MIXED_HEAD_DIM && !isVarLen && "
        "(t % MIXED_CHUNK_SIZE) == 0 && h == hv && "
        "platform.GetCurNpuArch() == NpuArch::DAV_2201;"
    ) in host_flat
    assert (
        "const bool useGroupedFast = ENABLE_GROUPED_SAFE && useFastDomain;"
    ) in host_flat
    assert (
        "const bool useMixedFast = !useGroupedFast && ENABLE_MIXED_SAFE && useFastDomain;"
    ) in host_flat
    assert (
        "context->SetTilingKey(useGroupedFast ? GROUPED_TILING_KEY : "
        "(useMixedFast ? MIXED_TILING_KEY : baseTilingKey + "
        "(safeGate && ENABLE_BLOCKWISE_SAFE ? 4 : 0)));"
    ) in host_flat
    assert host.index("const bool useGroupedFast =") < host.index(
        "const bool useMixedFast ="
    )
    assert "TILING_KEY_IS(15)" in kernel, "pair-wise rollback key must remain compiled"
    assert "TILING_KEY_IS(23)" in kernel
    assert "KERNEL_TASK_TYPE(23, KERNEL_TYPE_MIX_AIC_1_2)" in kernel
    assert "Directional endpoint references keep the feature-side gate <= 1." in kernel
    assert "BuildGate(gate, gRightRef, gSource, curK);" in kernel
    assert "BuildGate(gate, gSource, gLeftRef, curK);" in kernel
    assert "BuildGate(gate, gSelf, gRightRef, curK);" in kernel
    assert "BuildGate(gate, gLeftRef, gSelf, curK);" in kernel
    assert "MulAddDst(acc, common, coefficientBrcb" in kernel
    assert "MulAddDst(outG, qCache[rowBegin * BK], dqAcc" in kernel
    assert "MulAddDst(outG, dkRight, kCache[rowBegin * BK]" in kernel
    multi_include = '#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"'
    grouped_include = '#include "chunk_kda_bwd_intra_grouped.hpp"'
    assert multi_include in kernel
    assert grouped_include in kernel
    assert kernel.index(multi_include) < kernel.index(grouped_include)
    assert "struct MmadPingpongTlaMulti" in multi_mmad
    assert "void preSetFlags()" in multi_mmad
    assert "void finalWaitFlags()" in multi_mmad
    assert "void RunFromL1(" in multi_mmad
    assert "AscendC::SetHF32Mode(true);" in multi_mmad
    assert "AscendC::SetHF32Mode(false);" in multi_mmad
    assert "uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);" in multi_mmad
    assert "BlockMmadTla(Arch::Resource<ArchTag> &resource," in multi_mmad
    assert "uint32_t l1BufAddrStart = 0)" in multi_mmad
    assert "protected:" in multi_mmad
    profile_analyzer_flat = " ".join(profile_analyzer.split())
    assert (
        "mixed_aic_aiv=(aic_us > 0.0 and aiv_us > 0.0 and "
        "mix_block_num > 0),"
    ) in profile_analyzer_flat
    assert '{"ieee": "NO", "hf32": "YES"}.get(' in profile_analyzer
    assert "def _hf32_mode_evidence(" in profile_analyzer
    assert "hf32_mode_matches" in profile_analyzer
    assert '"process_device": 0' in profile_analyzer
    assert '("gate_dtype", "FP32")' in profile_analyzer
    assert '("accumulator_dtype", "FP32")' in profile_analyzer
    assert 'gate_scale != 0.2' in profile_analyzer
    assert 'launch_manifest["warmup"] != args.discard_first' in profile_analyzer
    assert 'launch_manifest["repeat"] != args.expected_rows' in profile_analyzer
    assert '"visible_devices"' in profile_analyzer
    assert 'aiv_mte3_us=_number(row, "aiv_mte3_time(us)")' in profile_analyzer
    assert 'aiv_vec_us=_number(row, "aiv_vec_time(us)")' in profile_analyzer
    assert 'aic_cube_us=_number(row, "aic_cube_time(us)", "aic_mac_time(us)")' in profile_analyzer
    assert '"profile_evidence.sha256"' in profile_comparator
    assert '"profile_evidence.pass"' in profile_comparator
    assert "pair_scratch_selection=" in profile_comparator
    assert "IDENTITY_INVARIANTS" in profile_comparator
    assert "MANIFEST_INVARIANTS" in profile_comparator
    assert "def _verify_hash_manifest(" in profile_comparator
    assert "verify_unfiltered_wheel_build()" in validation_runner
    assert "cmake_value_is_false()" in validation_runner
    assert '""|0|OFF|NO|FALSE|N|IGNORE|NOTFOUND|*-NOTFOUND)' in validation_runner
    assert "custom_compile_options.ini" in validation_runner
    assert "custom_tiling_keys.ini" in validation_runner
    assert "CMakeCache.txt" in validation_runner
    assert "generated_wheel_tiling_filter=none" in validation_runner
    assert "grep -RIEq -- '--tiling_key='" not in validation_runner
    assert "--quick-build" in validation_runner
    assert "--stable-build" in validation_runner
    assert 'BUILD_TILING_KEYS="7"' in validation_runner
    assert 'BUILD_TILING_KEYS="0,2,5,7"' in validation_runner
    assert "verify_filtered_tiling_keys" in validation_runner

    # Exercise the manifest parser and HF32 evidence mapping, rather than only
    # checking that their source tokens exist.  Keep this inside the existing
    # source-contract item so the device regression count remains 37.
    import importlib.util
    import json
    import tempfile
    from types import SimpleNamespace

    analyzer_path = ROOT / "scripts" / "analyze_chunk_kda_bwd_intra_profile.py"
    analyzer_spec = importlib.util.spec_from_file_location(
        "_kda_profile_analyzer_contract", analyzer_path
    )
    assert analyzer_spec is not None and analyzer_spec.loader is not None
    analyzer = importlib.util.module_from_spec(analyzer_spec)
    sys.modules[analyzer_spec.name] = analyzer
    try:
        analyzer_spec.loader.exec_module(analyzer)
        single_variant = (
            "pair-factor_setup-overlap_epilogue-tail_scratch-single_"
            "tail-batch_mmad-persistent_vmask-reuse_dbr-coalesced_"
            "store-serial_stagea-split_cube-ieee_io-gm"
        )
        manifest = {
            "operator": "chunk_kda_bwd_intra",
            "batch": 1,
            "seqlen": 8192,
            "heads": 32,
            "head_dim": 128,
            "chunk_size": 64,
            "safe_gate": True,
            "layout": "BNSD",
            "q_dtype": "BF16",
            "k_dtype": "BF16",
            "gate_dtype": "FP32",
            "accumulator_dtype": "FP32",
            "process_device": 0,
            "visible_devices": "2",
            "gate_scale": 0.2,
            "warmup": 3,
            "repeat": 10,
            "build_variant": single_variant,
            "pair_gates": "factor",
            "shared_setup": "overlap",
            "stage_epilogue": "tail",
            "pair_scratch": "single",
            "tail_blocks": "batch",
            "vector_mask": "reuse",
            "mmad_engines": "persistent",
            "db_reduce": "coalesced",
            "task_store": "serial",
            "stage_a": "split",
            "cube_mode": "ieee",
            "stage_io": "gm",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = pathlib.Path(temp_dir) / "launch.json"

            def load_manifest(payload):
                manifest_path.write_text(
                    json.dumps(payload, allow_nan=False), encoding="utf-8"
                )
                return analyzer._load_launch_manifest(manifest_path)

            loaded_single = load_manifest(manifest)
            assert loaded_single["pair_scratch"] == "single"
            assert loaded_single["cube_mode"] == "ieee"
            assert loaded_single["visible_devices"] == "2"

            mismatched_scratch = dict(manifest, pair_scratch="pingpong")
            try:
                load_manifest(mismatched_scratch)
            except ValueError as error:
                assert "inconsistent build identity" in str(error)
            else:
                raise AssertionError("pair-scratch identity mismatch was accepted")

            pingpong_manifest = dict(
                manifest,
                pair_scratch="pingpong",
                build_variant=single_variant.replace(
                    "_scratch-single_", "_scratch-pingpong_"
                ),
            )
            loaded_pingpong = load_manifest(pingpong_manifest)
            assert loaded_pingpong["pair_scratch"] == "pingpong"

            invalid_device = dict(manifest, process_device=2)
            try:
                load_manifest(invalid_device)
            except ValueError as error:
                assert "process_device=0" in str(error)
            else:
                raise AssertionError("nonzero process device was accepted")

            expected, observed, matches = analyzer._hf32_mode_evidence(
                [SimpleNamespace(hf32="NO")], loaded_single
            )
            assert (expected, observed, matches) == ("NO", ("NO",), True)
            assert analyzer._hf32_mode_evidence(
                [SimpleNamespace(hf32="YES")], loaded_single
            ) == ("NO", ("YES",), False)

            hf32_manifest = dict(
                pingpong_manifest,
                cube_mode="hf32",
                build_variant=pingpong_manifest["build_variant"].replace(
                    "_cube-ieee_", "_cube-hf32_"
                ),
            )
            loaded_hf32 = load_manifest(hf32_manifest)
            assert analyzer._hf32_mode_evidence(
                [SimpleNamespace(hf32="YES")], loaded_hf32
            ) == ("YES", ("YES",), True)
    finally:
        sys.modules.pop(analyzer_spec.name, None)

    comparator_path = ROOT / "scripts" / "compare_chunk_kda_bwd_intra_profiles.py"
    comparator_spec = importlib.util.spec_from_file_location(
        "_kda_profile_comparator_contract", comparator_path
    )
    assert comparator_spec is not None and comparator_spec.loader is not None
    comparator = importlib.util.module_from_spec(comparator_spec)
    sys.modules[comparator_spec.name] = comparator
    try:
        comparator_spec.loader.exec_module(comparator)
        single_variant = (
            "pair-factor_setup-overlap_epilogue-tail_scratch-single_"
            "tail-batch_mmad-persistent_vmask-reuse_dbr-coalesced_"
            "store-serial_stagea-split_cube-ieee_io-gm"
        )
        pingpong_variant = single_variant.replace(
            "_scratch-single_", "_scratch-pingpong_"
        )
        assert comparator._normalized_variant(single_variant) == (
            comparator._normalized_variant(pingpong_variant)
        )
        metric = comparator._metric_row(
            [{"aiv_mte3_us": 4000.0}, {"aiv_mte3_us": 4200.0}],
            [{"aiv_mte3_us": 3800.0}, {"aiv_mte3_us": 4000.0}],
            "aiv_mte3_us",
            1.0 / 1000.0,
        )
        assert metric["single"] == 4.1
        assert metric["pingpong"] == 3.9
        assert math.isclose(
            metric["delta_pingpong_minus_single"], -0.2,
            rel_tol=0.0, abs_tol=1e-12,
        )
        assert math.isclose(
            metric["improvement_percent"], 100.0 * 0.2 / 4.1,
            rel_tol=0.0, abs_tol=1e-12,
        )
    finally:
        sys.modules.pop(comparator_spec.name, None)

    for member in (
        "l1AEventList",
        "l1BEventList",
        "l0ATensorList",
        "l0AEventList",
        "l0BTensorList",
        "l0BEventList",
        "l0CTensorList",
        "l0CEventList",
    ):
        assert member in multi_mmad
    assert "constexpr uint32_t KDA_GROUPED_SLOT_ELEMENTS =" in grouped
    assert "KDA_GROUPED_SLOT_ELEMENTS == 49152" in grouped
    expected_pair_gates = os.environ.get("KDA_EXPECT_PAIR_GATES", "factor")
    expected_shared_setup = os.environ.get("KDA_EXPECT_SHARED_SETUP", "overlap")
    expected_stage_epilogue = os.environ.get("KDA_EXPECT_STAGE_EPILOGUE", "tail")
    expected_pair_scratch = os.environ.get("KDA_EXPECT_PAIR_SCRATCH", "single")
    expected_tail_blocks = os.environ.get("KDA_EXPECT_TAIL_BLOCKS", "batch")
    expected_task_store = os.environ.get("KDA_EXPECT_TASK_STORE", "serial")
    expected_mmad_engines = os.environ.get("KDA_EXPECT_MMAD_ENGINES", "scoped")
    expected_vector_mask = os.environ.get("KDA_EXPECT_VECTOR_MASK", "reuse")
    expected_db_reduce = os.environ.get("KDA_EXPECT_DB_REDUCE", "coalesced")
    expected_stage_a = os.environ.get("KDA_EXPECT_STAGE_A", "split")
    expected_cube_mode = os.environ.get("KDA_EXPECT_CUBE_MODE", "ieee")
    expected_stage_io = os.environ.get("KDA_EXPECT_STAGE_IO", "gm")
    expected_aic_diagnostic = os.environ.get("KDA_EXPECT_AIC_DIAGNOSTIC", "full")
    assert expected_pair_gates in {"factor", "direct"}
    assert expected_shared_setup in {"overlap", "prologue"}
    assert expected_stage_epilogue in {"overlap", "tail"}
    assert expected_pair_scratch in {"pingpong", "single"}
    assert expected_tail_blocks in {"batch", "scalar"}
    assert expected_task_store in {"overlap", "serial"}
    assert expected_mmad_engines in {"persistent", "scoped"}
    assert expected_vector_mask in {"reuse", "per-call"}
    assert expected_db_reduce in {"coalesced", "per-row"}
    assert expected_stage_a in {"packed", "split"}
    assert expected_cube_mode in {"ieee", "hf32"}
    assert expected_stage_io in {"tscm", "gm"}
    assert expected_aic_diagnostic in {
        "full", "handshake", "stage0-right", "stage0-left",
        "stage0-both", "through-stage1", "through-stage2",
    }
    factor_literal = "true" if expected_pair_gates == "factor" else "false"
    overlap_literal = "true" if expected_shared_setup == "overlap" else "false"
    epilogue_literal = "true" if expected_stage_epilogue == "overlap" else "false"
    pair_scratch_literal = "true" if expected_pair_scratch == "pingpong" else "false"
    tail_blocks_literal = "true" if expected_tail_blocks == "batch" else "false"
    task_store_literal = "true" if expected_task_store == "overlap" else "false"
    persistent_mmad_literal = "true" if expected_mmad_engines == "persistent" else "false"
    vector_mask_literal = "true" if expected_vector_mask == "reuse" else "false"
    db_reduce_literal = "true" if expected_db_reduce == "coalesced" else "false"
    stage_a_literal = "true" if expected_stage_a == "packed" else "false"
    cube_mode_literal = "true" if expected_cube_mode == "hf32" else "false"
    stage_io_literal = "true" if expected_stage_io == "tscm" else "false"
    aic_diagnostic_literal = {
        "full": "0",
        "handshake": "1",
        "stage0-right": "2",
        "stage0-left": "3",
        "stage0-both": "4",
        "through-stage1": "5",
        "through-stage2": "6",
    }[expected_aic_diagnostic]
    assert (
        f"constexpr bool KDA_GROUPED_FACTOR_PAIR_GATES = {factor_literal};"
        in grouped
    )
    assert (
        f"constexpr bool KDA_GROUPED_OVERLAP_SHARED_SETUP = {overlap_literal};"
        in grouped
    )
    assert (
        f"constexpr bool KDA_GROUPED_OVERLAP_STAGE_EPILOGUE = {epilogue_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH = "
        f"{pair_scratch_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_BATCH_TAIL_BLOCKS = "
        f"{tail_blocks_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_OVERLAP_TASK_STORE = "
        f"{task_store_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_PERSISTENT_MMAD_ENGINES = "
        f"{persistent_mmad_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_REUSE_VECTOR_MASK = "
        f"{vector_mask_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_COALESCE_DB_REDUCE = "
        f"{db_reduce_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME = true;"
        in grouped
    )
    assert "constexpr bool KDA_GROUPED_COALESCE_RIGHT_B_WRITES = true;" in grouped
    assert "constexpr bool KDA_GROUPED_COALESCE_LEFT_C_READS = true;" in grouped
    assert (
        "constexpr bool KDA_GROUPED_PACK_STAGE_A = "
        f"{stage_a_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_USE_HF32_CUBE = "
        f"{cube_mode_literal};"
        in grouped
    )
    assert (
        "constexpr bool KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER = "
        f"{stage_io_literal};"
        in grouped
    )
    assert (
        "constexpr uint32_t KDA_GROUPED_AIC_DIAGNOSTIC_MODE = "
        f"{aic_diagnostic_literal};"
        in grouped
    )
    assert "--aic-diagnostic MODE" in validation_runner
    assert "partial AIC diagnostic artifacts are build-only" in validation_runner
    assert "kda_grouped_preflight_card${PHYSICAL_DEVICE}.log" in validation_runner
    assert (
        '"${test_file}::test_chunk_kda_bwd_intra_safe_gate_grouped_fastpath_bf16"'
        in validation_runner
    )
    assert (
        '"${test_file}::test_chunk_kda_bwd_intra_grouped_fastpath_'
        'off_right_and_pair_bridge_ftz_guard"'
        in validation_runner
    )
    assert "timeout --kill-after=15s 120s env" in validation_runner
    assert "preflight_rc=${PIPESTATUS[0]}" in validation_runner
    assert 'preflight_label="target BF16/safe"' in validation_runner
    assert 'preflight_label="key7 target/bridge"' in validation_runner
    assert "[PASS] $preflight_label preflight completed" in validation_runner
    assert "[PASS] quick key7 target and pair-bridge validation complete" in validation_runner
    assert "CopyStageAbNzRows(" in grouped
    assert "Catlass::layout::zN" in grouped
    assert "Catlass::layout::nZ" in grouped
    assert "constexpr bool KDA_GROUPED_ENABLE_UNIT_FLAG = false;" in grouped
    assert (
        "KdaBwdGroupedArchTag, KDA_GROUPED_ENABLE_UNIT_FLAG,\n"
        "        KDA_GROUPED_USE_HF32_CUBE>"
        in grouped
    )
    assert "KDA_GROUPED_FP32_REPEAT_ELEMENTS = 64" in grouped
    assert "KDA_GROUPED_PAIR_BRIDGE_PADDING_BYTES = 128" in grouped
    assert "KDA_GROUPED_PAIR_BRIDGE_UB % 512 == 128" in grouped
    assert "constexpr uint32_t KDA_GROUPED_PACKED_A = 6144;" in grouped
    assert "constexpr uint32_t KDA_GROUPED_PACKED_B_OFF_RIGHT = 8192;" in grouped
    assert "constexpr uint32_t KDA_GROUPED_PACKED_B_DIAG_RIGHT = 20480;" in grouped
    assert "constexpr uint32_t KDA_GROUPED_DB_LOCAL_UB = 82048;" in grouped
    assert "constexpr uint32_t KDA_GROUPED_REF_UB = 82176;" in grouped
    assert "KDA_GROUPED_REF_UB % 512 == 256" in grouped
    assert "KDA_GROUPED_FIRST_REF_COUNT = KDA_GROUPED_BLOCKS - 1" in grouped
    assert "KDA_GROUPED_MIDDLE_REF_COUNT = KDA_GROUPED_BLOCKS" in grouped
    assert "KDA_GROUPED_RIGHT_REF_COUNT = KDA_GROUPED_BLOCKS - 1" in grouped
    assert "KDA_GROUPED_RIGHT_REF_PADDING_ELEMENTS = 32" in grouped
    assert "KDA_GROUPED_REF_VALUE_ELEMENTS" in grouped
    assert "KDA_GROUPED_REF_VALUE_ELEMENTS == 10 * KDA_GROUPED_K" in grouped
    assert "KDA_GROUPED_REF_STORAGE_ELEMENTS" in grouped
    assert "10 * KDA_GROUPED_K + 32" in grouped
    assert "KDA_GROUPED_RIGHT_REF_OFFSET * sizeof(float)) % 512 == 384" in grouped
    assert "KDA_GROUPED_G_CACHE_UB % 512 == 128" in grouped
    assert "KDA_GROUPED_DK_RIGHT_PADDING_BYTES = 256" in grouped
    assert "KDA_GROUPED_DK_RIGHT_ACC_UB % 512 == 128" in grouped
    assert "KDA_GROUPED_ACC_ZERO_ELEMENTS ==" in grouped
    assert "3 * KDA_GROUPED_SELECTED_ELEMENTS + 64" in grouped
    assert "KDA_GROUPED_FP32_DUPLICATE_MAX_ELEMENTS = 64 * 255" in grouped
    assert (
        "KDA_GROUPED_ACC_ZERO_ELEMENTS <=\n"
        "                  KDA_GROUPED_FP32_DUPLICATE_MAX_ELEMENTS"
        in grouped
    )
    assert "KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES" in grouped
    assert "KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES == 179936" in grouped
    assert "KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES" in grouped
    assert "KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES == 192256" in grouped
    assert "KDA_GROUPED_PAIR_SCRATCH_ALT_UB" in grouped
    assert "KDA_GROUPED_PAIR_SCRATCH_ALT_BYTES == 3 * 4096" in grouped
    assert "KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES <= 192 * 1024" in grouped
    assert "KDA_GROUPED_BATCH_OUT_Q_UB" in grouped
    assert "KDA_GROUPED_BATCH_OUT_K_UB" in grouped
    assert "KDA_GROUPED_BATCH_OUT_G_UB" in grouped
    assert "KDA_GROUPED_BATCH_ROW_TMP_UB" in grouped
    assert "KDA_GROUPED_BATCH_OUT_K_UB = KDA_GROUPED_REF_UB" in grouped
    assert "KDA_GROUPED_BATCH_OUT_G_UB = KDA_GROUPED_K_BETA_CACHE_UB" in grouped
    assert "Retired k-beta storage must hold one full staged dk input" in grouped
    assert "Batched row scratch must fit in retired tail scratch" in grouped
    assert "struct KdaBwdGroupedPartitionedBlockMmad" in grouped
    assert "static_assert(!BaseMmad::ENABLE_UNIT_FLAG" in grouped
    assert "this->l0CEventList[i] = static_cast<int32_t>(EVENT_BASE + i);" in grouped
    assert "void SyncFixpipeToM()" not in grouped
    copy_out_begin = multi_mmad.index("// copy block out")
    copy_out_end = multi_mmad.index(
        "/// Perform a block-scoped matrix multiply", copy_out_begin
    )
    copy_out = multi_mmad[copy_out_begin:copy_out_end]
    assert "if constexpr (!ENABLE_UNIT_FLAG)" in copy_out
    assert "AscendC::SetFlag<AscendC::HardEvent::M_FIX>(" in copy_out
    assert "AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(" in copy_out
    assert "AscendC::SetFlag<AscendC::HardEvent::FIX_M>(" in copy_out
    finish_mmad_begin = grouped.index(
        "__aicore__ inline void FinishStageMmad"
    )
    finish_mmad_end = grouped.index(
        "__aicore__ inline uint32_t CacheBlockOffset", finish_mmad_begin
    )
    finish_mmad = grouped[finish_mmad_begin:finish_mmad_end]
    assert "if constexpr (MANAGE_FLAGS)" in finish_mmad
    assert "SyncFixpipeToM" not in finish_mmad
    assert finish_mmad.count("blockMmad.finalWaitFlags();") == 1
    assert grouped.count(
        "RunStageMmad(blockMmad, blockA, blockB, blockC, shape);"
    ) == 6
    assert grouped.count("FinishStageMmad<MANAGE_FLAGS>(blockMmad);") == 6
    assert grouped.count("blockMmad.finalWaitFlags();") == 1
    assert "KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES == 40 * 1024" in grouped
    assert "KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES == 36 * 1024" in grouped
    assert "KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES == 32 * 1024" in grouped
    assert "KDA_GROUPED_PERSISTENT_RIGHT_L0B_BYTES == 32 * 1024" in grouped
    assert "KDA_GROUPED_PERSISTENT_EVENT_BASE = 0" in grouped
    assert grouped.count("KDA_GROUPED_PERSISTENT_EVENT_BASE>") == 2
    assert "right16.preSetFlags();" not in grouped
    assert "right16.finalWaitFlags();" not in grouped
    assert "left32.preSetFlags();" in grouped
    assert "left32.finalWaitFlags();" in grouped
    assert "KDA_GROUPED_C_OFF_LEFT =\n    KDA_GROUPED_INPUT_SLOT_ELEMENTS" in grouped
    assert "workspace_[slotBase + cBase + pairOffset]" in grouped
    assert "workspace_[slotBase + rightBase + pairOffset]" not in grouped
    assert grouped.index("KDA_GROUPED_ROW_BLOCK_ELEMENTS =") < grouped.index(
        "KDA_GROUPED_RIGHT_OUTER_ELEMENTS ="
    )
    assert "RunOffRightPairAic<STAGE, 0>" in grouped
    assert "BuildPairInnerGates<STAGE, EARLY, SCRATCH_SET>();" in grouped
    assert "BuildPairBridgeGates();" in grouped
    assert "RunPackedOffLeftAic" in grouped
    assert "RunPackedDiagLeftAic" in grouped
    assert "RunPackedDiagRightAic" in grouped
    assert "PairBridgeGate<STAGE, EARLY>()" in grouped
    assert "__aicore__ inline void MultiplyBridgeAcrossRows" in grouped
    assert "BinaryRepeatParams bridgeParams{1, 1, 1, 16, 0, 16};" in grouped
    assert "MultiplyBridgeAcrossRows(" in grouped
    assert "__aicore__ inline void ReferenceMinusRows" in grouped
    assert "__aicore__ inline void RowsMinusReference" in grouped
    assert "BinaryRepeatParams params{1, 1, 1, 16, 0, 16};" in grouped
    assert "BinaryRepeatParams params{1, 1, 1, 16, 16, 0};" in grouped
    assert "Sub<float, false>(dst, reference, rows," in grouped
    assert "Sub<float, false>(dst, rows, reference," in grouped
    assert "__aicore__ inline void BeginReusableFp32Mask()" in grouped
    assert "SetVectorMask<float, MaskMode::NORMAL>(" in grouped
    assert "__aicore__ inline void EndReusableFp32Mask()" in grouped
    assert "ResetMask();" in grouped
    assert "Muls<float, false>(" in grouped
    assert "Exp<float, false>(" in grouped
    assert "Mul<float, false>(" in grouped
    assert "Add<float, false>(" in grouped
    assert "MulAddDst<float, float, false>(" in grouped
    assert grouped.count("MASK_PLACEHOLDER") == 12
    assert "static_cast<uint64_t>(0)" not in grouped
    consume_begin = grouped.index("__aicore__ inline void ConsumeStageAiv")
    consume_end = grouped.index(
        "__aicore__ inline void StageTaskOutputsInAccumulators", consume_begin
    )
    consume = grouped[consume_begin:consume_end]
    assert "DataCopyExtParams offRightTiles" in consume
    assert "static_cast<uint16_t>(STAGE), rowTileBytes" in consume
    assert "pairGapBytes, 0, 0" in consume
    assert "STAGE * KDA_GROUPED_ROW_BLOCK_ELEMENTS" in consume
    # The one strided GM->UB request must select exactly the same physical-AIV
    # rows as the former per-pair copies.  DataCopyExtParams srcStride is the
    # byte gap from one block tail to the next block head.
    row_elements = 8 * 128
    pair_elements = 2 * 16 * 128
    for stage in range(1, 4):
        for row_start in (0, 8):
            first = row_start * 128
            per_pair = [
                element
                for pair in range(stage)
                for element in range(
                    pair * pair_elements + first,
                    pair * pair_elements + first + row_elements,
                )
            ]
            block_gap = pair_elements - row_elements
            coalesced = []
            source = first
            for _ in range(stage):
                coalesced.extend(range(source, source + row_elements))
                source += row_elements + block_gap
            assert coalesced == per_pair
    assert "__aicore__ inline void CopyStageRightBRows(" in grouped
    right_b_begin = grouped.index(
        "__aicore__ inline void CopyStageRightBRows("
    )
    right_b_end = grouped.index(
        "template <typename Layout>", right_b_begin
    )
    right_b_copy = grouped[right_b_begin:right_b_end]
    assert "DataCopyExtParams rightRows" in right_b_copy
    assert "2, rowTileBytes, 0, rightHalfGapBytes, 0" in right_b_copy
    assert "firstHalf[KDA_GROUPED_ROW_BLOCK_ELEMENTS]" in right_b_copy
    # One CopyOut reads two adjacent 8x128 UB tiles and lands them in the
    # q-gated and beta*k-gated halves of the same 32x128 GM matrix.
    for row_start in (0, 8):
        old_src = [*range(0, row_elements), *range(row_elements, 2 * row_elements)]
        old_dst = [
            *range(row_start * 128, row_start * 128 + row_elements),
            *range(
                (16 + row_start) * 128,
                (16 + row_start) * 128 + row_elements,
            ),
        ]
        new_src = []
        new_dst = []
        src = 0
        dst = row_start * 128
        for _ in range(2):
            new_src.extend(range(src, src + row_elements))
            new_dst.extend(range(dst, dst + row_elements))
            src += row_elements
            dst += 2 * row_elements
        assert new_src == old_src
        assert new_dst == old_dst
    assert "__aicore__ inline void CopyStageLeftCRows(" in grouped
    left_c_begin = grouped.index("__aicore__ inline void CopyStageLeftCRows(")
    left_c_end = grouped.index("template <typename Layout>", left_c_begin)
    left_c_copy = grouped[left_c_begin:left_c_end]
    assert "DataCopyExtParams leftRows" in left_c_copy
    assert "2, rowTileBytes, rowTileBytes, 0, 0" in left_c_copy
    # GM rows [r:r+8] and [r+16:r+24] become adjacent scratch tiles.
    for row_start in (0, 8):
        per_call = [
            *range(row_start * 128, row_start * 128 + row_elements),
            *range(
                (16 + row_start) * 128,
                (16 + row_start) * 128 + row_elements,
            ),
        ]
        strided = []
        source = row_start * 128
        for _ in range(2):
            strided.extend(range(source, source + row_elements))
            source += 2 * row_elements
        assert strided == per_call
    # STAGE>0 opens the mask before off-diagonal accumulation; STAGE==0 has
    # no such path and opens it immediately before the diagonal accumulation.
    # The two compile-time-exclusive entries intentionally share one exit.
    assert consume.count("BeginReusableFp32Mask();") == 2
    assert consume.count("EndReusableFp32Mask();") == 1
    assert consume.index("if constexpr (STAGE > 0)") < consume.index(
        "BeginReusableFp32Mask();"
    )
    stage_zero_mask = consume.index("if constexpr (STAGE == 0)")
    assert stage_zero_mask < consume.index("BeginReusableFp32Mask();", stage_zero_mask)
    assert consume.rindex("BeginReusableFp32Mask();") < consume.index(
        "EndReusableFp32Mask();"
    )
    assert grouped.count("BeginReusableFp32Mask();") == 11
    assert grouped.count("EndReusableFp32Mask();") == 10
    assert "BroadcastReference(" not in grouped
    assert "BroadcastReferenceTwice(" not in grouped
    assert "DataCopyPad(refs," in grouped
    assert "DataCopyPad(refs[KDA_GROUPED_MIDDLE_REF_OFFSET]" in grouped
    assert "DataCopyPad(refs[KDA_GROUPED_RIGHT_REF_OFFSET]" in grouped
    zero_begin = grouped.index("__aicore__ inline void ZeroAccumulators")
    zero_end = grouped.index("__aicore__ inline void ReferenceMinusRows", zero_begin)
    assert "KDA_GROUPED_ACC_ZERO_ELEMENTS" in grouped[zero_begin:zero_end]
    pair_inner_begin = grouped.index("__aicore__ inline void BuildPairInnerGates")
    pair_inner_end = grouped.index(
        "__aicore__ inline void BuildAndWritePairB", pair_inner_begin
    )
    pair_inner = grouped[pair_inner_begin:pair_inner_end]
    assert "ReferenceMinusRows(leftInner, FirstReference<STAGE>()" in pair_inner
    assert "RowsMinusReference(rightInner, gCache[CacheBlockOffset(STAGE)]" in pair_inner
    assert "BuildPairCrossGate" not in grouped
    assert "RightOuterGate<0>()" in grouped
    assert "RightOuterGate<3>()" not in grouped
    assert "KDA_GROUPED_RIGHT_OUTER_ELEMENTS" in grouped
    assert (
        "constexpr uint32_t KDA_GROUPED_CAUSAL_MASK_UB = KDA_GROUPED_TAIL_UB;"
        in grouped
    )
    assert "BuildCausalSelectMask" not in grouped
    load_begin = grouped.index("__aicore__ inline void LoadTaskFeatures")
    load_end = grouped.index("__aicore__ inline void ZeroAccumulators", load_begin)
    load_features = grouped[load_begin:load_end]
    assert load_features.count("BroadcastSelectedBetas(betaBrcb, betaLocal);") == 1
    assert "UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>()" in load_features
    store_begin = grouped.index("__aicore__ inline void StoreTaskOutputs")
    store_end = grouped.index("GlobalTensor<T> q_;", store_begin)
    store_outputs = grouped[store_begin:store_end]
    assert "BroadcastSelectedBetas(" not in store_outputs
    assert "UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>()" in store_outputs
    reduce_begin = grouped.index("__aicore__ inline void ReduceDbProductRows")
    reduce_end = grouped.index(
        "__aicore__ inline void FinalizeStageLeft", reduce_begin
    )
    reduce_rows = grouped[reduce_begin:reduce_end]
    assert "if constexpr (KDA_GROUPED_COALESCE_DB_REDUCE)" in reduce_rows
    assert "WholeReduceSum(dbAcc, product, 64, ROWS," in reduce_rows
    assert "WholeReduceSum(dbAcc[1], product[64], 64, ROWS," in reduce_rows
    assert "product[row * KDA_GROUPED_K]" in reduce_rows
    assert "WholeReduceSum(dbCompact, dbAcc" in reduce_rows
    assert grouped.count("ReduceDbProductRows<") == 3
    assert (
        "ReduceDbProductRows<KDA_GROUPED_ROWS_PER_AIV>(" in grouped
    )
    assert "ReduceDbProductRows<KDA_GROUPED_SELECTED_ROWS>(" in grouped
    task_store_begin = grouped.index(
        "__aicore__ inline void ProcessAivWithTaskStoreOverlap"
    )
    task_store_end = grouped.index(
        "__aicore__ inline void AllocAivSyncEvents", task_store_begin
    )
    task_store = grouped[task_store_begin:task_store_end]
    assert "SyncMte3ToMte2();" not in task_store
    assert task_store.index("PrepareStageAiv<0>") < task_store.index(
        "IssueStagedTaskOutputs(completedB"
    )
    assert task_store.index("PrepareStageAiv<1>") < task_store.index(
        "IssueStagedTaskOutputs(completedB"
    )
    assert task_store.index("IssueStagedTaskOutputs(completedB") < task_store.index(
        "SyncMte3ToV();", task_store.index("IssueStagedTaskOutputs(completedB")
    )
    assert "ZeroAccumulators();" in task_store
    steady_prepare0 = task_store.rindex("PrepareStageAiv<0>")
    staged_outputs = task_store.index("StageTaskOutputsInAccumulators(")
    preserve_outputs = task_store.index("SyncVToMte2();", staged_outputs)
    advance_next_task = task_store.index(
        "AdvanceTaskCoordinates(b, hv, chunkStart", preserve_outputs
    )
    steady_shared_gates = task_store.rindex(
        "BuildSharedTaskGates();", steady_prepare0
    )
    steady_prepare1 = task_store.rindex(
        "PrepareStageAiv<1>", steady_shared_gates
    )
    steady_issue = task_store.rindex(
        "IssueStagedTaskOutputs(completedB", steady_prepare1
    )
    steady_wait = task_store.rindex("SyncMte3ToV();", steady_issue)
    steady_zero = task_store.rindex("ZeroAccumulators();", steady_wait)
    assert (
        staged_outputs
        < preserve_outputs
        < advance_next_task
        < steady_prepare0
        < steady_shared_gates
        < steady_prepare1
        < steady_issue
        < steady_wait
        < steady_zero
    )
    # The old output DMA is ordered to Vector before the accumulator clear.
    # ConsumeStageAiv<0> then provides the unconditional V->MTE2 edge before
    # the first future workspace read, so no direct MTE3->MTE2 edge is needed.
    diag_workspace_load = consume.index(
        "slotBase + diagLeftBase + rowStart * KDA_GROUPED_K"
    )
    assert consume.rindex("SyncVToMte2();", 0, diag_workspace_load) < diag_workspace_load
    stage_outputs_begin = grouped.index(
        "__aicore__ inline void StageTaskOutputsInAccumulators"
    )
    stage_outputs_end = grouped.index(
        "__aicore__ inline void IssueStagedTaskOutputs", stage_outputs_begin
    )
    stage_outputs = grouped[stage_outputs_begin:stage_outputs_end]
    assert "UbTensor<float, KDA_GROUPED_DQ_ACC_UB>()" in stage_outputs
    assert "UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>()" in stage_outputs
    assert "UbTensor<float, KDA_GROUPED_DK_RIGHT_ACC_UB>()" in stage_outputs
    assert "UbTensor<float, KDA_GROUPED_BATCH_ROW_TMP_UB>()" in stage_outputs
    stage_outputs_flat = " ".join(stage_outputs.split())
    db_load = stage_outputs_flat.index("DataCopyPad(dbLocal,")
    db_product = stage_outputs_flat.index("Mul(product, dkLeftAcc, kCache")
    db_reduce = stage_outputs_flat.index(
        "ReduceDbProductRows<KDA_GROUPED_SELECTED_ROWS>"
    )
    db_add = stage_outputs_flat.index("Add(dbLocal, dbLocal, dbCompact")
    beta_scale = stage_outputs_flat.index(
        "ScaleRowsByBeta<KDA_GROUPED_SELECTED_ROWS>"
    )
    assert db_load < db_product < db_reduce < db_add < beta_scale
    dk_sub = stage_outputs_flat.index(
        "ContiguousSub<KDA_GROUPED_SELECTED_ELEMENTS>( rowTmp, dkLeftAcc, dkRightAcc);"
    )
    dk_left_add = stage_outputs_flat.index(
        "ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>( dkLeftAcc, dkInput, dkLeftAcc);"
    )
    dk_right_add = stage_outputs_flat.index(
        "ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>( dkLeftAcc, dkLeftAcc, dkRightAcc);"
    )
    dg_load = stage_outputs_flat.index("DataCopyPad(dkRightAcc,")
    dg_q = stage_outputs_flat.index(
        "ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>( dkRightAcc, qCache, dqAcc);"
    )
    dq_add = stage_outputs_flat.index(
        "ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>( dqAcc, dqInput, dqAcc);"
    )
    dg_k = stage_outputs_flat.index(
        "ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>( dkRightAcc, rowTmp, kCache);"
    )
    assert dk_sub < dk_left_add < dk_right_add < dg_load < dg_q < dq_add < dg_k
    # The direct dg load may overwrite dk-right only after both consumers of
    # its old value have retired.  q*dq must likewise read dq before dq input
    # is accumulated in place.
    assert (
        "PipeBarrier<PIPE_V>(); EndReusableFp32Mask(); SyncVToMte2(); "
        "DataCopyPad(dkRightAcc," in stage_outputs_flat
    )
    assert "dkRightAcc, qCache, dqAcc); PipeBarrier<PIPE_V>();" in stage_outputs_flat
    issue_outputs_begin = grouped.index(
        "__aicore__ inline void IssueStagedTaskOutputs"
    )
    issue_outputs_end = grouped.index(
        "__aicore__ inline void StoreTaskOutputs", issue_outputs_begin
    )
    issue_outputs = grouped[issue_outputs_begin:issue_outputs_end]
    assert issue_outputs.count("DataCopyPad(") == 4
    assert issue_outputs.index("SyncVToMte3();") < issue_outputs.index(
        "DataCopyPad(dqOut_"
    )
    assert "SyncMte3ToMte2();" not in issue_outputs
    assert "SyncMte3ToV();" not in issue_outputs
    per_row_layout = [
        (row * 8 + half, row * 128 + half * 64)
        for row in range(32)
        for half in range(2)
    ]
    coalesced_layout = [
        (row * 8 + half, row * 128 + half * 64)
        for half in range(2)
        for row in range(32)
    ]
    assert sorted(coalesced_layout) == sorted(per_row_layout)
    assert "BuildAndWritePairB<STAGE, 0, 0, true>" in grouped
    assert "BuildAndWritePairB<STAGE, 1, 1, true>" in grouped
    assert "WaitPairScratchDone<0>();" in grouped
    assert "WaitPairScratchDone<1>();" in grouped
    assert "BuildAndWriteDiagonalB<STAGE, 1>" in grouped
    aic_begin = grouped.index("__aicore__ inline void ProcessAic()")
    aic_end = grouped.index("__aicore__ inline void ProcessAiv()", aic_begin)
    aic = grouped[aic_begin:aic_end]
    persistent_begin = aic.index(
        "if constexpr (KDA_GROUPED_PERSISTENT_MMAD_ENGINES)"
    )
    scoped_begin = aic.index("} else {", persistent_begin)
    persistent_aic = aic[persistent_begin:scoped_begin]
    persistent_loop = persistent_aic.index("for (uint64_t task")
    persistent_last_done = persistent_aic.rindex("doneFlag_")
    assert persistent_aic.count(".preSetFlags();") == 1
    assert persistent_aic.count(".finalWaitFlags();") == 1
    assert persistent_aic.index("left32.preSetFlags();") < persistent_loop
    assert persistent_aic.index("left32.finalWaitFlags();") > persistent_last_done
    assert "right16.preSetFlags();" not in persistent_aic
    assert "right16.finalWaitFlags();" not in persistent_aic
    assert "ComputeDiagonalPersistentAic(left32, right16, 0);" in persistent_aic
    assert "ComputeGroupedStagePersistentAic<1>(left32, right16, 1);" in persistent_aic
    assert "ComputeGroupedStagePersistentAic<2>(left32, right16, 0);" in persistent_aic
    assert "ComputeGroupedStagePersistentAic<3>(left32, right16, 1);" in persistent_aic
    scoped_aic = aic[scoped_begin:]
    assert "KdaBwdGroupedScopedLeft16Mmad left16(resource);" in scoped_aic
    assert "KdaBwdGroupedScopedLeft32Mmad left32(resource);" in scoped_aic
    assert "KdaBwdGroupedScopedRight16Mmad right16(resource);" in scoped_aic
    assert "static_assert(!BaseMmad::ENABLE_L1_RESIDENT" in grouped
    assert "BaseMmad::L1A_STAGES == 2 && BaseMmad::L1B_STAGES == 2" in grouped
    assert "KdaBwdGroupedPersistentLeftBase::L1_TILE_K == 32" in grouped
    assert "KdaBwdGroupedPersistentRightBase::L0_TILE_K == 32" in grouped
    init_begin = grouped.index("__aicore__ inline void Init(")
    init_end = grouped.index("__aicore__ inline void ProcessAic()", init_begin)
    init = grouped[init_begin:init_end]
    assert init.count("AllocEventID<HardEvent::MTE3_V>()") == 2
    assert init.index("pipe_->InitBuffer(ubBuf_, KDA_GROUPED_UB_BYTES);") < init.index(
        "AllocAivSyncEvents();"
    )
    assert init.index("AllocAivSyncEvents();") < init.index(
        "pairScratchDone0_ ="
    )
    process_begin = grouped.index("__aicore__ inline void ProcessAiv()")
    process_end = grouped.index("private:", process_begin)
    process = grouped[process_begin:process_end]
    dispatch = process.index("if constexpr (KDA_GROUPED_OVERLAP_TASK_STORE)")
    subblock_read = "static_cast<uint32_t>(GetSubBlockIdx())"
    assert process.count(subblock_read) == 1
    assert grouped.count(subblock_read) == 1
    assert grouped.count("subBlockIdx * KDA_GROUPED_ROWS_PER_AIV") == 1
    assert process.index("InitializeCausalSelectState(rowStart);") < dispatch
    assert process.index("ProcessAivWithTaskStoreOverlap(rowStart);") > dispatch
    assert process.index("ProcessAivSerialTaskStore(rowStart);") > dispatch
    assert process.count("ReleaseEventID<HardEvent::MTE3_V>") == 2
    assert process.rindex("ReleaseEventID<HardEvent::MTE3_V>") > process.rindex(
        "ProcessAivSerialTaskStore(rowStart);"
    )
    assert process.rindex("ReleaseEventID<HardEvent::MTE3_V>") > process.rindex(
        "ProcessAivWithTaskStoreOverlap(rowStart);"
    )
    assert process.rindex("ReleaseAivSyncEvents();") > process.rindex(
        "ReleaseEventID<HardEvent::MTE3_V>"
    )

    advance_begin = grouped.index(
        "__aicore__ inline void AdvanceTaskCoordinates("
    )
    advance_end = grouped.index(
        "__aicore__ inline void BuildSharedTaskGates()", advance_begin
    )
    advance = grouped[advance_begin:advance_end]
    assert "chunkStart += chunkStartStep;" in advance
    assert "if (hv >= heads_)" in advance
    assert "if (chunkStart >= sequenceSpan)" in advance

    serial_begin = grouped.index(
        "__aicore__ inline void ProcessAivSerialTaskStore("
    )
    serial_end = grouped.index(
        "__aicore__ inline void ProcessAivWithTaskStoreOverlap(", serial_begin
    )
    serial = grouped[serial_begin:serial_end]
    task_loop = serial.index("while (true)")
    assert serial.index("ResolveTask(task, b, hv, chunkStart);") < task_loop
    assert "InitializeCausalSelectState(" not in serial
    assert "ResolveTask(" not in serial[task_loop:]
    assert "const uint64_t flatChunkStep = usedCoreNum_ / heads_;" in serial
    assert "const uint64_t batchStep = flatChunkStep / chunks_;" in serial
    assert "AdvanceTaskCoordinates(b, hv, chunkStart, batchStep, headStep," in serial[
        task_loop:
    ]

    # Slot 0/1 reuse is legal only after the matching AIC result has been
    # consumed. Keep the exact two-slot producer/consumer order guarded here;
    # otherwise a harmless-looking schedule edit can overwrite stage-0/1
    # workspace while either AIC or AIV is still reading it.
    serial_flat = " ".join(serial.split())
    for prepared in (
        "PrepareStageAiv<0>(b, hv, chunkStart, rowStart, 0);",
        "PrepareStageAiv<1>(b, hv, chunkStart, rowStart, 1);",
        "PrepareStageAiv<2>(b, hv, chunkStart, rowStart, 0);",
        "PrepareStageAiv<3>(b, hv, chunkStart, rowStart, 1);",
    ):
        assert prepared in serial_flat
    slot_flow = (
        "PrepareStageAiv<0>",
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>",
        "PrepareStageAiv<1>",
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>",
        "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>",
        "ConsumeStageAiv<0>(0, rowStart)",
        "PrepareStageAiv<2>",
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>",
        "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>",
        "ConsumeStageAiv<1>(1, rowStart)",
        "PrepareStageAiv<3>",
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>",
        "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>",
        "ConsumeStageAiv<2>(0, rowStart)",
        "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>",
        "ConsumeStageAiv<3>(1, rowStart)",
    )
    cursor = task_loop
    for marker in slot_flow:
        cursor = serial.index(marker, cursor) + len(marker)

    alloc_begin = grouped.index("__aicore__ inline void AllocAivSyncEvents()")
    alloc_end = grouped.index(
        "__aicore__ inline void ReleaseAivSyncEvents()", alloc_begin
    )
    alloc_events = grouped[alloc_begin:alloc_end]
    release_end = grouped.index("template <typename U", alloc_end)
    release_events = grouped[alloc_end:release_end]
    for hard_event, member in (
        ("MTE2_V", "mte2ToVEvent_"),
        ("V_MTE2", "vToMte2Event_"),
        ("S_V", "sToVEvent_"),
        ("V_MTE3", "vToMte3Event_"),
        ("MTE3_MTE2", "mte3ToMte2Event_"),
        ("MTE3_V", "mte3ToVEvent_"),
    ):
        assert (
            f"{member} = GetTPipePtr()->AllocEventID<HardEvent::{hard_event}>();"
            in " ".join(alloc_events.split())
        )
        assert (
            f"ReleaseEventID<HardEvent::{hard_event}>({member});" in release_events
        )
    sync_begin = grouped.index("__aicore__ inline void SyncMte2ToV()")
    sync_end = grouped.index(
        "__aicore__ inline void BroadcastSelectedBetas", sync_begin
    )
    sync_helpers = grouped[sync_begin:sync_end]
    assert "AllocEventID" not in sync_helpers
    assert "ReleaseEventID" not in sync_helpers
    assert "TEventID dAReady" not in grouped
    assert "HardEvent::V_S" not in grouped
    assert "SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);" in grouped
    assert "WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);" in grouped
    select_begin = grouped.index(
        "__aicore__ inline void InitializeCausalSelectState"
    )
    select_end = grouped.index(
        "__aicore__ inline void LoadTaskFeatures", select_begin
    )
    select_state = grouped[select_begin:select_end]
    assert "zeroPtr[word] = 0;" in select_state
    assert "SyncSToV();" in select_state
    assert "SyncVToS" not in select_state
    assert "Duplicate(" not in select_state

    # AIC output aliases retired workspace inputs.  Both layouts preserve the
    # numerical call order; packed-A changes only the A subview and right-B
    # offsets, never either gate reference or the 17 logical GEMMs.
    grouped_aic_begin = grouped.index(
        "__aicore__ inline void ComputeGroupedStageAic"
    )
    grouped_aic_end = grouped.index(
        "__aicore__ inline void ComputeDiagonalPersistentAic",
        grouped_aic_begin,
    )
    grouped_aic = grouped[grouped_aic_begin:grouped_aic_end]
    packed_begin = grouped_aic.index(
        "if constexpr (KDA_GROUPED_PACK_STAGE_A)"
    )
    split_begin = grouped_aic.index("} else {", packed_begin)
    packed_aic = grouped_aic[packed_begin:split_begin]
    split_aic = grouped_aic[split_begin:]
    packed_off_left = packed_aic.index("RunPackedOffLeftAic<STAGE>")
    packed_off_right = packed_aic.index("RunOffRightPairAic<STAGE, 0>")
    packed_diag_right = packed_aic.index("RunPackedDiagRightAic<STAGE>")
    packed_diag_left = packed_aic.index("RunPackedDiagLeftAic<STAGE>")
    assert packed_off_left < packed_off_right < packed_diag_right
    assert packed_diag_right < packed_diag_left
    assert "rightMmad.preSetFlags();" not in packed_aic
    assert "rightMmad.finalWaitFlags();" not in packed_aic
    assert "RunOffRightPairAic<STAGE, 0, false>" not in packed_aic
    assert "RunPackedDiagRightAic<STAGE, false>" not in packed_aic
    off_left = split_aic.index("RunRowMajorAic<32, offPrefix>")
    off_right = split_aic.index("RunOffRightPairAic<STAGE, 0>")
    diag_right = split_aic.index("RunColumnMajorAic<16, 32>")
    diag_left = split_aic.index("RunRowMajorAic<32, 16>", diag_right)
    assert off_left < off_right < diag_right < diag_left
    assert "rightMmad.preSetFlags();" not in split_aic
    assert "rightMmad.finalWaitFlags();" not in split_aic
    assert "RunOffRightPairAic<STAGE, 0, false>" not in split_aic
    assert "RunColumnMajorAic<16, 32, false>" not in split_aic

    # Prove the alternate packed-A address mapping independently of CATLASS
    # execution.  A RowMajor [32,prefix] buffer reinterpreted as ColumnMajor
    # [prefix,32] is its exact transpose because both use offset
    # column*prefix + row.  Every stage must therefore expose the same four
    # mathematical A operands as the split Aoff/Adiag layout.
    for stage in range(4):
        prefix = (stage + 1) * 16
        off_prefix = stage * 16
        rows = [
            [row * 1000 + col for col in range(prefix)]
            for row in range(32)
        ]
        packed_flat = [value for row in rows for value in row]

        packed_off_left = [
            packed_flat[row * prefix : row * prefix + off_prefix]
            for row in range(32)
        ]
        split_off_left = [row[:off_prefix] for row in rows]
        assert packed_off_left == split_off_left

        packed_diag_left = [
            packed_flat[
                row * prefix + off_prefix : row * prefix + prefix
            ]
            for row in range(32)
        ]
        split_diag_left = [row[off_prefix:prefix] for row in rows]
        assert packed_diag_left == split_diag_left

        packed_diag_right = [
            [packed_flat[column * prefix + row] for column in range(32)]
            for row in range(off_prefix, prefix)
        ]
        split_diag_right = [
            [rows[column][row] for column in range(32)]
            for row in range(off_prefix, prefix)
        ]
        assert packed_diag_right == split_diag_right

        for early in range(stage):
            begin = early * 16
            end = begin + 16
            packed_off_right = [
                [packed_flat[column * prefix + row] for column in range(32)]
                for row in range(begin, end)
            ]
            split_off_right = [
                [rows[column][row] for column in range(32)]
                for row in range(begin, end)
            ]
            assert packed_off_right == split_off_right

    # Mirror the C++ mixed-radix carry for representative head/chunk ratios.
    # This remains inside the source-contract item so the runner's fixed case
    # count does not change when a pure-host scheduling invariant is added.
    domains = [
        (1, 1, 1, 1),
        (1, 128, 32, 20),
        (3, 7, 1, 20),
        (2, 5, 128, 20),
        (4, 3, 7, 12),
    ]
    for batch, chunks, heads, used_cores in domains:
        task_count = batch * chunks * heads
        used_cores = min(used_cores, task_count)
        for core in range(used_cores):
            task = core
            flat_chunk = task // heads
            hv = task % heads
            b = flat_chunk // chunks
            chunk_start = (flat_chunk % chunks) * 64
            flat_chunk_step = used_cores // heads
            head_step = used_cores - flat_chunk_step * heads
            batch_step = flat_chunk_step // chunks
            chunk_step = flat_chunk_step - batch_step * chunks
            chunk_start_step = chunk_step * 64
            sequence_span = chunks * 64
            while task < task_count:
                expected_flat_chunk = task // heads
                assert (b, hv, chunk_start) == (
                    expected_flat_chunk // chunks,
                    task % heads,
                    (expected_flat_chunk % chunks) * 64,
                )
                task += used_cores
                if task >= task_count:
                    break
                b += batch_step
                chunk_start += chunk_start_step
                hv += head_step
                if hv >= heads:
                    hv -= heads
                    chunk_start += 64
                if chunk_start >= sequence_span:
                    chunk_start -= sequence_span
                    b += 1
    ready0 = serial.index(
        "Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);"
    )
    deferred_gates = serial.index("BuildSharedTaskGates();", ready0)
    deferred_zero = serial.index("ZeroAccumulators();", deferred_gates)
    prepare1 = serial.index("PrepareStageAiv<1>", deferred_zero)
    assert ready0 < deferred_gates < deferred_zero < prepare1
    shared_gate_begin = grouped.index(
        "__aicore__ inline void BuildSharedTaskGates()"
    )
    shared_gate_end = grouped.index(
        "__aicore__ inline void ProcessAivSerialTaskStore(", shared_gate_begin
    )
    shared_gates = grouped[shared_gate_begin:shared_gate_end]
    persistent_gates = shared_gates.index("BuildPersistentBlockGates();")
    pair_bridge = shared_gates.index("BuildPairBridgeGates();", persistent_gates)
    assert persistent_gates < pair_bridge
    persistent_begin = grouped.index(
        "__aicore__ inline void BuildPersistentBlockGates()"
    )
    persistent_end = grouped.index(
        "__aicore__ inline void BuildPairBridgeDifference", persistent_begin
    )
    persistent_body = grouped[persistent_begin:persistent_end]
    persistent_flat = " ".join(persistent_body.split())
    assert "for (" not in persistent_body
    assert persistent_body.count("ReferenceMinusRows(") == 3
    for block in range(3):
        assert f"rightOuter[CacheBlockOffset({block})]" in persistent_flat
        assert f"gCache[CacheBlockOffset({block})]" in persistent_flat
    stage_b_begin = grouped.index("__aicore__ inline void BuildAndWriteStageB")
    stage_b_end = grouped.index("__aicore__ inline void PrepareStageAiv", stage_b_begin)
    stage_b = grouped[stage_b_begin:stage_b_end]
    stage3_pair0 = stage_b.index("BuildAndWritePairB<STAGE, 0, 0, true>")
    stage3_pair1 = stage_b.index(
        "BuildAndWritePairB<STAGE, 1, 1, true>", stage3_pair0
    )
    stage3_wait0 = stage_b.index("WaitPairScratchDone<0>();", stage3_pair1)
    stage3_pair2 = stage_b.index(
        "BuildAndWritePairB<STAGE, 2, 0, false>", stage3_wait0
    )
    stage3_wait1 = stage_b.index("WaitPairScratchDone<1>();", stage3_pair2)
    stage3_diag = stage_b.index(
        "BuildAndWriteDiagonalB<STAGE, 1>", stage3_wait1
    )
    assert (
        stage3_pair0
        < stage3_pair1
        < stage3_wait0
        < stage3_pair2
        < stage3_wait1
        < stage3_diag
    )
    diag_begin = grouped.index("__aicore__ inline void BuildAndWriteDiagonalB")
    diag_end = grouped.index("__aicore__ inline void BuildAndWriteStageB", diag_begin)
    diag = grouped[diag_begin:diag_end]
    final_set = diag.index("SetFlag<HardEvent::MTE3_V>(pairScratchDone0_);")
    final_wait = diag.index(
        "WaitFlag<HardEvent::MTE3_V>(pairScratchDone0_);", final_set
    )
    assert final_set < final_wait
    assert "if constexpr (!KDA_GROUPED_OVERLAP_SHARED_SETUP)" in grouped
    consume_begin = grouped.index("__aicore__ inline void ConsumeStageAiv")
    consume_end = grouped.index(
        "__aicore__ inline void StageTaskOutputsInAccumulators", consume_begin
    )
    consume = grouped[consume_begin:consume_end]
    diag_accumulate = consume.index(
        "dkRightAcc[CacheBlockOffset(STAGE)], ScratchBank<2>()"
    )
    stage_finalize = consume.index("FinalizeStageLeft<STAGE>();", diag_accumulate)
    assert diag_accumulate < stage_finalize
    assert consume.count("BeginReusableFp32Mask();") == 2
    assert consume.count("EndReusableFp32Mask();") == 1
    assert "if constexpr (KDA_GROUPED_OVERLAP_STAGE_EPILOGUE)" in consume
    store = grouped[consume_end:]
    assert "if constexpr (!KDA_GROUPED_OVERLAP_STAGE_EPILOGUE)" in store
    assert "if constexpr (KDA_GROUPED_BATCH_TAIL_BLOCKS)" in store
    assert "ScaleRowsByBeta<KDA_GROUPED_ROWS_PER_AIV>" in grouped
    assert "DataCopyExtParams featureInputs" in store
    assert "DataCopyExtParams featureOutputs" in store
    assert "DataCopyExtParams dbOutputs" in store
    assert "featureBlockGapBytes" in store
    assert "dbBlockGapBytes" in store
    assert "KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV" in store
    assert "KDA_GROUPED_SELECTED_ELEMENTS" in store
    db_batch_begin = store.index("if constexpr (KDA_GROUPED_BATCH_TAIL_BLOCKS)")
    batch_begin = store.index(
        "if constexpr (KDA_GROUPED_BATCH_TAIL_BLOCKS)", db_batch_begin + 1
    )
    scalar_begin = store.index("} else {", batch_begin)
    batch_tail = store[batch_begin:scalar_begin]
    assert batch_tail.count("DataCopyPad(") == 7
    batch_tail_flat = " ".join(batch_tail.split())
    batch_dq = batch_tail_flat.index("batchOutQ, batchOutQ, dqAcc")
    batch_dk_left = batch_tail_flat.index("batchOutK, batchOutK, dkLeftAcc")
    batch_dk_right = batch_tail_flat.index("batchOutK, batchOutK, dkRightAcc")
    batch_dg_q = batch_tail_flat.index("batchOutG, qCache, dqAcc")
    batch_dg_sub = batch_tail_flat.index("batchRowTmp, dkLeftAcc, dkRightAcc")
    batch_dg_k = batch_tail_flat.index("batchOutG, batchRowTmp, kCache")
    assert (
        batch_dq
        < batch_dk_left
        < batch_dk_right
        < batch_dg_q
        < batch_dg_sub
        < batch_dg_k
    )
    batch_mask_end = batch_tail_flat.index("EndReusableFp32Mask();", batch_dg_k)
    batch_v_to_mte3 = batch_tail_flat.index("SyncVToMte3();", batch_mask_end)
    batch_mte3_to_mte2 = batch_tail_flat.index("SyncMte3ToMte2();", batch_v_to_mte3)
    batch_mte3_to_v = batch_tail_flat.index("SyncMte3ToV();", batch_mte3_to_mte2)
    assert (
        batch_dg_k
        < batch_mask_end
        < batch_v_to_mte3
        < batch_mte3_to_mte2
        < batch_mte3_to_v
    )
    reuse_guard = grouped.index("if constexpr (STAGE >= KDA_GROUPED_QUEUE_DEPTH)")
    next_copy = grouped.index("DataCopyExtParams rowParams", reuse_guard)
    assert "SyncVToMte2();" in grouped[reuse_guard:next_copy]


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


def test_chunk_kda_bwd_intra_safe_gate_grouped_fastpath_bf16():
    """Exercise the target BF16/safe shape on the current delivery path."""
    device = _device()
    inputs = _case(t=64, h=2, hv=2, kdim=128, dtype=torch.bfloat16, gate_scale=0.2)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)

    # Rows 7 and 8 are owned by different AIV subblocks. Upper-triangular dA
    # entries straddling that boundary must be removed by both causal Select
    # masks; otherwise they produce an unmistakable dq/dk signal.
    mask_inputs = list(_empty_grouped_precision_case())
    q, k, _, _, dAqk, dAkk, _, _, _, _ = mask_inputs
    q[0, 7, 0, 0] = 1.0
    k[0, 8, 0, 1] = 1.0
    k[0, 8, 0, 2] = 1.0
    k[0, 9, 0, 2] = 1.0
    dAqk[0, 7, 0, 8] = 3.0
    dAkk[0, 8, 0, 9] = -2.0
    masked = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in mask_inputs), chunk_size=64, safe_gate=True
    )
    for actual, zero in zip(masked, mask_inputs[6:10]):
        torch.testing.assert_close(
            actual.detach().cpu(), zero, rtol=0.0, atol=0.0, check_dtype=False
        )


def test_chunk_kda_bwd_intra_grouped_fastpath_right_diag_ftz_guard():
    _run_grouped_precision_guard(_right_diag_ftz_case(), "dk", 15, 2.0 ** -18)


def test_chunk_kda_bwd_intra_grouped_fastpath_left_diag_overflow_guard():
    _run_grouped_precision_guard(_left_diag_overflow_case(), "dq", 15, 2.0 ** 20)


def test_chunk_kda_bwd_intra_grouped_fastpath_off_right_and_pair_bridge_ftz_guard():
    inputs, expected = _off_right_cross_block_ftz_case()
    _run_grouped_precision_guard(inputs, "dk", 0, expected)

    # The cross-block guard above has a unit left-outer factor.  This second
    # launch puts both left/right inner gates just above the FP32 normal floor
    # and requires a non-unit outer * non-unit bridge on opposite AIV rows.
    inputs, expected = _pair_bridge_reassociation_case()
    device = _device()
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    for name, token in (("dq", 33), ("dk", 14)):
        _assert_sparse_signal(ref, name, token, expected, rtol=1e-12)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=0.0)
    for output, name, token in ((got[0], "dq", 33), (got[1], "dk", 14)):
        signal = output.detach().cpu()[0, token, 0, 0]
        assert torch.isfinite(signal) and signal != 0, (
            f"NPU {name} bridge signal was lost"
        )
        torch.testing.assert_close(
            signal,
            torch.tensor(expected, dtype=signal.dtype),
            rtol=5e-3,
            atol=0.0,
            msg=f"NPU {name}[{token},0] changed at the pair-bridge FTZ boundary",
        )


def test_chunk_kda_bwd_intra_grouped_fastpath_cancellation_guard():
    _run_grouped_precision_guard(_grouped_cancellation_case(), "dq", 48, 1.0)


def test_chunk_kda_bwd_intra_grouped_fastpath_cross_task_slot_canary():
    """Use 128 unique batch/chunk/head tasks to catch slot reuse and workspace bleed."""
    device = _device()
    inputs = list(
        _case(b=2, t=128, h=32, hv=32, kdim=128, dtype=torch.bfloat16, gate_scale=0.0)
    )
    for tensor in inputs:
        tensor.zero_()

    q, k, _, _, dAqk, _, _, _, _, _ = inputs
    expected = tuple(tensor.clone() for tensor in inputs[6:10])
    # The final stage reaches source row 47 and features 126/127, while the
    # four entries alternate slot 0/1 and then reuse both slots in each task.
    stage_markers = (
        (7, 7, 0, 1),
        (24, 3, 2, 3),
        (40, 19, 4, 5),
        (63, 47, 126, 127),
    )
    for batch in range(2):
        for chunk in range(2):
            chunk_start = chunk * 64
            for head in range(32):
                task = (batch * 2 + chunk) * 32 + head
                for stage, (target, source, k_feature, q_feature) in enumerate(stage_markers):
                    marker = 1.0 + stage / 4.0 + task / 512.0
                    q[batch, chunk_start + target, head, q_feature] = 1.0
                    k[batch, chunk_start + source, head, k_feature] = 1.0
                    dAqk[batch, chunk_start + target, head, source] = marker
                    expected[0][batch, chunk_start + target, head, k_feature] = marker
                    expected[1][batch, chunk_start + source, head, q_feature] = marker

    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(
            actual.detach().cpu(), reference, rtol=0.0, atol=0.0, check_dtype=False
        )


def test_chunk_kda_bwd_intra_grouped_fastpath_repeated_large_negative_gate():
    device = _device()
    inputs = _case(t=64, h=1, hv=1, kdim=128, dtype=torch.bfloat16, gate_scale=4.0)
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)
    first = fla_ascendc.chunk_kda_bwd_intra(
        *device_inputs, chunk_size=64, safe_gate=True
    )
    second = fla_ascendc.chunk_kda_bwd_intra(
        *device_inputs, chunk_size=64, safe_gate=True
    )
    # Both launches are enqueued before the first host synchronization.
    _assert_outputs(first, ref, rtol=5e-3, atol=5e-3)
    _assert_outputs(second, ref, rtol=5e-3, atol=5e-3)
    for first_output, second_output in zip(first, second):
        torch.testing.assert_close(
            first_output.detach().cpu(),
            second_output.detach().cpu(),
            rtol=0.0,
            atol=0.0,
            check_dtype=False,
        )


@pytest.mark.parametrize("source", ["dAqk", "dAkk"])
def test_chunk_kda_bwd_intra_grouped_fastpath_cross_aiv_one_hot(source):
    """Cover grouped stage-1/stage-3 data owned by opposite AIV subblocks."""
    device = _device()
    inputs = list(_case(t=64, h=1, hv=1, kdim=128, dtype=torch.bfloat16, gate_scale=0.2))
    inputs[4].zero_()
    inputs[5].zero_()
    inputs[4 if source == "dAqk" else 5][0, 24, 0, 3] = 0.5
    # Stage 3 groups all three early blocks; row 55 and column 43 are owned by
    # opposite AIV subblocks and exercise pair 2's stride-48 ColumnMajor view.
    inputs[4 if source == "dAqk" else 5][0, 55, 0, 43] = -0.25
    ref = _golden(*inputs, chunk_size=64, safe_gate=True)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=True
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)

    if source == "dAkk":
        # Keep this inside an existing pytest item so the runner's 37-case
        # provenance stays stable.  One exact diagonal marker in every
        # stage/AIV half makes dk-left complete at ConsumeStage, while beta=0
        # distinguishes "reduce then scale" from an accidental early scale.
        epilogue_inputs = list(_empty_grouped_precision_case())
        _, epilogue_k, _, epilogue_beta, epilogue_daqk, epilogue_dakk, \
            epilogue_dq, epilogue_dk, epilogue_db, epilogue_dg = epilogue_inputs
        epilogue_beta.zero_()
        beta_values = (0.0, 0.25, 0.5, 1.0, 0.0, 0.75, 0.125, 1.0)
        k_values = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0)
        for marker, token in enumerate((0, 8, 16, 24, 32, 40, 48, 56)):
            epilogue_beta[0, token, 0] = beta_values[marker]
            epilogue_k[0, token, 0, marker] = k_values[marker]
            epilogue_dakk[0, token, 0, token] = 1.0
            epilogue_db[0, token, 0] = (marker - 4) / 8.0
        assert torch.count_nonzero(epilogue_daqk) == 0
        assert torch.count_nonzero(epilogue_dq) == 0
        assert torch.count_nonzero(epilogue_dk) == 0
        assert torch.count_nonzero(epilogue_dg) == 0
        epilogue_ref = _golden(
            *epilogue_inputs, chunk_size=64, safe_gate=True
        )
        epilogue_got = fla_ascendc.chunk_kda_bwd_intra(
            *(tensor.to(device) for tensor in epilogue_inputs),
            chunk_size=64,
            safe_gate=True,
        )
        _assert_outputs(epilogue_got, epilogue_ref, rtol=0.0, atol=0.0)


def test_chunk_kda_bwd_intra_unsafe_branch():
    device = _device()
    inputs = _case(t=31, kdim=48, gate_scale=0.01)
    ref = _golden(*inputs, chunk_size=64, safe_gate=False)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=False
    )
    _assert_outputs(got, ref)


def test_chunk_kda_bwd_intra_unsafe_target_shape_falls_back_from_grouped():
    device = _device()
    inputs = _case(
        t=64, h=2, hv=2, kdim=128, dtype=torch.bfloat16, gate_scale=0.01
    )
    ref = _golden(*inputs, chunk_size=64, safe_gate=False)
    got = fla_ascendc.chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in inputs), chunk_size=64, safe_gate=False
    )
    _assert_outputs(got, ref, rtol=5e-3, atol=5e-3)


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
