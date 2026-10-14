# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GDN Phase6 公共接口的同输入布局、view 和门控 dtype 等价回归。

默认只校验并收集用例定义，不导入 NPU runtime，也不报告 NPU 通过。
设备运行示例：python test_gdn_phase6_native_layout.py --npu --device 0 --output result.json
--device 指物理卡号；存在 ASCEND_RT_VISIBLE_DEVICES 时按其映射绑定逻辑卡。
数值门禁为逐元素相等，A 只比较真实 chunk 内定义的有效下三角。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any


SEEDS = (42, 20260813, 20260827)
INPUT_RANGES = {"q": (-0.05, 0.05), "k": (-0.05, 0.05), "v": (-0.05, 0.05),
                "g": (-1.0, 1.0), "beta": (-1.0, 1.0)}
OUTPUT_NAMES = ("o", "final_state", "g_cumsum", "A")


@dataclass(frozen=True)
class Case:
    name: str
    dtype: str
    batch: int
    tokens: int
    key_heads: int
    value_heads: int
    value_dim: int
    chunk_size: int
    initial_state: bool
    final_state: bool
    state_dtype: str = "fp32"
    cu_seqlens: tuple[int, ...] | None = None

    @property
    def sequences(self) -> int:
        return self.batch if self.cu_seqlens is None else len(self.cu_seqlens) - 1


@dataclass(frozen=True)
class Variant:
    name: str
    layout: str
    gate_dtype: str
    beta_dtype: str
    view: str = "dense"
    export_intermediates: bool = True


CASES = (
    Case("bf16_dense_gva_v128_bt64", "bf16", 2, 65, 2, 4, 128, 64, False, False),
    Case("bf16_varlen_mha_v256_bt128", "bf16", 1, 129, 2, 2, 256, 128,
         True, True, "public", (0, 1, 129)),
    Case("fp16_dense_mha_v128_bt128", "fp16", 2, 129, 2, 2, 128, 128, True, False),
    Case("fp16_varlen_gva_v256_bt64", "fp16", 1, 65, 2, 4, 256, 64,
         False, True, "fp32", (0, 1, 65)),
    Case("bf16_varlen_empty_sequences", "bf16", 1, 130, 2, 4, 128, 64,
         True, True, "fp32", (0, 0, 3, 3, 130)),
)
VARIANTS = (
    Variant("bnsd_public_public", "BNSD", "public", "public"),
    Variant("ntd_fp32_public", "NTD", "fp32", "public"),
    Variant("bsnd_public_fp32", "BSND", "public", "fp32"),
    Variant("tnd_fp32_fp32", "TND", "fp32", "fp32"),
    Variant("bsnd_offset", "BSND", "public", "public", "offset"),
    Variant("tnd_offset", "TND", "public", "fp32", "offset"),
    Variant("bsnd_strided", "BSND", "fp32", "public", "strided"),
    Variant("tnd_strided", "TND", "public", "public", "strided"),
    Variant("bsnd_inference", "BSND", "public", "public", export_intermediates=False),
)


def _variants(case: Case) -> tuple[Variant, ...]:
    # 空序列只验证同输入布局/view 等价与状态有限值，不另加数学标杆。
    if case.cu_seqlens is not None and any(a == b for a, b in
                                          zip(case.cu_seqlens, case.cu_seqlens[1:])):
        return tuple(replace(v, gate_dtype="fp32", beta_dtype="fp32") for v in VARIANTS[:8])
    return VARIANTS


def _chunk_indices(case: Case) -> tuple[int, ...] | None:
    if case.cu_seqlens is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(case.cu_seqlens, case.cu_seqlens[1:])):
        for chunk in range((end - begin + case.chunk_size - 1) // case.chunk_size):
            result.extend((sequence, chunk))
    return tuple(result)


def _chunk_ranges(case: Case):
    ranges = ((b, 0, case.tokens) for b in range(case.batch)) if case.cu_seqlens is None else (
        (0, begin, end) for begin, end in zip(case.cu_seqlens, case.cu_seqlens[1:]))
    for batch, begin, end in ranges:
        for start in range(begin, end, case.chunk_size):
            yield batch, start, min(start + case.chunk_size, end)


def _definition_check() -> dict[str, Any]:
    assert len(SEEDS) == 3 and len(set(SEEDS)) == 3
    assert {(c.chunk_size, c.value_dim) for c in CASES} == {
        (64, 128), (64, 256), (128, 128), (128, 256)}
    assert {(c.initial_state, c.final_state) for c in CASES} == {
        (False, False), (False, True), (True, False), (True, True)}
    definitions = []
    for case in CASES:
        assert case.dtype in ("bf16", "fp16") and case.state_dtype in ("public", "fp32")
        assert case.batch > 0 and case.tokens > 0 and case.key_heads > 0
        assert case.value_heads % case.key_heads == 0
        if case.cu_seqlens is not None:
            assert case.batch == 1 and len(case.cu_seqlens) >= 2
            assert case.cu_seqlens[0] == 0 and case.cu_seqlens[-1] == case.tokens
            assert all(a <= b for a, b in zip(case.cu_seqlens, case.cu_seqlens[1:]))
            assert len(_chunk_indices(case)) == 2 * len(tuple(_chunk_ranges(case)))
        definitions.append({**asdict(case), "q_shape": [case.batch, case.key_heads, case.tokens, 128],
                            "v_shape": [case.batch, case.value_heads, case.tokens, case.value_dim],
                            "gate_shape": [case.batch, case.tokens, case.value_heads],
                            "state_shape": [case.sequences, case.value_heads, 128, case.value_dim],
                            "chunk_indices": _chunk_indices(case),
                            "variants": [asdict(v) for v in _variants(case)]})
    return {"result": "passed", "scope": "case_definitions_only", "seeds": SEEDS,
            "input_ranges": INPUT_RANGES, "cases": definitions,
            "reference_calls": len(CASES) * len(SEEDS),
            "logical_checks": sum(len(_variants(c)) for c in CASES) * len(SEEDS)}


def _json_number(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def _tensor_summary(tensor, torch) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    values = cpu.double()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "stride": list(tensor.stride()), "storage_offset": int(tensor.storage_offset()),
            "storage_bytes": int(tensor.untyped_storage().nbytes()),
            "contiguous": bool(tensor.is_contiguous()), "logical_sha256": digest,
            "finite": bool(torch.isfinite(values).all().item()),
            "min": _json_number(float(values.min().item())),
            "max": _json_number(float(values.max().item()))}


def _make_inputs(case: Case, seed: int, torch):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    shapes = {"q": (case.batch, case.key_heads, case.tokens, 128),
              "k": (case.batch, case.key_heads, case.tokens, 128),
              "v": (case.batch, case.value_heads, case.tokens, case.value_dim),
              "g": (case.batch, case.tokens, case.value_heads),
              "beta": (case.batch, case.tokens, case.value_heads)}
    inputs = {name: torch.empty(shape, dtype=torch.float32).uniform_(
        *INPUT_RANGES[name], generator=generator).to(dtype) for name, shape in shapes.items()}
    inputs["initial_state"] = None
    if case.initial_state:
        state_shape = (case.sequences, case.value_heads, 128, case.value_dim)
        # 与既有 ATK deterministic_initial_state 使用相同值域和公式。
        state = ((torch.arange(math.prod(state_shape), dtype=torch.float32).remainder(257) - 128)
                 * 1.0e-4).reshape(state_shape)
        inputs["initial_state"] = state.to(dtype if case.state_dtype == "public" else torch.float32)
    return inputs


def _make_view(tensor, kind: str, torch):
    if kind == "dense":
        return tensor.contiguous().clone()
    if kind == "offset":
        offset = 13
        backing = torch.full((tensor.numel() + 2 * offset,), -7.0,
                             dtype=tensor.dtype, device=tensor.device)
        view = backing[offset:offset + tensor.numel()].view(tensor.shape)
        view.copy_(tensor)
        assert view.is_contiguous() and view.storage_offset() == offset
        return view
    backing = torch.full((*tensor.shape[:-1], tensor.shape[-1] * 2), -7.0,
                         dtype=tensor.dtype, device=tensor.device)
    view = backing[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _mask_a(tensor, case: Case, torch):
    # 尾块 padding 和上三角不纳入精确比较，也不纳入有限值判定。
    masked = torch.zeros_like(tensor)
    for batch, begin, end in _chunk_ranges(case):
        valid = end - begin
        masked[batch, :, begin:end, :valid] = torch.tril(tensor[batch, :, begin:end, :valid])
    return masked


def _compare(actual, expected, torch) -> dict[str, Any]:
    if actual is None or expected is None:
        return {"equal": actual is None and expected is None,
                "actual_none": actual is None, "expected_none": expected is None}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"equal": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
                "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    equal = bool(torch.equal(actual, expected))
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    result = {"equal": equal and finite, "finite": finite, "max_abs_diff": 0.0}
    if not equal:
        unequal = actual != expected
        result["mismatch_count"] = int(unequal.sum().item())
        result["max_abs_diff"] = _json_number(float((actual.double() - expected.double()).abs().max().item()))
        flat_index = int(unequal.reshape(-1).nonzero()[0].item())
        coordinates = []
        for size in reversed(actual.shape):
            flat_index, coordinate = divmod(flat_index, int(size))
            coordinates.append(coordinate)
        index = tuple(reversed(coordinates))
        result.update(first_mismatch=list(index), actual_value=_json_number(float(actual[index].item())),
                      expected_value=_json_number(float(expected[index].item())))
    return result


def _execute(case: Case, seed: int, variant: Variant, frozen, operator, torch, device):
    record: dict[str, Any] = {"case": case.name, "seed": seed, "call": asdict(variant),
                             "status": "failed", "errors": []}
    start = time.monotonic()
    snapshots = None
    try:
        inputs = {}
        for name in ("q", "k", "v", "g", "beta"):
            value = frozen[name]
            if name in ("q", "k", "v") and variant.layout in ("BSND", "TND"):
                value = value.transpose(1, 2).contiguous()
            if ((name == "g" and variant.gate_dtype == "fp32") or
                    (name == "beta" and variant.beta_dtype == "fp32")):
                value = value.float()
            inputs[name] = _make_view(value.to(device), variant.view, torch)
            if not torch.equal(inputs[name].cpu(), value):
                raise AssertionError(f"{name} view 改变了冻结输入")
        # 每次调用都创建初态副本，避免任何调用污染下一组输入。
        state = frozen["initial_state"]
        inputs["initial_state"] = None if state is None else state.to(device).clone()
        record["inputs"] = {name: None if value is None else _tensor_summary(value, torch)
                            for name, value in inputs.items()}
        record["attributes"] = {"cu_seqlens": case.cu_seqlens, "chunk_indices": _chunk_indices(case),
                                "chunk_size": case.chunk_size, "scale": 128 ** -0.5,
                                "output_final_state": case.final_state,
                                "use_exp2": False, "use_qk_l2norm_in_kernel": False,
                                "state_v_first": False, "return_intermediate_states": False,
                                "disable_recompute": variant.export_intermediates}
        outputs = operator(inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
                           initial_state=inputs["initial_state"], layout=variant.layout,
                           **record["attributes"])
        torch.npu.synchronize()
        if len(outputs) != 10:
            raise AssertionError(f"公共返回应有10项，实际为{len(outputs)}")
        contract = {"o_present": outputs[0] is not None,
                    "q_hat_alias": outputs[6] is inputs["q"], "k_hat_alias": outputs[7] is inputs["k"],
                    "q_rstd_none": outputs[8] is None, "k_rstd_none": outputs[9] is None,
                    "beta_eff_none": outputs[4] is None, "h_none": outputs[5] is None,
                    "final_state_presence": (outputs[1] is not None) == case.final_state,
                    "g_cumsum_presence": (outputs[2] is not None) == variant.export_intermediates,
                    "A_presence": (outputs[3] is not None) == variant.export_intermediates,
                    "initial_state_unchanged": state is None or torch.equal(inputs["initial_state"].cpu(), state)}
        snapshots = tuple(None if value is None else value.detach().cpu().clone() for value in outputs[:4])
        if snapshots[3] is not None:
            snapshots = (*snapshots[:3], _mask_a(snapshots[3], case, torch))
        expected_shapes = ((case.batch, case.tokens, case.value_heads, case.value_dim),
                           (case.sequences, case.value_heads, 128, case.value_dim),
                           (case.batch, case.tokens, case.value_heads),
                           (case.batch, case.value_heads, case.tokens, case.chunk_size))
        expected_dtypes = (inputs["q"].dtype, torch.float32 if state is None else state.dtype,
                           torch.float32, inputs["q"].dtype)
        for name, value, shape, dtype in zip(OUTPUT_NAMES, snapshots, expected_shapes, expected_dtypes):
            if value is not None:
                contract[f"{name}_shape"] = tuple(value.shape) == shape
                contract[f"{name}_dtype"] = value.dtype == dtype
                contract[f"{name}_finite"] = bool(torch.isfinite(value).all().item())
        if case.cu_seqlens is not None and snapshots[1] is not None:
            empty = [i for i, (a, b) in enumerate(zip(case.cu_seqlens, case.cu_seqlens[1:])) if a == b]
            if empty:
                contract["empty_sequence_state_finite"] = bool(torch.isfinite(snapshots[1][empty]).all().item())
        record["contract"] = contract
        record["outputs"] = {name: None if value is None else _tensor_summary(value, torch)
                             for name, value in zip(OUTPUT_NAMES, snapshots)}
        record["errors"].extend(name for name, passed in contract.items() if not passed)
        record["status"] = "passed" if not record["errors"] else "failed"
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = round(time.monotonic() - start, 6)
    return record, snapshots


def _write_report(report: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _logical_device(physical: int) -> int:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return physical
    devices = [int(item.strip()) for item in visible.split(",")]
    if physical not in devices:
        raise ValueError(f"物理卡 {physical} 不在 ASCEND_RT_VISIBLE_DEVICES={visible} 中")
    return devices.index(physical)


def _run_npu(report: dict[str, Any], args) -> None:
    import torch
    import torch_npu
    import fla_npu
    from fla_npu.ops import ascendc

    logical = _logical_device(args.device)
    torch.npu.set_device(logical)
    device = torch.device(f"npu:{logical}")
    operator = ascendc.npu_chunk_gated_delta_rule_fwd
    report["runtime"] = {"physical_device": args.device, "logical_device": logical,
                         "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
                         "torch_version": torch.__version__, "torch_npu_version": getattr(torch_npu, "__version__", None),
                         "fla_npu_path": fla_npu.__file__, "public_api": "ascendc.npu_chunk_gated_delta_rule_fwd",
                         "stable_abi_setting": os.environ.get("FLA_NPU_STABLE_ABI", "auto")}
    reference_variant = Variant("reference_bnsd_fp32_fp32", "BNSD", "fp32", "fp32")
    for case in CASES:
        for seed in SEEDS:
            frozen = _make_inputs(case, seed, torch)
            report["npu_executed"] = True
            reference_record, reference = _execute(case, seed, reference_variant, frozen, operator, torch, device)
            report["references"].append(reference_record)
            failed = 0
            for variant in _variants(case):
                record, actual = _execute(case, seed, variant, frozen, operator, torch, device)
                if actual is not None and reference is not None:
                    comparisons = {}
                    for index, name in enumerate(OUTPUT_NAMES):
                        expected = reference[index] if index < 2 or variant.export_intermediates else None
                        comparisons[name] = _compare(actual[index], expected, torch)
                    record["comparisons"] = comparisons
                    record["errors"].extend(name for name, result in comparisons.items() if not result["equal"])
                else:
                    record["errors"].append("reference_or_actual_output_missing")
                if reference_record["status"] != "passed":
                    record["errors"].append("reference_contract_failed")
                record["status"] = "passed" if not record["errors"] else "failed"
                failed += record["status"] == "failed"
                report["checks"].append(record)
            print(f"{case.name} seed={seed} checks={len(_variants(case))} failed={failed}", flush=True)
            _write_report(report, args.output)
    report["result"] = "failed" if any(item["status"] != "passed" for item in
                                        (*report["references"], *report["checks"])) else "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", action="store_true", help="执行实际 NPU 公共接口回归")
    parser.add_argument("--device", type=int, default=0, help="NPU 物理卡号")
    parser.add_argument("--output", type=Path, help="原子写入 JSON 证据，失败记录也会保留")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device 必须为非负整数")
    report: dict[str, Any] = {"schema": "gdn-phase6-native-layout/v1",
                             "mode": "npu" if args.npu else "cpu-definition-check",
                             "result": "running" if args.npu else "collected",
                             "npu_executed": False,
                             "comparison": "elementwise_exact_valid_A_lower_triangle",
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "references": [], "checks": []}
    try:
        report["definitions"] = _definition_check()
        if args.npu:
            _write_report(report, args.output)
            _run_npu(report, args)
    except Exception as exc:
        report["result"] = "failed"
        report["setup_or_runner_error"] = {"error": f"{type(exc).__name__}: {exc}",
                                           "traceback": traceback.format_exc()}
    except KeyboardInterrupt:
        report["result"] = "interrupted"
    report["summary"] = {"planned_reference_calls": report.get("definitions", {}).get("reference_calls", 0),
                         "planned_checks": report.get("definitions", {}).get("logical_checks", 0),
                         "completed_references": len(report["references"]),
                         "completed_checks": len(report["checks"]),
                         "failed_references": sum(r["status"] == "failed" for r in report["references"]),
                         "failed_checks": sum(r["status"] == "failed" for r in report["checks"])}
    _write_report(report, args.output)
    print(json.dumps({key: report[key] for key in ("mode", "result", "npu_executed", "summary")},
                     ensure_ascii=False), flush=True)
    return 130 if report["result"] == "interrupted" else int(report["result"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
