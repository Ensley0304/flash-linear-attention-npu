#!/usr/bin/env python3
"""Replay the fused KDA backward from a directory of model ``*.pt`` dumps."""

from __future__ import annotations

import argparse
import gc
import math
import os
import pathlib
import sys
import time

import torch
import torch_npu  # noqa: F401

import fla_npu
from fla_npu.ops.ascendc import npu_chunk_kda_bwd


INPUT_FILES = {
    "q": "q_head.pt",
    "k": "k_head.pt",
    "v": "v_head.pt",
    "beta": "beta_head.pt",
    "gk": "gk_head.pt",
    "aqk": "Aqk_head.pt",
    "akk": "Akk_head.pt",
    "w": "w_head.pt",
    "qg": "qg_head.pt",
    "kg": "kg_head.pt",
    "v_new": "v_new_head.pt",
    "h": "h_head.pt",
    "d_o": "do_head.pt",
    "raw_g": "raw_g_head.pt",
    "a_log": "gate_A_log.pt",
    "dt_bias": "gate_dt_bias.pt",
}
OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias")


def load_pt(path: pathlib.Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scalar(value, cast):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape={tuple(value.shape)}")
        value = value.item()
    elif isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return cast(value)


def load_scalar(root: pathlib.Path, filename: str, cast):
    return scalar(load_pt(root / filename), cast)


def load_cu_seqlens(root: pathlib.Path):
    value = load_pt(root / "cu_seqlens_host.pt")
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    return tuple(int(item) for item in value)


def finite_stats(kind: str, name: str, value: torch.Tensor) -> bool:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor: {type(value)!r}")
    if value.is_floating_point() or value.is_complex():
        finite = torch.isfinite(value)
        nan_count = int(torch.isnan(value).sum().cpu())
        inf_count = int(torch.isinf(value).sum().cpu())
        bad_count = int((~finite).sum().cpu())
    else:
        finite = torch.ones_like(value, dtype=torch.bool)
        nan_count = 0
        inf_count = 0
        bad_count = 0
    print(
        f"{kind} {name:<8} shape={tuple(value.shape)} dtype={value.dtype} "
        f"stride={tuple(value.stride())} bad={bad_count}/{value.numel()} "
        f"nan={nan_count} inf={inf_count}",
        flush=True,
    )
    if bad_count:
        flat_bad = (~finite).reshape(-1).nonzero(as_tuple=False)[:8].reshape(-1)
        print(f"  first_bad_flat={flat_bad.cpu().tolist()}", flush=True)
    return bad_count == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0,
                        help="process-local NPU id; usually 0 after visible-device mapping")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    root = args.dump_dir.resolve()
    print(f"DUMP_DIR={root}", flush=True)
    print(f"FLA_NPU={fla_npu.__file__}", flush=True)
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}", flush=True)
    print(f"FLA_NPU_OP_API_LIB={os.environ.get('FLA_NPU_OP_API_LIB', '')}", flush=True)

    inputs = {name: load_pt(root / filename) for name, filename in INPUT_FILES.items()}
    raw_do_path = root / "do.pt"
    if raw_do_path.is_file():
        finite_stats("DIAG", "do", load_pt(raw_do_path))

    chunk_size = load_scalar(root, "chunk_size.pt", int)
    scale = load_scalar(root, "scale.pt", float)
    safe_gate = load_scalar(root, "safe_gate.pt", bool)
    lower_bound = load_scalar(root, "lower_bound.pt", float)
    use_gate = load_scalar(root, "use_gate_in_kernel.pt", bool)
    cu_seqlens = load_cu_seqlens(root)
    print(
        f"PARAM chunk_size={chunk_size} scale={scale:.12g} safe_gate={safe_gate} "
        f"lower_bound={lower_bound:.12g} use_gate_in_kernel={use_gate} "
        f"cu_seqlens={cu_seqlens}",
        flush=True,
    )
    if chunk_size != 64:
        raise ValueError(f"fused backward requires chunk_size=64, got {chunk_size}")
    if not math.isfinite(scale) or not math.isfinite(lower_bound):
        raise ValueError("scale/lower_bound must be finite")

    required = tuple(INPUT_FILES)
    if not use_gate:
        required = tuple(name for name in required if name not in {"raw_g", "a_log", "dt_bias"})
    input_finite = True
    for name in required:
        input_finite &= finite_stats("INPUT", name, inputs[name])

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    npu_inputs = {
        name: value.contiguous().to(device)
        for name, value in inputs.items()
        if name in required
    }

    output_finite = True
    for iteration in range(args.repeats):
        start = time.perf_counter()
        outputs = npu_chunk_kda_bwd(
            npu_inputs["q"], npu_inputs["k"], npu_inputs["v"],
            npu_inputs["beta"], npu_inputs["gk"], npu_inputs["aqk"],
            npu_inputs["akk"], npu_inputs["w"], npu_inputs["qg"],
            npu_inputs["kg"], npu_inputs["v_new"], npu_inputs["h"],
            npu_inputs["d_o"], scale,
            raw_g=npu_inputs.get("raw_g"), A_log=npu_inputs.get("a_log"),
            dt_bias=npu_inputs.get("dt_bias"), initial_state=None, dht=None,
            cu_seqlens=cu_seqlens, chunk_indices=None, chunk_size=chunk_size,
            safe_gate=safe_gate, lower_bound=lower_bound,
            use_gate_in_kernel=use_gate, disable_recompute=True,
            use_exp2=True, state_v_first=False,
        )
        torch.npu.synchronize()
        print(
            f"RUN {iteration + 1}/{args.repeats} elapsed_ms="
            f"{(time.perf_counter() - start) * 1e3:.3f}",
            flush=True,
        )
        for name, output in zip(OUTPUT_NAMES, outputs):
            if output is None:
                print(f"OUTPUT {name:<8} None", flush=True)
                continue
            output_finite &= finite_stats("OUTPUT", name, output)
        del outputs
        gc.collect()
        torch.npu.empty_cache()

    if not input_finite:
        result = "FAIL_INPUT_NONFINITE"
        status = 2
    elif not output_finite:
        result = "FAIL_OUTPUT_NONFINITE"
        status = 3
    else:
        result = "PASS_NO_NAN_INF"
        status = 0
    print(f"FINAL_RESULT={result}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
