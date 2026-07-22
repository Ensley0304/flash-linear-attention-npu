#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.

"""Validate and summarize msprof rows for ChunkKdaBwdIntra.

The script deliberately separates claims that are easy to conflate: the exact
target launch and build identity were used, the profile contains mixed AIC/AIV
execution evidence, the requested IEEE/HF32 Cube mode is visible in msprof,
and the 4 ms performance target was met.  Mixed AIC/AIV evidence alone does
not identify a particular tiling key or Cube precision mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_MS = 4.0
PHASE1_MS = 48.660472

_TARGET_BNSD_SHAPES = (
    (1, 32, 8192, 128),
    (1, 32, 8192, 128),
    (1, 32, 8192, 128),
    (1, 32, 8192),
    (1, 32, 8192, 64),
    (1, 32, 8192, 64),
    (1, 32, 8192, 128),
    (1, 32, 8192, 128),
    (1, 32, 8192),
    (1, 32, 8192, 128),
)
_TARGET_BSND_SHAPES = (
    (1, 8192, 32, 128),
    (1, 8192, 32, 128),
    (1, 8192, 32, 128),
    (1, 8192, 32),
    (1, 8192, 32, 64),
    (1, 8192, 32, 64),
    (1, 8192, 32, 128),
    (1, 8192, 32, 128),
    (1, 8192, 32),
    (1, 8192, 32, 128),
)
_TARGET_INPUT_DTYPES = (
    "BF16",
    "BF16",
    "FP32",
    "FP32",
    "FP32",
    "FP32",
    "FP32",
    "FP32",
    "FP32",
    "FP32",
)
_DTYPE_ALIASES = {
    "BF16": "BF16",
    "DT_BF16": "BF16",
    "FLOAT": "FP32",
    "FLOAT32": "FP32",
    "FP32": "FP32",
    "DT_FLOAT": "FP32",
}


def _field(row: Dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, "", "N/A"):
            return str(value).strip().strip('"')
    return ""


def _number(row: Dict[str, str], *names: str) -> float:
    value = _field(row, *names)
    if not value:
        return 0.0
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric value {value!r} for {'/'.join(names)}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value {value!r} for {'/'.join(names)}")
    return result


def _integer(row: Dict[str, str], *names: str) -> int:
    value = _number(row, *names)
    if not value.is_integer():
        raise ValueError(f"non-integral numeric value {value!r} for {'/'.join(names)}")
    return int(value)


def _optional_integer(row: Dict[str, str], *names: str) -> Optional[int]:
    if not _field(row, *names):
        return None
    return _integer(row, *names)


def _ratio(row: Dict[str, str], *names: str) -> float:
    value = _number(row, *names)
    return value / 100.0 if value > 1.0 else value


def _is_kda_row(row: Dict[str, str]) -> bool:
    identity = " ".join(
        _field(row, name)
        for name in ("Op Name", "Name", "OP Type", "Type")
    ).lower()
    compact = "".join(ch for ch in identity if ch.isalnum())
    return "chunkkdabwdintra" in compact


def _split_semicolon(raw: str) -> Tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip().strip('"').strip() for part in raw.split(";"))


def _parse_input_shapes(raw: str) -> Tuple[Tuple[int, ...], ...]:
    parsed = []
    for shape in _split_semicolon(raw):
        if not shape:
            return ()
        dims = []
        for dimension in shape.split(","):
            token = dimension.strip()
            try:
                value = int(token)
            except ValueError:
                return ()
            if value <= 0:
                return ()
            dims.append(value)
        parsed.append(tuple(dims))
    return tuple(parsed)


def _parse_input_dtypes(raw: str) -> Tuple[str, ...]:
    parsed = []
    for dtype in _split_semicolon(raw):
        normalized = _DTYPE_ALIASES.get(dtype.upper())
        if normalized is None:
            return ()
        parsed.append(normalized)
    return tuple(parsed)


def _target_input_layout(shapes: Tuple[Tuple[int, ...], ...]) -> str:
    if shapes == _TARGET_BNSD_SHAPES:
        return "BNSD"
    if shapes == _TARGET_BSND_SHAPES:
        return "BSND"
    return ""


def _load_launch_manifest(path: pathlib.Path) -> Dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"launch manifest contains non-finite JSON number {token!r}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream, parse_constant=reject_nonfinite)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read launch manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("launch manifest must be a JSON object")

    expected_ints = {
        "batch": 1,
        "seqlen": 8192,
        "heads": 32,
        "head_dim": 128,
        "chunk_size": 64,
        "process_device": 0,
    }
    for name, expected in expected_ints.items():
        value = manifest.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"launch manifest requires {name}={expected}, got {value!r}")
    if manifest.get("safe_gate") is not True:
        raise ValueError(
            f"launch manifest requires safe_gate=true, got {manifest.get('safe_gate')!r}"
        )
    for name, expected in (
        ("operator", "chunk_kda_bwd_intra"),
        ("layout", "BNSD"),
        ("q_dtype", "BF16"),
        ("k_dtype", "BF16"),
        ("gate_dtype", "FP32"),
        ("accumulator_dtype", "FP32"),
    ):
        value = manifest.get(name)
        if value != expected:
            raise ValueError(f"launch manifest requires {name}={expected}, got {value!r}")
    gate_scale = manifest.get("gate_scale")
    if (
        isinstance(gate_scale, bool)
        or not isinstance(gate_scale, (int, float))
        or not math.isfinite(gate_scale)
        or gate_scale != 0.2
    ):
        raise ValueError(
            f"launch manifest requires finite gate_scale=0.2, got {gate_scale!r}"
        )
    for name in ("warmup", "repeat"):
        value = manifest.get(name)
        if type(value) is not int or value < (0 if name == "warmup" else 1):
            raise ValueError(
                f"launch manifest requires a valid integer {name}, got {value!r}"
            )
    visible_devices = manifest.get("visible_devices")
    if not isinstance(visible_devices, str) or not visible_devices:
        raise ValueError(
            "launch manifest requires a non-empty ASCEND_RT_VISIBLE_DEVICES value"
        )
    # Stage I/O was added after Cube mode.  Strip its suffix first, then let
    # the existing compatibility chain validate every earlier dimension.
    stage_io_present = "stage_io" in manifest
    stage_io = manifest.get("stage_io")
    full_variant = manifest.get("build_variant")
    if stage_io_present:
        if "cube_mode" not in manifest:
            raise ValueError(
                "launch manifest has a partial build identity: stage_io is "
                "present while cube_mode is missing"
            )
        if stage_io == "unlabeled":
            pass
        elif stage_io in ("tscm", "gm"):
            if manifest.get("cube_mode") == "unlabeled":
                raise ValueError(
                    "launch manifest has a partial build identity: stage_io is "
                    "labeled while cube_mode is unlabeled"
                )
            stage_io_suffix = f"_io-{stage_io}"
            if (
                not isinstance(full_variant, str)
                or not full_variant.endswith(stage_io_suffix)
            ):
                raise ValueError(
                    "launch manifest has inconsistent stage-I/O identity: "
                    f"build_variant={full_variant!r}, stage_io={stage_io!r}"
                )
            manifest["build_variant"] = full_variant[: -len(stage_io_suffix)]
        else:
            raise ValueError(
                f"launch manifest has invalid stage_io={stage_io!r}"
            )

    # Cube mode was added after the original build-identity chain.  Validate
    # its suffix here, then let the compatibility logic below validate the
    # unchanged legacy prefix and restore the full identity before returning.
    cube_mode_present = "cube_mode" in manifest
    cube_mode = manifest.get("cube_mode")
    variant_without_stage_io = manifest.get("build_variant")
    if cube_mode_present:
        if "task_store" not in manifest:
            raise ValueError(
                "launch manifest has a partial build identity: cube_mode is "
                "present while task_store is missing"
            )
        if cube_mode == "unlabeled":
            pass
        elif cube_mode in ("ieee", "hf32"):
            if manifest.get("task_store") == "unlabeled":
                raise ValueError(
                    "launch manifest has a partial build identity: cube_mode is "
                    "labeled while task_store is unlabeled"
                )
            cube_suffix = f"_cube-{cube_mode}"
            if (
                not isinstance(variant_without_stage_io, str)
                or not variant_without_stage_io.endswith(cube_suffix)
            ):
                raise ValueError(
                    "launch manifest has inconsistent Cube identity: "
                    f"build_variant={full_variant!r}, cube_mode={cube_mode!r}"
                )
            manifest["build_variant"] = variant_without_stage_io[
                : -len(cube_suffix)
            ]
        else:
            raise ValueError(
                f"launch manifest has invalid cube_mode={cube_mode!r}"
            )
    # Stage-A packing was added after Cube mode.  Strip its suffix from the
    # already cube-stripped identity so the compatibility chain below can
    # continue validating every older prefix unchanged.
    stage_a_present = "stage_a" in manifest
    stage_a = manifest.get("stage_a")
    if stage_a_present:
        if "task_store" not in manifest:
            raise ValueError(
                "launch manifest has a partial build identity: stage_a is "
                "present while task_store is missing"
            )
        if stage_a == "unlabeled":
            pass
        elif stage_a in ("packed", "split"):
            if manifest.get("task_store") == "unlabeled":
                raise ValueError(
                    "launch manifest has a partial build identity: stage_a "
                    "is labeled while task_store is unlabeled"
                )
            stage_a_suffix = f"_stagea-{stage_a}"
            variant_without_cube = manifest.get("build_variant")
            if (
                not isinstance(variant_without_cube, str)
                or not variant_without_cube.endswith(stage_a_suffix)
            ):
                raise ValueError(
                    "launch manifest has inconsistent stage-A identity: "
                    f"build_variant={full_variant!r}, stage_a={stage_a!r}"
                )
            manifest["build_variant"] = variant_without_cube[
                : -len(stage_a_suffix)
            ]
        else:
            raise ValueError(
                f"launch manifest has invalid stage_a={stage_a!r}"
            )
    legacy_identity_fields = ("build_variant", "pair_gates", "shared_setup")
    legacy_identity_present = tuple(name in manifest for name in legacy_identity_fields)
    stage_epilogue_present = "stage_epilogue" in manifest
    pair_scratch_present = "pair_scratch" in manifest
    tail_blocks_present = "tail_blocks" in manifest
    mmad_engines_present = "mmad_engines" in manifest
    vector_mask_present = "vector_mask" in manifest
    db_reduce_present = "db_reduce" in manifest
    task_store_present = "task_store" in manifest
    variant = manifest.get("build_variant")
    pair_gates = manifest.get("pair_gates")
    shared_setup = manifest.get("shared_setup")
    stage_epilogue = manifest.get("stage_epilogue")
    pair_scratch = manifest.get("pair_scratch")
    tail_blocks = manifest.get("tail_blocks")
    mmad_engines = manifest.get("mmad_engines")
    vector_mask = manifest.get("vector_mask")
    db_reduce = manifest.get("db_reduce")
    task_store = manifest.get("task_store")
    legacy_identity = (variant, pair_gates, shared_setup)
    if not any(legacy_identity_present):
        newer_fields = tuple(
            name
            for name, present in (
                ("stage_epilogue", stage_epilogue_present),
                ("pair_scratch", pair_scratch_present),
                ("tail_blocks", tail_blocks_present),
                ("mmad_engines", mmad_engines_present),
                ("vector_mask", vector_mask_present),
                ("db_reduce", db_reduce_present),
                ("task_store", task_store_present),
            )
            if present
        )
        if newer_fields:
            raise ValueError(
                "launch manifest has a partial build identity: newer fields "
                f"{newer_fields!r} are present while the legacy identity fields "
                "are missing"
            )
    elif not all(legacy_identity_present):
        raise ValueError(
            "launch manifest has a partial legacy build identity: "
            f"present={dict(zip(legacy_identity_fields, legacy_identity_present))}"
        )
    elif legacy_identity == ("unlabeled", "unlabeled", "unlabeled"):
        if pair_scratch_present and not stage_epilogue_present:
            raise ValueError(
                "launch manifest has a partial build identity: pair_scratch is "
                "present while stage_epilogue is missing"
            )
        if tail_blocks_present and not pair_scratch_present:
            raise ValueError(
                "launch manifest has a partial build identity: tail_blocks is "
                "present while pair_scratch is missing"
            )
        if mmad_engines_present and not tail_blocks_present:
            raise ValueError(
                "launch manifest has a partial build identity: MMAD engines is "
                "present while tail_blocks is missing"
            )
        if vector_mask_present and not mmad_engines_present:
            raise ValueError(
                "launch manifest has a partial build identity: Vector mask is "
                "present while MMAD engines is missing"
            )
        if db_reduce_present and not vector_mask_present:
            raise ValueError(
                "launch manifest has a partial build identity: db reduction is "
                "present while Vector mask is missing"
            )
        if task_store_present and not db_reduce_present:
            raise ValueError(
                "launch manifest has a partial build identity: task store is "
                "present while db reduction is missing"
            )
        labeled_newer_fields = {
            name: value
            for name, present, value in (
                ("stage_epilogue", stage_epilogue_present, stage_epilogue),
                ("pair_scratch", pair_scratch_present, pair_scratch),
                ("tail_blocks", tail_blocks_present, tail_blocks),
                ("mmad_engines", mmad_engines_present, mmad_engines),
                ("vector_mask", vector_mask_present, vector_mask),
                ("db_reduce", db_reduce_present, db_reduce),
                ("task_store", task_store_present, task_store),
            )
            if present and value != "unlabeled"
        }
        if labeled_newer_fields:
            raise ValueError(
                "launch manifest has a partial build identity: newer fields are "
                "labeled while the legacy identity fields are unlabeled: "
                f"{labeled_newer_fields!r}"
            )
    else:
        if any(value is None or value == "unlabeled" for value in legacy_identity):
            raise ValueError(
                "launch manifest has a partial legacy build identity: "
                f"build_variant={variant!r}, pair_gates={pair_gates!r}, "
                f"shared_setup={shared_setup!r}"
            )
        if pair_gates not in ("factor", "direct"):
            raise ValueError(f"launch manifest has invalid pair_gates={pair_gates!r}")
        if shared_setup not in ("overlap", "prologue"):
            raise ValueError(f"launch manifest has invalid shared_setup={shared_setup!r}")
        if not stage_epilogue_present:
            if (
                pair_scratch_present
                or tail_blocks_present
                or mmad_engines_present
                or vector_mask_present
                or db_reduce_present
                or task_store_present
            ):
                raise ValueError(
                    "launch manifest has a partial build identity: newer fields are "
                    "present while stage_epilogue is missing"
                )
            # Compatibility for a fully labeled two-dimensional launch manifest.
            expected_variant = f"pair-{pair_gates}_setup-{shared_setup}"
        elif stage_epilogue == "unlabeled":
            if tail_blocks_present and not pair_scratch_present:
                raise ValueError(
                    "launch manifest has a partial build identity: tail_blocks is "
                    "present while pair_scratch is missing"
                )
            if mmad_engines_present and not tail_blocks_present:
                raise ValueError(
                    "launch manifest has a partial build identity: MMAD engines is "
                    "present while tail_blocks is missing"
                )
            if vector_mask_present and not mmad_engines_present:
                raise ValueError(
                    "launch manifest has a partial build identity: Vector mask is "
                    "present while MMAD engines is missing"
                )
            if db_reduce_present and not vector_mask_present:
                raise ValueError(
                    "launch manifest has a partial build identity: db reduction is "
                    "present while Vector mask is missing"
                )
            if task_store_present and not db_reduce_present:
                raise ValueError(
                    "launch manifest has a partial build identity: task store is "
                    "present while db reduction is missing"
                )
            if (
                (pair_scratch_present and pair_scratch != "unlabeled")
                or (tail_blocks_present and tail_blocks != "unlabeled")
                or (mmad_engines_present and mmad_engines != "unlabeled")
                or (vector_mask_present and vector_mask != "unlabeled")
                or (db_reduce_present and db_reduce != "unlabeled")
                or (task_store_present and task_store != "unlabeled")
            ):
                raise ValueError(
                    "launch manifest has a partial build identity: a newer field is "
                    "labeled while stage_epilogue is unlabeled"
                )
            # Explicit unlabeled suffixes also represent the legacy 2D identity.
            expected_variant = f"pair-{pair_gates}_setup-{shared_setup}"
        elif stage_epilogue in ("overlap", "tail"):
            expected_variant = (
                f"pair-{pair_gates}_setup-{shared_setup}_epilogue-{stage_epilogue}"
            )
            if not pair_scratch_present:
                if (
                    tail_blocks_present
                    or mmad_engines_present
                    or vector_mask_present
                    or db_reduce_present
                    or task_store_present
                ):
                    raise ValueError(
                        "launch manifest has a partial build identity: newer fields "
                        "are present while pair_scratch is missing"
                    )
                # Compatibility for the previous fully labeled 3D identity.
                pass
            elif pair_scratch == "unlabeled":
                if mmad_engines_present and not tail_blocks_present:
                    raise ValueError(
                        "launch manifest has a partial build identity: MMAD engines is "
                        "present while tail_blocks is missing"
                    )
                if vector_mask_present and not mmad_engines_present:
                    raise ValueError(
                        "launch manifest has a partial build identity: Vector mask "
                        "is present while MMAD engines is missing"
                    )
                if db_reduce_present and not vector_mask_present:
                    raise ValueError(
                        "launch manifest has a partial build identity: db reduction "
                        "is present while Vector mask is missing"
                    )
                if task_store_present and not db_reduce_present:
                    raise ValueError(
                        "launch manifest has a partial build identity: task store "
                        "is present while db reduction is missing"
                    )
                if (
                    tail_blocks_present and tail_blocks != "unlabeled"
                ) or (
                    mmad_engines_present and mmad_engines != "unlabeled"
                ) or (
                    vector_mask_present and vector_mask != "unlabeled"
                ) or (
                    db_reduce_present and db_reduce != "unlabeled"
                ) or (
                    task_store_present and task_store != "unlabeled"
                ):
                    raise ValueError(
                        "launch manifest has a partial build identity: a newer field "
                        "is labeled while pair_scratch is unlabeled"
                    )
                # Explicit unlabeled suffixes also represent the legacy 3D identity.
            elif pair_scratch in ("pingpong", "single"):
                expected_variant += f"_scratch-{pair_scratch}"
                if not tail_blocks_present:
                    if (
                        mmad_engines_present
                        or vector_mask_present
                        or db_reduce_present
                        or task_store_present
                    ):
                        raise ValueError(
                            "launch manifest has a partial build identity: MMAD "
                            "engines or Vector mask is present while tail_blocks "
                            "is missing"
                        )
                    # Compatibility for the previous fully labeled 4D identity.
                    pass
                elif tail_blocks == "unlabeled":
                    if (
                        mmad_engines_present and mmad_engines != "unlabeled"
                    ) or (vector_mask_present and vector_mask != "unlabeled") or (
                        db_reduce_present and db_reduce != "unlabeled"
                    ) or (
                        task_store_present and task_store != "unlabeled"
                    ):
                        raise ValueError(
                            "launch manifest has a partial build identity: MMAD "
                            "engines or Vector mask is labeled while tail_blocks "
                            "is unlabeled"
                        )
                    # Explicit unlabeled suffixes also represent the legacy 4D identity.
                elif tail_blocks in ("batch", "scalar"):
                    expected_variant += f"_tail-{tail_blocks}"
                    if not mmad_engines_present or mmad_engines == "unlabeled":
                        # Compatibility for the previous fully labeled 5D identity.
                        if (
                            vector_mask_present and vector_mask != "unlabeled"
                        ) or (db_reduce_present and db_reduce != "unlabeled") or (
                            task_store_present and task_store != "unlabeled"
                        ):
                            raise ValueError(
                                "launch manifest has a partial build identity: "
                                "Vector mask is labeled while MMAD engines is "
                                "missing or unlabeled"
                            )
                    elif mmad_engines in ("persistent", "scoped"):
                        expected_variant += f"_mmad-{mmad_engines}"
                        if not vector_mask_present or vector_mask == "unlabeled":
                            # Compatibility for the previous fully labeled 6D identity.
                            if (
                                db_reduce_present and db_reduce != "unlabeled"
                            ) or (task_store_present and task_store != "unlabeled"):
                                raise ValueError(
                                    "launch manifest has a partial build identity: "
                                    "db reduction is labeled while Vector mask is "
                                    "missing or unlabeled"
                                )
                        elif vector_mask in ("reuse", "per-call"):
                            expected_variant += f"_vmask-{vector_mask}"
                            if not db_reduce_present or db_reduce == "unlabeled":
                                # Compatibility for the previous fully labeled 7D identity.
                                if task_store_present and task_store != "unlabeled":
                                    raise ValueError(
                                        "launch manifest has a partial build identity: "
                                        "task store is labeled while db reduction is "
                                        "missing or unlabeled"
                                    )
                            elif db_reduce in ("coalesced", "per-row"):
                                expected_variant += f"_dbr-{db_reduce}"
                                if not task_store_present or task_store == "unlabeled":
                                    # Compatibility for the previous fully labeled 8D identity.
                                    pass
                                elif task_store in ("overlap", "serial"):
                                    expected_variant += f"_store-{task_store}"
                                else:
                                    raise ValueError(
                                        "launch manifest has invalid "
                                        f"task_store={task_store!r}"
                                    )
                            else:
                                raise ValueError(
                                    "launch manifest has invalid "
                                    f"db_reduce={db_reduce!r}"
                                )
                        else:
                            raise ValueError(
                                "launch manifest has invalid "
                                f"vector_mask={vector_mask!r}"
                            )
                    else:
                        raise ValueError(
                            "launch manifest has invalid "
                            f"mmad_engines={mmad_engines!r}"
                        )
                else:
                    raise ValueError(
                        f"launch manifest has invalid tail_blocks={tail_blocks!r}"
                    )
            else:
                raise ValueError(
                    f"launch manifest has invalid pair_scratch={pair_scratch!r}"
                )
        else:
            raise ValueError(
                "launch manifest has invalid "
                f"stage_epilogue={stage_epilogue!r}"
            )
        if variant != expected_variant:
            raise ValueError(
                "launch manifest has inconsistent build identity: "
                f"build_variant={variant!r}, expected={expected_variant!r}"
            )
        if task_store == "overlap" and (
            tail_blocks != "batch" or stage_epilogue != "tail"
        ):
            raise ValueError(
                "launch manifest task_store='overlap' requires "
                "tail_blocks='batch' and stage_epilogue='tail'"
            )
    if stage_io_present or cube_mode_present or stage_a_present:
        manifest["build_variant"] = full_variant
    return manifest


def _classify_bound(pipelines: Dict[str, float]) -> str:
    largest = max(pipelines.values(), default=0.0)
    mte2 = pipelines["mte2"]
    cube = pipelines["cube"]
    if mte2 > 0.80 or (math.isclose(mte2, largest) and mte2 > 0.70):
        return "MTE2 BOUND"
    if cube > 0.80 or (math.isclose(cube, largest) and cube > 0.70):
        return "CUBE BOUND"
    if pipelines["vec"] > 0.80:
        return "VEC BOUND"
    if pipelines["fixpipe"] > 0.80:
        return "FIXP BOUND"
    if pipelines["mte3"] > 0.80:
        return "MTE3 BOUND"
    if pipelines["scalar"] > 0.80:
        return "SCALAR BOUND"
    return "NO SINGLE PIPE BOUND"


@dataclass
class ProfileRow:
    source: str
    device_id: Optional[int]
    op_name: str
    op_type: str
    task_type: str
    start_us: float
    task_duration_us: float
    effective_duration_us: float
    wait_us: float
    aic_us: float
    aiv_us: float
    block_num: int
    mix_block_num: int
    hf32: str
    input_shapes: str
    input_data_types: str
    input_layout: str
    target_shape: bool
    target_dtypes: bool
    target_tensor_signature: bool
    mixed_aic_aiv: bool
    bound: str
    top_pipeline: str
    top_pipeline_ratio: float
    aic_cube_us: float
    aic_cube_ratio: float
    aic_scalar_us: float
    aic_scalar_ratio: float
    aic_mte1_us: float
    aic_mte1_ratio: float
    aic_mte2_us: float
    aic_mte2_ratio: float
    aic_mte3_us: float
    aic_mte3_ratio: float
    aic_fixpipe_us: float
    aic_fixpipe_ratio: float
    aiv_vec_us: float
    aiv_vec_ratio: float
    aiv_scalar_us: float
    aiv_scalar_ratio: float
    aiv_mte2_us: float
    aiv_mte2_ratio: float
    aiv_mte3_us: float
    aiv_mte3_ratio: float


def _parse_profile_row(path: pathlib.Path, row: Dict[str, str]) -> ProfileRow:
    pipelines = {
        "mte2": max(
            _ratio(row, "aic_mte2_ratio"),
            _ratio(row, "aiv_mte2_ratio"),
        ),
        "cube": max(
            _ratio(row, "aic_cube_ratio"),
            _ratio(row, "aic_mac_ratio"),
        ),
        "vec": _ratio(row, "aiv_vec_ratio"),
        "fixpipe": _ratio(row, "aic_fixpipe_ratio"),
        "mte3": max(
            _ratio(row, "aic_mte3_ratio"),
            _ratio(row, "aiv_mte3_ratio"),
        ),
        "scalar": max(
            _ratio(row, "aic_scalar_ratio"),
            _ratio(row, "aiv_scalar_ratio"),
        ),
    }
    top_pipeline = max(pipelines, key=pipelines.get)
    aic_us = _number(row, "aicore_time(us)", "aic_time(us)")
    aiv_us = _number(row, "aiv_time(us)")
    task_duration_us = _number(row, "Task Duration(us)", "Duration(us)")
    mix_block_num = _integer(row, "Mix Block Num", "Mix Block Dim")
    hf32 = _field(row, "HF32 Eligible").upper()
    shapes = _field(row, "Input Shapes")
    dtypes = _field(row, "Input Data Types")
    parsed_shapes = _parse_input_shapes(shapes)
    parsed_dtypes = _parse_input_dtypes(dtypes)
    input_layout = _target_input_layout(parsed_shapes)
    target_shape = bool(input_layout)
    target_dtypes = parsed_dtypes == _TARGET_INPUT_DTYPES
    return ProfileRow(
        source=str(path),
        device_id=_optional_integer(row, "Device_id", "Device ID"),
        op_name=_field(row, "Op Name", "Name"),
        op_type=_field(row, "OP Type", "Type"),
        task_type=_field(row, "Task Type", "Accelerator Core"),
        start_us=_number(row, "Task Start Time(us)", "Start Time(us)"),
        task_duration_us=task_duration_us,
        effective_duration_us=max(task_duration_us, aic_us, aiv_us),
        wait_us=_number(row, "Task Wait Time(us)", "Wait Time(us)"),
        aic_us=aic_us,
        aiv_us=aiv_us,
        block_num=_integer(row, "Block Num", "Block Dim"),
        mix_block_num=mix_block_num,
        hf32=hf32,
        input_shapes=shapes,
        input_data_types=dtypes,
        input_layout=input_layout,
        target_shape=target_shape,
        target_dtypes=target_dtypes,
        target_tensor_signature=target_shape and target_dtypes,
        mixed_aic_aiv=(aic_us > 0.0 and aiv_us > 0.0 and mix_block_num > 0),
        bound=_classify_bound(pipelines),
        top_pipeline=top_pipeline,
        top_pipeline_ratio=pipelines[top_pipeline],
        aic_cube_us=_number(row, "aic_cube_time(us)", "aic_mac_time(us)"),
        aic_cube_ratio=max(
            _ratio(row, "aic_cube_ratio"), _ratio(row, "aic_mac_ratio")
        ),
        aic_scalar_us=_number(row, "aic_scalar_time(us)"),
        aic_scalar_ratio=_ratio(row, "aic_scalar_ratio"),
        aic_mte1_us=_number(row, "aic_mte1_time(us)"),
        aic_mte1_ratio=_ratio(row, "aic_mte1_ratio"),
        aic_mte2_us=_number(row, "aic_mte2_time(us)"),
        aic_mte2_ratio=_ratio(row, "aic_mte2_ratio"),
        aic_mte3_us=_number(row, "aic_mte3_time(us)"),
        aic_mte3_ratio=_ratio(row, "aic_mte3_ratio"),
        aic_fixpipe_us=_number(row, "aic_fixpipe_time(us)"),
        aic_fixpipe_ratio=_ratio(row, "aic_fixpipe_ratio"),
        aiv_vec_us=_number(row, "aiv_vec_time(us)"),
        aiv_vec_ratio=_ratio(row, "aiv_vec_ratio"),
        aiv_scalar_us=_number(row, "aiv_scalar_time(us)"),
        aiv_scalar_ratio=_ratio(row, "aiv_scalar_ratio"),
        aiv_mte2_us=_number(row, "aiv_mte2_time(us)"),
        aiv_mte2_ratio=_ratio(row, "aiv_mte2_ratio"),
        aiv_mte3_us=_number(row, "aiv_mte3_time(us)"),
        aiv_mte3_ratio=_ratio(row, "aiv_mte3_ratio"),
    )


def _profile_files(path: pathlib.Path) -> Iterable[pathlib.Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        yield from sorted(path.rglob("op_summary_*.csv"))


def _load_rows(path: pathlib.Path) -> List[ProfileRow]:
    results: List[ProfileRow] = []
    for csv_path in _profile_files(path):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for raw in csv.DictReader(stream):
                row = {
                    str(key).lstrip("\ufeff"): value
                    for key, value in raw.items()
                    if key is not None
                }
                if _is_kda_row(row):
                    results.append(_parse_profile_row(csv_path, row))
    results.sort(key=lambda row: (row.start_us, row.source))
    return results


def _hf32_mode_evidence(
    rows: Sequence[ProfileRow], launch_manifest: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Tuple[str, ...], bool]:
    expected = (
        {"ieee": "NO", "hf32": "YES"}.get(launch_manifest.get("cube_mode"))
        if launch_manifest is not None
        else None
    )
    observed = tuple(sorted({row.hf32 or "<missing>" for row in rows}))
    matches = expected is None or all(row.hf32 == expected for row in rows)
    return expected, observed, matches


def _print_human(
    rows: Sequence[ProfileRow],
    target_ms: float,
    baseline_ms: float,
    total_rows: int,
    discarded_rows: int,
    launch_manifest: Optional[pathlib.Path],
    launch_manifest_data: Optional[Dict[str, Any]],
) -> None:
    durations_ms = [row.effective_duration_us / 1000.0 for row in rows]
    median_ms = statistics.median(durations_ms)
    print(f"total_kda_rows={total_rows}")
    print(f"discarded_rows={discarded_rows}")
    print(f"sample_count={len(rows)}")
    if launch_manifest is None:
        print("launch_manifest=NOT_PROVIDED")
    else:
        print(f"launch_manifest=PASS ({launch_manifest})")
        print(
            "launch_signature=B1/T8192/H32/K128/BT64/BNSD/BF16/safe_gate=true"
        )
        print(
            "launch_variant="
            f"{launch_manifest_data.get('build_variant', '<missing>')}, "
            f"pair_gates={launch_manifest_data.get('pair_gates', '<missing>')}, "
            f"shared_setup={launch_manifest_data.get('shared_setup', '<missing>')}, "
            f"stage_epilogue={launch_manifest_data.get('stage_epilogue', '<missing>')}, "
            f"pair_scratch={launch_manifest_data.get('pair_scratch', '<missing>')}, "
            f"tail_blocks={launch_manifest_data.get('tail_blocks', '<missing>')}, "
            f"task_store={launch_manifest_data.get('task_store', '<missing>')}, "
            f"mmad_engines={launch_manifest_data.get('mmad_engines', '<missing>')}, "
            f"vector_mask={launch_manifest_data.get('vector_mask', '<missing>')}, "
            f"db_reduce={launch_manifest_data.get('db_reduce', '<missing>')}, "
            f"stage_a={launch_manifest_data.get('stage_a', '<missing>')}, "
            f"cube_mode={launch_manifest_data.get('cube_mode', '<missing>')}, "
            f"stage_io={launch_manifest_data.get('stage_io', '<missing>')}"
        )
        expected_hf32, observed_hf32, marker_matches = _hf32_mode_evidence(
            rows, launch_manifest_data
        )
        if expected_hf32 is not None:
            marker_status = "PASS" if marker_matches else "FAIL"
            print(
                f"hf32_mode_evidence={marker_status}, expected={expected_hf32}, "
                f"observed={','.join(observed_hf32)}"
            )
    for index, row in enumerate(rows, 1):
        print(
            f"[{index}] task_duration_ms={row.task_duration_us / 1000.0:.6f}, "
            f"effective_duration_ms={row.effective_duration_us / 1000.0:.6f}, "
            f"wait_ms={row.wait_us / 1000.0:.6f}, "
            f"aic_ms={row.aic_us / 1000.0:.6f}, aiv_ms={row.aiv_us / 1000.0:.6f}, "
            f"device_id={row.device_id if row.device_id is not None else '<missing>'}, "
            f"blocks={row.block_num}, "
            f"mix_blocks={row.mix_block_num}, hf32={row.hf32 or '<missing>'}"
        )
        print(
            f"    target_shape={row.target_shape}, target_dtypes={row.target_dtypes}, "
            f"target_tensor_signature={row.target_tensor_signature}, layout={row.input_layout or '<none>'}"
        )
        print(
            f"    mixed_aic_aiv_evidence={row.mixed_aic_aiv}, "
            f"bound={row.bound}, top={row.top_pipeline}:{row.top_pipeline_ratio:.1%}"
        )
        print(f"    op={row.op_name} ({row.op_type}/{row.task_type})")
        print(f"    source={row.source}")
    print(f"min_kernel_ms={min(durations_ms):.6f}")
    print(f"median_kernel_ms={median_ms:.6f}")
    print(f"max_kernel_ms={max(durations_ms):.6f}")
    if baseline_ms > 0.0:
        print(f"speedup_vs_phase1={baseline_ms / median_ms:.3f}x")
    print(f"target_ms={target_ms:.6f}")
    print("performance_target=PASS" if median_ms <= target_ms else "performance_target=NOT_MET")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=pathlib.Path, help="op_summary CSV or msprof output directory")
    parser.add_argument("--target-ms", type=float, default=TARGET_MS)
    parser.add_argument("--baseline-ms", type=float, default=PHASE1_MS)
    parser.add_argument("--discard-first", type=int, default=0, metavar="N")
    parser.add_argument("--min-rows", type=int, default=1, metavar="N")
    parser.add_argument("--expected-rows", type=int, metavar="N")
    parser.add_argument("--expected-device-id", type=int, metavar="ID")
    parser.add_argument(
        "--launch-manifest",
        type=pathlib.Path,
        help="JSON evidence for the exact BNSD/BF16/safe-gate launch attributes",
    )
    parser.add_argument("--expected-build-variant")
    parser.add_argument("--expected-pair-gates", choices=("factor", "direct"))
    parser.add_argument("--expected-shared-setup", choices=("overlap", "prologue"))
    parser.add_argument("--expected-stage-epilogue", choices=("overlap", "tail"))
    parser.add_argument("--expected-pair-scratch", choices=("pingpong", "single"))
    parser.add_argument("--expected-tail-blocks", choices=("batch", "scalar"))
    parser.add_argument("--expected-task-store", choices=("overlap", "serial"))
    parser.add_argument("--expected-mmad-engines", choices=("persistent", "scoped"))
    parser.add_argument("--expected-vector-mask", choices=("reuse", "per-call"))
    parser.add_argument("--expected-db-reduce", choices=("coalesced", "per-row"))
    parser.add_argument("--expected-stage-a", choices=("packed", "split"))
    parser.add_argument("--expected-cube-mode", choices=("ieee", "hf32"))
    parser.add_argument("--expected-stage-io", choices=("tscm", "gm"))
    parser.add_argument("--require-target-shape", action="store_true")
    parser.add_argument(
        "--require-mixed",
        action="store_true",
        help="require mixed AIC/AIV execution evidence (does not identify a tiling key)",
    )
    parser.add_argument("--require-under-target", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.target_ms) or not math.isfinite(args.baseline_ms):
        raise ValueError("target-ms and baseline-ms must be finite")
    if args.target_ms <= 0.0 or args.baseline_ms < 0.0:
        raise ValueError("target-ms must be positive and baseline-ms must be non-negative")
    if args.discard_first < 0 or args.min_rows <= 0:
        raise ValueError("discard-first must be non-negative and min-rows must be positive")
    if args.expected_rows is not None and args.expected_rows <= 0:
        raise ValueError("expected-rows must be positive")
    if args.expected_device_id is not None and args.expected_device_id < 0:
        raise ValueError("expected-device-id must be non-negative")
    launch_manifest = (
        _load_launch_manifest(args.launch_manifest) if args.launch_manifest is not None else None
    )
    variant_expectations = {
        "build_variant": args.expected_build_variant,
        "pair_gates": args.expected_pair_gates,
        "shared_setup": args.expected_shared_setup,
        "stage_epilogue": args.expected_stage_epilogue,
        "pair_scratch": args.expected_pair_scratch,
        "tail_blocks": args.expected_tail_blocks,
        "task_store": args.expected_task_store,
        "mmad_engines": args.expected_mmad_engines,
        "vector_mask": args.expected_vector_mask,
        "db_reduce": args.expected_db_reduce,
        "stage_a": args.expected_stage_a,
        "cube_mode": args.expected_cube_mode,
        "stage_io": args.expected_stage_io,
    }
    if launch_manifest is None and any(value is not None for value in variant_expectations.values()):
        raise ValueError("expected build variant fields require --launch-manifest")
    if launch_manifest is not None:
        for name, expected in variant_expectations.items():
            if expected is not None and launch_manifest.get(name) != expected:
                raise ValueError(
                    f"launch manifest requires {name}={expected!r}, "
                    f"got {launch_manifest.get(name)!r}"
                )
        if launch_manifest["warmup"] != args.discard_first:
            raise ValueError(
                "launch manifest warmup/discard mismatch: "
                f"warmup={launch_manifest['warmup']}, "
                f"discard_first={args.discard_first}"
            )
        if (
            args.expected_rows is not None
            and launch_manifest["repeat"] != args.expected_rows
        ):
            raise ValueError(
                "launch manifest repeat/expected-rows mismatch: "
                f"repeat={launch_manifest['repeat']}, "
                f"expected_rows={args.expected_rows}"
            )
        if args.expected_device_id is not None and launch_manifest[
            "visible_devices"
        ] != str(args.expected_device_id):
            raise ValueError(
                "launch manifest physical-device mismatch: "
                f"visible_devices={launch_manifest['visible_devices']!r}, "
                f"expected_device_id={args.expected_device_id}"
            )
    all_rows = _load_rows(args.profile)
    if not all_rows:
        print(f"[FAIL] no ChunkKdaBwdIntra row found under {args.profile}", file=sys.stderr)
        return 1
    rows = all_rows[args.discard_first :]
    discarded_rows = len(all_rows) - len(rows)
    if not rows:
        print(
            f"[FAIL] discard-first={args.discard_first} removed all {len(all_rows)} KDA rows",
            file=sys.stderr,
        )
        return 2
    if any(row.task_duration_us <= 0.0 for row in rows):
        print("[FAIL] one or more KDA rows have a missing/non-positive duration", file=sys.stderr)
        return 1
    if any(row.aic_us < 0.0 or row.aiv_us < 0.0 for row in rows):
        print("[FAIL] one or more KDA rows have a negative AIC/AIV duration", file=sys.stderr)
        return 1
    if launch_manifest is not None and any(row.input_layout != "BNSD" for row in rows):
        print(
            "[FAIL] launch manifest says BNSD but one or more profiled rows do not have "
            "the exact BNSD input signature",
            file=sys.stderr,
        )
        return 2

    durations_ms = [row.effective_duration_us / 1000.0 for row in rows]
    median_ms = statistics.median(durations_ms)
    expected_hf32, observed_hf32, hf32_mode_matches = _hf32_mode_evidence(
        rows, launch_manifest
    )
    if not math.isfinite(median_ms):
        print("[FAIL] median kernel duration is not finite", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "rows": [asdict(row) for row in rows],
                    "total_kda_rows": len(all_rows),
                    "discarded_rows": discarded_rows,
                    "sample_count": len(rows),
                    "launch_manifest": {
                        "provided": args.launch_manifest is not None,
                        "valid": launch_manifest is not None,
                        "path": str(args.launch_manifest) if args.launch_manifest is not None else None,
                        "validated_fields": {
                            name: launch_manifest.get(name)
                            for name in (
                                "batch",
                                "seqlen",
                                "heads",
                                "head_dim",
                                "chunk_size",
                                "safe_gate",
                                "layout",
                                "q_dtype",
                                "k_dtype",
                                "gate_dtype",
                                "accumulator_dtype",
                                "gate_scale",
                                "process_device",
                                "visible_devices",
                                "warmup",
                                "repeat",
                                "build_variant",
                                "pair_gates",
                                "shared_setup",
                                "stage_epilogue",
                                "pair_scratch",
                                "tail_blocks",
                                "task_store",
                                "mmad_engines",
                                "vector_mask",
                                "db_reduce",
                                "stage_a",
                                "cube_mode",
                                "stage_io",
                            )
                        }
                        if launch_manifest is not None
                        else None,
                    },
                    "median_kernel_ms": median_ms,
                    "min_kernel_ms": min(durations_ms),
                    "max_kernel_ms": max(durations_ms),
                    "duration_basis": "max(task_duration_us, aic_us, aiv_us)",
                    "hf32_mode_evidence": {
                        "expected": expected_hf32,
                        "observed": observed_hf32,
                        "matches": hf32_mode_matches,
                    },
                    "median_effective_duration_ms": median_ms,
                    "min_effective_duration_ms": min(durations_ms),
                    "max_effective_duration_ms": max(durations_ms),
                    "target_ms": args.target_ms,
                    "performance_target": median_ms <= args.target_ms,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
    else:
        _print_human(
            rows,
            args.target_ms,
            args.baseline_ms,
            len(all_rows),
            discarded_rows,
            args.launch_manifest,
            launch_manifest,
        )

    failures = []
    if len(rows) < args.min_rows:
        failures.append(f"sample_count={len(rows)} is below min-rows={args.min_rows}")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        failures.append(f"sample_count={len(rows)} does not equal expected-rows={args.expected_rows}")
    if args.expected_device_id is not None:
        unexpected_devices = sorted(
            {
                "<missing>" if row.device_id is None else str(row.device_id)
                for row in rows
                if row.device_id != args.expected_device_id
            }
        )
        if unexpected_devices:
            failures.append(
                f"one or more rows do not have expected-device-id={args.expected_device_id}; "
                f"observed unexpected device ids: {unexpected_devices}"
            )
    if args.require_target_shape and not all(row.target_tensor_signature for row in rows):
        failures.append(
            "one or more rows do not match the exact 10-input "
            "B1/T8192/H32/K128/BT64 BF16/FP32 tensor signature"
        )
    if args.require_mixed and not all(row.mixed_aic_aiv for row in rows):
        failures.append(
            "mixed AIC/AIV evidence requires positive AIC/AIV time and Mix Block Num; "
            "this gate does not identify a tiling key or Cube precision mode"
        )
    if not hf32_mode_matches:
        failures.append(
            f"cube_mode={launch_manifest.get('cube_mode')!r} requires "
            f"HF32 Eligible={expected_hf32}, observed={observed_hf32}"
        )
    if args.require_under_target and median_ms > args.target_ms:
        failures.append(f"median {median_ms:.6f} ms exceeds {args.target_ms:.6f} ms")
    for failure in failures:
        print(f"[FAIL] {failure}", file=sys.stderr)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
