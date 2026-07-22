#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.

"""Compare clean-wheel KDA single/ping-pong profile evidence.

Both inputs are ``PROF_GROUP_*`` directories emitted by
``run_chunk_kda_bwd_intra_grouped_validation.sh``.  The comparison is rejected
unless the two groups have valid hashes, complete 37-test evidence, the exact
target launch signature, and identical build/runtime identity except for the
pair-scratch switch and artifacts derived from that switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any


TARGET_MS = 4.0
DEFAULT_MIN_IMPROVEMENT_PERCENT = 1.0

IDENTITY_INVARIANTS = (
    "commit",
    "physical_device",
    "cann_env",
    "cann_env_sha256",
    "cann_version_digest",
    "catlass_commit",
    "catlass_tree",
    "runner_sha256",
    "pair_gates",
    "shared_setup",
    "stage_epilogue",
    "tail_blocks",
    "task_store",
    "mmad_engines",
    "vector_mask",
    "db_reduce",
    "stage_a",
    "cube_mode",
    "stage_io",
)

MANIFEST_INVARIANTS = (
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
    "pair_gates",
    "shared_setup",
    "stage_epilogue",
    "tail_blocks",
    "task_store",
    "mmad_engines",
    "vector_mask",
    "db_reduce",
    "stage_a",
    "cube_mode",
    "stage_io",
)

TIME_FIELDS_US = (
    "effective_duration_us",
    "task_duration_us",
    "wait_us",
    "aic_us",
    "aiv_us",
    "aic_cube_us",
    "aic_scalar_us",
    "aic_mte1_us",
    "aic_mte2_us",
    "aic_mte3_us",
    "aic_fixpipe_us",
    "aiv_vec_us",
    "aiv_scalar_us",
    "aiv_mte2_us",
    "aiv_mte3_us",
)

RATIO_FIELDS = (
    "aic_cube_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_mte3_ratio",
    "aic_fixpipe_ratio",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)


class EvidenceError(ValueError):
    """Raised when two profile groups are not comparable evidence."""


@dataclass(frozen=True)
class ProfileEvidence:
    root: pathlib.Path
    identity: dict[str, str]
    summary: dict[str, Any]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_identity(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        raise EvidenceError(f"missing profile identity: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            raise EvidenceError(f"invalid identity line {line_number}: {raw!r}")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise EvidenceError(f"duplicate/empty identity key on line {line_number}")
        result[key] = value
    return result


def _verify_hash_manifest(root: pathlib.Path, identity: dict[str, str]) -> None:
    manifest = root / "profile_evidence.sha256"
    if not manifest.is_file():
        raise EvidenceError(f"missing hash manifest: {manifest}")
    expected_manifest_digest = identity.get("profile_evidence_sha256")
    observed_manifest_digest = _sha256(manifest)
    if observed_manifest_digest != expected_manifest_digest:
        raise EvidenceError(
            "profile hash-manifest digest mismatch: "
            f"expected={expected_manifest_digest}, observed={observed_manifest_digest}"
        )
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise EvidenceError(f"invalid SHA256 line {line_number}: {raw!r}")
        expected, raw_path = fields
        artifact = pathlib.Path(raw_path.lstrip(" *"))
        if not artifact.is_absolute():
            artifact = root / artifact
        if not artifact.is_file():
            raise EvidenceError(f"profile artifact is missing: {artifact}")
        observed = _sha256(artifact)
        if observed != expected:
            raise EvidenceError(
                f"profile artifact digest mismatch for {artifact}: "
                f"expected={expected}, observed={observed}"
            )


def _normalized_variant(value: str) -> str:
    for scratch in ("single", "pingpong"):
        marker = f"_scratch-{scratch}_"
        if marker in value:
            return value.replace(marker, "_scratch-<PAIR>_", 1)
    raise EvidenceError(f"build variant has no pair-scratch identity: {value!r}")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite: {value!r}")
    return result


def _load_profile_group(
    path: pathlib.Path, expected_scratch: str, verify_hashes: bool
) -> ProfileEvidence:
    root = path.resolve()
    if not root.is_dir():
        raise EvidenceError(f"profile group is not a directory: {root}")
    identity = _parse_identity(root / "profile_evidence.pass")
    if identity.get("profile_complete") != "true":
        raise EvidenceError(f"profile evidence is incomplete: {root}")
    if identity.get("pair_scratch") != expected_scratch:
        raise EvidenceError(
            f"expected pair_scratch={expected_scratch}, got "
            f"{identity.get('pair_scratch')!r}: {root}"
        )
    if identity.get("performance_target") not in {"PASS", "NOT_MET"}:
        raise EvidenceError(f"invalid performance_target in {root}")
    if verify_hashes:
        _verify_hash_manifest(root, identity)

    summary_path = root / "kda_profile_summary.json"
    if not summary_path.is_file():
        raise EvidenceError(f"missing machine-readable profile summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid profile summary {summary_path}: {error}") from error

    launch = summary.get("launch_manifest")
    if not isinstance(launch, dict) or launch.get("valid") is not True:
        raise EvidenceError(f"launch manifest was not validated: {summary_path}")
    fields = launch.get("validated_fields")
    if not isinstance(fields, dict):
        raise EvidenceError(f"validated launch fields are missing: {summary_path}")
    if fields.get("pair_scratch") != expected_scratch:
        raise EvidenceError("JSON and profile identity disagree on pair_scratch")
    if fields.get("build_variant") != identity.get("variant"):
        raise EvidenceError("JSON and profile identity disagree on build_variant")
    if fields.get("process_device") != 0:
        raise EvidenceError("profile process_device must be logical device 0")
    if not fields.get("visible_devices"):
        raise EvidenceError("profile visible_devices must identify the physical card")
    if summary.get("sample_count") != 10 or summary.get("discarded_rows") != 3:
        raise EvidenceError(
            "pair-scratch comparison requires exactly 3 discarded warmups and 10 samples"
        )
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        raise EvidenceError("profile JSON must contain exactly 10 measured rows")
    expected_device = int(identity["physical_device"])
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvidenceError(f"profile row {index} is not an object")
        if row.get("device_id") != expected_device:
            raise EvidenceError(f"profile row {index} has the wrong physical device")
        if row.get("target_tensor_signature") is not True:
            raise EvidenceError(f"profile row {index} is not the exact target signature")
        if row.get("mixed_aic_aiv") is not True:
            raise EvidenceError(f"profile row {index} has no mixed AIC/AIV evidence")
        for field in (*TIME_FIELDS_US, *RATIO_FIELDS):
            _finite_number(row.get(field), f"row[{index}].{field}")
    hf32 = summary.get("hf32_mode_evidence")
    if not isinstance(hf32, dict) or hf32.get("matches") is not True:
        raise EvidenceError("HF32/IEEE profiler marker does not match the build identity")
    median_ms = _finite_number(summary.get("median_kernel_ms"), "median_kernel_ms")
    recomputed_ms = statistics.median(
        _finite_number(row["effective_duration_us"], "effective_duration_us")
        for row in rows
    ) / 1000.0
    if not math.isclose(median_ms, recomputed_ms, rel_tol=0.0, abs_tol=5e-7):
        raise EvidenceError(
            f"reported/recomputed median mismatch: {median_ms} vs {recomputed_ms}"
        )
    return ProfileEvidence(root=root, identity=identity, summary=summary)


def _require_comparable(single: ProfileEvidence, pingpong: ProfileEvidence) -> None:
    for field in IDENTITY_INVARIANTS:
        if single.identity.get(field) != pingpong.identity.get(field):
            raise EvidenceError(
                f"profile identity differs outside pair_scratch: {field}: "
                f"{single.identity.get(field)!r} != {pingpong.identity.get(field)!r}"
            )
    if _normalized_variant(single.identity["variant"]) != _normalized_variant(
        pingpong.identity["variant"]
    ):
        raise EvidenceError("build variants differ outside pair_scratch")

    single_fields = single.summary["launch_manifest"]["validated_fields"]
    pingpong_fields = pingpong.summary["launch_manifest"]["validated_fields"]
    for field in MANIFEST_INVARIANTS:
        if single_fields.get(field) != pingpong_fields.get(field):
            raise EvidenceError(
                f"launch manifests differ outside pair_scratch: {field}: "
                f"{single_fields.get(field)!r} != {pingpong_fields.get(field)!r}"
            )
    if _normalized_variant(single_fields["build_variant"]) != _normalized_variant(
        pingpong_fields["build_variant"]
    ):
        raise EvidenceError("manifest build variants differ outside pair_scratch")


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def _metric_row(
    single_rows: list[dict[str, Any]],
    pingpong_rows: list[dict[str, Any]],
    field: str,
    scale: float,
) -> dict[str, float | None]:
    baseline = _median(single_rows, field) * scale
    candidate = _median(pingpong_rows, field) * scale
    improvement = (
        (baseline - candidate) / baseline * 100.0 if baseline != 0.0 else None
    )
    return {
        "single": baseline,
        "pingpong": candidate,
        "delta_pingpong_minus_single": candidate - baseline,
        "improvement_percent": improvement,
    }


def compare_profiles(
    single: ProfileEvidence,
    pingpong: ProfileEvidence,
    target_ms: float,
    min_improvement_percent: float,
) -> dict[str, Any]:
    _require_comparable(single, pingpong)
    single_rows = single.summary["rows"]
    pingpong_rows = pingpong.summary["rows"]
    metrics = {
        field.removesuffix("_us") + "_ms": _metric_row(
            single_rows, pingpong_rows, field, 1.0 / 1000.0
        )
        for field in TIME_FIELDS_US
    }
    metrics.update(
        {
            field: _metric_row(single_rows, pingpong_rows, field, 1.0)
            for field in RATIO_FIELDS
        }
    )
    single_ms = float(single.summary["median_kernel_ms"])
    pingpong_ms = float(pingpong.summary["median_kernel_ms"])
    improvement_percent = (single_ms - pingpong_ms) / single_ms * 100.0
    if improvement_percent >= min_improvement_percent:
        selection = "pingpong"
    elif improvement_percent <= -min_improvement_percent:
        selection = "single"
    else:
        selection = "inconclusive"
    return {
        "comparison": "chunk_kda_bwd_intra_pair_scratch_single_vs_pingpong",
        "identity": {
            field: single.identity[field] for field in IDENTITY_INVARIANTS
        },
        "single_profile": str(single.root),
        "pingpong_profile": str(pingpong.root),
        "sample_count_each": 10,
        "single_median_kernel_ms": single_ms,
        "pingpong_median_kernel_ms": pingpong_ms,
        "pingpong_improvement_percent": improvement_percent,
        "minimum_decisive_improvement_percent": min_improvement_percent,
        "selection": selection,
        "target_ms": target_ms,
        "single_under_target": single_ms <= target_ms,
        "pingpong_under_target": pingpong_ms <= target_ms,
        "metrics": metrics,
    }


def _print_human(report: dict[str, Any]) -> None:
    print("profile_identity=PASS (only pair_scratch differs)")
    print(f"single_profile={report['single_profile']}")
    print(f"pingpong_profile={report['pingpong_profile']}")
    print(f"single_median_kernel_ms={report['single_median_kernel_ms']:.6f}")
    print(f"pingpong_median_kernel_ms={report['pingpong_median_kernel_ms']:.6f}")
    print(f"pingpong_improvement_percent={report['pingpong_improvement_percent']:.3f}")
    print(f"pair_scratch_selection={report['selection']}")
    print(
        "performance_target="
        + ("PASS" if report["pingpong_under_target"] else "NOT_MET")
    )
    print("metric                         single      pingpong       delta     improve")
    for field in (
        "aic_ms",
        "aiv_ms",
        "aic_cube_ms",
        "aiv_vec_ms",
        "aiv_scalar_ms",
        "aiv_mte2_ms",
        "aiv_mte3_ms",
        "wait_ms",
    ):
        row = report["metrics"][field]
        improvement = row["improvement_percent"]
        improvement_text = "N/A" if improvement is None else f"{improvement:8.3f}%"
        print(
            f"{field:<28} {row['single']:>10.6f}  {row['pingpong']:>10.6f}  "
            f"{row['delta_pingpong_minus_single']:>10.6f}  {improvement_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_profile", type=pathlib.Path)
    parser.add_argument("pingpong_profile", type=pathlib.Path)
    parser.add_argument("--target-ms", type=float, default=TARGET_MS)
    parser.add_argument(
        "--min-improvement-percent",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT_PERCENT,
    )
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--require-pingpong-faster", action="store_true")
    parser.add_argument("--require-under-target", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.target_ms <= 0.0 or not math.isfinite(args.target_ms):
        parser.error("--target-ms must be finite and positive")
    if args.min_improvement_percent < 0.0 or not math.isfinite(
        args.min_improvement_percent
    ):
        parser.error("--min-improvement-percent must be finite and non-negative")
    try:
        single = _load_profile_group(
            args.single_profile, "single", not args.skip_hash_verification
        )
        pingpong = _load_profile_group(
            args.pingpong_profile, "pingpong", not args.skip_hash_verification
        )
        report = compare_profiles(
            single, pingpong, args.target_ms, args.min_improvement_percent
        )
    except (EvidenceError, KeyError, ValueError) as error:
        print(f"[FAIL] incomparable profile evidence: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        _print_human(report)
    failures = []
    if args.require_pingpong_faster and report["selection"] != "pingpong":
        failures.append(
            "ping-pong does not meet the configured decisive improvement threshold"
        )
    if args.require_under_target and not report["pingpong_under_target"]:
        failures.append(
            f"ping-pong median exceeds target {args.target_ms:.6f} ms"
        )
    for failure in failures:
        print(f"[FAIL] {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
