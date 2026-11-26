"""Check archived paired device measurements; never invent missing benchmarks.

Input JSON: {"comparisons": [{"name": str, "kind": "k256"|"c32"|"regression",
"baseline": {"meta": {...}, "rounds_us": [[...], ...]}, "candidate": {...}}]}.
Every round must contain at least ten measured calls after warmup. Use three or
more interleaved rounds on the same physical device. Each comparison describes
one stage or whole-chain metric, including all interface tasks for chain sums.
The caller must supply the complete required matrix; PASS is only for that input.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


REQUIRED = (
    "device", "soc", "cann", "dtype", "batch", "tokens", "heads", "value_heads",
    "keys", "values", "chunk_size", "layout", "mode", "stage", "metric",
    "gate", "norm", "beta", "state", "segments", "scale", "seed",
    "source_revision", "package_sha256", "warmups", "baseline_role",
)


def check(case):
    base, cand = case["baseline"], case["candidate"]
    bm, cm = base["meta"], cand["meta"]
    for label, run in (("baseline", base), ("candidate", cand)):
        missing = set(REQUIRED) - run["meta"].keys()
        if missing:
            raise ValueError(f"{label}: missing metadata {sorted(missing)}")
        if run["meta"]["warmups"] < 2:
            raise ValueError(f"{label}: fewer than two warmups")
        if len(run["rounds_us"]) < 3:
            raise ValueError(f"{label}: fewer than three rounds")
        for values in run["rounds_us"]:
            if len(values) < 10 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"{label}: insufficient or invalid duration samples")
    if len(base["rounds_us"]) != len(cand["rounds_us"]):
        raise ValueError("round counts differ")
    kind = case["kind"]
    ignore = {"source_revision", "package_sha256", "warmups", "baseline_role"}
    if kind == "k256":
        ignore |= {"keys", "values"}
        if (bm["keys"], bm["values"], cm["keys"], cm["values"]) != (128, 128, 256, 256):
            raise ValueError("k256 comparison requires KV128 versus KV256")
        if bm["baseline_role"] != "original_fast":
            raise ValueError("k256 baseline must be original_fast")
        limit = 5.0
    elif kind == "c32":
        ignore.add("chunk_size")
        if (bm["chunk_size"], cm["chunk_size"]) != (64, 32):
            raise ValueError("c32 comparison requires C64 versus C32")
        if bm["baseline_role"] != "optimized_c64_accepted":
            raise ValueError("C64 baseline has not passed its performance gate")
        limit = 1.0
    elif kind == "regression":
        if bm["baseline_role"] != "original_fast":
            raise ValueError("regression baseline must be original_fast")
        limit = 1.05
    else:
        raise ValueError(f"unknown comparison kind: {kind}")
    different = [k for k in REQUIRED if k not in ignore and bm[k] != cm[k]]
    if different:
        raise ValueError(f"non-comparable conditions: {different}")
    ratios = [statistics.median(c) / statistics.median(b)
              for b, c in zip(base["rounds_us"], cand["rounds_us"])]
    passed = all(x < limit if kind == "c32" else x <= limit for x in ratios)
    return {"name": case["name"], "passed": passed, "round_ratios": ratios,
            "worst_ratio": max(ratios), "limit": limit,
            "note": "C32 near noise floor needs additional samples before acceptance"
            if kind == "c32" and max(ratios) > .95 else ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.report.read_text(encoding="utf-8-sig"))["comparisons"]
    if not cases:
        raise ValueError("empty comparison matrix")
    results = []
    for case in cases:
        try:
            results.append(check(case))
        except (KeyError, TypeError, ValueError) as exc:
            results.append({"name": case.get("name", "unnamed"), "passed": False,
                            "error": str(exc)})
    print(json.dumps({"scope": "provided comparisons only", "results": results}, indent=2))
    raise SystemExit(0 if all(x["passed"] for x in results) else 1)


if __name__ == "__main__":
    main()
