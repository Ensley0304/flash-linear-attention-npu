#!/usr/bin/env python3
"""Summarize per-core lanes in an msprof-op simulator trace.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPECTED = {
    "aic-only": {"core0.cubecore0"},
    "mix": {"core0.cubecore0", "core0.veccore0", "core0.veccore1"},
}


def find_trace(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "trace.json"
    if candidate.is_file():
        return candidate
    matches = sorted(path.rglob("trace.json"))
    combined = [p for p in matches if p.parent.name == "simulator"]
    if len(combined) == 1:
        return combined[0]
    if not combined:
        raise FileNotFoundError(f"No combined simulator trace.json under {path}")
    raise RuntimeError(
        "Multiple simulator trace.json files found; pass the desired simulator directory"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="trace.json or its simulator directory")
    parser.add_argument("--expect", choices=sorted(EXPECTED))
    args = parser.parse_args()

    try:
        trace_path = find_trace(args.path.resolve())
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stats: dict[str, dict[str, float | int]] = {}
    for event in payload.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        core = str(event.get("pid", "unknown"))
        item = stats.setdefault(core, {"events": 0, "max_end_us": 0.0})
        item["events"] = int(item["events"]) + 1
        end = float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))
        item["max_end_us"] = max(float(item["max_end_us"]), end)

    print(f"trace: {trace_path}")
    print(f"{'core':<20} {'events':>10} {'max_end_us':>14}")
    for core in sorted(stats):
        item = stats[core]
        print(f"{core:<20} {int(item['events']):>10} {item['max_end_us']:>14.3f}")

    if not args.expect:
        return 0

    expected = EXPECTED[args.expect]
    present = set(stats)
    missing = sorted(expected - present)
    if missing:
        print(f"missing expected lanes: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"lane check: PASS ({args.expect})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
