"""Repeat original ATK tail profiles to detect intermittent pipeline errors.

Run with the intended wheel and custom OPP selected in the environment.
This checks bytewise stability; it does not replace ATK dual precision tests.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401: registers the NPU device for the test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atk-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=int, nargs="+", default=[143, 163, 154, 153, 146])
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[0, 10000, 20000])
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    combinations = [(case, offset) for offset in args.seed_offsets for case in args.profiles]
    if args.repeats < len(combinations):
        parser.error("--repeats must cover every profile/seed combination")
    args.output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.atk_root.resolve()))
    spec = importlib.util.spec_from_file_location("finalize_atk", args.atk_root / "Finalize.py")
    atk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atk)
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    references = {}
    failures = []
    for iteration in range(args.repeats):
        case, offset = combinations[iteration % len(combinations)]
        profile = dict(atk.PROFILES[case])
        profile["seed"] += offset
        inputs = atk.build_inputs(profile)
        outputs = [tensor.cpu() for tensor in atk.npu_call(inputs, profile, device)]
        assert len(outputs) == 7
        key = (case, offset)
        if key not in references:
            references[key] = outputs
        differing_bytes = [
            int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())
            for actual, expected in zip(outputs, references[key])
        ]
        if any(differing_bytes):
            record = dict(iteration=iteration, case=case, seed=profile["seed"],
                          differing_bytes=differing_bytes)
            failures.append(record)
            torch.save(dict(inputs=inputs, profile=profile, reference=references[key],
                            actual=outputs), args.output / f"failure_{iteration}.pt")
            print("FAIL", record, flush=True)
        if iteration % 500 == 0:
            print(f"{iteration}/{args.repeats}, failures={len(failures)}", flush=True)

    report = dict(launches=args.repeats, combinations=combinations, failures=failures)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{args.repeats} launches, {len(failures)} failures", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
