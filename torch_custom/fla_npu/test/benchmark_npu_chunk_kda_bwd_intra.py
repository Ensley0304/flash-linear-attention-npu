# Copyright (c) 2026 Tianjin University, Ltd.

"""Reproducible NPU benchmark for ``chunk_kda_bwd_intra``.

The defaults match the performance target used by this operator's design
document.  Select the physical device with ``ASCEND_RT_VISIBLE_DEVICES`` and
pass the process-local device through ``--device``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception as error:  # pragma: no cover - requires an Ascend runtime
    raise RuntimeError("torch_npu is required for this benchmark") from error

from fla_npu.ops import ascendc as fla_ascendc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="process-local NPU id")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--gate-scale", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        help="write the parsed launch contract as JSON before allocating inputs",
    )
    parser.add_argument(
        "--safe-gate", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _build_identity() -> tuple[
    str, str, str, str, str, str, str, str, str, str, str, str, str
]:
    variant = os.environ.get("KDA_BUILD_VARIANT", "unlabeled")
    pair_gates = os.environ.get("KDA_BUILD_PAIR_GATES", "unlabeled")
    shared_setup = os.environ.get("KDA_BUILD_SHARED_SETUP", "unlabeled")
    stage_epilogue = os.environ.get("KDA_BUILD_STAGE_EPILOGUE", "unlabeled")
    pair_scratch = os.environ.get("KDA_BUILD_PAIR_SCRATCH", "unlabeled")
    tail_blocks = os.environ.get("KDA_BUILD_TAIL_BLOCKS", "unlabeled")
    vector_mask = os.environ.get("KDA_BUILD_VECTOR_MASK", "unlabeled")
    mmad_engines = os.environ.get("KDA_BUILD_MMAD_ENGINES", "unlabeled")
    db_reduce = os.environ.get("KDA_BUILD_DB_REDUCE", "unlabeled")
    task_store = os.environ.get("KDA_BUILD_TASK_STORE", "unlabeled")
    stage_a = os.environ.get("KDA_BUILD_STAGE_A", "unlabeled")
    cube_mode = os.environ.get("KDA_BUILD_CUBE_MODE", "unlabeled")
    stage_io = os.environ.get("KDA_BUILD_STAGE_IO", "unlabeled")
    identity = (
        variant,
        pair_gates,
        shared_setup,
        stage_epilogue,
        pair_scratch,
        tail_blocks,
        vector_mask,
        mmad_engines,
        db_reduce,
        task_store,
        stage_a,
        cube_mode,
        stage_io,
    )
    legacy_identity = identity[:3]
    if legacy_identity == ("unlabeled", "unlabeled", "unlabeled"):
        if any(value != "unlabeled" for value in identity[3:]):
            raise ValueError(
                "partial KDA build identity: a newer build field is labeled "
                "while the legacy identity fields are unlabeled"
            )
        return identity
    if pair_gates not in {"factor", "direct"}:
        raise ValueError(f"invalid KDA_BUILD_PAIR_GATES={pair_gates!r}")
    if shared_setup not in {"overlap", "prologue"}:
        raise ValueError(f"invalid KDA_BUILD_SHARED_SETUP={shared_setup!r}")
    if stage_epilogue == "unlabeled":
        # Compatibility for fully labeled two-dimensional benchmark identities.
        if any(value != "unlabeled" for value in identity[4:]):
            raise ValueError(
                "partial KDA build identity: a newer build field is labeled "
                "while stage epilogue is unlabeled"
            )
        expected_variant = f"pair-{pair_gates}_setup-{shared_setup}"
    elif stage_epilogue in {"overlap", "tail"}:
        expected_variant = (
            f"pair-{pair_gates}_setup-{shared_setup}_epilogue-{stage_epilogue}"
        )
        if pair_scratch == "unlabeled":
            # Compatibility for the previous fully labeled three-dimensional identity.
            if (
                tail_blocks != "unlabeled"
                or mmad_engines != "unlabeled"
                or vector_mask != "unlabeled"
                or db_reduce != "unlabeled"
                or task_store != "unlabeled"
                or stage_a != "unlabeled"
                or cube_mode != "unlabeled"
            ):
                raise ValueError(
                    "partial KDA build identity: a newer build field is labeled "
                    "while pair scratch is unlabeled"
                )
        elif pair_scratch in {"pingpong", "single"}:
            expected_variant += f"_scratch-{pair_scratch}"
            if tail_blocks == "unlabeled":
                # Compatibility for the previous fully labeled 4D identity.
                if (
                    mmad_engines != "unlabeled"
                    or vector_mask != "unlabeled"
                    or db_reduce != "unlabeled"
                    or task_store != "unlabeled"
                    or stage_a != "unlabeled"
                    or cube_mode != "unlabeled"
                ):
                    raise ValueError(
                        "partial KDA build identity: MMAD engines or Vector mask "
                        "is labeled while tail blocks is unlabeled"
                    )
            elif tail_blocks in {"batch", "scalar"}:
                expected_variant += f"_tail-{tail_blocks}"
                if mmad_engines == "unlabeled":
                    # Compatibility for the previous fully labeled 5D identity.
                    if (
                        vector_mask != "unlabeled"
                        or db_reduce != "unlabeled"
                        or task_store != "unlabeled"
                        or stage_a != "unlabeled"
                        or cube_mode != "unlabeled"
                    ):
                        raise ValueError(
                            "partial KDA build identity: Vector mask is labeled "
                            "while MMAD engines is unlabeled"
                        )
                elif mmad_engines in {"persistent", "scoped"}:
                    expected_variant += f"_mmad-{mmad_engines}"
                    if vector_mask == "unlabeled":
                        # Compatibility for the previous fully labeled 6D identity.
                        if (
                            db_reduce != "unlabeled"
                            or task_store != "unlabeled"
                            or stage_a != "unlabeled"
                            or cube_mode != "unlabeled"
                        ):
                            raise ValueError(
                                "partial KDA build identity: db reduction is labeled "
                                "while Vector mask is unlabeled"
                            )
                    elif vector_mask in {"reuse", "per-call"}:
                        expected_variant += f"_vmask-{vector_mask}"
                        if db_reduce == "unlabeled":
                            # Compatibility for the previous fully labeled 7D identity.
                            if (
                                task_store != "unlabeled"
                                or stage_a != "unlabeled"
                                or cube_mode != "unlabeled"
                            ):
                                raise ValueError(
                                    "partial KDA build identity: task store is labeled "
                                    "while db reduction is unlabeled"
                                )
                        elif db_reduce in {"coalesced", "per-row"}:
                            expected_variant += f"_dbr-{db_reduce}"
                            if task_store == "unlabeled":
                                # Compatibility for the previous fully labeled 8D identity.
                                if (
                                    stage_a != "unlabeled"
                                    or cube_mode != "unlabeled"
                                ):
                                    raise ValueError(
                                        "partial KDA build identity: Cube mode is labeled "
                                        "while task store is unlabeled"
                                    )
                            elif task_store in {"overlap", "serial"}:
                                expected_variant += f"_store-{task_store}"
                                if stage_a == "unlabeled":
                                    # Compatibility for identities created before
                                    # the stage-A packing A/B dimension existed.
                                    if cube_mode == "unlabeled":
                                        pass
                                    elif cube_mode in {"ieee", "hf32"}:
                                        expected_variant += f"_cube-{cube_mode}"
                                    else:
                                        raise ValueError(
                                            f"invalid KDA_BUILD_CUBE_MODE={cube_mode!r}"
                                        )
                                elif stage_a in {"packed", "split"}:
                                    expected_variant += f"_stagea-{stage_a}"
                                    if cube_mode == "unlabeled":
                                        pass
                                    elif cube_mode in {"ieee", "hf32"}:
                                        expected_variant += f"_cube-{cube_mode}"
                                    else:
                                        raise ValueError(
                                            f"invalid KDA_BUILD_CUBE_MODE={cube_mode!r}"
                                        )
                                else:
                                    raise ValueError(
                                        f"invalid KDA_BUILD_STAGE_A={stage_a!r}"
                                    )
                            else:
                                raise ValueError(
                                    f"invalid KDA_BUILD_TASK_STORE={task_store!r}"
                                )
                        else:
                            raise ValueError(
                                f"invalid KDA_BUILD_DB_REDUCE={db_reduce!r}"
                            )
                    else:
                        raise ValueError(
                            f"invalid KDA_BUILD_VECTOR_MASK={vector_mask!r}"
                        )
                else:
                    raise ValueError(
                        f"invalid KDA_BUILD_MMAD_ENGINES={mmad_engines!r}"
                    )
            else:
                raise ValueError(f"invalid KDA_BUILD_TAIL_BLOCKS={tail_blocks!r}")
        else:
            raise ValueError(f"invalid KDA_BUILD_PAIR_SCRATCH={pair_scratch!r}")
    else:
        raise ValueError(
            f"invalid KDA_BUILD_STAGE_EPILOGUE={stage_epilogue!r}"
        )
    if stage_io == "unlabeled":
        pass
    elif stage_io in {"tscm", "gm"}:
        if cube_mode == "unlabeled":
            raise ValueError(
                "partial KDA build identity: stage I/O is labeled while "
                "Cube mode is unlabeled"
            )
        expected_variant += f"_io-{stage_io}"
    else:
        raise ValueError(f"invalid KDA_BUILD_STAGE_IO={stage_io!r}")
    if variant != expected_variant:
        raise ValueError(
            "inconsistent KDA build identity: "
            f"variant={variant!r}, expected={expected_variant!r}"
        )
    if task_store == "overlap" and (
        tail_blocks != "batch" or stage_epilogue != "tail"
    ):
        raise ValueError(
            "KDA task-store overlap requires tail_blocks='batch' and "
            "stage_epilogue='tail'"
        )
    return identity


def _write_launch_manifest(
    args: argparse.Namespace,
    build_identity: tuple[
        str, str, str, str, str, str, str, str, str, str, str, str, str
    ],
) -> None:
    if args.manifest_out is None:
        return
    manifest = {
        "operator": "chunk_kda_bwd_intra",
        "batch": args.batch,
        "seqlen": args.seqlen,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "chunk_size": args.chunk_size,
        "safe_gate": args.safe_gate,
        "layout": "BNSD",
        "q_dtype": "BF16",
        "k_dtype": "BF16",
        "gate_dtype": "FP32",
        "accumulator_dtype": "FP32",
        "process_device": args.device,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "gate_scale": args.gate_scale,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "build_variant": build_identity[0],
        "pair_gates": build_identity[1],
        "shared_setup": build_identity[2],
        "stage_epilogue": build_identity[3],
        "pair_scratch": build_identity[4],
        "tail_blocks": build_identity[5],
        "vector_mask": build_identity[6],
        "mmad_engines": build_identity[7],
        "db_reduce": build_identity[8],
        "task_store": build_identity[9],
        "stage_a": build_identity[10],
        "cube_mode": build_identity[11],
        "stage_io": build_identity[12],
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _make_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260721)
    b, h, t, k = args.batch, args.heads, args.seqlen, args.head_dim
    q = torch.randn(b, h, t, k, dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(q) * 0.1
    gate = -torch.rand(b, h, t, k, dtype=torch.float32).cumsum(dim=2) * args.gate_scale
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float32))
    d_a_qk = torch.randn(b, h, t, args.chunk_size, dtype=torch.float32) * 0.02
    d_a_kk = torch.randn_like(d_a_qk) * 0.02
    d_q = torch.randn(b, h, t, k, dtype=torch.float32) * 0.01
    d_k = torch.randn_like(d_q) * 0.01
    d_beta = torch.randn(b, h, t, dtype=torch.float32) * 0.01
    d_gate = torch.randn_like(d_q) * 0.01
    return tuple(
        tensor.to(device)
        for tensor in (q, key, gate, beta, d_a_qk, d_a_kk, d_q, d_k, d_beta, d_gate)
    )


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if not math.isfinite(args.gate_scale):
        raise ValueError("gate-scale must be finite")
    build_identity = _build_identity()
    _write_launch_manifest(args, build_identity)

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    inputs = _make_inputs(args, device)

    def launch() -> tuple[torch.Tensor, ...]:
        return fla_ascendc.chunk_kda_bwd_intra(
            *inputs,
            chunk_size=args.chunk_size,
            layout="BNSD",
            safe_gate=args.safe_gate,
        )

    with torch.no_grad():
        for _ in range(args.warmup):
            outputs = launch()
        torch.npu.synchronize()

        samples_ms = []
        for _ in range(args.repeat):
            begin = time.perf_counter()
            outputs = launch()
            torch.npu.synchronize()
            samples_ms.append((time.perf_counter() - begin) * 1e3)

    # Keep the final outputs live until all asynchronous work has completed.
    if not all(torch.isfinite(output).all().item() for output in outputs):
        raise RuntimeError("benchmark output contains NaN or Inf")

    mode = "safe" if args.safe_gate else "unsafe"
    print(
        f"shape=B{args.batch}_T{args.seqlen}_H{args.heads}_K{args.head_dim}_"
        f"BT{args.chunk_size}_BF16_{mode}"
    )
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print(
        "variant="
        f"{build_identity[0]}, pair_gates={build_identity[1]}, "
        f"shared_setup={build_identity[2]}, stage_epilogue={build_identity[3]}, "
        f"pair_scratch={build_identity[4]}, tail_blocks={build_identity[5]}, "
        f"task_store={build_identity[9]}, "
        f"mmad_engines={build_identity[7]}, vector_mask={build_identity[6]}, "
        f"db_reduce={build_identity[8]}, stage_a={build_identity[10]}, "
        f"stage_io={build_identity[12]}"
    )
    print(
        "e2e_ms: "
        f"median={statistics.median(samples_ms):.3f}, "
        f"min={min(samples_ms):.3f}, max={max(samples_ms):.3f}"
    )
    print(f"cube_mode={build_identity[11]}")
    print("samples_ms: " + ",".join(f"{sample:.3f}" for sample in samples_ms))


if __name__ == "__main__":
    main()
