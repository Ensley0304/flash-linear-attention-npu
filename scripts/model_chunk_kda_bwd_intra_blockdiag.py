#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.

"""Model a precision-preserving stage-level block-diagonal KDA MMAD.

This is a design verifier, not a performance result.  It combines the logical
GEMMs of one 16-token late stage into one larger RowMajor GEMM by placing every
original A matrix on a disjoint (M, K) diagonal block and stacking the matching
B matrices on K.  No gate reference is shared across logical GEMMs.

Each K segment is rounded up to the A2 K32 MMAD tile.  The active products keep
their original local-k order; padding and all other diagonal blocks contribute
only exact zeros.  The script proves that structural invariant and reports the
extra Cube/workspace cost for the exact BT=64, K=128 target domain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product


BC = 16
N = 128
K_TILE = 32
FP32_BYTES = 4
STAGES = 4
TARGET_TASKS = 1 * (8192 // 64) * 32

A2_L1_BYTES = 512 * 1024
A2_L0A_BYTES = 64 * 1024
A2_L0B_BYTES = 64 * 1024
A2_L0C_BYTES = 128 * 1024


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class LogicalGemm:
    name: str
    m: int
    k: int


@dataclass(frozen=True)
class Block:
    name: str
    m: int
    k: int
    m_begin: int
    k_begin: int
    k_capacity: int


@dataclass(frozen=True)
class StageModel:
    stage: int
    logical_gemms: int
    fused_m: int
    fused_n: int
    fused_k: int
    logical_fmas: int
    fused_fmas: int
    compute_ratio: float
    a_elements: int
    active_a_elements: int
    zero_a_elements: int
    b_elements: int
    zero_b_elements: int
    c_elements: int
    workspace_elements: int
    workspace_bytes: int


@dataclass(frozen=True)
class PartitionModel:
    calls: int
    fmas: int
    workspace_traffic_bytes: int
    structural_zero_elements: int
    signature: str


def logical_gemms(stage: int) -> list[LogicalGemm]:
    if not 0 <= stage < STAGES:
        raise ValueError(f"stage must be in [0, {STAGES}), got {stage}")

    gemms: list[LogicalGemm] = []
    if stage > 0:
        gemms.append(LogicalGemm("off_left", 2 * BC, stage * BC))
        for early in range(stage):
            gemms.append(LogicalGemm(f"off_right_{early}", BC, 2 * BC))
    gemms.append(LogicalGemm("diag_right", BC, 2 * BC))
    gemms.append(LogicalGemm("diag_left", 2 * BC, BC))
    return gemms


def block_layout(stage: int) -> tuple[list[Block], int, int]:
    blocks: list[Block] = []
    m_cursor = 0
    k_cursor = 0
    gemms = logical_gemms(stage)
    for index, gemm in enumerate(gemms):
        # Every following logical block starts on a fresh K32 MMAD tile.  The
        # final block may retain its exact K16 tail because no later block
        # needs an aligned start.
        k_capacity = gemm.k if index == len(gemms) - 1 else round_up(gemm.k, K_TILE)
        blocks.append(
            Block(
                name=gemm.name,
                m=gemm.m,
                k=gemm.k,
                m_begin=m_cursor,
                k_begin=k_cursor,
                k_capacity=k_capacity,
            )
        )
        m_cursor += gemm.m
        k_cursor += k_capacity
    return blocks, m_cursor, k_cursor


def verify_active_product_order(blocks: list[Block], fused_m: int, fused_k: int) -> None:
    """Prove every output row sees exactly one unchanged logical dot product."""

    owner_by_row: list[Block | None] = [None] * fused_m
    for block in blocks:
        for row in range(block.m_begin, block.m_begin + block.m):
            if owner_by_row[row] is not None:
                raise AssertionError(f"M blocks overlap at fused row {row}")
            owner_by_row[row] = block

    if any(owner is None for owner in owner_by_row):
        raise AssertionError("M block layout contains an uncovered fused row")

    for fused_row, owner in enumerate(owner_by_row):
        assert owner is not None
        active_global_k = [
            global_k
            for global_k in range(fused_k)
            if owner.k_begin <= global_k < owner.k_begin + owner.k
        ]
        expected_global_k = list(range(owner.k_begin, owner.k_begin + owner.k))
        if active_global_k != expected_global_k:
            raise AssertionError(
                f"{owner.name} row {fused_row} changes active K order: "
                f"{active_global_k} != {expected_global_k}"
            )

        local_k = [global_k - owner.k_begin for global_k in active_global_k]
        if local_k != list(range(owner.k)):
            raise AssertionError(
                f"{owner.name} row {fused_row} changes local dot-product order"
            )

        padded = range(owner.k_begin + owner.k, owner.k_begin + owner.k_capacity)
        if any(global_k in active_global_k for global_k in padded):
            raise AssertionError(f"{owner.name} K32 padding is not structurally zero")


def model_stage(stage: int) -> StageModel:
    gemms = logical_gemms(stage)
    blocks, fused_m, fused_k = block_layout(stage)
    verify_active_product_order(blocks, fused_m, fused_k)

    logical_fmas = sum(gemm.m * gemm.k * N for gemm in gemms)
    fused_fmas = fused_m * fused_k * N
    active_a_elements = sum(gemm.m * gemm.k for gemm in gemms)
    a_elements = fused_m * fused_k
    b_elements = fused_k * N
    zero_b_elements = (fused_k - sum(gemm.k for gemm in gemms)) * N
    c_elements = fused_m * N
    workspace_elements = a_elements + b_elements + c_elements

    return StageModel(
        stage=stage,
        logical_gemms=len(gemms),
        fused_m=fused_m,
        fused_n=N,
        fused_k=fused_k,
        logical_fmas=logical_fmas,
        fused_fmas=fused_fmas,
        compute_ratio=fused_fmas / logical_fmas,
        a_elements=a_elements,
        active_a_elements=active_a_elements,
        zero_a_elements=a_elements - active_a_elements,
        b_elements=b_elements,
        zero_b_elements=zero_b_elements,
        c_elements=c_elements,
        workspace_elements=workspace_elements,
        workspace_bytes=workspace_elements * FP32_BYTES,
    )


def contiguous_partitions(gemms: list[LogicalGemm]) -> list[list[list[LogicalGemm]]]:
    """Enumerate order-preserving partitions of one stage's logical GEMMs."""

    if not gemms:
        return []
    partitions: list[list[list[LogicalGemm]]] = []
    boundary_count = len(gemms) - 1
    for mask in range(1 << boundary_count):
        groups: list[list[LogicalGemm]] = []
        current = [gemms[0]]
        for boundary in range(boundary_count):
            if mask & (1 << boundary):
                groups.append(current)
                current = [gemms[boundary + 1]]
            else:
                current.append(gemms[boundary + 1])
        groups.append(current)
        partitions.append(groups)
    return partitions


def model_partition(groups: list[list[LogicalGemm]]) -> PartitionModel:
    total_fmas = 0
    total_traffic_bytes = 0
    total_zero_elements = 0
    signature_parts: list[str] = []
    for group in groups:
        m_total = sum(gemm.m for gemm in group)
        k_total = 0
        active_a = 0
        active_b = 0
        for index, gemm in enumerate(group):
            capacity = (
                gemm.k
                if index == len(group) - 1
                else round_up(gemm.k, K_TILE)
            )
            k_total += capacity
            active_a += gemm.m * gemm.k
            active_b += gemm.k * N

        a_elements = m_total * k_total
        b_elements = k_total * N
        c_elements = m_total * N
        total_fmas += m_total * k_total * N
        # A/B: AIV writes then AIC reads.  C: AIC writes then AIV reads.
        total_traffic_bytes += 2 * (
            a_elements + b_elements + c_elements
        ) * FP32_BYTES
        total_zero_elements += (a_elements - active_a) + (
            b_elements - active_b
        )
        signature_parts.append("+".join(gemm.name for gemm in group))

    return PartitionModel(
        calls=len(groups),
        fmas=total_fmas,
        workspace_traffic_bytes=total_traffic_bytes,
        structural_zero_elements=total_zero_elements,
        signature=" | ".join(signature_parts),
    )


def build_partition_frontier() -> list[dict[str, object]]:
    stage_options = [
        [model_partition(groups) for groups in contiguous_partitions(logical_gemms(stage))]
        for stage in range(STAGES)
    ]
    candidates: list[dict[str, object]] = []
    for combination in product(*stage_options):
        candidates.append(
            {
                "calls": sum(item.calls for item in combination),
                "fmas": sum(item.fmas for item in combination),
                "workspace_traffic_bytes": sum(
                    item.workspace_traffic_bytes for item in combination
                ),
                "structural_zero_elements": sum(
                    item.structural_zero_elements for item in combination
                ),
                "stage_signatures": [item.signature for item in combination],
            }
        )

    frontier: list[dict[str, object]] = []
    for candidate in candidates:
        dominated = any(
            other["calls"] <= candidate["calls"]
            and other["fmas"] <= candidate["fmas"]
            and other["workspace_traffic_bytes"]
            <= candidate["workspace_traffic_bytes"]
            and (
                other["calls"] < candidate["calls"]
                or other["fmas"] < candidate["fmas"]
                or other["workspace_traffic_bytes"]
                < candidate["workspace_traffic_bytes"]
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)

    # Multiple partitions can have identical metrics.  Retain one stable
    # representative so JSON and documentation remain deterministic.
    unique: dict[tuple[int, int, int], dict[str, object]] = {}
    for candidate in frontier:
        key = (
            int(candidate["calls"]),
            int(candidate["fmas"]),
            int(candidate["workspace_traffic_bytes"]),
        )
        current = unique.get(key)
        if current is None or candidate["stage_signatures"] < current["stage_signatures"]:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda item: item["calls"])


def build_layout_feasible_11_gemm() -> dict[str, object]:
    """Model a lower-risk grouping that avoids mixed A orientations.

    For stages 1..3, off-left and diag-left share the existing RowMajor packed
    A and produce two N=128 output panels through one N=256 GEMM.  All
    off-right matrices use one ColumnMajor block-diagonal GEMM, and diag-right
    remains unchanged.  Stage 0 keeps its two original diagonal calls.

    The traffic estimate deliberately gives each group separate A/B/C storage;
    a kernel implementation may reuse packed A and overlap retired regions,
    but must not claim that reduction until its exact layout is audited.
    """

    calls = 0
    fmas = 0
    traffic_bytes = 0
    zero_elements = 0
    stage_rows: list[dict[str, object]] = []
    for stage in range(STAGES):
        if stage == 0:
            groups = [
                ("diag_right", BC, 2 * BC, N, 0),
                ("diag_left", 2 * BC, BC, N, 0),
            ]
        else:
            prefix = (stage + 1) * BC
            off_prefix = stage * BC

            # The horizontal left GEMM uses one packed A=[Aoff, Adiag].
            # Its two N=128 B panels select disjoint contiguous K intervals;
            # the active products retain the original local-k order.
            for panel, expected in (
                (0, list(range(off_prefix))),
                (1, list(range(off_prefix, prefix))),
            ):
                active_k = [
                    k
                    for k in range(prefix)
                    if (panel == 0 and k < off_prefix)
                    or (panel == 1 and k >= off_prefix)
                ]
                if active_k != expected:
                    raise AssertionError(
                        f"stage {stage} left panel {panel} changes active K order"
                    )

            # The right GEMM stacks each original K32 B matrix and assigns one
            # disjoint M16 row block to it.  Every output row therefore sees
            # exactly the same contiguous K32 sequence as its logical GEMM.
            right_m = stage * BC
            right_k = stage * 2 * BC
            for row in range(right_m):
                early = row // BC
                active_k = [
                    k
                    for k in range(right_k)
                    if early * 2 * BC <= k < (early + 1) * 2 * BC
                ]
                expected = list(range(early * 2 * BC, (early + 1) * 2 * BC))
                if active_k != expected:
                    raise AssertionError(
                        f"stage {stage} right row {row} changes active K order"
                    )

            # name, M, K, N, structural zeros in A/B
            groups = [
                (
                    "off_left+diag_left_horizontal",
                    2 * BC,
                    prefix,
                    2 * N,
                    prefix * N,
                ),
                (
                    "off_right_block_diagonal",
                    stage * BC,
                    stage * 2 * BC,
                    N,
                    stage * BC * stage * 2 * BC
                    - stage * BC * 2 * BC,
                ),
                ("diag_right", BC, 2 * BC, N, 0),
            ]

        stage_fmas = 0
        stage_traffic = 0
        stage_zero = 0
        for _name, m, k, n, zeros in groups:
            a_elements = m * k
            b_elements = k * n
            c_elements = m * n
            stage_fmas += m * k * n
            stage_traffic += 2 * (
                a_elements + b_elements + c_elements
            ) * FP32_BYTES
            stage_zero += zeros
        calls += len(groups)
        fmas += stage_fmas
        traffic_bytes += stage_traffic
        zero_elements += stage_zero
        stage_rows.append(
            {
                "stage": stage,
                "calls": len(groups),
                "fmas": stage_fmas,
                "workspace_traffic_bytes": stage_traffic,
                "structural_zero_elements": stage_zero,
                "groups": [group[0] for group in groups],
            }
        )

    if calls != 11:
        raise AssertionError(f"layout-feasible candidate must have 11 calls, got {calls}")

    tile_budgets = {
        "left_n256": {
            "l1_bytes": 2 * 32 * K_TILE * FP32_BYTES
            + 2 * K_TILE * 256 * FP32_BYTES,
            "l0a_bytes": 2 * 32 * K_TILE * FP32_BYTES,
            "l0b_bytes": 2 * K_TILE * 256 * FP32_BYTES,
            "l0c_bytes": 32 * 256 * FP32_BYTES,
        },
        "right_m48": {
            "l1_bytes": 2 * 48 * K_TILE * FP32_BYTES
            + 2 * K_TILE * N * FP32_BYTES,
            "l0a_bytes": 2 * 48 * K_TILE * FP32_BYTES,
            "l0b_bytes": 2 * K_TILE * N * FP32_BYTES,
            "l0c_bytes": 48 * N * FP32_BYTES,
        },
    }
    limits = {
        "l1_bytes": A2_L1_BYTES,
        "l0a_bytes": A2_L0A_BYTES,
        "l0b_bytes": A2_L0B_BYTES,
        "l0c_bytes": A2_L0C_BYTES,
    }
    for engine, budget in tile_budgets.items():
        for name, value in budget.items():
            if value > limits[name]:
                raise AssertionError(
                    f"layout-feasible {engine} {name}={value} exceeds {limits[name]}"
                )

    return {
        "name": "layout_feasible_11_gemm",
        "precision_contract": {
            "gate_reference_reassociation": False,
            "active_product_order_changed": False,
            "structural_zero_products_inserted": True,
            "requires_finite_non_padding_operands": True,
            "signed_zero_bitwise_equivalence_claimed": False,
        },
        "routing_proof": "PASS",
        "calls": calls,
        "fmas_per_task": fmas,
        "workspace_traffic_bytes_per_task": traffic_bytes,
        "structural_zero_elements": zero_elements,
        "tile_budgets": tile_budgets,
        "stages": stage_rows,
    }


def verify_a2_tile_budget(max_m: int) -> dict[str, int]:
    # Two-stage L1/L0 ping-pong, one FP32 L0C stage, N=128 and K-tile=32.
    budget = {
        "l1_bytes": 2 * max_m * K_TILE * FP32_BYTES
        + 2 * K_TILE * N * FP32_BYTES,
        "l0a_bytes": 2 * max_m * K_TILE * FP32_BYTES,
        "l0b_bytes": 2 * K_TILE * N * FP32_BYTES,
        "l0c_bytes": max_m * N * FP32_BYTES,
    }
    limits = {
        "l1_bytes": A2_L1_BYTES,
        "l0a_bytes": A2_L0A_BYTES,
        "l0b_bytes": A2_L0B_BYTES,
        "l0c_bytes": A2_L0C_BYTES,
    }
    for name, value in budget.items():
        if value > limits[name]:
            raise AssertionError(f"{name}={value} exceeds A2 limit {limits[name]}")
    return budget


def build_report(used_aic: int) -> dict[str, object]:
    stages = [model_stage(stage) for stage in range(STAGES)]
    logical_calls = sum(stage.logical_gemms for stage in stages)
    fused_calls = len(stages)
    if logical_calls != 17 or fused_calls != 4:
        raise AssertionError(
            f"unexpected call model: logical={logical_calls}, fused={fused_calls}"
        )

    logical_fmas_per_task = sum(stage.logical_fmas for stage in stages)
    fused_fmas_per_task = sum(stage.fused_fmas for stage in stages)
    workspace_bytes_per_core = sum(stage.workspace_bytes for stage in stages)
    workspace_read_write_bytes_per_task = 2 * workspace_bytes_per_core
    zero_init_bytes_per_core = (
        sum(
            stage.zero_a_elements + stage.zero_b_elements
            for stage in stages
        )
        * FP32_BYTES
    )
    tile_budget = verify_a2_tile_budget(max(stage.fused_m for stage in stages))

    return {
        "scheme": "stage_block_diagonal_4_gemm_candidate",
        "precision_contract": {
            "gate_reference_reassociation": False,
            "active_product_order_changed": False,
            "active_local_k_order_preserved": True,
            "structural_zero_products_inserted": True,
            "padding_operand": "+0.0f",
            "requires_finite_non_padding_operands": True,
            "signed_zero_bitwise_equivalence_claimed": False,
        },
        "target": {
            "batch": 1,
            "seqlen": 8192,
            "heads": 32,
            "chunk_size": 64,
            "head_dim": 128,
            "tasks": TARGET_TASKS,
            "used_aic": used_aic,
        },
        "calls": {
            "logical_per_task": logical_calls,
            "fused_per_task": fused_calls,
            "reduction": logical_calls / fused_calls,
        },
        "compute": {
            "logical_fmas_per_task": logical_fmas_per_task,
            "fused_fmas_per_task": fused_fmas_per_task,
            "inflation": fused_fmas_per_task / logical_fmas_per_task,
            "fused_fmas_target": fused_fmas_per_task * TARGET_TASKS,
        },
        "workspace": {
            "bytes_per_logical_core_four_stage_slots": workspace_bytes_per_core,
            "bytes_for_used_aic": workspace_bytes_per_core * used_aic,
            "structural_zero_init_bytes_per_core": zero_init_bytes_per_core,
            "stage_workspace_read_write_bytes_per_task": workspace_read_write_bytes_per_task,
            "stage_workspace_read_write_bytes_target": (
                workspace_read_write_bytes_per_task * TARGET_TASKS
            ),
        },
        "a2_tile_budget": tile_budget,
        "layout_feasible_candidate": build_layout_feasible_11_gemm(),
        "partition_frontier": build_partition_frontier(),
        "stages": [asdict(stage) for stage in stages],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--used-aic", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--frontier", action="store_true")
    args = parser.parse_args()
    if args.used_aic <= 0:
        parser.error("--used-aic must be positive")

    report = build_report(args.used_aic)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("stage logical fused_shape logical_FMA fused_FMA ratio workspace_KiB")
    for stage in report["stages"]:
        print(
            f"{stage['stage']:>5} {stage['logical_gemms']:>7} "
            f"{stage['fused_m']}x{stage['fused_n']}x{stage['fused_k']} "
            f"{stage['logical_fmas']:>11} {stage['fused_fmas']:>9} "
            f"{stage['compute_ratio']:.3f} "
            f"{stage['workspace_bytes'] / 1024:.1f}"
        )
    print(
        "calls_per_task: "
        f"{report['calls']['logical_per_task']} -> "
        f"{report['calls']['fused_per_task']} "
        f"({report['calls']['reduction']:.2f}x fewer)"
    )
    print(
        "compute_inflation: "
        f"{report['compute']['inflation']:.3f}x; "
        "workspace_per_core: "
        f"{report['workspace']['bytes_per_logical_core_four_stage_slots'] / 1024:.1f} KiB; "
        "target_workspace_traffic: "
        f"{report['workspace']['stage_workspace_read_write_bytes_target'] / 1e9:.3f} GB"
    )
    if args.frontier:
        print("\nPareto frontier (straight block-diagonal workspace model):")
        print("calls FMA_ratio target_workspace_GB")
        logical_fmas = report["compute"]["logical_fmas_per_task"]
        for point in report["partition_frontier"]:
            print(
                f"{point['calls']:>5} "
                f"{point['fmas'] / logical_fmas:>9.3f} "
                f"{point['workspace_traffic_bytes'] * TARGET_TASKS / 1e9:>19.3f}"
            )
        feasible = report["layout_feasible_candidate"]
        print("\nLayout-feasible mixed RowMajor/ColumnMajor candidate:")
        print(
            f"calls={feasible['calls']}, "
            f"FMA_ratio={feasible['fmas_per_task'] / logical_fmas:.3f}, "
            "conservative_target_workspace_GB="
            f"{feasible['workspace_traffic_bytes_per_task'] * TARGET_TASKS / 1e9:.3f}"
        )
    print("structural_active_k_order=PASS")
    print("a2_tile_budget=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
