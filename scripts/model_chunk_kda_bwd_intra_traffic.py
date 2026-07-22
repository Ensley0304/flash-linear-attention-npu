#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.

"""Model physical GM traffic for the grouped KDA backward target path.

This is a source-level traffic model, not a profiler result.  It counts every
logical AIV->GM, GM->AIC, AIC->GM and GM->AIV transfer made by grouped key 23
for BT=64 and K=128.  In particular, the two workspace slots are capacity
ping-pong; they do not make any of those transfers disappear.

The tensor side counts the selected q/k/g/beta rows, the causal dA prefixes,
the reference rows redundantly loaded by the two AIV halves, accumulator
inputs and final outputs.  Cache hits may reduce HBM traffic on hardware, but
they do not change the bytes issued by the kernel, so the report deliberately
uses "physical GM traffic" rather than claiming measured HBM bandwidth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


BT = 64
BC = 16
K = 128
STAGES = BT // BC
ROWS_PER_AIV = BC // 2
FP32_BYTES = 4
DEFAULT_QK_BYTES = 2
DEFAULT_USED_AIC = 20

WORKSPACE_SLOT_ELEMENTS = 26_624
WORKSPACE_QUEUE_DEPTH = 2

A2_UB_BYTES = 192 * 1024
A2_L1_BYTES = 512 * 1024
GROUPED_SINGLE_UB_BYTES = 179_936
GROUPED_PAIR_PINGPONG_UB_BYTES = 192_256
PAIR_SCRATCH_BANKS = 3
PERSISTENT_MMAD_L1_BYTES = (40 + 36) * 1024
# The delivery path gives each of the 17 logical GEMMs its own complete local
# event transaction.  Stage-level batching of 2--4 right MMAD calls under one
# flag envelope deadlocks on A2 and is intentionally not counted as scoped.
SCOPED_MMAD_ENVELOPES_PER_TASK = 17
PERSISTENT_MMAD_ENGINES_PER_AIC = 2
FP32_C0_ELEMENTS = 32 // FP32_BYTES
FP32_CUBE_ROWS = 16


@dataclass(frozen=True)
class StageTraffic:
    stage: int
    logical_gemms: int
    a_write_elements: int
    a_read_elements: int
    b_write_elements: int
    b_read_elements: int
    c_write_elements: int
    c_read_elements: int
    a_bytes: int
    b_bytes: int
    c_bytes: int
    total_bytes: int


def model_stage(stage: int) -> StageTraffic:
    if not 0 <= stage < STAGES:
        raise ValueError(f"stage must be in [0, {STAGES}), got {stage}")

    off_prefix = stage * BC
    prefix = (stage + 1) * BC

    # AIV writes Aoff + Adiag (or the equal-size packed A slab).  AIC reads
    # Aoff once from the left GEMM, once through all M16 right subviews, and
    # Adiag once for each of the two diagonal GEMMs.
    a_write = 2 * BC * prefix
    a_read = (2 * BC * off_prefix) + (stage * BC * 2 * BC) + 2 * (2 * BC * BC)

    # For every off-diagonal pair, AIV publishes K-left [16,K] and the two
    # right operands [16,K] each.  The diagonal publishes the same three
    # matrix heights.  Every B matrix is consumed exactly once by AIC.
    b_write = (stage + 1) * (3 * BC * K)
    b_read = b_write

    # Diagonal outputs are [16,K] + [32,K].  A non-zero stage additionally
    # has one [32,K] off-left output and ``stage`` [16,K] off-right outputs.
    c_write = 3 * BC * K
    if stage > 0:
        c_write += 2 * BC * K + stage * BC * K
    c_read = c_write

    a_bytes = (a_write + a_read) * FP32_BYTES
    b_bytes = (b_write + b_read) * FP32_BYTES
    c_bytes = (c_write + c_read) * FP32_BYTES
    return StageTraffic(
        stage=stage,
        logical_gemms=2 if stage == 0 else stage + 3,
        a_write_elements=a_write,
        a_read_elements=a_read,
        b_write_elements=b_write,
        b_read_elements=b_read,
        c_write_elements=c_write,
        c_read_elements=c_read,
        a_bytes=a_bytes,
        b_bytes=b_bytes,
        c_bytes=c_bytes,
        total_bytes=a_bytes + b_bytes + c_bytes,
    )


def tensor_traffic_per_task(qk_bytes: int) -> dict[str, int]:
    selected_feature_elements = BT * K
    # Each physical AIV loads F1..F3, M0..M3 and R0..R2.  The two halves use
    # the same ten K-wide references, so these 20 rows are issued twice in GM
    # traffic even though they overlap the selected g tensor logically.
    duplicated_reference_elements = 2 * 10 * K
    # Each late 16-row block loads only its causal prefix: 16, 32, 48, 64.
    causal_da_elements_one_tensor = BC * sum((stage + 1) * BC for stage in range(STAGES))

    rows = {
        "q_read": selected_feature_elements * qk_bytes,
        "k_read": selected_feature_elements * qk_bytes,
        "g_selected_read": selected_feature_elements * FP32_BYTES,
        "g_reference_reload": duplicated_reference_elements * FP32_BYTES,
        "beta_read": BT * FP32_BYTES,
        "dAqk_causal_read": causal_da_elements_one_tensor * FP32_BYTES,
        "dAkk_causal_read": causal_da_elements_one_tensor * FP32_BYTES,
        "dq_input_read": selected_feature_elements * FP32_BYTES,
        "dk_input_read": selected_feature_elements * FP32_BYTES,
        "dg_input_read": selected_feature_elements * FP32_BYTES,
        "db_input_read": BT * FP32_BYTES,
        "dq_output_write": selected_feature_elements * FP32_BYTES,
        "dk_output_write": selected_feature_elements * FP32_BYTES,
        "dg_output_write": selected_feature_elements * FP32_BYTES,
        "db_output_write": BT * FP32_BYTES,
    }
    return rows


def required_gbps(byte_count: int, target_ms: float) -> float:
    return byte_count / (target_ms / 1000.0) / 1e9


def floor_ms(byte_count: int, bandwidth_tbps: float) -> float:
    return byte_count / (bandwidth_tbps * 1e12) * 1000.0


def scenario(name: str, byte_count: int, target_ms: float) -> dict[str, object]:
    return {
        "name": name,
        "bytes_target": byte_count,
        "gb_decimal_target": byte_count / 1e9,
        "required_gbps_at_target_ms": required_gbps(byte_count, target_ms),
    }


def zn_offset(row: int, column: int, matrix_rows: int) -> int:
    """Return the FP32 zN physical element offset for aligned dimensions."""

    return (
        (column // FP32_C0_ELEMENTS) * matrix_rows * FP32_C0_ELEMENTS
        + (row // FP32_CUBE_ROWS)
        * FP32_CUBE_ROWS
        * FP32_C0_ELEMENTS
        + (row % FP32_CUBE_ROWS) * FP32_C0_ELEMENTS
        + column % FP32_C0_ELEMENTS
    )


def nz_offset(row: int, column: int, matrix_columns: int) -> int:
    """Return the FP32 nZ physical element offset for aligned dimensions."""

    return (
        (row // FP32_C0_ELEMENTS) * matrix_columns * FP32_C0_ELEMENTS
        + (column // FP32_CUBE_ROWS)
        * FP32_CUBE_ROWS
        * FP32_C0_ELEMENTS
        + row % FP32_C0_ELEMENTS
        + (column % FP32_CUBE_ROWS) * FP32_C0_ELEMENTS
    )


def validate_tscm_layouts() -> dict[str, object]:
    """Validate the A2 strided-copy image consumed by CATLASS from L1.

    A is emitted once as RowMajor->zN and is also consumed through the
    transpose ColumnMajor->nZ view.  For every logical (m, k), zN(M, K)
    and nZ(K, M) at (k, m) must name the same physical element.
    """

    shapes = {
        (32, 16),
        (32, 32),
        (32, 48),
        (32, 64),
        (16, 128),
        (32, 128),
        (48, 128),
    }
    for rows, columns in shapes:
        if rows % FP32_CUBE_ROWS or columns % FP32_C0_ELEMENTS:
            raise AssertionError("grouped TSCM matrices must be fractal-aligned")
        offsets: set[int] = set()
        for row in range(rows):
            for column in range(columns):
                offset = zn_offset(row, column, rows)
                if offset != nz_offset(column, row, rows):
                    raise AssertionError(
                        f"zN/nZ transpose mismatch for {rows}x{columns} "
                        f"at ({row}, {column})"
                    )
                offsets.add(offset)
        if offsets != set(range(rows * columns)):
            raise AssertionError(
                f"zN image for {rows}x{columns} is not a dense permutation"
            )

        # Two grouped AIVs write rows 0..7 and 8..15 in every 16-row
        # partition.  The helper loops over 8-column C0 blocks and issues one
        # strided 32-byte block per row.  Confirm those writes cover the zN
        # image exactly once.
        produced: list[int] = []
        for row_block in range(0, rows, FP32_CUBE_ROWS):
            for aiv_half in (0, 8):
                destination_row = row_block + aiv_half
                for column in range(0, columns, FP32_C0_ELEMENTS):
                    for local_row in range(8):
                        for lane in range(FP32_C0_ELEMENTS):
                            produced.append(
                                column * rows
                                + (destination_row + local_row)
                                * FP32_C0_ELEMENTS
                                + lane
                            )
        if len(produced) != rows * columns or set(produced) != offsets:
            raise AssertionError(
                f"two-AIV strided zN writes do not cover {rows}x{columns}"
            )

    return {
        "validated": True,
        "element_type": "float32",
        "c0_elements": FP32_C0_ELEMENTS,
        "cube_rows": FP32_CUBE_ROWS,
        "matrix_shapes": [list(shape) for shape in sorted(shapes)],
        "row_major_image": "zN",
        "transpose_column_major_image": "nZ",
        "same_physical_image_for_transpose": True,
        "producer": "two AIV halves using strided 32-byte UB-to-TSCM copies",
        "local_nd2nz_required": False,
    }


def build_report(
    batch: int,
    seqlen: int,
    heads: int,
    qk_bytes: int,
    target_ms: float,
    bandwidth_tbps: list[float],
    stage_io: str,
    used_aic: int,
    pair_scratch: str = "single",
) -> dict[str, object]:
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")
    if seqlen <= 0 or seqlen % BT != 0:
        raise ValueError(f"seqlen must be a positive multiple of {BT}")
    if qk_bytes not in {2, 4}:
        raise ValueError("qk-bytes must be 2 or 4")
    if target_ms <= 0:
        raise ValueError("target-ms must be positive")
    if any(value <= 0 for value in bandwidth_tbps):
        raise ValueError("bandwidth values must be positive")
    if stage_io not in {"tscm", "gm"}:
        raise ValueError("stage-io must be tscm or gm")
    if used_aic <= 0:
        raise ValueError("used-aic must be positive")
    if pair_scratch not in {"single", "pingpong"}:
        raise ValueError("pair-scratch must be single or pingpong")

    tasks = batch * (seqlen // BT) * heads
    stages = [model_stage(stage) for stage in range(STAGES)]
    if sum(item.logical_gemms for item in stages) != 17:
        raise AssertionError("grouped key 23 must issue 17 logical GEMMs per task")

    workspace_a_per_task = sum(item.a_bytes for item in stages)
    workspace_b_per_task = sum(item.b_bytes for item in stages)
    workspace_c_per_task = sum(item.c_bytes for item in stages)
    workspace_per_task = workspace_a_per_task + workspace_b_per_task + workspace_c_per_task

    tensors = tensor_traffic_per_task(qk_bytes)
    tensor_per_task = sum(tensors.values())

    workspace_a_target = workspace_a_per_task * tasks
    workspace_b_target = workspace_b_per_task * tasks
    workspace_c_target = workspace_c_per_task * tasks
    workspace_target = workspace_per_task * tasks
    tensor_target = tensor_per_task * tasks
    current_target = workspace_target + tensor_target

    # AIC could, in principle, transform each distinct Aoff/Adiag image from
    # GM into a stage-local L1 cache once and reuse that image across the left
    # and right MMAD calls.  The AIV publication and one AIC read still remain;
    # only ``a_read - a_write`` is removable.  B has no analogous reuse in the
    # current factorization: every published B matrix feeds exactly one GEMM.
    a_l1_reuse_saved_per_task = sum(
        (item.a_read_elements - item.a_write_elements) * FP32_BYTES
        for item in stages
    )
    if a_l1_reuse_saved_per_task < 0:
        raise AssertionError("stage-local A reuse cannot increase GM traffic")
    a_l1_reuse_saved_target = a_l1_reuse_saved_per_task * tasks

    scenarios = {
        "current_gm_bridge": scenario(
            "current_gm_bridge", current_target, target_ms
        ),
        "aic_l1_a_reuse": scenario(
            "aic_l1_a_reuse",
            current_target - a_l1_reuse_saved_target,
            target_ms,
        ),
        "a_on_chip": scenario(
            "a_on_chip", current_target - workspace_a_target, target_ms
        ),
        "b_on_chip": scenario(
            "b_on_chip", current_target - workspace_b_target, target_ms
        ),
        "c_on_chip": scenario(
            "c_on_chip", current_target - workspace_c_target, target_ms
        ),
        "ab_on_chip": scenario(
            "ab_on_chip",
            current_target - workspace_a_target - workspace_b_target,
            target_ms,
        ),
        "all_workspace_on_chip": scenario(
            "all_workspace_on_chip", tensor_target, target_ms
        ),
    }
    # DAV_2201 implements AIV UB->TSCM through a GM/KFC relay.  Selecting the
    # retained source experiment therefore does not make A/B traffic on-chip on
    # A2; keep the ideal ab_on_chip row only as an architectural lower bound.
    active_scenario_name = "current_gm_bridge"
    active_scenario = scenarios[active_scenario_name]

    off_right_pair_tiles_per_aiv = sum(range(1, STAGES))
    off_right_stage_requests_per_aiv = STAGES - 1
    off_right_aiv_workers_per_task = 2
    off_right_bytes_per_task = (
        off_right_pair_tiles_per_aiv
        * ROWS_PER_AIV
        * K
        * FP32_BYTES
        * off_right_aiv_workers_per_task
    )
    off_right_old_calls_target = (
        tasks * off_right_aiv_workers_per_task * off_right_pair_tiles_per_aiv
    )
    off_right_coalesced_calls_target = (
        tasks * off_right_aiv_workers_per_task * off_right_stage_requests_per_aiv
    )
    right_b_matrices_per_task = sum(stage + 1 for stage in range(STAGES))
    right_b_bytes_per_task = (
        right_b_matrices_per_task * 2 * BC * K * FP32_BYTES
    )
    right_b_old_calls_target = (
        tasks * off_right_aiv_workers_per_task * right_b_matrices_per_task * 2
    )
    right_b_coalesced_calls_target = (
        tasks * off_right_aiv_workers_per_task * right_b_matrices_per_task
    )
    left_c_matrices_per_task = (STAGES - 1) + STAGES
    left_c_bytes_per_task = left_c_matrices_per_task * 2 * BC * K * FP32_BYTES
    left_c_old_calls_target = (
        tasks * off_right_aiv_workers_per_task * left_c_matrices_per_task * 2
    )
    left_c_coalesced_calls_target = (
        tasks * off_right_aiv_workers_per_task * left_c_matrices_per_task
    )

    # The whole-tail epilogue owns four disjoint 8xK row blocks per physical
    # AIV.  The batch path gathers/scatters the same bytes with strided copies
    # and evaluates each independent elementwise expression once over the
    # contiguous 4096-element image.  It does not merge any cross-row
    # reduction or change one element's FP32 operation order.
    tail_aiv_workers_per_task = 2
    tail_blocks_per_aiv = STAGES
    tail_feature_tensors = 3  # dq, dk, dg
    tail_vector_expressions = 6
    tail_output_tensors = 4  # dq, dk, dg, db
    tail_feature_block_bytes = ROWS_PER_AIV * K * FP32_BYTES
    tail_db_block_bytes = ROWS_PER_AIV * FP32_BYTES
    tail_db_input_bytes_per_task = (
        tail_aiv_workers_per_task
        * tail_blocks_per_aiv
        * tail_db_block_bytes
    )
    tail_feature_input_bytes_per_task = (
        tail_aiv_workers_per_task
        * tail_feature_tensors
        * tail_blocks_per_aiv
        * tail_feature_block_bytes
    )
    tail_output_bytes_per_task = tail_aiv_workers_per_task * (
        tail_feature_tensors
        * tail_blocks_per_aiv
        * tail_feature_block_bytes
        + tail_blocks_per_aiv * tail_db_block_bytes
    )
    tail_db_mte2_calls_before = (
        tasks * tail_aiv_workers_per_task * tail_blocks_per_aiv
    )
    tail_db_mte2_calls_after = tasks * tail_aiv_workers_per_task
    tail_feature_mte2_calls_before = (
        tasks
        * tail_aiv_workers_per_task
        * tail_feature_tensors
        * tail_blocks_per_aiv
    )
    tail_feature_mte2_calls_after = (
        tasks * tail_aiv_workers_per_task * tail_feature_tensors
    )
    tail_vector_calls_before = (
        tasks
        * tail_aiv_workers_per_task
        * tail_vector_expressions
        * tail_blocks_per_aiv
    )
    tail_vector_calls_after = (
        tasks * tail_aiv_workers_per_task * tail_vector_expressions
    )
    tail_mte3_calls_before = (
        tasks
        * tail_aiv_workers_per_task
        * tail_output_tensors
        * tail_blocks_per_aiv
    )
    tail_mte3_calls_after = (
        tasks * tail_aiv_workers_per_task * tail_output_tensors
    )
    task_aiv_pairs = tasks * tail_aiv_workers_per_task
    # Each off-diagonal pair publishes three 8xK FP32 tiles per physical AIV:
    # one left operand and the two adjacent right-operand halves.  The local
    # scratch ping-pong does not remove those GM writes.  It makes their MTE3
    # drain eligible to overlap the following pair/diagonal Vector work and
    # waits only before the same scratch set is reused.
    pair_scratch_pairs_per_task = sum(range(STAGES))
    pair_scratch_output_bytes_per_aiv_pair = (
        PAIR_SCRATCH_BANKS * ROWS_PER_AIV * K * FP32_BYTES
    )
    pair_scratch_mte3_bytes_per_task = (
        pair_scratch_pairs_per_task
        * tail_aiv_workers_per_task
        * pair_scratch_output_bytes_per_aiv_pair
    )
    pair_scratch_waits_per_aiv_before = pair_scratch_pairs_per_task
    pair_scratch_waits_per_aiv_after = sum(
        max(stage - 1, 0) for stage in range(1, STAGES)
    )
    pair_scratch_extra_ub_bytes = (
        GROUPED_PAIR_PINGPONG_UB_BYTES - GROUPED_SINGLE_UB_BYTES
    )
    row_start_source_expressions_per_task_before = 14
    consume_subblock_reads_per_task_before = STAGES
    persistent_gate_loop_iterations_per_task_before = STAGES - 1
    logical_gemm_calls_target = tasks * sum(
        item.logical_gemms for item in stages
    )
    scoped_mmad_envelopes_target = tasks * SCOPED_MMAD_ENVELOPES_PER_TASK
    persistent_mmad_envelopes_target = (
        min(tasks, used_aic) * PERSISTENT_MMAD_ENGINES_PER_AIC
    )

    max_stage = stages[-1]
    stage3_live_bytes = (
        max_stage.a_write_elements
        + max_stage.b_write_elements
        + max_stage.c_write_elements
    ) * FP32_BYTES
    stage3_double_bytes = 2 * stage3_live_bytes
    stage3_double_plus_mmad = stage3_double_bytes + PERSISTENT_MMAD_L1_BYTES
    stage3_ab_bytes = (
        max_stage.a_write_elements + max_stage.b_write_elements
    ) * FP32_BYTES
    tscm_ab_double_bytes = WORKSPACE_QUEUE_DEPTH * stage3_ab_bytes
    tscm_ab_double_plus_mmad = (
        tscm_ab_double_bytes + PERSISTENT_MMAD_L1_BYTES
    )

    report = {
        "model": "chunk_kda_bwd_intra_grouped_key23_physical_gm_traffic",
        "scope": {
            "profiler_result": False,
            "counts_kernel_issued_gm_bytes": True,
            "claims_hbm_bytes_after_cache": False,
            "workspace_slots_reduce_traffic": False,
        },
        "target": {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "chunk_size": BT,
            "head_dim": K,
            "qk_bytes": qk_bytes,
            "tasks": tasks,
            "target_ms": target_ms,
            "used_aic": min(tasks, used_aic),
        },
        "implementation": {
            "stage_io": stage_io,
            "pair_scratch": pair_scratch,
            "a2_physical_direct_ub_to_l1": False,
            "a2_route": (
                "software_emulated_via_gm_and_matmul_kfc"
                if stage_io == "tscm"
                else "explicit_two_slot_gm_bridge"
            ),
            "active_scenario": active_scenario_name,
            "active_bytes_target": active_scenario["bytes_target"],
            "active_required_gbps_at_target_ms": active_scenario[
                "required_gbps_at_target_ms"
            ],
        },
        "off_right_consume_coalescing": {
            "changes_fp32_arithmetic": False,
            "changes_gm_bytes": False,
            "pair_tiles_per_aiv": off_right_pair_tiles_per_aiv,
            "stage_requests_per_aiv": off_right_stage_requests_per_aiv,
            "aiv_workers_per_task": off_right_aiv_workers_per_task,
            "bytes_per_task": off_right_bytes_per_task,
            "bytes_target": off_right_bytes_per_task * tasks,
            "mte2_calls_target_before": off_right_old_calls_target,
            "mte2_calls_target_after": off_right_coalesced_calls_target,
            "mte2_calls_target_saved": (
                off_right_old_calls_target - off_right_coalesced_calls_target
            ),
            "mul_add_calls_target_before": off_right_old_calls_target,
            "mul_add_calls_target_after": off_right_coalesced_calls_target,
            "mul_add_calls_target_saved": (
                off_right_old_calls_target - off_right_coalesced_calls_target
            ),
        },
        "right_b_write_coalescing": {
            "changes_fp32_arithmetic": False,
            "changes_gm_bytes": False,
            "right_b_matrices_per_task": right_b_matrices_per_task,
            "bytes_per_task": right_b_bytes_per_task,
            "bytes_target": right_b_bytes_per_task * tasks,
            "mte3_calls_target_before": right_b_old_calls_target,
            "mte3_calls_target_after": right_b_coalesced_calls_target,
            "mte3_calls_target_saved": (
                right_b_old_calls_target - right_b_coalesced_calls_target
            ),
        },
        "left_c_read_coalescing": {
            "changes_fp32_arithmetic": False,
            "changes_gm_bytes": False,
            "left_c_matrices_per_task": left_c_matrices_per_task,
            "bytes_per_task": left_c_bytes_per_task,
            "bytes_target": left_c_bytes_per_task * tasks,
            "mte2_calls_target_before": left_c_old_calls_target,
            "mte2_calls_target_after": left_c_coalesced_calls_target,
            "mte2_calls_target_saved": (
                left_c_old_calls_target - left_c_coalesced_calls_target
            ),
        },
        "batch_tail_coalescing": {
            "changes_per_element_fp32_order": False,
            "changes_gm_bytes": False,
            "aiv_workers_per_task": tail_aiv_workers_per_task,
            "blocks_per_aiv": tail_blocks_per_aiv,
            "db_input_bytes_target": tail_db_input_bytes_per_task * tasks,
            "feature_input_bytes_target": (
                tail_feature_input_bytes_per_task * tasks
            ),
            "output_bytes_target": tail_output_bytes_per_task * tasks,
            "total_bytes_target": (
                tail_db_input_bytes_per_task
                + tail_feature_input_bytes_per_task
                + tail_output_bytes_per_task
            ) * tasks,
            "db_input_mte2_calls_target_before": tail_db_mte2_calls_before,
            "db_input_mte2_calls_target_after": tail_db_mte2_calls_after,
            "feature_input_mte2_calls_target_before": (
                tail_feature_mte2_calls_before
            ),
            "feature_input_mte2_calls_target_after": (
                tail_feature_mte2_calls_after
            ),
            "total_mte2_calls_target_before": (
                tail_db_mte2_calls_before + tail_feature_mte2_calls_before
            ),
            "total_mte2_calls_target_after": (
                tail_db_mte2_calls_after + tail_feature_mte2_calls_after
            ),
            "vector_expression_calls_target_before": tail_vector_calls_before,
            "vector_expression_calls_target_after": tail_vector_calls_after,
            "mte3_calls_target_before": tail_mte3_calls_before,
            "mte3_calls_target_after": tail_mte3_calls_after,
        },
        "scalar_hot_path_hoisting": {
            "source_level_model": True,
            "changes_fp32_arithmetic": False,
            "changes_vector_instructions": False,
            "changes_gm_bytes": False,
            "task_aiv_pairs_target": task_aiv_pairs,
            "row_start_source_expressions_target_before": (
                task_aiv_pairs * row_start_source_expressions_per_task_before
            ),
            "row_start_source_expressions_target_after": 0,
            "consume_get_subblock_reads_target_before": (
                task_aiv_pairs * consume_subblock_reads_per_task_before
            ),
            "consume_get_subblock_reads_target_after": 0,
            "persistent_gate_loop_iterations_target_before": (
                task_aiv_pairs * persistent_gate_loop_iterations_per_task_before
            ),
            "persistent_gate_loop_iterations_target_after": 0,
            "kernel_entry_get_subblock_reads_unchanged": True,
        },
        "persistent_mmad_scheduling": {
            "source_level_model": True,
            "source_default_enabled": False,
            "changes_fp32_arithmetic": False,
            "changes_logical_gemm_order": False,
            "changes_gm_bytes": False,
            "logical_gemm_calls_target": logical_gemm_calls_target,
            "scoped_envelopes_per_task": SCOPED_MMAD_ENVELOPES_PER_TASK,
            "scoped_envelopes_target": scoped_mmad_envelopes_target,
            "persistent_engines_per_aic": PERSISTENT_MMAD_ENGINES_PER_AIC,
            "persistent_envelopes_target": persistent_mmad_envelopes_target,
            "envelope_reduction": (
                scoped_mmad_envelopes_target
                / persistent_mmad_envelopes_target
            ),
            "scoped_one_envelope_per_logical_gemm": True,
            "persistent_compile_time_experiment_retained": True,
        },
        "pair_scratch_pingpong": {
            "source_default_enabled": False,
            "active": pair_scratch == "pingpong",
            "changes_fp32_arithmetic": False,
            "changes_gm_bytes": False,
            "changes_workspace_layout": False,
            "pairs_per_task": pair_scratch_pairs_per_task,
            "scratch_banks_per_set": PAIR_SCRATCH_BANKS,
            "aiv_workers_per_task": tail_aiv_workers_per_task,
            "mte3_bytes_per_task": pair_scratch_mte3_bytes_per_task,
            "mte3_bytes_target_eligible_for_overlap": (
                pair_scratch_mte3_bytes_per_task * tasks
            ),
            "pair_completion_waits_per_aiv_before": (
                pair_scratch_waits_per_aiv_before
            ),
            "pair_completion_waits_per_aiv_after": (
                pair_scratch_waits_per_aiv_after
            ),
            "pair_completion_waits_target_before": (
                task_aiv_pairs * pair_scratch_waits_per_aiv_before
            ),
            "pair_completion_waits_target_after": (
                task_aiv_pairs * pair_scratch_waits_per_aiv_after
            ),
            "pair_completion_waits_target_saved": (
                task_aiv_pairs
                * (
                    pair_scratch_waits_per_aiv_before
                    - pair_scratch_waits_per_aiv_after
                )
            ),
            "single_ub_bytes": GROUPED_SINGLE_UB_BYTES,
            "pingpong_ub_bytes": GROUPED_PAIR_PINGPONG_UB_BYTES,
            "active_ub_bytes": (
                GROUPED_PAIR_PINGPONG_UB_BYTES
                if pair_scratch == "pingpong"
                else GROUPED_SINGLE_UB_BYTES
            ),
            "extra_ub_bytes": pair_scratch_extra_ub_bytes,
            "a2_ub_headroom_bytes": (
                A2_UB_BYTES - GROUPED_PAIR_PINGPONG_UB_BYTES
            ),
            "baseline_live_mte3_v_event_ids": 1,
            "pingpong_live_mte3_v_event_ids": 3,
            "a2_mte3_v_event_id_capacity": 8,
            "requires_device_profile_to_claim_hidden_bytes": True,
        },
        "aic_l1_a_reuse_candidate": {
            "implemented": False,
            "preserves_aiv_publication": True,
            "preserves_one_aic_read_per_distinct_a_matrix": True,
            "b_reuse_available": False,
            "saved_bytes_per_task": a_l1_reuse_saved_per_task,
            "saved_bytes_target": a_l1_reuse_saved_target,
            "saved_fraction_of_current": (
                a_l1_reuse_saved_target / current_target
            ),
        },
        "a2_cv_direct_paths": {
            "target_arch": "DAV_2201",
            "l0c_to_aiv_ub_supported": False,
            "aiv_ub_to_l1_supported": False,
            "direct_paths_start_at": "DAV_3510",
            "c_workspace_round_trip_still_required": True,
        },
        "workspace": {
            "queue_depth": WORKSPACE_QUEUE_DEPTH,
            "slot_elements": WORKSPACE_SLOT_ELEMENTS,
            "slot_bytes": WORKSPACE_SLOT_ELEMENTS * FP32_BYTES,
            "capacity_bytes_per_logical_core": (
                WORKSPACE_QUEUE_DEPTH * WORKSPACE_SLOT_ELEMENTS * FP32_BYTES
            ),
            "a_bytes_per_task": workspace_a_per_task,
            "b_bytes_per_task": workspace_b_per_task,
            "c_bytes_per_task": workspace_c_per_task,
            "total_bytes_per_task": workspace_per_task,
            "a_bytes_target": workspace_a_target,
            "b_bytes_target": workspace_b_target,
            "c_bytes_target": workspace_c_target,
            "total_bytes_target": workspace_target,
        },
        "tensors": {
            "breakdown_bytes_per_task": tensors,
            "total_bytes_per_task": tensor_per_task,
            "total_bytes_target": tensor_target,
        },
        "total": {
            "bytes_per_task": workspace_per_task + tensor_per_task,
            "bytes_target": current_target,
            "workspace_fraction": workspace_target / current_target,
            "required_gbps_at_target_ms": required_gbps(current_target, target_ms),
            "bandwidth_floors_ms": {
                str(value): floor_ms(current_target, value)
                for value in bandwidth_tbps
            },
        },
        "on_chip_capacity_only": {
            "warning": (
                "The TSCM capacity/layout arithmetic is counterfactual on A2: "
                "DAV_2201 software-emulates AIV UB->TSCM through GM/Matmul KFC."
            ),
            "a2_ub_bytes": A2_UB_BYTES,
            "grouped_single_ub_bytes": GROUPED_SINGLE_UB_BYTES,
            "grouped_pair_pingpong_ub_bytes": GROUPED_PAIR_PINGPONG_UB_BYTES,
            "full_ub_double_bytes": 2 * GROUPED_SINGLE_UB_BYTES,
            "full_ub_double_fits": 2 * GROUPED_SINGLE_UB_BYTES <= A2_UB_BYTES,
            "a2_l1_bytes": A2_L1_BYTES,
            "stage3_a_b_c_live_bytes": stage3_live_bytes,
            "stage3_a_b_c_double_bytes": stage3_double_bytes,
            "persistent_mmad_l1_bytes": PERSISTENT_MMAD_L1_BYTES,
            "stage3_double_plus_mmad_l1_bytes": stage3_double_plus_mmad,
            "stage3_double_plus_mmad_l1_headroom_bytes": (
                A2_L1_BYTES - stage3_double_plus_mmad
            ),
            "stage3_double_plus_mmad_l1_fits": (
                stage3_double_plus_mmad <= A2_L1_BYTES
            ),
        },
        "tscm_ab_double_buffer": {
            "a2_supported_direct_path": False,
            "a2_model_only": True,
            "queue_depth": WORKSPACE_QUEUE_DEPTH,
            "slot_bytes": stage3_ab_bytes,
            "total_bytes": tscm_ab_double_bytes,
            "persistent_mmad_l1_bytes": PERSISTENT_MMAD_L1_BYTES,
            "total_plus_persistent_mmad_bytes": tscm_ab_double_plus_mmad,
            "l1_headroom_bytes": A2_L1_BYTES - tscm_ab_double_plus_mmad,
            "fits_l1": tscm_ab_double_plus_mmad <= A2_L1_BYTES,
            "physical_layout": validate_tscm_layouts(),
        },
        "scenarios": scenarios,
        "stages": [asdict(item) for item in stages],
    }

    # Exact target invariants make source-model drift visible in clean builds.
    if (batch, seqlen, heads, qk_bytes) == (1, 8192, 32, 2):
        assert workspace_per_task == 946_176
        assert workspace_target == 3_875_536_896
        assert tensor_per_task == 293_632
        assert a_l1_reuse_saved_per_task == 20_480
        assert a_l1_reuse_saved_target == 83_886_080
        assert tensor_target == 1_202_716_672
        assert current_target == 5_078_253_568
        assert off_right_bytes_per_task * tasks == 201_326_592
        assert off_right_old_calls_target == 49_152
        assert off_right_coalesced_calls_target == 24_576
        assert right_b_bytes_per_task * tasks == 671_088_640
        assert right_b_old_calls_target == 163_840
        assert right_b_coalesced_calls_target == 81_920
        assert left_c_bytes_per_task * tasks == 469_762_048
        assert left_c_old_calls_target == 114_688
        assert left_c_coalesced_calls_target == 57_344
        assert tail_db_input_bytes_per_task * tasks == 1_048_576
        assert tail_feature_input_bytes_per_task * tasks == 402_653_184
        assert tail_output_bytes_per_task * tasks == 403_701_760
        assert tail_db_mte2_calls_before == 32_768
        assert tail_db_mte2_calls_after == 8_192
        assert tail_feature_mte2_calls_before == 98_304
        assert tail_feature_mte2_calls_after == 24_576
        assert tail_vector_calls_before == 196_608
        assert tail_vector_calls_after == 49_152
        assert tail_mte3_calls_before == 131_072
        assert tail_mte3_calls_after == 32_768
        assert task_aiv_pairs == 8_192
        assert logical_gemm_calls_target == 69_632
        assert scoped_mmad_envelopes_target == 69_632
        assert persistent_mmad_envelopes_target == 40
        assert pair_scratch_pairs_per_task == 6
        assert pair_scratch_mte3_bytes_per_task == 147_456
        assert pair_scratch_mte3_bytes_per_task * tasks == 603_979_776
        assert pair_scratch_waits_per_aiv_before == 6
        assert pair_scratch_waits_per_aiv_after == 3
        assert task_aiv_pairs * pair_scratch_waits_per_aiv_before == 49_152
        assert task_aiv_pairs * pair_scratch_waits_per_aiv_after == 24_576
        assert pair_scratch_extra_ub_bytes == 12_320
        assert (
            task_aiv_pairs * row_start_source_expressions_per_task_before
            == 114_688
        )
        assert task_aiv_pairs * consume_subblock_reads_per_task_before == 32_768
        assert (
            task_aiv_pairs * persistent_gate_loop_iterations_per_task_before
            == 24_576
        )
    return report


def print_text(report: dict[str, object]) -> None:
    target = report["target"]
    implementation = report["implementation"]
    workspace = report["workspace"]
    tensors = report["tensors"]
    total = report["total"]
    scenarios = report["scenarios"]
    capacity = report["on_chip_capacity_only"]
    tscm = report["tscm_ab_double_buffer"]
    off_right = report["off_right_consume_coalescing"]
    right_b = report["right_b_write_coalescing"]
    left_c = report["left_c_read_coalescing"]
    batch_tail = report["batch_tail_coalescing"]
    scalar_hot = report["scalar_hot_path_hoisting"]
    persistent_mmad = report["persistent_mmad_scheduling"]
    pair_scratch = report["pair_scratch_pingpong"]
    a_l1_reuse = report["aic_l1_a_reuse_candidate"]
    cv_paths = report["a2_cv_direct_paths"]

    print(
        "target: "
        f"B={target['batch']} T={target['seqlen']} H={target['heads']} "
        f"BT={target['chunk_size']} K={target['head_dim']} tasks={target['tasks']}"
    )
    print("stage  gemms  A-bytes  B-bytes  C-bytes  workspace-bytes")
    for stage in report["stages"]:
        print(
            f"{stage['stage']:>5}  {stage['logical_gemms']:>5}  "
            f"{stage['a_bytes']:>7,}  {stage['b_bytes']:>7,}  "
            f"{stage['c_bytes']:>7,}  {stage['total_bytes']:>15,}"
        )
    print(
        f"workspace/task={workspace['total_bytes_per_task']:,} B, "
        f"tensor/task={tensors['total_bytes_per_task']:,} B, "
        f"total/target={total['bytes_target'] / 1e9:.6f} GB"
    )
    print(
        f"workspace share={100 * total['workspace_fraction']:.2f}%, "
        f"required@{target['target_ms']:.3f}ms="
        f"{total['required_gbps_at_target_ms']:.3f} GB/s"
    )
    print(
        f"active stage_io={implementation['stage_io']}: "
        f"route={implementation['a2_route']}, "
        f"scenario={implementation['active_scenario']}, "
        f"bytes={implementation['active_bytes_target'] / 1e9:.6f} GB, "
        f"required@{target['target_ms']:.3f}ms="
        f"{implementation['active_required_gbps_at_target_ms']:.3f} GB/s"
    )
    print(
        "off-right consume coalescing: "
        f"bytes={off_right['bytes_target'] / 1e9:.6f} GB unchanged, "
        f"MTE2 calls={off_right['mte2_calls_target_before']:,}->"
        f"{off_right['mte2_calls_target_after']:,}, "
        f"MulAdd calls={off_right['mul_add_calls_target_before']:,}->"
        f"{off_right['mul_add_calls_target_after']:,}"
    )
    print(
        "right-B write coalescing: "
        f"bytes={right_b['bytes_target'] / 1e9:.6f} GB unchanged, "
        f"MTE3 calls={right_b['mte3_calls_target_before']:,}->"
        f"{right_b['mte3_calls_target_after']:,}"
    )
    print(
        "left-C read coalescing: "
        f"bytes={left_c['bytes_target'] / 1e9:.6f} GB unchanged, "
        f"MTE2 calls={left_c['mte2_calls_target_before']:,}->"
        f"{left_c['mte2_calls_target_after']:,}"
    )
    print(
        "batch-tail coalescing: "
        f"bytes={batch_tail['total_bytes_target'] / 1e9:.6f} GB unchanged, "
        f"MTE2 calls={batch_tail['total_mte2_calls_target_before']:,}->"
        f"{batch_tail['total_mte2_calls_target_after']:,}, "
        "Vector expressions="
        f"{batch_tail['vector_expression_calls_target_before']:,}->"
        f"{batch_tail['vector_expression_calls_target_after']:,}, "
        f"MTE3 calls={batch_tail['mte3_calls_target_before']:,}->"
        f"{batch_tail['mte3_calls_target_after']:,}"
    )
    print(
        "Scalar hot-path hoisting (source-level): "
        "rowStart expressions="
        f"{scalar_hot['row_start_source_expressions_target_before']:,}->0, "
        "Consume GetSubBlockIdx reads="
        f"{scalar_hot['consume_get_subblock_reads_target_before']:,}->0, "
        "persistent-gate loop iterations="
        f"{scalar_hot['persistent_gate_loop_iterations_target_before']:,}->0"
    )
    print(
        "persistent MMAD scheduling (source-level experiment): "
        f"logical GEMMs={persistent_mmad['logical_gemm_calls_target']:,} unchanged, "
        f"flag envelopes={persistent_mmad['scoped_envelopes_target']:,}->"
        f"{persistent_mmad['persistent_envelopes_target']:,} "
        f"({persistent_mmad['envelope_reduction']:.1f}x fewer)"
    )
    print(
        "pair-scratch ping-pong (source-level candidate): "
        f"eligible MTE3 bytes={pair_scratch['mte3_bytes_target_eligible_for_overlap'] / 1e9:.6f} GB, "
        "pair completion waits="
        f"{pair_scratch['pair_completion_waits_target_before']:,}->"
        f"{pair_scratch['pair_completion_waits_target_after']:,}, "
        f"UB={pair_scratch['single_ub_bytes']:,}->"
        f"{pair_scratch['pingpong_ub_bytes']:,} B "
        f"(headroom={pair_scratch['a2_ub_headroom_bytes']:,} B); "
        "GM bytes unchanged"
    )
    print(
        "AIC stage-local A reuse candidate (not implemented): "
        f"saves={a_l1_reuse['saved_bytes_target'] / 1e9:.6f} GB "
        f"({100 * a_l1_reuse['saved_fraction_of_current']:.2f}% of current); "
        "B has no repeated consumer"
    )
    print(
        "A2 CV direct paths: "
        f"L0C->AIV_UB={cv_paths['l0c_to_aiv_ub_supported']}, "
        f"AIV_UB->L1={cv_paths['aiv_ub_to_l1_supported']}; "
        "C workspace round trip remains required"
    )
    for name in (
        "current_gm_bridge",
        "aic_l1_a_reuse",
        "a_on_chip",
        "b_on_chip",
        "c_on_chip",
        "ab_on_chip",
        "all_workspace_on_chip",
    ):
        row = scenarios[name]
        print(
            f"{name}: {row['gb_decimal_target']:.6f} GB, "
            f"{row['required_gbps_at_target_ms']:.3f} GB/s"
        )
    print(
        "UB full double: "
        f"{capacity['full_ub_double_bytes']:,}/{capacity['a2_ub_bytes']:,} B, "
        f"fits={capacity['full_ub_double_fits']}"
    )
    print(
        "L1 stage3 A/B/C double + persistent MMAD: "
        f"{capacity['stage3_double_plus_mmad_l1_bytes']:,}/"
        f"{capacity['a2_l1_bytes']:,} B, "
        f"headroom={capacity['stage3_double_plus_mmad_l1_headroom_bytes']:,} B, "
        f"capacity_fits={capacity['stage3_double_plus_mmad_l1_fits']}"
    )
    print(
        "L1 A/B TSCM ideal-only + persistent MMAD: "
        f"{tscm['total_plus_persistent_mmad_bytes']:,}/"
        f"{capacity['a2_l1_bytes']:,} B, "
        f"headroom={tscm['l1_headroom_bytes']:,} B, "
        f"layout_validated={tscm['physical_layout']['validated']}"
    )
    print(f"warning: {capacity['warning']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--qk-bytes", type=int, default=DEFAULT_QK_BYTES)
    parser.add_argument("--target-ms", type=float, default=4.0)
    parser.add_argument("--stage-io", choices=("tscm", "gm"), default="gm")
    parser.add_argument("--used-aic", type=int, default=DEFAULT_USED_AIC)
    parser.add_argument(
        "--pair-scratch",
        choices=("single", "pingpong"),
        default="single",
    )
    parser.add_argument(
        "--bandwidth-tbps",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.2, 1.6],
        help="effective decimal TB/s values used for traffic-only floors",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(
        batch=args.batch,
        seqlen=args.seqlen,
        heads=args.heads,
        qk_bytes=args.qk_bytes,
        target_ms=args.target_ms,
        bandwidth_tbps=args.bandwidth_tbps,
        stage_io=args.stage_io,
        used_aic=args.used_aic,
        pair_scratch=args.pair_scratch,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
