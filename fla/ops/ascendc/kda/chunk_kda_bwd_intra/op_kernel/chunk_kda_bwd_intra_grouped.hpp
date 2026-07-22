/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_KDA_BWD_INTRA_GROUPED_HPP
#define CHUNK_KDA_BWD_INTRA_GROUPED_HPP

constexpr uint32_t KDA_GROUPED_BT = 64;
constexpr uint32_t KDA_GROUPED_BC = 16;
constexpr uint32_t KDA_GROUPED_ROWS_PER_AIV = 8;
constexpr uint32_t KDA_GROUPED_K = 128;
constexpr uint32_t KDA_GROUPED_DATA_BLOCK_BYTES = 32;
constexpr uint32_t KDA_GROUPED_BLOCKS = KDA_GROUPED_BT / KDA_GROUPED_BC;
constexpr uint32_t KDA_GROUPED_STAGES = KDA_GROUPED_BLOCKS;
constexpr uint32_t KDA_GROUPED_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_GROUPED_MAX_OFF_PAIRS = KDA_GROUPED_BLOCKS - 1;
constexpr uint32_t KDA_GROUPED_PAIR_COUNT =
    KDA_GROUPED_MAX_OFF_PAIRS * (KDA_GROUPED_MAX_OFF_PAIRS + 1) / 2;
constexpr uint32_t KDA_GROUPED_PAIR_BRIDGE_ELEMENTS =
    KDA_GROUPED_PAIR_COUNT * KDA_GROUPED_K;
constexpr bool KDA_GROUPED_FACTOR_PAIR_GATES = true;
constexpr bool KDA_GROUPED_OVERLAP_SHARED_SETUP = true;
constexpr bool KDA_GROUPED_OVERLAP_STAGE_EPILOGUE = false;
// Gather the four disjoint physical-AIV row blocks into retired contiguous UB
// banks.  This preserves every element's FP32 expression order while reducing
// the whole-tail MTE2/MTE3 and Vector API submissions by 4x.  The scalar loop
// remains compiled as the clean-wheel rollback path.
constexpr bool KDA_GROUPED_BATCH_TAIL_BLOCKS = true;
// Stage the completed dq/dk/dg rows in the three accumulator banks.  The AIV
// may then prepare the next task's stage 0/1 workspace before draining the
// previous task's outputs, so that the final MTE3 writeback overlaps the next
// task's first Cube stages.  This candidate requires the batched tail layout
// and the whole-task db epilogue; both rollback paths remain compiled.
constexpr bool KDA_GROUPED_OVERLAP_TASK_STORE = false;
constexpr bool KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH = false;
// Keep the left32/right16 CATLASS engines and their disjoint L1/L0/event
// partitions live for the whole AIC Process.  The scoped branch remains a
// compile-time rollback, but is no longer the target-domain default: it would
// recreate and drain eleven MMAD flag envelopes for every 64-token task.
constexpr bool KDA_GROUPED_PERSISTENT_MMAD_ENGINES = true;
// Experimental non-A2 path: Vector builds every stage's A/B operands directly
// in two ping-pong TSCM slots and Cube consumes them from L1.  Keep this off on
// Atlas A2/910B.  On __NPU_ARCH__ == 2201 the AscendC UB->TSCM API is
// software-emulated through GM and requires a registered Matmul KFC client;
// this CATLASS direct-launch kernel has no such client, and the emulation would
// not remove the A/B GM round trip anyway.  The supported A2 path therefore
// retains the two-slot GM bridge below.  C also remains in GM for the AIV
// epilogue.
constexpr bool KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER = false;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
static_assert(!KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER,
              "DAV_2201 must use the explicit two-slot GM stage bridge");
#endif
// Pack the off-diagonal prefix and causal diagonal into one 32x(prefix) A
// slab.  Both GEMMs keep their original references and arithmetic; only two
// redundant AIV MTE3 row-copy submissions per nonzero stage are removed.
constexpr bool KDA_GROUPED_PACK_STAGE_A = false;
// IEEE FP32 is the precision baseline.  HF32 is kept as an explicit clean-wheel
// A/B candidate because it rounds both FP32 Cube inputs before MMAD: it may be
// the main throughput lever on A2, but it must pass the cancellation and
// extreme safe-gate guards before it can become a delivery default.
constexpr bool KDA_GROUPED_USE_HF32_CUBE = false;
// K=128 db reduction keeps the same two 64-element FP32 reduction trees per
// row, but issues the first and second halves across all rows in two calls.
// Keep the row-wise launch path as a clean-wheel rollback candidate.
constexpr bool KDA_GROUPED_COALESCE_DB_REDUCE = true;
// For one late stage, off-right C tiles are equally spaced in GM while the
// corresponding AIV scratch, right accumulators and outer gates are all
// contiguous in UB.  Gather the STAGE physical-AIV row tiles with one strided
// MTE2 request, then issue one contiguous FP32 MulAddDst.  This only coalesces
// independent elementwise work: every destination still evaluates
// `dst += C * right_outer` with the same operands and FP32 instruction.
constexpr bool KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME = true;
// The two right-hand B halves are adjacent in pair scratch and occupy two
// eight-row windows of one 32xK GM matrix.  One strided UB->GM request replaces
// the two independent CopyOut submissions without changing either row image.
constexpr bool KDA_GROUPED_COALESCE_RIGHT_B_WRITES = true;
// A physical AIV owns two eight-row windows of every 32xK left C result.
// Gather both windows into adjacent scratch banks with one strided GM->UB
// request; the following dq and dk-left arithmetic remains independent.
constexpr bool KDA_GROUPED_COALESCE_LEFT_C_READS = true;
// All grouped safe-gate Vector operands are FP32 and every optimized span is
// an exact multiple of one 64-element repeat.  Reuse one externally configured
// normal mask inside each local arithmetic envelope instead of making every
// Level-2 API call reprogram the mask/counter state.  The low-level calls keep
// the same instruction, operand order, repeat grouping and FP32 arithmetic.
constexpr bool KDA_GROUPED_REUSE_VECTOR_MASK = true;
constexpr uint32_t KDA_GROUPED_FP32_REPEAT_ELEMENTS = 64;
constexpr uint32_t KDA_GROUPED_SELECTED_ROWS =
    KDA_GROUPED_BLOCKS * KDA_GROUPED_ROWS_PER_AIV;
constexpr uint32_t KDA_GROUPED_SELECTED_ELEMENTS =
    KDA_GROUPED_SELECTED_ROWS * KDA_GROUPED_K;
constexpr uint32_t KDA_GROUPED_BETA_BRCB_ELEMENTS =
    KDA_GROUPED_SELECTED_ROWS * 8;
constexpr uint32_t KDA_GROUPED_ROW_BLOCK_ELEMENTS =
    KDA_GROUPED_ROWS_PER_AIV * KDA_GROUPED_K;
static_assert(KDA_GROUPED_BC == 2 * KDA_GROUPED_ROWS_PER_AIV,
              "Two grouped AIVs must partition each 16-row block exactly");
static_assert(!KDA_GROUPED_OVERLAP_TASK_STORE ||
                  KDA_GROUPED_BATCH_TAIL_BLOCKS,
              "Cross-task store overlap requires contiguous batched tail rows");
static_assert(!KDA_GROUPED_OVERLAP_TASK_STORE ||
                  !KDA_GROUPED_OVERLAP_STAGE_EPILOGUE,
              "Cross-task store overlap requires the whole-task db epilogue");
constexpr uint32_t KDA_GROUPED_RIGHT_OUTER_ELEMENTS =
    KDA_GROUPED_MAX_OFF_PAIRS * KDA_GROUPED_ROW_BLOCK_ELEMENTS;
constexpr uint32_t KDA_GROUPED_RIGHT_PAIR_ELEMENTS =
    2 * KDA_GROUPED_BC * KDA_GROUPED_K;

// The rollback layout keeps each stage's off-diagonal and diagonal A matrices
// separate.  Aoff and Adiag are each written once from the same UB dA slab,
// then reinterpreted as RowMajor for the left GEMM and ColumnMajor for the
// right GEMM.  KDA_GROUPED_PACK_STAGE_A selects the alternate packed layout
// below without changing either mathematical matrix.
constexpr uint32_t KDA_GROUPED_B_OFF_LEFT = 0;
constexpr uint32_t KDA_GROUPED_A_OFF = 6144;
constexpr uint32_t KDA_GROUPED_B_OFF_RIGHT = 7680;
constexpr uint32_t KDA_GROUPED_B_DIAG_RIGHT = 19968;
constexpr uint32_t KDA_GROUPED_A_DIAG = 24064;
constexpr uint32_t KDA_GROUPED_B_DIAG_LEFT = 24576;
constexpr uint32_t KDA_GROUPED_SLOT_ELEMENTS = 26624;
constexpr uint32_t KDA_GROUPED_SLOT_BYTES =
    KDA_GROUPED_SLOT_ELEMENTS * sizeof(float);
// Integer TSCM mask bit 2 maps VECTOR0 and VECTOR1 onto the same Cube L1
// addresses.  The two AIVs write disjoint eight-row halves of each matrix.
constexpr uint32_t KDA_GROUPED_TSCM_BLOCK_GROUP_MASK = 1U << 2;

// Outputs deliberately overwrite inputs that have already been fully loaded
// into L1.  Keep this exact call order on AIC:
// off-left -> off-right -> diag-right -> diag-left.
constexpr uint32_t KDA_GROUPED_C_OFF_LEFT = 0;
constexpr uint32_t KDA_GROUPED_C_DIAG_RIGHT = 4096;
constexpr uint32_t KDA_GROUPED_C_OFF_RIGHT = KDA_GROUPED_B_OFF_RIGHT;
constexpr uint32_t KDA_GROUPED_C_DIAG_LEFT = KDA_GROUPED_B_DIAG_RIGHT;

// Packed-A retains the existing Boff-left and Bdiag-left capacities.  The
// maximum 32x64 A slab consumes exactly Aoff+Adiag's former total space, so
// shifting the two right-B regions by 512 elements preserves the slot size.
constexpr uint32_t KDA_GROUPED_PACKED_A = 6144;
constexpr uint32_t KDA_GROUPED_PACKED_B_OFF_RIGHT = 8192;
constexpr uint32_t KDA_GROUPED_PACKED_B_DIAG_RIGHT = 20480;
constexpr uint32_t KDA_GROUPED_PACKED_C_OFF_RIGHT =
    KDA_GROUPED_PACKED_B_OFF_RIGHT;
constexpr uint32_t KDA_GROUPED_PACKED_C_DIAG_LEFT =
    KDA_GROUPED_PACKED_B_DIAG_RIGHT;

static_assert(KDA_GROUPED_A_OFF == 48 * KDA_GROUPED_K,
              "Boff-left must reserve the maximum 48 source rows");
static_assert(KDA_GROUPED_B_OFF_RIGHT == KDA_GROUPED_A_OFF + 32 * 48,
              "Aoff must reserve the maximum compact 32x48 matrix");
static_assert(KDA_GROUPED_B_DIAG_RIGHT == KDA_GROUPED_B_OFF_RIGHT +
                  KDA_GROUPED_MAX_OFF_PAIRS * KDA_GROUPED_RIGHT_PAIR_ELEMENTS,
              "Boff-right must reserve three independent 32x128 pair matrices");
static_assert(KDA_GROUPED_A_DIAG == KDA_GROUPED_B_DIAG_RIGHT + 32 * KDA_GROUPED_K,
              "Bdiag-right must reserve 32x128 elements");
static_assert(KDA_GROUPED_B_DIAG_LEFT == KDA_GROUPED_A_DIAG + 32 * KDA_GROUPED_BC,
              "Adiag must reserve 32x16 elements");
static_assert(KDA_GROUPED_SLOT_ELEMENTS == KDA_GROUPED_B_DIAG_LEFT +
                  KDA_GROUPED_BC * KDA_GROUPED_K,
              "Bdiag-left must end exactly at the slot boundary");
static_assert(KDA_GROUPED_C_DIAG_RIGHT + KDA_GROUPED_BC * KDA_GROUPED_K ==
                  KDA_GROUPED_A_OFF,
              "Diagonal-right output must reuse only retired Boff-left storage");
static_assert(KDA_GROUPED_C_DIAG_LEFT + 2 * KDA_GROUPED_BC * KDA_GROUPED_K ==
                  KDA_GROUPED_A_DIAG,
              "Diagonal-left output must reuse only retired Bdiag-right storage");
static_assert(KDA_GROUPED_C_OFF_RIGHT == KDA_GROUPED_B_OFF_RIGHT,
              "Each off-right output must overwrite its own retired B pair");
static_assert(KDA_GROUPED_PACKED_A == KDA_GROUPED_A_OFF,
              "Packed A must begin at the retired Aoff boundary");
static_assert(KDA_GROUPED_PACKED_B_OFF_RIGHT ==
                  KDA_GROUPED_PACKED_A +
                      2 * KDA_GROUPED_BC * KDA_GROUPED_BT,
              "Packed A must reserve one maximum 32x64 matrix");
static_assert(KDA_GROUPED_PACKED_B_DIAG_RIGHT ==
                  KDA_GROUPED_PACKED_B_OFF_RIGHT +
                      KDA_GROUPED_MAX_OFF_PAIRS *
                          KDA_GROUPED_RIGHT_PAIR_ELEMENTS,
              "Packed off-right storage must retain three 32x128 pairs");
static_assert(KDA_GROUPED_B_DIAG_LEFT ==
                  KDA_GROUPED_PACKED_B_DIAG_RIGHT +
                      2 * KDA_GROUPED_BC * KDA_GROUPED_K,
              "Packed diagonal-right B must end at Bdiag-left");
static_assert(KDA_GROUPED_C_DIAG_RIGHT +
                      KDA_GROUPED_BC * KDA_GROUPED_K ==
                  KDA_GROUPED_PACKED_A,
              "Packed diagonal-right output must end at packed A");
static_assert(KDA_GROUPED_PACKED_C_OFF_RIGHT ==
                  KDA_GROUPED_PACKED_B_OFF_RIGHT,
              "Packed off-right output must overwrite its retired B pair");
static_assert(KDA_GROUPED_PACKED_C_DIAG_LEFT +
                      2 * KDA_GROUPED_BC * KDA_GROUPED_K ==
                  KDA_GROUPED_B_DIAG_LEFT,
              "Packed diagonal-left output must overwrite only Bdiag-right");
static_assert(KDA_GROUPED_SLOT_BYTES % 512 == 0,
              "Each grouped workspace slot must be 512-byte aligned");
static_assert(KDA_GROUPED_STAGES % KDA_GROUPED_QUEUE_DEPTH == 0,
              "Reverse flag phases must return to zero at every task boundary");
static_assert((KDA_GROUPED_BC * sizeof(float)) % KDA_GROUPED_DATA_BLOCK_BYTES == 0,
              "Grouped UB row strides must be whole 32-byte data blocks");

// A single UB slab gives deterministic relative bank-group coloring on A2.
// Offsets 0/128/256/384 modulo 512 bytes map gate/cache/scratch/accumulator
// starts to four different bank groups while preserving 32-byte alignment.
constexpr uint32_t KDA_GROUPED_Q_TYPED_UB = 0;
constexpr uint32_t KDA_GROUPED_K_TYPED_UB = 8192;
constexpr uint32_t KDA_GROUPED_Q_CACHE_UB = 16512;
constexpr uint32_t KDA_GROUPED_K_CACHE_UB = 32896;
constexpr uint32_t KDA_GROUPED_K_BETA_CACHE_UB = 49280;
constexpr uint32_t KDA_GROUPED_G_CACHE_UB = 65664;
// Only F1..F3, M0..M3 and R0..R2 are consumed.  Keep the 32 input-db rows in
// the 128-byte prefix after gCache, then compact those ten references with one
// 128-byte class pad: first/middle references use color 8 and right references
// use color 12.  This keeps direct row Sub operands distinct from color-4
// gCache/color-0 destinations and removes the color-8/color-8 dual read from
// every pair-bridge Sub.
constexpr uint32_t KDA_GROUPED_DB_LOCAL_UB = 82048;
constexpr uint32_t KDA_GROUPED_REF_UB = 82176;
constexpr uint32_t KDA_GROUPED_FIRST_REF_COUNT = KDA_GROUPED_BLOCKS - 1;
constexpr uint32_t KDA_GROUPED_MIDDLE_REF_COUNT = KDA_GROUPED_BLOCKS;
constexpr uint32_t KDA_GROUPED_RIGHT_REF_COUNT = KDA_GROUPED_BLOCKS - 1;
constexpr uint32_t KDA_GROUPED_MIDDLE_REF_OFFSET =
    KDA_GROUPED_FIRST_REF_COUNT * KDA_GROUPED_K;
constexpr uint32_t KDA_GROUPED_RIGHT_REF_PADDING_ELEMENTS = 32;
constexpr uint32_t KDA_GROUPED_RIGHT_REF_OFFSET =
    (KDA_GROUPED_FIRST_REF_COUNT + KDA_GROUPED_MIDDLE_REF_COUNT) *
        KDA_GROUPED_K + KDA_GROUPED_RIGHT_REF_PADDING_ELEMENTS;
constexpr uint32_t KDA_GROUPED_REF_VALUE_ELEMENTS =
    (KDA_GROUPED_FIRST_REF_COUNT + KDA_GROUPED_MIDDLE_REF_COUNT +
     KDA_GROUPED_RIGHT_REF_COUNT) * KDA_GROUPED_K;
constexpr uint32_t KDA_GROUPED_REF_STORAGE_ELEMENTS =
    KDA_GROUPED_REF_VALUE_ELEMENTS + KDA_GROUPED_RIGHT_REF_PADDING_ELEMENTS;
constexpr uint32_t KDA_GROUPED_BETA_UB = 88192;
constexpr uint32_t KDA_GROUPED_GATE_SLOTS_UB = 88576;
constexpr uint32_t KDA_GROUPED_OPERAND0_UB = 104960;
constexpr uint32_t KDA_GROUPED_OPERAND1_UB = 109056;
constexpr uint32_t KDA_GROUPED_DQ_ACC_UB = 113536;
constexpr uint32_t KDA_GROUPED_DK_LEFT_ACC_UB = 129920;
// Move dk-right from color 12 to color 4.  The final dg expression reads
// dk-left and dk-right together; distinct colors avoid a DAV_2201 read/read
// bank-group conflict without changing any arithmetic or accumulator lifetime.
constexpr uint32_t KDA_GROUPED_DK_RIGHT_PADDING_BYTES = 256;
constexpr uint32_t KDA_GROUPED_DK_RIGHT_ACC_UB =
    KDA_GROUPED_DK_LEFT_ACC_UB +
    KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) +
    KDA_GROUPED_DK_RIGHT_PADDING_BYTES;
constexpr uint32_t KDA_GROUPED_ACC_ZERO_ELEMENTS =
    (KDA_GROUPED_DK_RIGHT_ACC_UB +
     KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) -
     KDA_GROUPED_DQ_ACC_UB) / sizeof(float);
// The three-argument Duplicate overload takes an FP32 element count and
// internally emits at most 255 256-byte vector repeats.  Keep the combined
// accumulator clear below that documented limit so later UB recoloring cannot
// silently overflow into the explicit uint8_t repeat-time constraint.
constexpr uint32_t KDA_GROUPED_FP32_DUPLICATE_MAX_ELEMENTS = 64 * 255;
constexpr uint32_t KDA_GROUPED_RESULT_UB = 163072;
constexpr uint32_t KDA_GROUPED_RAW_DA_UB = 171264;
constexpr uint32_t KDA_GROUPED_REDUCE_UB = 175616;
constexpr uint32_t KDA_GROUPED_BETA_BRCB_UB = KDA_GROUPED_REDUCE_UB;
// Keep the K-wide bridge broadcast source on color 4. Pair products write
// color-8 scratch while reading color-0 outer gates, so all three operands of
// the explicit repeat-stride broadcast use different A2 UB bank groups.  Its
// bridge-building Sub now reads color-8 first and color-12 right references.
constexpr uint32_t KDA_GROUPED_PAIR_BRIDGE_PADDING_BYTES = 128;
constexpr uint32_t KDA_GROUPED_PAIR_BRIDGE_UB =
    KDA_GROUPED_BETA_BRCB_UB +
    KDA_GROUPED_BETA_BRCB_ELEMENTS * sizeof(float) +
    KDA_GROUPED_PAIR_BRIDGE_PADDING_BYTES;
constexpr uint32_t KDA_GROUPED_TAIL_UB = KDA_GROUPED_PAIR_BRIDGE_UB +
    KDA_GROUPED_PAIR_BRIDGE_ELEMENTS * sizeof(float);
// The triangular Select state depends only on the physical AIV half, not on
// batch/chunk/head.  Keep it after the six pair bridges so it survives every
// task instead of rebuilding it in retired typed storage after each Cast.
constexpr uint32_t KDA_GROUPED_CAUSAL_MASK_UB = KDA_GROUPED_TAIL_UB;
constexpr uint32_t KDA_GROUPED_CAUSAL_MASK_BYTES =
    KDA_GROUPED_ROWS_PER_AIV * sizeof(uint64_t);
constexpr uint32_t KDA_GROUPED_CAUSAL_ZERO_UB =
    KDA_GROUPED_CAUSAL_MASK_UB + KDA_GROUPED_CAUSAL_MASK_BYTES;
constexpr uint32_t KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES =
    KDA_GROUPED_CAUSAL_ZERO_UB + 8 * sizeof(float);
// The pair-output pipeline needs only a second set of its three 4 KiB
// scratch banks, not a copy of the full 179,936-byte UB data path. Start the
// alternate set at color 8 (256 modulo 512 bytes), matching the original
// RESULT/RAW_DA scratch coloring and leaving 4,352 bytes below the A2 limit.
constexpr uint32_t KDA_GROUPED_PAIR_SCRATCH_ALT_UB =
    KDA_GROUPED_TAIL_UB + 128;
constexpr uint32_t KDA_GROUPED_PAIR_SCRATCH_ALT_BYTES =
    3 * KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
constexpr uint32_t KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES =
    KDA_GROUPED_PAIR_SCRATCH_ALT_UB + KDA_GROUPED_PAIR_SCRATCH_ALT_BYTES;
constexpr uint32_t KDA_GROUPED_UB_BYTES =
    KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH ?
        KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES :
        KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES;
// Optional tail batching gathers the four disjoint 8x128 row blocks into
// retired UB banks, runs each elementwise expression once over 4096 elements,
// then scatters the four blocks back to GM.  These are aliases only: the
// compile-time experiment does not increase the grouped-path UB footprint.
constexpr uint32_t KDA_GROUPED_BATCH_OUT_Q_UB = KDA_GROUPED_Q_TYPED_UB;
constexpr uint32_t KDA_GROUPED_BATCH_OUT_K_UB = KDA_GROUPED_REF_UB;
constexpr uint32_t KDA_GROUPED_BATCH_OUT_G_UB = KDA_GROUPED_K_BETA_CACHE_UB;
constexpr uint32_t KDA_GROUPED_BATCH_ROW_TMP_UB = KDA_GROUPED_RESULT_UB;
// The overlap-stage-epilogue experiment uses the compact reference cache's
// 128-byte prefix for persistent db rows and its 896-byte suffix for one
// block's reduction scratch.  REDUCE_UB holds the once-per-task beta broadcast
// until all four blocks have been finalized.
constexpr uint32_t KDA_GROUPED_DB_STAGE_SCRATCH_PADDING_BYTES = 384;
constexpr uint32_t KDA_GROUPED_DB_STAGE_SCRATCH_UB = KDA_GROUPED_REF_UB +
    KDA_GROUPED_REF_STORAGE_ELEMENTS * sizeof(float) +
    KDA_GROUPED_DB_STAGE_SCRATCH_PADDING_BYTES;
constexpr uint32_t KDA_GROUPED_DB_STAGE_ACC_ELEMENTS =
    KDA_GROUPED_ROWS_PER_AIV * 8;

static_assert(KDA_GROUPED_REF_UB == KDA_GROUPED_DB_LOCAL_UB +
                  KDA_GROUPED_SELECTED_ROWS * sizeof(float) &&
                  KDA_GROUPED_REF_UB % 512 == 256,
              "Compact reference cache must start after db rows on UB color 8");
static_assert(KDA_GROUPED_FIRST_REF_COUNT == 3 &&
                  KDA_GROUPED_MIDDLE_REF_COUNT == 4 &&
                  KDA_GROUPED_RIGHT_REF_COUNT == 3 &&
                  KDA_GROUPED_REF_VALUE_ELEMENTS == 10 * KDA_GROUPED_K &&
                  KDA_GROUPED_REF_STORAGE_ELEMENTS ==
                      10 * KDA_GROUPED_K + 32,
              "Compact reference cache must contain exactly F1..F3/M0..M3/R0..R2");
static_assert((KDA_GROUPED_REF_UB +
                   KDA_GROUPED_RIGHT_REF_OFFSET * sizeof(float)) % 512 == 384,
              "Right references must use color 12 for pair-bridge Sub");
static_assert(KDA_GROUPED_Q_TYPED_UB % 512 == 0 &&
                  KDA_GROUPED_GATE_SLOTS_UB % 512 == 0 &&
                  KDA_GROUPED_G_CACHE_UB % 512 == 128 &&
                  KDA_GROUPED_REF_UB % 512 == 256,
              "Direct reference Sub operands must use distinct UB bank colors");
static_assert(KDA_GROUPED_BETA_UB >= KDA_GROUPED_REF_UB +
                  KDA_GROUPED_REF_STORAGE_ELEMENTS * sizeof(float),
              "Reference cache must hold the ten consumed first/middle/last rows");
static_assert(KDA_GROUPED_OPERAND0_UB == KDA_GROUPED_GATE_SLOTS_UB +
                  4 * KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float),
              "Gate slots must hold four stage-parity gate banks");
static_assert(KDA_GROUPED_OPERAND1_UB == KDA_GROUPED_OPERAND0_UB +
                  KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float),
              "Operand-backed gate banks must remain contiguous");
static_assert(KDA_GROUPED_DK_LEFT_ACC_UB == KDA_GROUPED_DQ_ACC_UB +
                  KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) &&
                  KDA_GROUPED_DK_RIGHT_ACC_UB == KDA_GROUPED_DK_LEFT_ACC_UB +
                      KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) +
                      KDA_GROUPED_DK_RIGHT_PADDING_BYTES &&
                  KDA_GROUPED_DK_RIGHT_ACC_UB +
                      KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                      KDA_GROUPED_RESULT_UB,
              "Grouped accumulators and dk-right color pad must remain disjoint");
static_assert(KDA_GROUPED_ACC_ZERO_ELEMENTS ==
                  3 * KDA_GROUPED_SELECTED_ELEMENTS + 64,
              "Accumulator zero span must include the 256-byte color pad");
static_assert(KDA_GROUPED_ACC_ZERO_ELEMENTS <=
                  KDA_GROUPED_FP32_DUPLICATE_MAX_ELEMENTS,
              "Combined FP32 Duplicate clear must fit the count overload");
static_assert(KDA_GROUPED_REDUCE_UB >= KDA_GROUPED_RAW_DA_UB +
                   2 * KDA_GROUPED_ROWS_PER_AIV * KDA_GROUPED_BT * sizeof(float),
               "Raw dA scratch must hold two maximum-width row slabs");
static_assert(KDA_GROUPED_Q_CACHE_UB >= KDA_GROUPED_Q_TYPED_UB +
                   KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float),
               "Retired typed storage must hold one full output-reduction product");
static_assert(KDA_GROUPED_G_CACHE_UB >= KDA_GROUPED_K_BETA_CACHE_UB +
                   KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float),
               "Retired k-beta storage must hold one full staged dk input");
static_assert(KDA_GROUPED_CAUSAL_MASK_UB == KDA_GROUPED_TAIL_UB &&
                  KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES ==
                      KDA_GROUPED_CAUSAL_ZERO_UB + 8 * sizeof(float),
              "Persistent causal Select state must exactly end the single slab");
static_assert(KDA_GROUPED_CAUSAL_ZERO_UB + 8 * sizeof(float) <=
                  KDA_GROUPED_PAIR_SCRATCH_ALT_UB,
              "Persistent causal Select state must precede alternate scratch");
static_assert(KDA_GROUPED_DB_LOCAL_UB +
                  KDA_GROUPED_SELECTED_ROWS * sizeof(float) <=
                  KDA_GROUPED_REF_UB,
              "Persistent db rows must fit before the compact reference cache");
static_assert(KDA_GROUPED_DB_STAGE_SCRATCH_UB +
                   (KDA_GROUPED_DB_STAGE_ACC_ELEMENTS +
                    KDA_GROUPED_ROWS_PER_AIV) * sizeof(float) <=
                   KDA_GROUPED_BETA_UB,
              "Stage db reduction scratch must fit after the compact reference cache");
static_assert(KDA_GROUPED_DB_STAGE_SCRATCH_UB % 512 == 256,
              "Stage db scratch must keep dbCompact distinct from dbLocal");
static_assert(KDA_GROUPED_DB_LOCAL_UB % KDA_GROUPED_DATA_BLOCK_BYTES == 0 &&
                  KDA_GROUPED_DB_STAGE_SCRATCH_UB %
                      KDA_GROUPED_DATA_BLOCK_BYTES == 0 &&
                  KDA_GROUPED_BETA_BRCB_UB %
                      KDA_GROUPED_DATA_BLOCK_BYTES == 0,
              "Stage epilogue aliases must remain 32-byte aligned");
static_assert(KDA_GROUPED_BETA_BRCB_UB +
                   KDA_GROUPED_BETA_BRCB_ELEMENTS * sizeof(float) +
                   KDA_GROUPED_PAIR_BRIDGE_PADDING_BYTES ==
                   KDA_GROUPED_PAIR_BRIDGE_UB,
              "Persistent beta broadcast and bank padding must precede pair bridges");
static_assert(KDA_GROUPED_BETA_UB >= KDA_GROUPED_REF_UB +
                   (KDA_GROUPED_SELECTED_ROWS +
                    KDA_GROUPED_SELECTED_ROWS * 8 +
                    KDA_GROUPED_SELECTED_ROWS +
                    KDA_GROUPED_BETA_BRCB_ELEMENTS) * sizeof(float),
              "Retired reference storage must hold fused db and beta scratch");
static_assert(KDA_GROUPED_PAIR_COUNT == 6 &&
                  KDA_GROUPED_PAIR_BRIDGE_UB % 512 == 128,
              "Six K-wide pair bridges must start on UB color 4");
static_assert(KDA_GROUPED_SINGLE_SCRATCH_UB_BYTES == 179936,
              "Update grouped single-scratch UB accounting after layout changes");
static_assert(KDA_GROUPED_PAIR_SCRATCH_ALT_UB % 512 == 256,
              "Alternate pair scratch must preserve the original color-8 phase");
static_assert(KDA_GROUPED_PAIR_SCRATCH_ALT_BYTES == 3 * 4096,
              "Pair scratch ping-pong must reserve exactly three 8x128 banks");
static_assert(KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES == 192256,
              "Update grouped pair-scratch UB accounting after layout changes");
static_assert(KDA_GROUPED_DOUBLE_SCRATCH_UB_BYTES <= 192 * 1024,
              "Pair scratch ping-pong exceeds the A2 UB budget");
static_assert(KDA_GROUPED_BATCH_OUT_Q_UB +
                  KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                  KDA_GROUPED_Q_CACHE_UB,
              "Batched dq must reuse only retired typed/mask storage");
static_assert(KDA_GROUPED_BATCH_OUT_K_UB +
                   KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                   KDA_GROUPED_OPERAND0_UB,
               "Batched dk must fit after db rows in retired reference/gate storage");
static_assert(KDA_GROUPED_BATCH_OUT_G_UB +
                   KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                   KDA_GROUPED_G_CACHE_UB,
               "Batched dg must fit in the retired k*beta cache");
static_assert(KDA_GROUPED_BATCH_OUT_G_UB +
                  KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                  KDA_GROUPED_BATCH_OUT_K_UB,
              "Batched dk and dg aliases must remain disjoint");
static_assert(KDA_GROUPED_BATCH_ROW_TMP_UB +
                  KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float) <=
                  KDA_GROUPED_TAIL_UB,
              "Batched row scratch must fit in retired tail scratch");
static_assert(KDA_GROUPED_Q_TYPED_UB % 512 == 0 && KDA_GROUPED_K_TYPED_UB % 512 == 0,
              "Typed buffers double as color-0 gate banks after Cast");
static_assert(KDA_GROUPED_Q_CACHE_UB % 512 == 128 &&
                  KDA_GROUPED_K_CACHE_UB % 512 == 128 &&
                  KDA_GROUPED_K_BETA_CACHE_UB % 512 == 128 &&
                  KDA_GROUPED_G_CACHE_UB % 512 == 128,
              "Feature caches must share color 4");
static_assert(KDA_GROUPED_GATE_SLOTS_UB % 512 == 0 &&
                   KDA_GROUPED_OPERAND0_UB % 512 == 0 &&
                   KDA_GROUPED_OPERAND1_UB % 512 == 0,
              "All six stage-parity gate banks must share color 0");
static_assert(KDA_GROUPED_DQ_ACC_UB % 512 == 384 &&
                  KDA_GROUPED_DK_LEFT_ACC_UB % 512 == 384 &&
                  KDA_GROUPED_DK_RIGHT_ACC_UB % 512 == 128,
              "dq/dk-left must use color 12 and dk-right color 4");
static_assert(KDA_GROUPED_RESULT_UB % 512 == 256 &&
                  KDA_GROUPED_RAW_DA_UB % 512 == 256,
              "Result and raw scratch buffers must use color 8");
static_assert(KDA_GROUPED_RAW_DA_UB == KDA_GROUPED_RESULT_UB +
                  2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float),
              "Three coalesced off-right scratch tiles must remain contiguous");
static_assert(KDA_GROUPED_K_TYPED_UB == KDA_GROUPED_Q_TYPED_UB +
                  2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float) &&
                  KDA_GROUPED_K_TYPED_UB +
                          KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float) <=
                      KDA_GROUPED_Q_CACHE_UB,
              "Three persistent right-outer gate tiles must remain contiguous");
static_assert(KDA_GROUPED_UB_BYTES <= 192 * 1024,
              "ChunkKdaBwdIntra grouped path exceeds the A2 UB budget");

constexpr uint8_t KDA_GROUPED_DONE_FLAG0 = 2;
constexpr uint8_t KDA_GROUPED_DONE_FLAG1 = 3;
constexpr uint8_t KDA_GROUPED_READY_FLAG0 = 4;
constexpr uint8_t KDA_GROUPED_READY_FLAG1 = 5;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaBwdGroupedArchTag = Catlass::Arch::Ascend950;
#else
using KdaBwdGroupedArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaBwdGroupedDispatchPolicy =
    Catlass::Gemm::MmadPingpongTlaMulti<
        KdaBwdGroupedArchTag, true, KDA_GROUPED_USE_HF32_CUBE>;
static_assert(KdaBwdGroupedDispatchPolicy::USE_HF32_MODE ==
                  KDA_GROUPED_USE_HF32_CUBE,
              "ChunkKdaBwdIntra grouped Cube mode must match its A/B switch");

using KdaBwdGroupedLeftTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
    KdaBwdGroupedArchTag, float, Catlass::layout::RowMajor, float,
    Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;
using KdaBwdGroupedRightTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
    KdaBwdGroupedArchTag, float, Catlass::layout::ColumnMajor, float,
    Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;

template <uint32_t M, uint32_t K>
using KdaBwdGroupedTileShape =
    tla::Shape<tla::Int<M>, tla::Int<KDA_GROUPED_K>, tla::Int<K>>;

template <uint32_t K>
using KdaBwdGroupedLeftBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
    KdaBwdGroupedDispatchPolicy, KdaBwdGroupedTileShape<32, K>,
    KdaBwdGroupedTileShape<32, K>, float, float, float, void,
    KdaBwdGroupedLeftTileCopy>;

template <uint32_t M>
using KdaBwdGroupedRightBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
    KdaBwdGroupedDispatchPolicy, KdaBwdGroupedTileShape<M, 32>,
    KdaBwdGroupedTileShape<M, 32>, float, float, float, void,
    KdaBwdGroupedRightTileCopy>;

constexpr uint32_t KDA_GROUPED_LOCAL_EVENT_CAPACITY = 8;
constexpr uint32_t KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT = 512;

// The repository-local MmadPingpongTlaMulti extension deliberately keeps flag
// setup/drain out of its constructor/destructor and exposes preSetFlags() /
// finalWaitFlags().  Its BlockMmad constructor accepts only an L1 start address.
// This operator-private wrapper keeps those contracts, then partitions
// L0A/L0B/L0C and the local hard-event IDs as well.  No CATLASS-wide API or
// specialization is changed.
template <typename BaseMmad, uint32_t L1_BASE, uint32_t L0A_BASE,
          uint32_t L0B_BASE, uint32_t L0C_BASE, uint32_t EVENT_BASE>
struct KdaBwdGroupedPartitionedBlockMmad : BaseMmad {
    using ArchTag = typename BaseMmad::ArchTag;
    using ElementA = typename BaseMmad::ElementA;
    using ElementB = typename BaseMmad::ElementB;
    using ElementAccumulator = typename BaseMmad::ElementAccumulator;

    static constexpr uint32_t L1_BYTES =
        BaseMmad::L1A_TILE_SIZE * BaseMmad::L1A_STAGES +
        BaseMmad::L1B_TILE_SIZE * BaseMmad::L1B_STAGES;
    static constexpr uint32_t L0A_BYTES =
        BaseMmad::L0A_TILE_SIZE * BaseMmad::L0A_STAGES;
    static constexpr uint32_t L0B_BYTES =
        BaseMmad::L0B_TILE_SIZE * BaseMmad::L0B_STAGES;
    static constexpr uint32_t L0C_BYTES =
        BaseMmad::L0C_TILE_SIZE * BaseMmad::L0C_STAGES;

    static_assert(BaseMmad::ENABLE_UNIT_FLAG,
                  "Persistent grouped MMAD requires unit-flag completion");
    static_assert(!BaseMmad::HAS_BIAS,
                  "Persistent grouped MMAD does not reserve bias resources");
    static_assert(!BaseMmad::ENABLE_L1_RESIDENT,
                  "Grouped workspace slots are rewritten between MMAD calls");
    static_assert(BaseMmad::L1A_STAGES == 2 && BaseMmad::L1B_STAGES == 2 &&
                      BaseMmad::L0A_STAGES == 2 && BaseMmad::L0B_STAGES == 2 &&
                      BaseMmad::L0C_STAGES == 1,
                  "Persistent grouped MMAD requires the audited ping-pong geometry");
    static_assert(L1_BASE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      L0A_BASE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      L0B_BASE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      L0C_BASE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0,
                  "Persistent grouped MMAD bases must be 512-byte aligned");
    static_assert(BaseMmad::L1A_TILE_SIZE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      BaseMmad::L1B_TILE_SIZE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      BaseMmad::L0A_TILE_SIZE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      BaseMmad::L0B_TILE_SIZE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0 &&
                      BaseMmad::L0C_TILE_SIZE % KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT == 0,
                  "Persistent grouped MMAD tiles must preserve 512-byte partitions");
    static_assert(L1_BASE + L1_BYTES <= ArchTag::L1_SIZE,
                  "Persistent grouped MMAD exceeds L1 capacity");
    static_assert(L0A_BASE + L0A_BYTES <= ArchTag::L0A_SIZE,
                  "Persistent grouped MMAD exceeds L0A capacity");
    static_assert(L0B_BASE + L0B_BYTES <= ArchTag::L0B_SIZE,
                  "Persistent grouped MMAD exceeds L0B capacity");
    static_assert(L0C_BASE + L0C_BYTES <= ArchTag::L0C_SIZE,
                  "Persistent grouped MMAD exceeds L0C capacity");
    static_assert(EVENT_BASE + BaseMmad::L1A_STAGES +
                          BaseMmad::L1B_STAGES <=
                      KDA_GROUPED_LOCAL_EVENT_CAPACITY,
                  "Persistent grouped MMAD exceeds L1 hard-event IDs");
    static_assert(EVENT_BASE + BaseMmad::L0A_STAGES +
                          BaseMmad::L0B_STAGES <=
                      KDA_GROUPED_LOCAL_EVENT_CAPACITY,
                  "Persistent grouped MMAD exceeds L0 hard-event IDs");
    static_assert(EVENT_BASE + BaseMmad::L0C_STAGES <=
                      KDA_GROUPED_LOCAL_EVENT_CAPACITY,
                  "Persistent grouped MMAD exceeds L0C hard-event IDs");

    CATLASS_DEVICE
    KdaBwdGroupedPartitionedBlockMmad(
        Catlass::Arch::Resource<ArchTag> &resource)
        : BaseMmad(resource, L1_BASE)
    {
        if ASCEND_IS_AIC {
            // BaseMmad(resource, L1_BASE) already binds a disjoint packed L1
            // range.  Re-number its events and explicitly partition every L0
            // tensor, which the shared constructor otherwise binds at zero.
            for (uint32_t i = 0; i < BaseMmad::L1A_STAGES; ++i) {
                this->l1AEventList[i] = static_cast<int32_t>(EVENT_BASE + i);
            }
            for (uint32_t i = 0; i < BaseMmad::L1B_STAGES; ++i) {
                this->l1BEventList[i] = static_cast<int32_t>(
                    EVENT_BASE + BaseMmad::L1A_STAGES + i);
            }
            for (uint32_t i = 0; i < BaseMmad::L0A_STAGES; ++i) {
                this->l0ATensorList[i] =
                    resource.l0ABuf.template GetBufferByByte<ElementA>(
                        L0A_BASE + BaseMmad::L0A_TILE_SIZE * i);
                this->l0AEventList[i] = static_cast<int32_t>(EVENT_BASE + i);
            }
            for (uint32_t i = 0; i < BaseMmad::L0B_STAGES; ++i) {
                this->l0BTensorList[i] =
                    resource.l0BBuf.template GetBufferByByte<ElementB>(
                        L0B_BASE + BaseMmad::L0B_TILE_SIZE * i);
                this->l0BEventList[i] = static_cast<int32_t>(
                    EVENT_BASE + BaseMmad::L0A_STAGES + i);
            }
            for (uint32_t i = 0; i < BaseMmad::L0C_STAGES; ++i) {
                this->l0CTensorList[i] =
                    resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                        L0C_BASE + BaseMmad::L0C_TILE_SIZE * i);
                // The unit-flag branch of the shared constructor initializes
                // l0C storage but not this event list; make it deterministic.
                this->l0CEventList[i] = static_cast<int32_t>(EVENT_BASE + i);
            }
        }
    }
};

using KdaBwdGroupedPersistentLeftBase = KdaBwdGroupedLeftBlockMmad<32>;
using KdaBwdGroupedPersistentRightBase = KdaBwdGroupedRightBlockMmad<16>;
static_assert(KdaBwdGroupedPersistentLeftBase::L1_TILE_K == 32 &&
                  KdaBwdGroupedPersistentLeftBase::L0_TILE_K == 32 &&
                  KdaBwdGroupedPersistentRightBase::L1_TILE_K == 32 &&
                  KdaBwdGroupedPersistentRightBase::L0_TILE_K == 32,
              "Persistent grouped MMAD K16/K32/K48 proof assumes K32 tiles");

// The rollback scoped path intentionally keeps all three MMAD objects aliased at
// local-memory/event base zero and brackets every use with its own flag
// envelope.  Use the private wrapper there as well so unit-flag L0C event zero
// is initialized deterministically without changing the shared implementation.
using KdaBwdGroupedScopedLeft16Mmad =
    KdaBwdGroupedPartitionedBlockMmad<
        KdaBwdGroupedLeftBlockMmad<16>, 0, 0, 0, 0, 0>;
using KdaBwdGroupedScopedLeft32Mmad =
    KdaBwdGroupedPartitionedBlockMmad<
        KdaBwdGroupedLeftBlockMmad<32>, 0, 0, 0, 0, 0>;
using KdaBwdGroupedScopedRight16Mmad =
    KdaBwdGroupedPartitionedBlockMmad<
        KdaBwdGroupedRightBlockMmad<16>, 0, 0, 0, 0, 0>;

constexpr uint32_t KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES =
    KdaBwdGroupedPersistentLeftBase::L1A_TILE_SIZE *
        KdaBwdGroupedPersistentLeftBase::L1A_STAGES +
    KdaBwdGroupedPersistentLeftBase::L1B_TILE_SIZE *
        KdaBwdGroupedPersistentLeftBase::L1B_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_LEFT_L0A_BYTES =
    KdaBwdGroupedPersistentLeftBase::L0A_TILE_SIZE *
    KdaBwdGroupedPersistentLeftBase::L0A_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES =
    KdaBwdGroupedPersistentLeftBase::L0B_TILE_SIZE *
    KdaBwdGroupedPersistentLeftBase::L0B_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_LEFT_L0C_BYTES =
    KdaBwdGroupedPersistentLeftBase::L0C_TILE_SIZE *
    KdaBwdGroupedPersistentLeftBase::L0C_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES =
    KdaBwdGroupedPersistentRightBase::L1A_TILE_SIZE *
        KdaBwdGroupedPersistentRightBase::L1A_STAGES +
    KdaBwdGroupedPersistentRightBase::L1B_TILE_SIZE *
        KdaBwdGroupedPersistentRightBase::L1B_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_RIGHT_L0A_BYTES =
    KdaBwdGroupedPersistentRightBase::L0A_TILE_SIZE *
    KdaBwdGroupedPersistentRightBase::L0A_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_RIGHT_L0B_BYTES =
    KdaBwdGroupedPersistentRightBase::L0B_TILE_SIZE *
    KdaBwdGroupedPersistentRightBase::L0B_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_RIGHT_L0C_BYTES =
    KdaBwdGroupedPersistentRightBase::L0C_TILE_SIZE *
    KdaBwdGroupedPersistentRightBase::L0C_STAGES;
constexpr uint32_t KDA_GROUPED_PERSISTENT_LEFT_EVENT_BASE = 0;
constexpr uint32_t KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE = 4;

static_assert(KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES == 40 * 1024 &&
                  KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES == 36 * 1024,
              "Persistent grouped MMAD L1 geometry changed unexpectedly");
static_assert(KDA_GROUPED_PERSISTENT_LEFT_L0A_BYTES == 8 * 1024 &&
                  KDA_GROUPED_PERSISTENT_RIGHT_L0A_BYTES == 4 * 1024,
              "Persistent grouped MMAD L0A geometry changed unexpectedly");
static_assert(KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES == 32 * 1024 &&
                  KDA_GROUPED_PERSISTENT_RIGHT_L0B_BYTES == 32 * 1024,
              "Persistent grouped MMAD L0B geometry changed unexpectedly");
static_assert(KDA_GROUPED_PERSISTENT_LEFT_L0C_BYTES == 16 * 1024 &&
                   KDA_GROUPED_PERSISTENT_RIGHT_L0C_BYTES == 8 * 1024,
               "Persistent grouped MMAD L0C geometry changed unexpectedly");
static_assert(
    KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_LEFT_L0A_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_RIGHT_L0A_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_RIGHT_L0B_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_LEFT_L0C_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0 &&
        KDA_GROUPED_PERSISTENT_RIGHT_L0C_BYTES %
                KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
            0,
    "Persistent grouped MMAD partition boundaries must stay 512-byte aligned");
static_assert(KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES +
                       KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES <=
                   KdaBwdGroupedArchTag::L1_SIZE &&
                  KDA_GROUPED_PERSISTENT_LEFT_L0A_BYTES +
                          KDA_GROUPED_PERSISTENT_RIGHT_L0A_BYTES <=
                      KdaBwdGroupedArchTag::L0A_SIZE &&
                  KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES +
                          KDA_GROUPED_PERSISTENT_RIGHT_L0B_BYTES <=
                      KdaBwdGroupedArchTag::L0B_SIZE &&
                  KDA_GROUPED_PERSISTENT_LEFT_L0C_BYTES +
                          KDA_GROUPED_PERSISTENT_RIGHT_L0C_BYTES <=
                       KdaBwdGroupedArchTag::L0C_SIZE,
               "Persistent grouped MMAD partitions exceed local memory");
static_assert(
    KDA_GROUPED_PERSISTENT_LEFT_EVENT_BASE +
                KdaBwdGroupedPersistentLeftBase::L1A_STAGES +
                KdaBwdGroupedPersistentLeftBase::L1B_STAGES <=
            KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE &&
        KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE +
                KdaBwdGroupedPersistentRightBase::L1A_STAGES +
                KdaBwdGroupedPersistentRightBase::L1B_STAGES <=
            KDA_GROUPED_LOCAL_EVENT_CAPACITY &&
        KDA_GROUPED_PERSISTENT_LEFT_EVENT_BASE +
                KdaBwdGroupedPersistentLeftBase::L0A_STAGES +
                KdaBwdGroupedPersistentLeftBase::L0B_STAGES <=
            KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE &&
        KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE +
                KdaBwdGroupedPersistentRightBase::L0A_STAGES +
                KdaBwdGroupedPersistentRightBase::L0B_STAGES <=
            KDA_GROUPED_LOCAL_EVENT_CAPACITY &&
        KDA_GROUPED_PERSISTENT_LEFT_EVENT_BASE +
                KdaBwdGroupedPersistentLeftBase::L0C_STAGES <=
            KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE &&
        KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE +
                KdaBwdGroupedPersistentRightBase::L0C_STAGES <=
            KDA_GROUPED_LOCAL_EVENT_CAPACITY,
    "Persistent grouped MMAD event partitions overlap or exceed IDs 0..7");

using KdaBwdGroupedPersistentLeftMmad =
    KdaBwdGroupedPartitionedBlockMmad<
        KdaBwdGroupedPersistentLeftBase, 0, 0, 0, 0,
        KDA_GROUPED_PERSISTENT_LEFT_EVENT_BASE>;
using KdaBwdGroupedPersistentRightMmad =
    KdaBwdGroupedPartitionedBlockMmad<
        KdaBwdGroupedPersistentRightBase,
        KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES,
        KDA_GROUPED_PERSISTENT_LEFT_L0A_BYTES,
        KDA_GROUPED_PERSISTENT_LEFT_L0B_BYTES,
        KDA_GROUPED_PERSISTENT_LEFT_L0C_BYTES,
        KDA_GROUPED_PERSISTENT_RIGHT_EVENT_BASE>;

constexpr uint32_t KDA_GROUPED_TSCM_AB_BYTES =
    KDA_GROUPED_QUEUE_DEPTH * KDA_GROUPED_SLOT_BYTES;
constexpr uint32_t KDA_GROUPED_TSCM_AB_BASE =
    KdaBwdGroupedArchTag::L1_SIZE - KDA_GROUPED_TSCM_AB_BYTES;
static_assert(KDA_GROUPED_TSCM_AB_BYTES < KdaBwdGroupedArchTag::L1_SIZE,
              "Grouped A/B ping-pong slots exceed L1 capacity");
static_assert(KDA_GROUPED_TSCM_AB_BASE %
                      KDA_GROUPED_LOCAL_BUFFER_ALIGNMENT ==
                  0,
              "Grouped TSCM base must be 512-byte aligned");
static_assert(KDA_GROUPED_PERSISTENT_LEFT_L1_BYTES +
                      KDA_GROUPED_PERSISTENT_RIGHT_L1_BYTES <=
                  KDA_GROUPED_TSCM_AB_BASE,
              "Grouped CATLASS L1 buffers overlap the top-down TSCM slots");

template <typename T>
class ChunkKdaBwdIntraGroupedKernel {
public:
    static_assert(sizeof(T) == 2,
                  "Grouped key 23 requires a two-byte BF16 input type");
    static_assert(KDA_GROUPED_K_TYPED_UB == KDA_GROUPED_Q_TYPED_UB +
                      KDA_GROUPED_SELECTED_ELEMENTS * sizeof(T),
                  "q/k typed banks must remain contiguous for one Cast");
    static_assert(KDA_GROUPED_K_CACHE_UB == KDA_GROUPED_Q_CACHE_UB +
                      KDA_GROUPED_SELECTED_ELEMENTS * sizeof(float),
                  "q/k FP32 caches must remain contiguous for one Cast");
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta,
                                GM_ADDR dAqk, GM_ADDR dAkk, GM_ADDR dq, GM_ADDR dk,
                                GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
                                GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR workspace,
                                const ChunkKdaBwdIntraTilingData &tiling, TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta));
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dAkk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk));
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk));
        db_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db));
        dg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg));
        dqOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqOut));
        dkOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkOut));
        dbOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbOut));
        dgOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgOut));
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        batch_ = static_cast<uint64_t>(tiling.batch);
        heads_ = static_cast<uint64_t>(tiling.vHeadNum);
        seqlen_ = static_cast<uint64_t>(tiling.seqlen);
        chunks_ = static_cast<uint64_t>(tiling.totalChunks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        pipe_ = pipe;

        if ASCEND_IS_AIC {
            logicalCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if ASCEND_IS_AIV {
            const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            logicalCoreIdx_ = subBlockNum == 0 ? 0 :
                static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
            pipe_->InitBuffer(ubBuf_, KDA_GROUPED_UB_BYTES);
            if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
                // depth=1 is sufficient because each publication is dequeued
                // immediately.  num=2 is the actual stage ping-pong storage.
                pipe_->InitBuffer(stageAbQueue_, KDA_GROUPED_QUEUE_DEPTH,
                                  KDA_GROUPED_SLOT_BYTES);
            }
            AllocAivSyncEvents();
            if constexpr (KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH) {
                pairScratchDone0_ =
                    GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
                pairScratchDone1_ =
                    GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            }
        }
    }

    __aicore__ inline void ProcessAic()
    {
        Catlass::Arch::Resource<KdaBwdGroupedArchTag> resource;
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            // TPipe allocates block-group TSCM from the top of L1.  The AIC
            // consumes the same physical slots directly; cross-core flags
            // below define their producer/consumer lifetime.
            stageAbSlots_[0] =
                resource.l1Buf.template GetBufferByByte<float>(
                    KDA_GROUPED_TSCM_AB_BASE);
            stageAbSlots_[1] =
                resource.l1Buf.template GetBufferByByte<float>(
                    KDA_GROUPED_TSCM_AB_BASE + KDA_GROUPED_SLOT_BYTES);
        }
        if constexpr (KDA_GROUPED_PERSISTENT_MMAD_ENGINES) {
            KdaBwdGroupedPersistentLeftMmad left32(resource);
            KdaBwdGroupedPersistentRightMmad right16(resource);
            // Both engines own disjoint L1/L0/event partitions, so their free
            // tokens can remain live across every stage and task on this AIC.
            left32.preSetFlags();
            right16.preSetFlags();
            const uint64_t taskCount = batch_ * chunks_ * heads_;
            for (uint64_t task = logicalCoreIdx_; task < taskCount;
                 task += usedCoreNum_) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeDiagonalPersistentAic(left32, right16, 0);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeGroupedStagePersistentAic<1>(left32, right16, 1);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeGroupedStagePersistentAic<2>(left32, right16, 0);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                // The left K32 engine still executes the 48-wide reduction as
                // K32 then K16, preserving the cancellation-sensitive order.
                ComputeGroupedStagePersistentAic<3>(left32, right16, 1);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);
            }
            // Unit-flag Fixpipe completion is carried by each PIPE_FIX done
            // signal.  These waits only drain the engines' local free tokens.
            right16.finalWaitFlags();
            left32.finalWaitFlags();
        } else {
            KdaBwdGroupedScopedLeft16Mmad left16(resource);
            KdaBwdGroupedScopedLeft32Mmad left32(resource);
            KdaBwdGroupedScopedRight16Mmad right16(resource);
            const uint64_t taskCount = batch_ * chunks_ * heads_;
            for (uint64_t task = logicalCoreIdx_; task < taskCount; task += usedCoreNum_) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeDiagonalAic(left16, right16, 0);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeGroupedStageAic<1>(left16, right16, left16, 1);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputeGroupedStageAic<2>(left32, right16, left16, 0);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);

                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                // Keep the 48-wide off-left reduction as a 32-wide tile plus a
                // 16-wide tail.  Besides reusing the stage-2 BlockMmad, this
                // preserves the upstream pair order for cancellation-sensitive
                // inputs: early blocks 0/1 reduce before early block 2 is added.
                ComputeGroupedStageAic<3>(left32, right16, left16, 1);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);
            }
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        // A physical AIV owns the same eight-row half for its entire kernel
        // lifetime.  Hoist this invariant once instead of recomputing
        // GetSubBlockIdx()*ROWS_PER_AIV throughout every task and stage.
        const uint32_t rowStart =
            subBlockIdx * KDA_GROUPED_ROWS_PER_AIV;
        InitializeCausalSelectState(rowStart);
        if constexpr (KDA_GROUPED_OVERLAP_TASK_STORE) {
            ProcessAivWithTaskStoreOverlap(rowStart);
        } else {
            ProcessAivSerialTaskStore(rowStart);
        }
        if constexpr (KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(pairScratchDone0_);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(pairScratchDone1_);
        }
        ReleaseAivSyncEvents();
    }

private:
    __aicore__ inline void AdvanceTaskCoordinates(
        uint64_t &b, uint64_t &hv, uint64_t &chunkStart,
        uint64_t batchStep, uint64_t headStep, uint64_t chunkStartStep,
        uint64_t sequenceSpan)
    {
        b += batchStep;
        chunkStart += chunkStartStep;
        hv += headStep;
        if (hv >= heads_) {
            hv -= heads_;
            chunkStart += KDA_GROUPED_BT;
        }
        if (chunkStart >= sequenceSpan) {
            chunkStart -= sequenceSpan;
            ++b;
        }
    }

    __aicore__ inline void BuildSharedTaskGates()
    {
        if constexpr (KDA_GROUPED_OVERLAP_SHARED_SETUP) {
            BuildPersistentBlockGates();
            if constexpr (KDA_GROUPED_FACTOR_PAIR_GATES) {
                BuildPairBridgeGates();
            }
        }
    }

    __aicore__ inline void ProcessAivSerialTaskStore(uint32_t rowStart)
    {
        const uint64_t taskCount = batch_ * chunks_ * heads_;
        uint64_t task = logicalCoreIdx_;

        // Resolve the mixed-radix (batch, chunk, head) coordinate once per
        // physical AIV.  A core always advances by the fixed usedCoreNum_
        // stride, so split that stride once and propagate coordinates with
        // add/carry operations in the hot loop.
        uint64_t b = 0;
        uint64_t hv = 0;
        uint64_t chunkStart = 0;
        ResolveTask(task, b, hv, chunkStart);
        const uint64_t flatChunkStep = usedCoreNum_ / heads_;
        const uint64_t headStep = usedCoreNum_ - flatChunkStep * heads_;
        const uint64_t batchStep = flatChunkStep / chunks_;
        const uint64_t chunkStep = flatChunkStep - batchStep * chunks_;
        const uint64_t chunkStartStep = chunkStep * KDA_GROUPED_BT;
        const uint64_t sequenceSpan = chunks_ * KDA_GROUPED_BT;

        // Host blockDim is min(taskCount, AIC cores), so every launched
        // logical core owns at least its initial task.  Keep the only hot-loop
        // termination check in the epilogue instead of testing twice.
        while (true) {
            LoadTaskFeatures(b, hv, chunkStart, rowStart);
            if constexpr (!KDA_GROUPED_OVERLAP_SHARED_SETUP) {
                ZeroAccumulators();
            }

            PrepareStageAiv<0>(b, hv, chunkStart, rowStart, 0);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            if constexpr (KDA_GROUPED_OVERLAP_SHARED_SETUP) {
                // Stage 0 consumes only diagonal gates and its completed
                // workspace slot.  Build state first used by stage 1 while
                // AIC computes stage 0, instead of extending the task prologue.
                BuildSharedTaskGates();
                ZeroAccumulators();
            }

            PrepareStageAiv<1>(b, hv, chunkStart, rowStart, 1);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(0);
            ConsumeStageAiv<0>(0, rowStart);

            PrepareStageAiv<2>(b, hv, chunkStart, rowStart, 0);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(1);
            ConsumeStageAiv<1>(1, rowStart);

            PrepareStageAiv<3>(b, hv, chunkStart, rowStart, 1);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(0);
            ConsumeStageAiv<2>(0, rowStart);

            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(1);
            ConsumeStageAiv<3>(1, rowStart);
            StoreTaskOutputs(b, hv, chunkStart, rowStart);

            task += usedCoreNum_;
            if (task >= taskCount) {
                break;
            }
            AdvanceTaskCoordinates(b, hv, chunkStart, batchStep, headStep,
                                   chunkStartStep, sequenceSpan);
        }
    }

    __aicore__ inline void ProcessAivWithTaskStoreOverlap(uint32_t rowStart)
    {
        const uint64_t taskCount = batch_ * chunks_ * heads_;
        uint64_t task = logicalCoreIdx_;
        uint64_t b = 0;
        uint64_t hv = 0;
        uint64_t chunkStart = 0;
        ResolveTask(task, b, hv, chunkStart);
        const uint64_t flatChunkStep = usedCoreNum_ / heads_;
        const uint64_t headStep = usedCoreNum_ - flatChunkStep * heads_;
        const uint64_t batchStep = flatChunkStep / chunks_;
        const uint64_t chunkStep = flatChunkStep - batchStep * chunks_;
        const uint64_t chunkStartStep = chunkStep * KDA_GROUPED_BT;
        const uint64_t sequenceSpan = chunks_ * KDA_GROUPED_BT;

        // Prologue: publish two stages before touching the accumulator slab.
        // Neither stage preparation aliases dq/dk-left/dk-right, so the same
        // ordering is also used in the steady state while the prior task's
        // completed outputs remain resident in those banks.
        LoadTaskFeatures(b, hv, chunkStart, rowStart);
        PrepareStageAiv<0>(b, hv, chunkStart, rowStart, 0);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
        BuildSharedTaskGates();
        PrepareStageAiv<1>(b, hv, chunkStart, rowStart, 1);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
        ZeroAccumulators();

        while (true) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(0);
            ConsumeStageAiv<0>(0, rowStart);

            PrepareStageAiv<2>(b, hv, chunkStart, rowStart, 0);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(1);
            ConsumeStageAiv<1>(1, rowStart);

            PrepareStageAiv<3>(b, hv, chunkStart, rowStart, 1);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(0);
            ConsumeStageAiv<2>(0, rowStart);

            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            ReleaseStageAbSlot(1);
            ConsumeStageAiv<3>(1, rowStart);
            StageTaskOutputsInAccumulators(b, hv, chunkStart, rowStart);

            const uint64_t completedB = b;
            const uint64_t completedHv = hv;
            const uint64_t completedChunkStart = chunkStart;
            task += usedCoreNum_;
            if (task >= taskCount) {
                IssueStagedTaskOutputs(completedB, completedHv,
                                       completedChunkStart, rowStart);
                SyncMte3ToV();
                break;
            }

            // The completed output rows live only in the three accumulator
            // banks plus the 128-byte db prefix.  All feature/gate/scratch
            // regions are therefore free to prepare two stages of the next
            // task before the previous output DMA is submitted.
            SyncVToMte2();
            AdvanceTaskCoordinates(b, hv, chunkStart, batchStep, headStep,
                                   chunkStartStep, sequenceSpan);
            LoadTaskFeatures(b, hv, chunkStart, rowStart);
            PrepareStageAiv<0>(b, hv, chunkStart, rowStart, 0);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            BuildSharedTaskGates();
            PrepareStageAiv<1>(b, hv, chunkStart, rowStart, 1);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);

            // Stage-0/1 workspace writes precede the old output on MTE3, so the
            // AIC can start both Cube stages while the old output drains.  Wait
            // only before Vector clears the accumulator banks for the new task.
            // The following MTE3->V wait, the V clear, and ConsumeStageAiv<0>'s
            // unconditional V->MTE2 dependency form the transitive ordering for
            // every later workspace load.  A direct MTE3->MTE2 wait here would
            // be correct but would unnecessarily serialize the two pipelines.
            IssueStagedTaskOutputs(completedB, completedHv,
                                   completedChunkStart, rowStart);
            SyncMte3ToV();
            ZeroAccumulators();
        }
    }

    __aicore__ inline void AllocAivSyncEvents()
    {
        // These dependencies are fully drained by every helper invocation.
        // Reserve one ID per HardEvent for the AIV lifetime instead of
        // repeating TPipe allocation bookkeeping hundreds of times per core.
        // Pair-scratch ping-pong owns two additional MTE3_V IDs, keeping the
        // maximum live count for that event type at three on A2 (limit eight).
        mte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        sToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        vToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte3ToMte2Event_ =
            GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        mte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void ReleaseAivSyncEvents()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(sToVEvent_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    template <typename U, uint32_t BYTE_OFFSET>
    __aicore__ inline LocalTensor<U> UbTensor()
    {
        static_assert(BYTE_OFFSET % sizeof(U) == 0, "UB offset must respect element alignment");
        return ubBuf_.Get<U>()[BYTE_OFFSET / sizeof(U)];
    }

    template <uint32_t BANK>
    __aicore__ inline LocalTensor<float> GateBank()
    {
        static_assert(BANK < 10, "Grouped gate bank index is out of range");
        if constexpr (BANK < 2) {
            return UbTensor<float, KDA_GROUPED_Q_TYPED_UB>()[
                BANK * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
        } else if constexpr (BANK < 4) {
            return UbTensor<float, KDA_GROUPED_K_TYPED_UB>()[
                (BANK - 2) * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
        } else if constexpr (BANK < 8) {
            return UbTensor<float, KDA_GROUPED_GATE_SLOTS_UB>()[
                (BANK - 4) * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
        } else if constexpr (BANK == 8) {
            return UbTensor<float, KDA_GROUPED_OPERAND0_UB>();
        } else {
            return UbTensor<float, KDA_GROUPED_OPERAND1_UB>();
        }
    }

    template <uint32_t INDEX>
    __aicore__ inline LocalTensor<float> ScratchBank()
    {
        static_assert(INDEX < 3, "Grouped scratch bank index is out of range");
        if constexpr (INDEX < 2) {
            return UbTensor<float, KDA_GROUPED_RESULT_UB>()[
                INDEX * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
        } else {
            return UbTensor<float, KDA_GROUPED_RAW_DA_UB>();
        }
    }

    template <uint32_t SET, uint32_t INDEX>
    __aicore__ inline LocalTensor<float> PairScratchBank()
    {
        static_assert(SET < 2 && INDEX < 3,
                      "Grouped pair scratch set/index is out of range");
        if constexpr (SET == 0) {
            return ScratchBank<INDEX>();
        } else {
            return UbTensor<float, KDA_GROUPED_PAIR_SCRATCH_ALT_UB>()[
                INDEX * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
        }
    }

    template <uint32_t SET>
    __aicore__ inline void PublishPairScratchDone()
    {
        static_assert(SET < 2, "Grouped pair scratch set is out of range");
        if constexpr (SET == 0) {
            SetFlag<HardEvent::MTE3_V>(pairScratchDone0_);
        } else {
            SetFlag<HardEvent::MTE3_V>(pairScratchDone1_);
        }
    }

    template <uint32_t SET>
    __aicore__ inline void WaitPairScratchDone()
    {
        static_assert(SET < 2, "Grouped pair scratch set is out of range");
        if constexpr (SET == 0) {
            WaitFlag<HardEvent::MTE3_V>(pairScratchDone0_);
        } else {
            WaitFlag<HardEvent::MTE3_V>(pairScratchDone1_);
        }
    }

    template <uint32_t STAGE>
    __aicore__ inline LocalTensor<float> LeftOuterGate()
    {
        constexpr uint32_t base = 4 + 3 * (STAGE & 1U);
        return GateBank<base>();
    }

    template <uint32_t STAGE>
    __aicore__ inline LocalTensor<float> DiagLateGate()
    {
        constexpr uint32_t base = 4 + 3 * (STAGE & 1U);
        return GateBank<base + 1>();
    }

    template <uint32_t STAGE>
    __aicore__ inline LocalTensor<float> DiagEarlyGate()
    {
        constexpr uint32_t base = 4 + 3 * (STAGE & 1U);
        return GateBank<base + 2>();
    }

    template <uint32_t BLOCK>
    __aicore__ inline LocalTensor<float> RightOuterGate()
    {
        static_assert(BLOCK < KDA_GROUPED_BLOCKS, "Grouped right-outer block is out of range");
        return UbTensor<float, KDA_GROUPED_Q_TYPED_UB>()[
            BLOCK * KDA_GROUPED_ROW_BLOCK_ELEMENTS];
    }

    template <uint32_t STAGE, uint32_t EARLY>
    __aicore__ inline LocalTensor<float> PairBridgeGate()
    {
        static_assert(STAGE > 0 && STAGE < KDA_GROUPED_STAGES && EARLY < STAGE,
                      "Grouped pair bridge is out of range");
        constexpr uint32_t pair = STAGE * (STAGE - 1) / 2 + EARLY;
        static_assert(pair < KDA_GROUPED_PAIR_COUNT,
                      "Grouped pair bridge index exceeds its UB slab");
        return UbTensor<float, KDA_GROUPED_PAIR_BRIDGE_UB>()[
            pair * KDA_GROUPED_K];
    }

    template <uint32_t STAGE>
    __aicore__ inline LocalTensor<float> FirstReference()
    {
        static_assert(STAGE > 0 && STAGE < KDA_GROUPED_STAGES,
                      "First reference exists only for off-diagonal stages");
        return UbTensor<float, KDA_GROUPED_REF_UB>()[
            (STAGE - 1) * KDA_GROUPED_K];
    }

    template <uint32_t BLOCK>
    __aicore__ inline LocalTensor<float> MiddleReference()
    {
        static_assert(BLOCK < KDA_GROUPED_BLOCKS,
                      "Middle reference block is out of range");
        return UbTensor<float, KDA_GROUPED_REF_UB>()[
            KDA_GROUPED_MIDDLE_REF_OFFSET + BLOCK * KDA_GROUPED_K];
    }

    template <uint32_t BLOCK>
    __aicore__ inline LocalTensor<float> RightReference()
    {
        static_assert(BLOCK < KDA_GROUPED_RIGHT_REF_COUNT,
                      "Right reference exists only for early blocks");
        return UbTensor<float, KDA_GROUPED_REF_UB>()[
            KDA_GROUPED_RIGHT_REF_OFFSET + BLOCK * KDA_GROUPED_K];
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t) const
    {
        return ((b * heads_ + h) * seqlen_ + t) * KDA_GROUPED_K;
    }

    __aicore__ inline uint64_t VOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_GROUPED_K;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * heads_ + hv) * seqlen_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t,
                                       uint64_t localColumn) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_GROUPED_BT + localColumn;
    }

    __aicore__ inline uint64_t SlotBase(uint32_t slot) const
    {
        return (logicalCoreIdx_ * KDA_GROUPED_QUEUE_DEPTH + slot) *
            KDA_GROUPED_SLOT_ELEMENTS;
    }

    __aicore__ inline uint32_t LocalSlotFromBase(uint64_t slotBase) const
    {
        return static_cast<uint32_t>(
            (slotBase / KDA_GROUPED_SLOT_ELEMENTS) %
            KDA_GROUPED_QUEUE_DEPTH);
    }

    __aicore__ inline void AcquireStageAbSlot(uint32_t slot)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            stageAbSlots_[slot] = stageAbQueue_.template AllocTensor<float>();
        }
    }

    __aicore__ inline void PublishStageAbSlot(uint32_t slot)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            stageAbQueue_.EnQue(stageAbSlots_[slot]);
            stageAbSlots_[slot] = stageAbQueue_.template DeQue<float>();
        }
    }

    __aicore__ inline void ReleaseStageAbSlot(uint32_t slot)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            stageAbQueue_.FreeTensor(stageAbSlots_[slot]);
        }
    }

    __aicore__ inline void CopyStageAb(uint32_t slot, uint32_t offset,
                                       LocalTensor<float> src,
                                       uint32_t elements)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            DataCopy(stageAbSlots_[slot][offset], src, elements);
        } else {
            DataCopy(workspace_[SlotBase(slot) + offset], src, elements);
        }
    }

    // Experimental non-A2 UB -> TSCM path.  Emit the zN image directly with
    // strided 32-byte block copies instead of relying on an Nd2Nz overload.
    // This helper must remain unreachable on DAV_2201: that architecture
    // software-emulates UB -> TSCM through GM/Matmul KFC, while this direct
    // CATLASS kernel has no registered KFC client.  For FP32, one C0 block is
    // eight values; zN stores a complete matrix column block as consecutive
    // row blocks:
    //
    //   dst[(column / 8) * matrixRows * 8 + row * 8 + column % 8]
    //
    // Every grouped matrix dimension is a multiple of the corresponding
    // 16x8 Cube fractal, so no padding or read-modify-write is required.  The
    // two AIVs call this helper for disjoint eight-row ranges in the same
    // block-group TSCM allocation.
    __aicore__ inline void CopyStageAbNzRows(
        uint32_t slot, uint32_t matrixOffset, uint32_t matrixRows,
        uint32_t matrixColumns, uint32_t destinationRow,
        LocalTensor<float> src, uint32_t sourceRowStride,
        uint32_t rowCount)
    {
        constexpr uint32_t C0_ELEMENTS =
            KDA_GROUPED_DATA_BLOCK_BYTES / sizeof(float);
        const DataCopyParams params{
            static_cast<uint16_t>(rowCount), 1,
            static_cast<uint16_t>(sourceRowStride / C0_ELEMENTS - 1), 0};
        for (uint32_t column = 0; column < matrixColumns;
             column += C0_ELEMENTS) {
            const uint32_t destinationOffset =
                matrixOffset + column * matrixRows +
                destinationRow * C0_ELEMENTS;
            DataCopy(stageAbSlots_[slot][destinationOffset], src[column],
                     params);
        }
    }

    __aicore__ inline void CopyStageRightBRows(
        uint32_t slot, uint32_t matrixOffset, uint32_t rowStart,
        LocalTensor<float> firstHalf)
    {
        constexpr uint32_t rowTileBytes =
            KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
        constexpr uint32_t rightHalfGapBytes =
            (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) *
            KDA_GROUPED_K * sizeof(float);
        static_assert(rowTileBytes == rightHalfGapBytes,
                      "Grouped right-B half-row gap must equal one AIV tile");
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            CopyStageAbNzRows(
                slot, matrixOffset, 2 * KDA_GROUPED_BC, KDA_GROUPED_K,
                rowStart, firstHalf, KDA_GROUPED_K,
                KDA_GROUPED_ROWS_PER_AIV);
            CopyStageAbNzRows(
                slot, matrixOffset, 2 * KDA_GROUPED_BC, KDA_GROUPED_K,
                KDA_GROUPED_BC + rowStart,
                firstHalf[KDA_GROUPED_ROW_BLOCK_ELEMENTS], KDA_GROUPED_K,
                KDA_GROUPED_ROWS_PER_AIV);
        } else if constexpr (KDA_GROUPED_COALESCE_RIGHT_B_WRITES) {
            // DataCopyExtParams uses data-block units on the UB source stride
            // and bytes on the GM destination stride.  The two scratch halves
            // are contiguous, hence srcStride=0; dstStride skips the eight GM
            // rows owned by the other AIV half.
            DataCopyExtParams rightRows{
                2, rowTileBytes, 0, rightHalfGapBytes, 0};
            DataCopyPad(
                workspace_[SlotBase(slot) + matrixOffset +
                           rowStart * KDA_GROUPED_K],
                firstHalf, rightRows);
        } else {
            CopyStageAb(slot, matrixOffset + rowStart * KDA_GROUPED_K,
                        firstHalf, KDA_GROUPED_ROW_BLOCK_ELEMENTS);
            CopyStageAb(
                slot,
                matrixOffset +
                    (KDA_GROUPED_BC + rowStart) * KDA_GROUPED_K,
                firstHalf[KDA_GROUPED_ROW_BLOCK_ELEMENTS],
                KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        }
    }

    __aicore__ inline void CopyStageLeftCRows(uint64_t sourceOffset)
    {
        if constexpr (KDA_GROUPED_COALESCE_LEFT_C_READS) {
            constexpr uint32_t rowTileBytes =
                KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
            DataCopyExtParams leftRows{
                2, rowTileBytes, rowTileBytes, 0, 0};
            DataCopyPadExtParams<float> noPad{
                false, 0, 0, 0.0f};
            DataCopyPad(ScratchBank<0>(), workspace_[sourceOffset],
                        leftRows, noPad);
        } else {
            DataCopy(ScratchBank<0>(), workspace_[sourceOffset],
                     KDA_GROUPED_ROW_BLOCK_ELEMENTS);
            DataCopy(
                ScratchBank<1>(),
                workspace_[sourceOffset +
                           KDA_GROUPED_BC * KDA_GROUPED_K],
                KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        }
    }

    template <typename Layout>
    __aicore__ inline auto MakeStageInputTensor(uint64_t slotBase,
                                                 uint32_t offset,
                                                 const Layout &layout)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            const uint32_t slot = LocalSlotFromBase(slotBase);
            if constexpr (tla::detail::isRowMajor<Layout>::value) {
                auto l1Layout = tla::MakeLayout<
                    float, Catlass::layout::zN>(
                        layout.template shape<0>(),
                        layout.template shape<1>());
                return tla::MakeTensor(stageAbSlots_[slot][offset], l1Layout,
                                       Catlass::Arch::PositionL1{});
            } else {
                static_assert(tla::detail::isColumnMajor<Layout>::value,
                              "Grouped TSCM inputs must be row- or column-major");
                auto l1Layout = tla::MakeLayout<
                    float, Catlass::layout::nZ>(
                        layout.template shape<0>(),
                        layout.template shape<1>());
                return tla::MakeTensor(stageAbSlots_[slot][offset], l1Layout,
                                       Catlass::Arch::PositionL1{});
            }
        } else {
            return tla::MakeTensor(workspace_[slotBase + offset], layout,
                                   Catlass::Arch::PositionGM{});
        }
    }

    template <typename BlockMmad, typename TensorA, typename TensorB,
              typename TensorC>
    __aicore__ inline void RunStageMmad(BlockMmad &blockMmad,
                                        TensorA &blockA, TensorB &blockB,
                                        TensorC &blockC,
                                        const Catlass::GemmCoord &shape)
    {
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            blockMmad.RunFromL1(blockA, blockB, blockC, shape);
        } else {
            blockMmad(blockA, blockB, blockC, shape);
        }
    }

    __aicore__ inline uint32_t CacheBlockOffset(uint32_t block) const
    {
        return block * KDA_GROUPED_ROW_BLOCK_ELEMENTS;
    }

    __aicore__ inline void ResolveTask(uint64_t task, uint64_t &b, uint64_t &hv,
                                       uint64_t &chunkStart) const
    {
        hv = task % heads_;
        const uint64_t flatChunk = task / heads_;
        b = flatChunk / chunks_;
        chunkStart = (flatChunk % chunks_) * KDA_GROUPED_BT;
    }

    __aicore__ inline void SyncMte2ToV()
    {
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void SyncVToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
    }

    __aicore__ inline void SyncSToV()
    {
        SetFlag<HardEvent::S_V>(sToVEvent_);
        WaitFlag<HardEvent::S_V>(sToVEvent_);
    }

    __aicore__ inline void SyncVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void BeginReusableFp32Mask()
    {
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(
                KDA_GROUPED_FP32_REPEAT_ELEMENTS);
        }
    }

    __aicore__ inline void EndReusableFp32Mask()
    {
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            ResetMask();
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousMuls(LocalTensor<float> dst,
                                           LocalTensor<float> src,
                                           float scalar)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            UnaryRepeatParams params{1, 1, 8, 8};
            Muls<float, false>(
                dst, src, scalar, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            Muls(dst, src, scalar, ELEMENTS);
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousExp(LocalTensor<float> dst,
                                          LocalTensor<float> src)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            UnaryRepeatParams params{1, 1, 8, 8};
            Exp<float, false>(
                dst, src, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            Exp(dst, src, ELEMENTS);
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousSub(LocalTensor<float> dst,
                                          LocalTensor<float> src0,
                                          LocalTensor<float> src1)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
            Sub<float, false>(
                dst, src0, src1, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            Sub(dst, src0, src1, ELEMENTS);
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousMul(LocalTensor<float> dst,
                                          LocalTensor<float> src0,
                                          LocalTensor<float> src1)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
            Mul<float, false>(
                dst, src0, src1, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            Mul(dst, src0, src1, ELEMENTS);
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousAdd(LocalTensor<float> dst,
                                          LocalTensor<float> src0,
                                          LocalTensor<float> src1)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
            Add<float, false>(
                dst, src0, src1, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            Add(dst, src0, src1, ELEMENTS);
        }
    }

    template <uint32_t ELEMENTS>
    __aicore__ inline void ContiguousMulAddDst(LocalTensor<float> dst,
                                                LocalTensor<float> src0,
                                                LocalTensor<float> src1)
    {
        static_assert(ELEMENTS % KDA_GROUPED_FP32_REPEAT_ELEMENTS == 0,
                      "Reusable FP32 mask requires whole 64-element repeats");
        static_assert(ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS <= 255,
                      "Reusable FP32 mask exceeds the Vector repeat limit");
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
            MulAddDst<float, float, false>(
                dst, src0, src1, MASK_PLACEHOLDER,
                static_cast<uint8_t>(
                    ELEMENTS / KDA_GROUPED_FP32_REPEAT_ELEMENTS),
                params);
        } else {
            MulAddDst(dst, src0, src1, ELEMENTS);
        }
    }

    __aicore__ inline void BroadcastSelectedBetas(LocalTensor<float> betaBrcb,
                                                   LocalTensor<float> betaLocal)
    {
        Brcb(betaBrcb, betaLocal,
             static_cast<uint8_t>(KDA_GROUPED_SELECTED_ROWS / 8), {1, 8});
        PipeBarrier<PIPE_V>();
    }

    template <uint32_t ROWS>
    __aicore__ inline void ScaleRowsByBeta(LocalTensor<float> dst,
                                            LocalTensor<float> src,
                                            LocalTensor<float> betaBrcb)
    {
        static_assert(ROWS > 0 && ROWS <= KDA_GROUPED_SELECTED_ROWS,
                      "Beta scale row count is out of range");
        // Brcb stores one eight-float block per row.  Reuse that block for all
        // eight FP32 data blocks in a 64-column half, then advance by one beta
        // block for the next row.  Two calls cover K=128 without scalar reads.
        BinaryRepeatParams betaParams{
            1, 1, 0,
            static_cast<uint8_t>(KDA_GROUPED_K * sizeof(float) /
                                 KDA_GROUPED_DATA_BLOCK_BYTES),
            static_cast<uint8_t>(KDA_GROUPED_K * sizeof(float) /
                                 KDA_GROUPED_DATA_BLOCK_BYTES),
            1};
        Mul(dst, src, betaBrcb, 64,
            static_cast<uint8_t>(ROWS), betaParams);
        Mul(dst[64], src[64], betaBrcb, 64,
            static_cast<uint8_t>(ROWS), betaParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeCausalSelectState(uint32_t rowStart)
    {
        LocalTensor<uint8_t> mask =
            UbTensor<uint8_t, KDA_GROUPED_CAUSAL_MASK_UB>();
        __ubuf__ uint64_t *maskPtr = reinterpret_cast<__ubuf__ uint64_t *>(
            mask.GetPhyAddr());
        for (uint32_t row = 0; row < KDA_GROUPED_ROWS_PER_AIV; ++row) {
            const uint32_t firstMaskedColumn = rowStart + row + 1;
            maskPtr[row] = ~0ULL << firstMaskedColumn;
        }
        __ubuf__ uint64_t *zeroPtr = reinterpret_cast<__ubuf__ uint64_t *>(
            UbTensor<float, KDA_GROUPED_CAUSAL_ZERO_UB>().GetPhyAddr());
        for (uint32_t word = 0; word < 4; ++word) {
            zeroPtr[word] = 0;
        }
        // No Vector instruction precedes this once-per-AIV initialization.
        // Publish both scalar-written regions once; all later task/stage
        // Select instructions only read them.
        SyncSToV();
    }

    __aicore__ inline void LoadTaskFeatures(uint64_t b, uint64_t hv,
                                             uint64_t chunkStart,
                                             uint32_t rowStart)
    {
        LocalTensor<T> qTyped = UbTensor<T, KDA_GROUPED_Q_TYPED_UB>();
        LocalTensor<T> kTyped = UbTensor<T, KDA_GROUPED_K_TYPED_UB>();
        LocalTensor<float> qCache = UbTensor<float, KDA_GROUPED_Q_CACHE_UB>();
        LocalTensor<float> kCache = UbTensor<float, KDA_GROUPED_K_CACHE_UB>();
        LocalTensor<float> kBetaCache =
            UbTensor<float, KDA_GROUPED_K_BETA_CACHE_UB>();
        LocalTensor<float> gCache = UbTensor<float, KDA_GROUPED_G_CACHE_UB>();
        LocalTensor<float> refs = UbTensor<float, KDA_GROUPED_REF_UB>();
        LocalTensor<float> betaLocal = UbTensor<float, KDA_GROUPED_BETA_UB>();
        const uint64_t firstSelectedToken = chunkStart + rowStart;
        constexpr uint32_t typedBlockBytes =
            KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(T);
        constexpr uint32_t floatBlockBytes =
            KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
        constexpr uint32_t betaBlockBytes =
            KDA_GROUPED_ROWS_PER_AIV * sizeof(float);
        constexpr uint32_t referenceBytes = KDA_GROUPED_K * sizeof(float);
        constexpr uint32_t referenceStrideBytes =
            (KDA_GROUPED_BC - 1) * referenceBytes;
        DataCopyExtParams typedBlocks{KDA_GROUPED_BLOCKS, typedBlockBytes,
                                      typedBlockBytes, 0, 0};
        DataCopyExtParams floatBlocks{KDA_GROUPED_BLOCKS, floatBlockBytes,
                                      floatBlockBytes, 0, 0};
        DataCopyExtParams betaBlocks{KDA_GROUPED_BLOCKS, betaBlockBytes,
                                     betaBlockBytes, 0, 0};
        DataCopyExtParams firstLastRefs{KDA_GROUPED_MAX_OFF_PAIRS,
                                        referenceBytes,
                                        referenceStrideBytes, 0, 0};
        DataCopyExtParams middleRefs{KDA_GROUPED_BLOCKS, referenceBytes,
                                     referenceStrideBytes, 0, 0};
        DataCopyPadExtParams<T> typedNoPad{false, 0, 0, 0};
        DataCopyPadExtParams<float> floatNoPad{false, 0, 0, 0.0f};

        // The two AIVs select rows [0:8] and [8:16] from every 16-row
        // sub-block.  One strided DMA per tensor replaces four small starts.
        DataCopyPad(qTyped, q_[QOffset(b, hv, firstSelectedToken)],
                    typedBlocks, typedNoPad);
        DataCopyPad(kTyped, k_[QOffset(b, hv, firstSelectedToken)],
                    typedBlocks, typedNoPad);
        DataCopyPad(gCache, g_[VOffset(b, hv, firstSelectedToken)],
                    floatBlocks, floatNoPad);
        DataCopyPad(betaLocal, beta_[BetaOffset(b, hv, firstSelectedToken)],
                    betaBlocks, floatNoPad);
        if constexpr (KDA_GROUPED_OVERLAP_STAGE_EPILOGUE) {
            // Keep the selected input accumulator rows resident in the compact
            // reference cache's 128-byte prefix. Each block can then finalize
            // db while AIC is already processing a later stage.
            LocalTensor<float> dbLocal =
                UbTensor<float, KDA_GROUPED_DB_LOCAL_UB>();
            DataCopyPad(dbLocal, db_[BetaOffset(b, hv, firstSelectedToken)],
                        betaBlocks, floatNoPad);
        }

        // Only F1..F3, M0..M3 and R0..R2 are consumed.  Store first/middle rows
        // on color 8 and right rows after the 128-byte class pad on color 12;
        // F0 feeds the unused stage-0 left gate and R3 can never be an
        // early-block reference.
        DataCopyPad(refs,
                    g_[VOffset(b, hv, chunkStart + KDA_GROUPED_BC)],
                    firstLastRefs, floatNoPad);
        DataCopyPad(refs[KDA_GROUPED_MIDDLE_REF_OFFSET],
                    g_[VOffset(b, hv, chunkStart + KDA_GROUPED_BC / 2)],
                    middleRefs, floatNoPad);
        DataCopyPad(refs[KDA_GROUPED_RIGHT_REF_OFFSET],
                    g_[VOffset(b, hv, chunkStart + KDA_GROUPED_BC - 1)],
                    firstLastRefs, floatNoPad);
        SyncMte2ToV();
        // q/k typed banks and q/k FP32 caches are pairwise contiguous.
        Cast(qCache, qTyped, RoundMode::CAST_NONE,
             2 * KDA_GROUPED_SELECTED_ELEMENTS);
        PipeBarrier<PIPE_V>();
        if constexpr (!KDA_GROUPED_OVERLAP_SHARED_SETUP) {
            BuildPersistentBlockGates();
            if constexpr (KDA_GROUPED_FACTOR_PAIR_GATES) {
                BuildPairBridgeGates();
            }
        }
        // REDUCE_UB is otherwise idle until task completion.  Keep the one
        // 32-row beta broadcast resident there for both the input k*beta cache
        // and the final dk-left beta scale.  This also serves every stage-local
        // finalizer when overlap-stage-epilogue is enabled.  The values and
        // FP32 Mul order are unchanged; only the duplicate tail Brcb is removed.
        LocalTensor<float> betaBrcb =
            UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>();
        BroadcastSelectedBetas(betaBrcb, betaLocal);
        ScaleRowsByBeta<KDA_GROUPED_SELECTED_ROWS>(
            kBetaCache, kCache, betaBrcb);
    }

    __aicore__ inline void ZeroAccumulators()
    {
        // One Duplicate covers dq, dk-left, the 256-byte color pad and
        // dk-right.  Zeroing the dead pad preserves a single Vector launch.
        Duplicate(UbTensor<float, KDA_GROUPED_DQ_ACC_UB>(), 0.0f,
                  KDA_GROUPED_ACC_ZERO_ELEMENTS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReferenceMinusRows(
        LocalTensor<float> dst, LocalTensor<float> reference,
        LocalTensor<float> rows)
    {
        // dst[row, k] = reference[k] - rows[row, k].  References are color 8
        // or 12, row caches color 4 and destinations color 0, avoiding a
        // same-bank source pair while removing the intermediate 8xK Copy.
        BinaryRepeatParams params{1, 1, 1, 16, 0, 16};
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            Sub<float, false>(dst, reference, rows,
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, params);
            Sub<float, false>(dst[64], reference[64], rows[64],
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, params);
        } else {
            Sub(dst, reference, rows, 64, KDA_GROUPED_ROWS_PER_AIV,
                params);
            Sub(dst[64], reference[64], rows[64], 64,
                KDA_GROUPED_ROWS_PER_AIV, params);
        }
    }

    __aicore__ inline void RowsMinusReference(
        LocalTensor<float> dst, LocalTensor<float> rows,
        LocalTensor<float> reference)
    {
        // Keep src0=rows and src1=reference to preserve Sub operand order.
        BinaryRepeatParams params{1, 1, 1, 16, 16, 0};
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            Sub<float, false>(dst, rows, reference,
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, params);
            Sub<float, false>(dst[64], rows[64], reference[64],
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, params);
        } else {
            Sub(dst, rows, reference, 64, KDA_GROUPED_ROWS_PER_AIV,
                params);
            Sub(dst[64], rows[64], reference[64], 64,
                KDA_GROUPED_ROWS_PER_AIV, params);
        }
    }

    __aicore__ inline void MultiplyBridgeAcrossRows(
        LocalTensor<float> dst, LocalTensor<float> bridge,
        LocalTensor<float> outer)
    {
        // FP32 Vector masks cover 64 elements. Two calls cover K=128 while
        // src0RepStride=0 broadcasts the same bridge half over all eight rows.
        // Keep src0=bridge and src1=outer to preserve the former FP32 Mul
        // operand order exactly; no intermediate broadcast tensor is needed.
        BinaryRepeatParams bridgeParams{1, 1, 1, 16, 0, 16};
        if constexpr (KDA_GROUPED_REUSE_VECTOR_MASK) {
            Mul<float, false>(dst, bridge, outer,
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, bridgeParams);
            Mul<float, false>(dst[64], bridge[64], outer[64],
                              MASK_PLACEHOLDER,
                              KDA_GROUPED_ROWS_PER_AIV, bridgeParams);
        } else {
            Mul(dst, bridge, outer, 64, KDA_GROUPED_ROWS_PER_AIV,
                bridgeParams);
            Mul(dst[64], bridge[64], outer[64], 64,
                KDA_GROUPED_ROWS_PER_AIV, bridgeParams);
        }
    }

    __aicore__ inline void BuildPersistentBlockGates()
    {
        LocalTensor<float> refs = UbTensor<float, KDA_GROUPED_REF_UB>();
        LocalTensor<float> gCache = UbTensor<float, KDA_GROUPED_G_CACHE_UB>();
        LocalTensor<float> rightOuter =
            UbTensor<float, KDA_GROUPED_Q_TYPED_UB>();
        BeginReusableFp32Mask();
        // Right-outer gates are consumed only by later stages, so block 3 can
        // never be an early block and has no consumer.  Spell out the three
        // fixed blocks so the A2 compiler sees constant UB offsets and cannot
        // spill a dynamic loop index in this per-task Scalar hot path.
        ReferenceMinusRows(
            rightOuter[CacheBlockOffset(0)],
            refs[KDA_GROUPED_RIGHT_REF_OFFSET],
            gCache[CacheBlockOffset(0)]);
        ReferenceMinusRows(
            rightOuter[CacheBlockOffset(1)],
            refs[KDA_GROUPED_RIGHT_REF_OFFSET + KDA_GROUPED_K],
            gCache[CacheBlockOffset(1)]);
        ReferenceMinusRows(
            rightOuter[CacheBlockOffset(2)],
            refs[KDA_GROUPED_RIGHT_REF_OFFSET + 2 * KDA_GROUPED_K],
            gCache[CacheBlockOffset(2)]);
        PipeBarrier<PIPE_V>();
        ContiguousMuls<KDA_GROUPED_RIGHT_OUTER_ELEMENTS>(
            rightOuter, rightOuter, LN2);
        PipeBarrier<PIPE_V>();
        ContiguousExp<KDA_GROUPED_RIGHT_OUTER_ELEMENTS>(rightOuter,
                                                         rightOuter);
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
    }

    template <uint32_t STAGE, uint32_t EARLY>
    __aicore__ inline void BuildPairBridgeDifference()
    {
        // For a cumulative KDA gate, F_stage <= R_early elementwise.  The
        // bridge is therefore bounded by one and can safely connect the two
        // already-stable outer factors without introducing a large common
        // reference:
        //   exp2(Fs - ge) = exp2(Re - ge) * exp2(Fs - Re)
        //   exp2(gs - Re) = exp2(gs - Fs) * exp2(Fs - Re).
        ContiguousSub<KDA_GROUPED_K>(PairBridgeGate<STAGE, EARLY>(),
                                     FirstReference<STAGE>(),
                                     RightReference<EARLY>());
    }

    __aicore__ inline void BuildPairBridgeGates()
    {
        LocalTensor<float> bridges =
            UbTensor<float, KDA_GROUPED_PAIR_BRIDGE_UB>();
        BeginReusableFp32Mask();
        BuildPairBridgeDifference<1, 0>();
        BuildPairBridgeDifference<2, 0>();
        BuildPairBridgeDifference<2, 1>();
        BuildPairBridgeDifference<3, 0>();
        BuildPairBridgeDifference<3, 1>();
        BuildPairBridgeDifference<3, 2>();
        PipeBarrier<PIPE_V>();
        ContiguousMuls<KDA_GROUPED_PAIR_BRIDGE_ELEMENTS>(
            bridges, bridges, LN2);
        PipeBarrier<PIPE_V>();
        ContiguousExp<KDA_GROUPED_PAIR_BRIDGE_ELEMENTS>(bridges, bridges);
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
    }

    template <uint32_t STAGE>
    __aicore__ inline void BuildStageGates()
    {
        LocalTensor<float> gCache = UbTensor<float, KDA_GROUPED_G_CACHE_UB>();
        LocalTensor<float> diagRef = MiddleReference<STAGE>();
        LocalTensor<float> leftOuter = LeftOuterGate<STAGE>();
        LocalTensor<float> diagLate = DiagLateGate<STAGE>();
        LocalTensor<float> diagEarly = DiagEarlyGate<STAGE>();
        LocalTensor<float> gLate = gCache[CacheBlockOffset(STAGE)];
        BeginReusableFp32Mask();
        if constexpr (STAGE > 0) {
            // Stage 0 has no off-left predecessor, so its left-outer gate has
            // no consumer.  Do not spend a full 8x128 Exp on it.
            RowsMinusReference(leftOuter, gLate, FirstReference<STAGE>());
        }
        RowsMinusReference(diagLate, gLate, diagRef);
        ReferenceMinusRows(diagEarly, diagRef, gLate);
        PipeBarrier<PIPE_V>();

        if constexpr (STAGE > 0) {
            ContiguousMuls<3 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                leftOuter, leftOuter, LN2);
            PipeBarrier<PIPE_V>();
            ContiguousExp<3 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(leftOuter,
                                                               leftOuter);
        } else {
            ContiguousMuls<2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                diagLate, diagLate, LN2);
            PipeBarrier<PIPE_V>();
            ContiguousExp<2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(diagLate,
                                                               diagLate);
        }
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
    }

    template <uint32_t STAGE>
    __aicore__ inline void LoadMaskAndWriteStageA(uint64_t b, uint64_t hv,
                                                   uint64_t chunkStart,
                                                   uint32_t rowStart,
                                                   uint32_t slot)
    {
        constexpr uint32_t prefix = (STAGE + 1) * KDA_GROUPED_BC;
        constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
        constexpr uint32_t oneMatrixElements =
            KDA_GROUPED_ROWS_PER_AIV * prefix;
        LocalTensor<float> raw = UbTensor<float, KDA_GROUPED_RAW_DA_UB>();
        const uint64_t sourceToken = chunkStart + STAGE * KDA_GROUPED_BC + rowStart;
        // Stages 2/3 reuse RAW_DA_UB immediately after ConsumeStageAiv<0/1>
        // used the same region as ScratchBank<2> on PIPE_V.  Order that final
        // vector read before MTE2 overwrites the slab for the next dA tile.
        if constexpr (STAGE >= KDA_GROUPED_QUEUE_DEPTH) {
            SyncVToMte2();
        }
        DataCopyExtParams rowParams{KDA_GROUPED_ROWS_PER_AIV,
                                    prefix * sizeof(float),
                                    (KDA_GROUPED_BT - prefix) * sizeof(float),
                                    0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(raw, dAqk_[AOffset(b, hv, sourceToken, 0)], rowParams, noPad);
        DataCopyPad(raw[oneMatrixElements],
                    dAkk_[AOffset(b, hv, sourceToken, 0)], rowParams, noPad);
        // Leave MTE2 in flight while the vector pipe builds gates.  The shared
        // event is safe here because every earlier MTE2_V dependency has been
        // waited and no nested MTE2_V synchronization occurs in BuildStageGates.
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        BuildStageGates<STAGE>();

        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        LocalTensor<uint8_t> causalMask =
            UbTensor<uint8_t, KDA_GROUPED_CAUSAL_MASK_UB>();
        LocalTensor<float> zero =
            UbTensor<float, KDA_GROUPED_CAUSAL_ZERO_UB>();
        const uint8_t rowStride = static_cast<uint8_t>(
            prefix * sizeof(float) / KDA_GROUPED_DATA_BLOCK_BYTES);
        BinaryRepeatParams maskParams{1, 0, 1, rowStride, 0, rowStride};
        Select(raw[offPrefix], causalMask, zero, raw[offPrefix],
               SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_GROUPED_BC,
               static_cast<uint8_t>(KDA_GROUPED_ROWS_PER_AIV), maskParams);
        Select(raw[oneMatrixElements + offPrefix], causalMask, zero,
               raw[oneMatrixElements + offPrefix],
               SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_GROUPED_BC,
               static_cast<uint8_t>(KDA_GROUPED_ROWS_PER_AIV), maskParams);
        PipeBarrier<PIPE_V>();

        SyncVToMte3();
        if constexpr (KDA_GROUPED_PACK_STAGE_A) {
            // The causal diagonal is already masked in raw.  Preserve the
            // complete prefix as one RowMajor [32,prefix] A matrix; its
            // transpose supplies every unchanged right-side GEMM as well.
            if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
                CopyStageAbNzRows(
                    slot, KDA_GROUPED_PACKED_A, 2 * KDA_GROUPED_BC,
                    prefix, rowStart, raw, prefix,
                    KDA_GROUPED_ROWS_PER_AIV);
                CopyStageAbNzRows(
                    slot, KDA_GROUPED_PACKED_A, 2 * KDA_GROUPED_BC,
                    prefix, KDA_GROUPED_BC + rowStart,
                    raw[oneMatrixElements], prefix,
                    KDA_GROUPED_ROWS_PER_AIV);
            } else {
                CopyStageAb(slot,
                            KDA_GROUPED_PACKED_A + rowStart * prefix,
                            raw, oneMatrixElements);
                CopyStageAb(slot,
                            KDA_GROUPED_PACKED_A +
                                (KDA_GROUPED_BC + rowStart) * prefix,
                            raw[oneMatrixElements], oneMatrixElements);
            }
        } else {
            if constexpr (STAGE > 0) {
                if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
                    CopyStageAbNzRows(
                        slot, KDA_GROUPED_A_OFF, 2 * KDA_GROUPED_BC,
                        offPrefix, rowStart, raw, prefix,
                        KDA_GROUPED_ROWS_PER_AIV);
                    CopyStageAbNzRows(
                        slot, KDA_GROUPED_A_OFF, 2 * KDA_GROUPED_BC,
                        offPrefix, KDA_GROUPED_BC + rowStart,
                        raw[oneMatrixElements], prefix,
                        KDA_GROUPED_ROWS_PER_AIV);
                } else {
                    DataCopyExtParams offParams{
                        KDA_GROUPED_ROWS_PER_AIV,
                        offPrefix * sizeof(float),
                        KDA_GROUPED_BC * sizeof(float) /
                            KDA_GROUPED_DATA_BLOCK_BYTES,
                        0, 0};
                    DataCopyPad(
                        workspace_[SlotBase(slot) + KDA_GROUPED_A_OFF +
                                   rowStart * offPrefix],
                        raw, offParams);
                    DataCopyPad(
                        workspace_[SlotBase(slot) + KDA_GROUPED_A_OFF +
                                   (KDA_GROUPED_BC + rowStart) * offPrefix],
                        raw[oneMatrixElements], offParams);
                }
            }
            if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
                CopyStageAbNzRows(
                    slot, KDA_GROUPED_A_DIAG, 2 * KDA_GROUPED_BC,
                    KDA_GROUPED_BC, rowStart, raw[offPrefix], prefix,
                    KDA_GROUPED_ROWS_PER_AIV);
                CopyStageAbNzRows(
                    slot, KDA_GROUPED_A_DIAG, 2 * KDA_GROUPED_BC,
                    KDA_GROUPED_BC, KDA_GROUPED_BC + rowStart,
                    raw[oneMatrixElements + offPrefix], prefix,
                    KDA_GROUPED_ROWS_PER_AIV);
            } else {
                DataCopyExtParams diagParams{
                    KDA_GROUPED_ROWS_PER_AIV,
                    KDA_GROUPED_BC * sizeof(float),
                    offPrefix * sizeof(float) /
                        KDA_GROUPED_DATA_BLOCK_BYTES,
                    0, 0};
                DataCopyPad(
                    workspace_[SlotBase(slot) + KDA_GROUPED_A_DIAG +
                               rowStart * KDA_GROUPED_BC],
                    raw[offPrefix], diagParams);
                DataCopyPad(
                    workspace_[SlotBase(slot) + KDA_GROUPED_A_DIAG +
                               (KDA_GROUPED_BC + rowStart) *
                                   KDA_GROUPED_BC],
                    raw[oneMatrixElements + offPrefix], diagParams);
            }
        }
        SyncMte3ToV();
    }

    template <uint32_t STAGE, uint32_t EARLY, uint32_t SCRATCH_SET = 0>
    __aicore__ inline void BuildPairInnerGates()
    {
        static_assert(STAGE > 0 && EARLY < STAGE,
                      "Grouped off-diagonal pair is out of range");
        LocalTensor<float> leftInner = PairScratchBank<SCRATCH_SET, 0>();
        LocalTensor<float> rightInner = PairScratchBank<SCRATCH_SET, 1>();
        if constexpr (KDA_GROUPED_FACTOR_PAIR_GATES) {
            // Broadcast the K-wide bridge directly as Mul source 0. This
            // removes two full 8xK Copy passes and their dependency barrier for
            // every pair while retaining the exact bridge*outer FP32 multiply.
            MultiplyBridgeAcrossRows(
                leftInner, PairBridgeGate<STAGE, EARLY>(),
                RightOuterGate<EARLY>());
            MultiplyBridgeAcrossRows(
                rightInner, PairBridgeGate<STAGE, EARLY>(),
                LeftOuterGate<STAGE>());
            PipeBarrier<PIPE_V>();
        } else {
            // Direct rollback path for precision bisects.
            LocalTensor<float> gCache = UbTensor<float, KDA_GROUPED_G_CACHE_UB>();
            ReferenceMinusRows(leftInner, FirstReference<STAGE>(),
                               gCache[CacheBlockOffset(EARLY)]);
            RowsMinusReference(rightInner, gCache[CacheBlockOffset(STAGE)],
                               RightReference<EARLY>());
            PipeBarrier<PIPE_V>();
            ContiguousMuls<2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                leftInner, leftInner, LN2);
            PipeBarrier<PIPE_V>();
            ContiguousExp<2 * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(leftInner,
                                                               leftInner);
            PipeBarrier<PIPE_V>();
        }
    }

    template <uint32_t STAGE, uint32_t EARLY, uint32_t SCRATCH_SET = 0,
              bool PUBLISH_DONE = false>
    __aicore__ inline void BuildAndWritePairB(uint32_t rowStart,
                                               uint32_t slot)
    {
        LocalTensor<float> qCache = UbTensor<float, KDA_GROUPED_Q_CACHE_UB>();
        LocalTensor<float> kCache = UbTensor<float, KDA_GROUPED_K_CACHE_UB>();
        LocalTensor<float> kBetaCache =
            UbTensor<float, KDA_GROUPED_K_BETA_CACHE_UB>();
        static_assert(!PUBLISH_DONE || KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH,
                      "Delayed pair completion requires scratch ping-pong");
        BeginReusableFp32Mask();
        BuildPairInnerGates<STAGE, EARLY, SCRATCH_SET>();

        // Once the left product has consumed ScratchBank<0>, that gate bank is
        // dead.  Likewise the final right product may overwrite its own gate.
        // Keep all three products resident, then submit one MTE3 batch instead
        // of three Mul -> event -> DataCopy -> event sequences.
        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 2>(),
            kCache[CacheBlockOffset(EARLY)],
            PairScratchBank<SCRATCH_SET, 0>());
        PipeBarrier<PIPE_V>();

        constexpr uint32_t pairBase =
            (KDA_GROUPED_PACK_STAGE_A ?
                 KDA_GROUPED_PACKED_B_OFF_RIGHT :
                 KDA_GROUPED_B_OFF_RIGHT) +
            EARLY * KDA_GROUPED_RIGHT_PAIR_ELEMENTS;
        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 0>(),
            qCache[CacheBlockOffset(STAGE)],
            PairScratchBank<SCRATCH_SET, 1>());
        PipeBarrier<PIPE_V>();

        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 1>(),
            kBetaCache[CacheBlockOffset(STAGE)],
            PairScratchBank<SCRATCH_SET, 1>());
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
        SyncVToMte3();
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
            CopyStageAbNzRows(
                slot, KDA_GROUPED_B_OFF_LEFT, offPrefix,
                KDA_GROUPED_K, EARLY * KDA_GROUPED_BC + rowStart,
                PairScratchBank<SCRATCH_SET, 2>(), KDA_GROUPED_K,
                KDA_GROUPED_ROWS_PER_AIV);
        } else {
            CopyStageAb(
                slot,
                KDA_GROUPED_B_OFF_LEFT +
                    (EARLY * KDA_GROUPED_BC + rowStart) * KDA_GROUPED_K,
                PairScratchBank<SCRATCH_SET, 2>(),
                KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        }
        CopyStageRightBRows(slot, pairBase, rowStart,
                            PairScratchBank<SCRATCH_SET, 0>());
        if constexpr (KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH) {
            if constexpr (PUBLISH_DONE) {
                // Do not wait here.  The vector pipe may build the next pair
                // or diagonal in the alternate set while MTE3 drains this set.
                PublishPairScratchDone<SCRATCH_SET>();
            }
        } else {
            SyncMte3ToV();
        }
    }

    template <uint32_t STAGE, uint32_t SCRATCH_SET = 0>
    __aicore__ inline void BuildAndWriteDiagonalB(uint32_t rowStart,
                                                   uint32_t slot)
    {
        LocalTensor<float> qCache = UbTensor<float, KDA_GROUPED_Q_CACHE_UB>();
        LocalTensor<float> kCache = UbTensor<float, KDA_GROUPED_K_CACHE_UB>();
        LocalTensor<float> kBetaCache =
            UbTensor<float, KDA_GROUPED_K_BETA_CACHE_UB>();
        BeginReusableFp32Mask();
        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 0>(),
            kCache[CacheBlockOffset(STAGE)], DiagEarlyGate<STAGE>());
        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 1>(),
            qCache[CacheBlockOffset(STAGE)], DiagLateGate<STAGE>());
        ContiguousMul<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            PairScratchBank<SCRATCH_SET, 2>(),
            kBetaCache[CacheBlockOffset(STAGE)], DiagLateGate<STAGE>());
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
        SyncVToMte3();
        constexpr uint32_t rightBase =
            KDA_GROUPED_PACK_STAGE_A ?
                KDA_GROUPED_PACKED_B_DIAG_RIGHT :
                KDA_GROUPED_B_DIAG_RIGHT;
        if constexpr (KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER) {
            CopyStageAbNzRows(
                slot, KDA_GROUPED_B_DIAG_LEFT, KDA_GROUPED_BC,
                KDA_GROUPED_K, rowStart,
                PairScratchBank<SCRATCH_SET, 0>(), KDA_GROUPED_K,
                KDA_GROUPED_ROWS_PER_AIV);
        } else {
            CopyStageAb(slot,
                        KDA_GROUPED_B_DIAG_LEFT +
                            rowStart * KDA_GROUPED_K,
                        PairScratchBank<SCRATCH_SET, 0>(),
                        KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        }
        CopyStageRightBRows(slot, rightBase, rowStart,
                            PairScratchBank<SCRATCH_SET, 1>());
        // This final drain covers every unsignalled pair DMA as well as the
        // diagonal.  It also preserves the original MTE3->MTE2 dependency for
        // the next stage's RAW_DA load.
        SyncMte3ToMte2();
        if constexpr (KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH) {
            // event0 has either not been published in this stage or has
            // already been consumed before its scratch set was reused.  Reuse
            // it for the final drain so the A2 MTE3_V pool never needs a third
            // event in addition to the two ping-pong completion events.
            SetFlag<HardEvent::MTE3_V>(pairScratchDone0_);
            WaitFlag<HardEvent::MTE3_V>(pairScratchDone0_);
        } else {
            SyncMte3ToV();
        }
    }

    template <uint32_t STAGE>
    __aicore__ inline void BuildAndWriteStageB(uint32_t rowStart,
                                                uint32_t slot)
    {
        if constexpr (KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH) {
            if constexpr (STAGE == 0) {
                BuildAndWriteDiagonalB<STAGE, 0>(rowStart, slot);
            } else if constexpr (STAGE == 1) {
                BuildAndWritePairB<STAGE, 0, 0, false>(rowStart, slot);
                BuildAndWriteDiagonalB<STAGE, 1>(rowStart, slot);
            } else if constexpr (STAGE == 2) {
                BuildAndWritePairB<STAGE, 0, 0, true>(rowStart, slot);
                BuildAndWritePairB<STAGE, 1, 1, false>(rowStart, slot);
                WaitPairScratchDone<0>();
                BuildAndWriteDiagonalB<STAGE, 0>(rowStart, slot);
            } else {
                BuildAndWritePairB<STAGE, 0, 0, true>(rowStart, slot);
                BuildAndWritePairB<STAGE, 1, 1, true>(rowStart, slot);
                WaitPairScratchDone<0>();
                BuildAndWritePairB<STAGE, 2, 0, false>(rowStart, slot);
                WaitPairScratchDone<1>();
                BuildAndWriteDiagonalB<STAGE, 1>(rowStart, slot);
            }
        } else {
            if constexpr (STAGE > 0) {
                BuildAndWritePairB<STAGE, 0>(rowStart, slot);
            }
            if constexpr (STAGE > 1) {
                BuildAndWritePairB<STAGE, 1>(rowStart, slot);
            }
            if constexpr (STAGE > 2) {
                BuildAndWritePairB<STAGE, 2>(rowStart, slot);
            }
            BuildAndWriteDiagonalB<STAGE>(rowStart, slot);
        }
    }

    template <uint32_t STAGE>
    __aicore__ inline void PrepareStageAiv(uint64_t b, uint64_t hv,
                                            uint64_t chunkStart,
                                            uint32_t rowStart,
                                            uint32_t slot)
    {
        AcquireStageAbSlot(slot);
        LoadMaskAndWriteStageA<STAGE>(b, hv, chunkStart, rowStart, slot);
        BuildAndWriteStageB<STAGE>(rowStart, slot);
        PublishStageAbSlot(slot);
    }

    template <uint32_t M, uint32_t K, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunRowMajorAic(BlockMmad &blockMmad,
                                          uint64_t slotBase,
                                          uint32_t aOffset,
                                          uint32_t bOffset,
                                          uint32_t cOffset)
    {
        auto layoutA = tla::MakeLayout<float, Catlass::layout::RowMajor>(M, K);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(K, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(M, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(slotBase, aOffset, layoutA);
        auto tensorB = MakeStageInputTensor(slotBase, bOffset, layoutB);
        auto tensorC = tla::MakeTensor(workspace_[slotBase + cOffset], layoutC,
                                       Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(M, K));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(K, KDA_GROUPED_K));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0),
                              tla::MakeShape(M, KDA_GROUPED_K));
        Catlass::GemmCoord shape{M, KDA_GROUPED_K, K};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <uint32_t M, uint32_t K, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunColumnMajorAic(BlockMmad &blockMmad,
                                             uint64_t slotBase,
                                             uint32_t aOffset,
                                             uint32_t bOffset,
                                             uint32_t cOffset)
    {
        auto layoutA = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(M, K);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(K, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(M, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(slotBase, aOffset, layoutA);
        auto tensorB = MakeStageInputTensor(slotBase, bOffset, layoutB);
        auto tensorC = tla::MakeTensor(workspace_[slotBase + cOffset], layoutC,
                                       Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(M, K));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(K, KDA_GROUPED_K));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0),
                              tla::MakeShape(M, KDA_GROUPED_K));
        Catlass::GemmCoord shape{M, KDA_GROUPED_K, K};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <uint32_t STAGE, uint32_t EARLY, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunOffRightPairAic(BlockMmad &blockMmad,
                                              uint64_t slotBase)
    {
        static_assert(STAGE > 0 && EARLY < STAGE,
                      "Grouped off-right pair is out of range");
        constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
        constexpr uint32_t aColumns =
            KDA_GROUPED_PACK_STAGE_A ?
                offPrefix + KDA_GROUPED_BC : offPrefix;
        constexpr uint32_t aBase =
            KDA_GROUPED_PACK_STAGE_A ?
                KDA_GROUPED_PACKED_A : KDA_GROUPED_A_OFF;
        constexpr uint32_t rightBase =
            KDA_GROUPED_PACK_STAGE_A ?
                KDA_GROUPED_PACKED_B_OFF_RIGHT :
                KDA_GROUPED_B_OFF_RIGHT;
        constexpr uint32_t pairOffset =
            EARLY * KDA_GROUPED_RIGHT_PAIR_ELEMENTS;
        // Split Aoff is RowMajor [32, offPrefix], while packed A is RowMajor
        // [32, prefix].  Reinterpret either storage as the matching
        // ColumnMajor [aColumns, 32] transpose; GetTile preserves that parent
        // leading stride while selecting one early 16-row block.
        auto layoutA = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(
            aColumns, 2 * KDA_GROUPED_BC);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            KDA_GROUPED_BC, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(slotBase, aBase, layoutA);
        auto tensorB = MakeStageInputTensor(
            slotBase, rightBase + pairOffset, layoutB);
        auto tensorC = tla::MakeTensor(
            workspace_[slotBase + rightBase + pairOffset],
            layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(EARLY * KDA_GROUPED_BC, 0),
            tla::MakeShape(KDA_GROUPED_BC, 2 * KDA_GROUPED_BC));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(0, 0),
            tla::MakeShape(2 * KDA_GROUPED_BC, KDA_GROUPED_K));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_GROUPED_BC, KDA_GROUPED_K));
        Catlass::GemmCoord shape{KDA_GROUPED_BC, KDA_GROUPED_K,
                                2 * KDA_GROUPED_BC};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <uint32_t STAGE, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunPackedDiagRightAic(BlockMmad &blockMmad,
                                                  uint64_t slotBase)
    {
        static_assert(STAGE < KDA_GROUPED_STAGES,
                      "Packed diagonal-right stage is out of range");
        constexpr uint32_t prefix = (STAGE + 1) * KDA_GROUPED_BC;
        auto layoutA = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(
            prefix, 2 * KDA_GROUPED_BC);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            KDA_GROUPED_BC, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(
            slotBase, KDA_GROUPED_PACKED_A, layoutA);
        auto tensorB = MakeStageInputTensor(
            slotBase, KDA_GROUPED_PACKED_B_DIAG_RIGHT, layoutB);
        auto tensorC = tla::MakeTensor(
            workspace_[slotBase + KDA_GROUPED_C_DIAG_RIGHT],
            layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(STAGE * KDA_GROUPED_BC, 0),
            tla::MakeShape(KDA_GROUPED_BC, 2 * KDA_GROUPED_BC));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(0, 0),
            tla::MakeShape(2 * KDA_GROUPED_BC, KDA_GROUPED_K));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_GROUPED_BC, KDA_GROUPED_K));
        Catlass::GemmCoord shape{KDA_GROUPED_BC, KDA_GROUPED_K,
                                2 * KDA_GROUPED_BC};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <uint32_t STAGE, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunPackedOffLeftAic(BlockMmad &blockMmad,
                                                uint64_t slotBase)
    {
        static_assert(STAGE > 0 && STAGE < KDA_GROUPED_STAGES,
                      "Packed off-left stage is out of range");
        constexpr uint32_t prefix = (STAGE + 1) * KDA_GROUPED_BC;
        constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
        auto layoutA = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, prefix);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            offPrefix, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(
            slotBase, KDA_GROUPED_PACKED_A, layoutA);
        auto tensorB = MakeStageInputTensor(
            slotBase, KDA_GROUPED_B_OFF_LEFT, layoutB);
        auto tensorC = tla::MakeTensor(
            workspace_[slotBase + KDA_GROUPED_C_OFF_LEFT], layoutC,
            Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(0, 0),
            tla::MakeShape(2 * KDA_GROUPED_BC, offPrefix));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(0, 0),
            tla::MakeShape(offPrefix, KDA_GROUPED_K));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(0, 0),
            tla::MakeShape(2 * KDA_GROUPED_BC, KDA_GROUPED_K));
        Catlass::GemmCoord shape{2 * KDA_GROUPED_BC, KDA_GROUPED_K,
                                offPrefix};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <uint32_t STAGE, bool MANAGE_FLAGS = true,
              typename BlockMmad>
    __aicore__ inline void RunPackedDiagLeftAic(BlockMmad &blockMmad,
                                                 uint64_t slotBase)
    {
        static_assert(STAGE < KDA_GROUPED_STAGES,
                      "Packed diagonal-left stage is out of range");
        constexpr uint32_t prefix = (STAGE + 1) * KDA_GROUPED_BC;
        auto layoutA = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, prefix);
        auto layoutB = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            KDA_GROUPED_BC, KDA_GROUPED_K);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(
            2 * KDA_GROUPED_BC, KDA_GROUPED_K);
        auto tensorA = MakeStageInputTensor(
            slotBase, KDA_GROUPED_PACKED_A, layoutA);
        auto tensorB = MakeStageInputTensor(
            slotBase, KDA_GROUPED_B_DIAG_LEFT, layoutB);
        auto tensorC = tla::MakeTensor(
            workspace_[slotBase + KDA_GROUPED_PACKED_C_DIAG_LEFT], layoutC,
            Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            tensorA, tla::MakeCoord(0, STAGE * KDA_GROUPED_BC),
            tla::MakeShape(2 * KDA_GROUPED_BC, KDA_GROUPED_BC));
        auto blockB = GetTile(
            tensorB, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_GROUPED_BC, KDA_GROUPED_K));
        auto blockC = GetTile(
            tensorC, tla::MakeCoord(0, 0),
            tla::MakeShape(2 * KDA_GROUPED_BC, KDA_GROUPED_K));
        Catlass::GemmCoord shape{2 * KDA_GROUPED_BC, KDA_GROUPED_K,
                                KDA_GROUPED_BC};
        if constexpr (MANAGE_FLAGS) {
            blockMmad.preSetFlags();
        }
        RunStageMmad(blockMmad, blockA, blockB, blockC, shape);
        if constexpr (MANAGE_FLAGS) {
            blockMmad.finalWaitFlags();
        }
    }

    template <typename LeftMmad, typename RightMmad>
    __aicore__ inline void ComputeDiagonalAic(LeftMmad &leftMmad,
                                               RightMmad &rightMmad,
                                               uint32_t slot)
    {
        const uint64_t slotBase = SlotBase(slot);
        if constexpr (KDA_GROUPED_PACK_STAGE_A) {
            RunPackedDiagRightAic<0>(rightMmad, slotBase);
            RunPackedDiagLeftAic<0>(leftMmad, slotBase);
        } else {
            RunColumnMajorAic<16, 32>(rightMmad, slotBase, KDA_GROUPED_A_DIAG,
                                      KDA_GROUPED_B_DIAG_RIGHT,
                                      KDA_GROUPED_C_DIAG_RIGHT);
            RunRowMajorAic<32, 16>(leftMmad, slotBase, KDA_GROUPED_A_DIAG,
                                   KDA_GROUPED_B_DIAG_LEFT,
                                   KDA_GROUPED_C_DIAG_LEFT);
        }
    }

    template <uint32_t STAGE, typename OffLeftMmad, typename RightMmad,
              typename DiagLeftMmad>
    __aicore__ inline void ComputeGroupedStageAic(OffLeftMmad &offLeftMmad,
                                                   RightMmad &rightMmad,
                                                   DiagLeftMmad &diagLeftMmad,
                                                   uint32_t slot)
    {
        static_assert(STAGE > 0 && STAGE < KDA_GROUPED_STAGES,
                      "Grouped off-diagonal stage is out of range");
        constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
        const uint64_t slotBase = SlotBase(slot);
        if constexpr (KDA_GROUPED_PACK_STAGE_A) {
            RunPackedOffLeftAic<STAGE>(offLeftMmad, slotBase);
            rightMmad.preSetFlags();
            RunOffRightPairAic<STAGE, 0, false>(rightMmad, slotBase);
            if constexpr (STAGE > 1) {
                RunOffRightPairAic<STAGE, 1, false>(rightMmad, slotBase);
            }
            if constexpr (STAGE > 2) {
                RunOffRightPairAic<STAGE, 2, false>(rightMmad, slotBase);
            }
            RunPackedDiagRightAic<STAGE, false>(rightMmad, slotBase);
            rightMmad.finalWaitFlags();
            RunPackedDiagLeftAic<STAGE>(diagLeftMmad, slotBase);
        } else {
            RunRowMajorAic<32, offPrefix>(offLeftMmad, slotBase,
                                          KDA_GROUPED_A_OFF,
                                          KDA_GROUPED_B_OFF_LEFT,
                                          KDA_GROUPED_C_OFF_LEFT);
            // Off-right pairs and diagonal-right all use the same M16/K32
            // BlockMmad instance.  Keep its L1/L0 event envelope alive across
            // consecutive calls instead of draining it 2--4 times.
            rightMmad.preSetFlags();
            RunOffRightPairAic<STAGE, 0, false>(rightMmad, slotBase);
            if constexpr (STAGE > 1) {
                RunOffRightPairAic<STAGE, 1, false>(rightMmad, slotBase);
            }
            if constexpr (STAGE > 2) {
                RunOffRightPairAic<STAGE, 2, false>(rightMmad, slotBase);
            }
            RunColumnMajorAic<16, 32, false>(
                rightMmad, slotBase, KDA_GROUPED_A_DIAG,
                KDA_GROUPED_B_DIAG_RIGHT, KDA_GROUPED_C_DIAG_RIGHT);
            rightMmad.finalWaitFlags();
            RunRowMajorAic<32, 16>(diagLeftMmad, slotBase,
                                   KDA_GROUPED_A_DIAG,
                                   KDA_GROUPED_B_DIAG_LEFT,
                                   KDA_GROUPED_C_DIAG_LEFT);
        }
    }

    template <typename LeftMmad, typename RightMmad>
    __aicore__ inline void ComputeDiagonalPersistentAic(
        LeftMmad &leftMmad, RightMmad &rightMmad, uint32_t slot)
    {
        const uint64_t slotBase = SlotBase(slot);
        if constexpr (KDA_GROUPED_PACK_STAGE_A) {
            RunPackedDiagRightAic<0, false>(rightMmad, slotBase);
            RunPackedDiagLeftAic<0, false>(leftMmad, slotBase);
        } else {
            RunColumnMajorAic<16, 32, false>(
                rightMmad, slotBase, KDA_GROUPED_A_DIAG,
                KDA_GROUPED_B_DIAG_RIGHT, KDA_GROUPED_C_DIAG_RIGHT);
            RunRowMajorAic<32, 16, false>(
                leftMmad, slotBase, KDA_GROUPED_A_DIAG,
                KDA_GROUPED_B_DIAG_LEFT, KDA_GROUPED_C_DIAG_LEFT);
        }
    }

    template <uint32_t STAGE, typename LeftMmad, typename RightMmad>
    __aicore__ inline void ComputeGroupedStagePersistentAic(
        LeftMmad &leftMmad, RightMmad &rightMmad, uint32_t slot)
    {
        static_assert(STAGE > 0 && STAGE < KDA_GROUPED_STAGES,
                      "Grouped off-diagonal stage is out of range");
        constexpr uint32_t offPrefix = STAGE * KDA_GROUPED_BC;
        const uint64_t slotBase = SlotBase(slot);

        if constexpr (KDA_GROUPED_PACK_STAGE_A) {
            RunPackedOffLeftAic<STAGE, false>(leftMmad, slotBase);
            RunOffRightPairAic<STAGE, 0, false>(rightMmad, slotBase);
            if constexpr (STAGE > 1) {
                RunOffRightPairAic<STAGE, 1, false>(rightMmad, slotBase);
            }
            if constexpr (STAGE > 2) {
                RunOffRightPairAic<STAGE, 2, false>(rightMmad, slotBase);
            }
            RunPackedDiagRightAic<STAGE, false>(rightMmad, slotBase);
            RunPackedDiagLeftAic<STAGE, false>(leftMmad, slotBase);
        } else {
            // Preserve the scoped path's GM/workspace order exactly.  The only
            // difference is that both engines' local event envelopes remain live.
            RunRowMajorAic<32, offPrefix, false>(
                leftMmad, slotBase, KDA_GROUPED_A_OFF,
                KDA_GROUPED_B_OFF_LEFT, KDA_GROUPED_C_OFF_LEFT);
            RunOffRightPairAic<STAGE, 0, false>(rightMmad, slotBase);
            if constexpr (STAGE > 1) {
                RunOffRightPairAic<STAGE, 1, false>(rightMmad, slotBase);
            }
            if constexpr (STAGE > 2) {
                RunOffRightPairAic<STAGE, 2, false>(rightMmad, slotBase);
            }
            RunColumnMajorAic<16, 32, false>(
                rightMmad, slotBase, KDA_GROUPED_A_DIAG,
                KDA_GROUPED_B_DIAG_RIGHT, KDA_GROUPED_C_DIAG_RIGHT);
            RunRowMajorAic<32, 16, false>(
                leftMmad, slotBase, KDA_GROUPED_A_DIAG,
                KDA_GROUPED_B_DIAG_LEFT, KDA_GROUPED_C_DIAG_LEFT);
        }
    }

    template <uint32_t ROWS>
    __aicore__ inline void ReduceDbProductRows(
        LocalTensor<float> dbCompact, LocalTensor<float> dbAcc,
        LocalTensor<float> product)
    {
        static_assert(ROWS > 0 && ROWS <= KDA_GROUPED_SELECTED_ROWS,
                      "Grouped db reduction row count is out of range");
        static_assert(ROWS <= 255,
                      "WholeReduceSum repeat count must fit uint8_t");
        static_assert(KDA_GROUPED_K == 128,
                      "Grouped db reduction assumes two 64-element halves");
        constexpr uint32_t partialsPerRow = 8;
        constexpr uint32_t rowStrideBlocks =
            KDA_GROUPED_K * sizeof(float) / KDA_GROUPED_DATA_BLOCK_BYTES;
        static_assert(rowStrideBlocks == 16,
                      "Grouped db source rows must be 16 data blocks apart");

        if constexpr (KDA_GROUPED_COALESCE_DB_REDUCE) {
            // WholeReduceSum dstRepStride is measured in reduced FP32 values,
            // while srcRepStride is measured in 32-byte data blocks.  These
            // two calls therefore write the same row*8 + {0,1} partials as the
            // rollback loop; only the independent row issue order changes.
            WholeReduceSum(dbAcc, product, 64, ROWS,
                           partialsPerRow, 1, rowStrideBlocks);
            WholeReduceSum(dbAcc[1], product[64], 64, ROWS,
                           partialsPerRow, 1, rowStrideBlocks);
        } else {
            for (uint32_t row = 0; row < ROWS; ++row) {
                WholeReduceSum(dbAcc[row * partialsPerRow],
                               product[row * KDA_GROUPED_K],
                               64, KDA_GROUPED_K / 64, 1, 1, 8);
            }
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dbCompact, dbAcc, KDA_GROUPED_K / 64,
                       ROWS, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    template <uint32_t STAGE>
    __aicore__ inline void FinalizeStageLeft()
    {
        static_assert(STAGE < KDA_GROUPED_STAGES,
                      "Grouped epilogue stage is out of range");
        constexpr uint32_t cacheOffset =
            STAGE * KDA_GROUPED_ROWS_PER_AIV * KDA_GROUPED_K;
        constexpr uint32_t dbOffset =
            STAGE * KDA_GROUPED_ROWS_PER_AIV;
        constexpr uint32_t betaBrcbOffset = dbOffset * 8;
        LocalTensor<float> product = ScratchBank<0>();
        LocalTensor<float> dbLocal =
            UbTensor<float, KDA_GROUPED_DB_LOCAL_UB>();
        LocalTensor<float> dbAcc =
            UbTensor<float, KDA_GROUPED_DB_STAGE_SCRATCH_UB>();
        LocalTensor<float> dbCompact =
            dbAcc[KDA_GROUPED_DB_STAGE_ACC_ELEMENTS];
        LocalTensor<float> betaBrcb =
            UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>();
        LocalTensor<float> dkLeftAcc =
            UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>();
        LocalTensor<float> kCache =
            UbTensor<float, KDA_GROUPED_K_CACHE_UB>();

        // dk-left for this block is complete immediately after its stage is
        // consumed.  Finalize the K reduction and beta scale now so stages
        // 0--2 can overlap this AIV work with a later AIC stage.
        Mul(product, dkLeftAcc[cacheOffset], kCache[cacheOffset],
            KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        PipeBarrier<PIPE_V>();
        ReduceDbProductRows<KDA_GROUPED_ROWS_PER_AIV>(
            dbCompact, dbAcc, product);
        Add(dbLocal[dbOffset], dbLocal[dbOffset], dbCompact,
            KDA_GROUPED_ROWS_PER_AIV);
        ScaleRowsByBeta<KDA_GROUPED_ROWS_PER_AIV>(
            dkLeftAcc[cacheOffset], dkLeftAcc[cacheOffset],
            betaBrcb[betaBrcbOffset]);
    }

    template <uint32_t STAGE>
    __aicore__ inline void ConsumeStageAiv(uint32_t slot,
                                            uint32_t rowStart)
    {
        const uint64_t slotBase = SlotBase(slot);
        constexpr uint32_t offRightBase =
            KDA_GROUPED_PACK_STAGE_A ?
                KDA_GROUPED_PACKED_C_OFF_RIGHT : KDA_GROUPED_C_OFF_RIGHT;
        constexpr uint32_t diagLeftBase =
            KDA_GROUPED_PACK_STAGE_A ?
                KDA_GROUPED_PACKED_C_DIAG_LEFT : KDA_GROUPED_C_DIAG_LEFT;
        LocalTensor<float> dqAcc = UbTensor<float, KDA_GROUPED_DQ_ACC_UB>();
        LocalTensor<float> dkLeftAcc =
            UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>();
        LocalTensor<float> dkRightAcc =
            UbTensor<float, KDA_GROUPED_DK_RIGHT_ACC_UB>();

        if constexpr (STAGE > 0) {
            SyncVToMte2();
            CopyStageLeftCRows(
                slotBase + KDA_GROUPED_C_OFF_LEFT +
                rowStart * KDA_GROUPED_K);
            SyncMte2ToV();
            BeginReusableFp32Mask();
            ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                dqAcc[CacheBlockOffset(STAGE)], ScratchBank<0>(),
                LeftOuterGate<STAGE>());
            ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                dkLeftAcc[CacheBlockOffset(STAGE)], ScratchBank<1>(),
                LeftOuterGate<STAGE>());
            PipeBarrier<PIPE_V>();

            SyncVToMte2();
            if constexpr (KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME) {
                constexpr uint32_t rowTileBytes =
                    KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
                constexpr uint32_t pairGapBytes =
                    (KDA_GROUPED_RIGHT_PAIR_ELEMENTS -
                     KDA_GROUPED_ROW_BLOCK_ELEMENTS) * sizeof(float);
                static_assert(rowTileBytes % KDA_GROUPED_DATA_BLOCK_BYTES == 0,
                              "Grouped off-right row tile must be 32-byte aligned");
                static_assert(pairGapBytes == 3 * rowTileBytes,
                              "Grouped off-right GM stride no longer matches the slot layout");
                DataCopyExtParams offRightTiles{
                    static_cast<uint16_t>(STAGE), rowTileBytes,
                    pairGapBytes, 0, 0};
                DataCopyPadExtParams<float> noPad{
                    false, 0, 0, 0.0f};
                DataCopyPad(
                    ScratchBank<0>(),
                    workspace_[slotBase + offRightBase +
                               rowStart * KDA_GROUPED_K],
                    offRightTiles, noPad);
            } else {
                DataCopy(ScratchBank<0>(),
                         workspace_[slotBase + offRightBase +
                             rowStart * KDA_GROUPED_K],
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                if constexpr (STAGE > 1) {
                    DataCopy(ScratchBank<1>(),
                              workspace_[slotBase + offRightBase +
                                  KDA_GROUPED_RIGHT_PAIR_ELEMENTS +
                                  rowStart * KDA_GROUPED_K],
                              KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                }
                if constexpr (STAGE > 2) {
                    DataCopy(ScratchBank<2>(),
                              workspace_[slotBase + offRightBase +
                                  2 * KDA_GROUPED_RIGHT_PAIR_ELEMENTS +
                                  rowStart * KDA_GROUPED_K],
                              KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                }
            }
            SyncMte2ToV();
            if constexpr (KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME) {
                ContiguousMulAddDst<
                    STAGE * KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    dkRightAcc[CacheBlockOffset(0)], ScratchBank<0>(),
                    RightOuterGate<0>());
            } else {
                ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    dkRightAcc[CacheBlockOffset(0)], ScratchBank<0>(),
                    RightOuterGate<0>());
                if constexpr (STAGE > 1) {
                    ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                        dkRightAcc[CacheBlockOffset(1)], ScratchBank<1>(),
                        RightOuterGate<1>());
                }
                if constexpr (STAGE > 2) {
                    ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                        dkRightAcc[CacheBlockOffset(2)], ScratchBank<2>(),
                        RightOuterGate<2>());
                }
            }
            PipeBarrier<PIPE_V>();
        }

        SyncVToMte2();
        CopyStageLeftCRows(
            slotBase + diagLeftBase + rowStart * KDA_GROUPED_K);
        DataCopy(ScratchBank<2>(),
                     workspace_[slotBase + KDA_GROUPED_C_DIAG_RIGHT +
                         rowStart * KDA_GROUPED_K],
                     KDA_GROUPED_ROW_BLOCK_ELEMENTS);
        SyncMte2ToV();
        if constexpr (STAGE == 0) {
            BeginReusableFp32Mask();
        }
        ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            dqAcc[CacheBlockOffset(STAGE)], ScratchBank<0>(),
            DiagLateGate<STAGE>());
        ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            dkLeftAcc[CacheBlockOffset(STAGE)], ScratchBank<1>(),
            DiagLateGate<STAGE>());
        ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
            dkRightAcc[CacheBlockOffset(STAGE)], ScratchBank<2>(),
            DiagEarlyGate<STAGE>());
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
        if constexpr (KDA_GROUPED_OVERLAP_STAGE_EPILOGUE) {
            FinalizeStageLeft<STAGE>();
        }
    }

    __aicore__ inline void StageTaskOutputsInAccumulators(
        uint64_t b, uint64_t hv, uint64_t chunkStart,
        uint32_t rowStart)
    {
        LocalTensor<float> dbLocal =
            UbTensor<float, KDA_GROUPED_DB_LOCAL_UB>();
        LocalTensor<float> dqAcc =
            UbTensor<float, KDA_GROUPED_DQ_ACC_UB>();
        LocalTensor<float> dkLeftAcc =
            UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>();
        LocalTensor<float> dkRightAcc =
            UbTensor<float, KDA_GROUPED_DK_RIGHT_ACC_UB>();
        LocalTensor<float> qCache =
            UbTensor<float, KDA_GROUPED_Q_CACHE_UB>();
        LocalTensor<float> kCache =
            UbTensor<float, KDA_GROUPED_K_CACHE_UB>();
        LocalTensor<float> product =
            UbTensor<float, KDA_GROUPED_Q_TYPED_UB>();
        LocalTensor<float> dbAcc =
            dbLocal[KDA_GROUPED_SELECTED_ROWS];
        LocalTensor<float> dbCompact =
            dbAcc[KDA_GROUPED_SELECTED_ROWS * 8];
        LocalTensor<float> betaBrcb =
            UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>();
        const uint64_t firstSelectedToken = chunkStart + rowStart;

        constexpr uint32_t featureBlockBytes =
            KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
        constexpr uint32_t featureBlockGapBytes =
            (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) *
            KDA_GROUPED_K * sizeof(float);
        constexpr uint32_t dbBlockBytes =
            KDA_GROUPED_ROWS_PER_AIV * sizeof(float);
        constexpr uint32_t dbBlockGapBytes =
            (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) * sizeof(float);
        DataCopyExtParams featureInputs{
            KDA_GROUPED_BLOCKS, featureBlockBytes,
            featureBlockGapBytes, 0, 0};
        DataCopyExtParams dbInputs{
            KDA_GROUPED_BLOCKS, dbBlockBytes,
            dbBlockGapBytes, 0, 0};
        DataCopyPadExtParams<float> floatNoPad{false, 0, 0, 0.0f};

        // Preserve the whole-tail reduction and beta order exactly.  Product
        // and reduction scratch reuse gate/reference regions that are dead
        // after stage 3, just like the serial batched-tail implementation.
        SyncVToMte2();
        DataCopyPad(dbLocal,
                    db_[BetaOffset(b, hv, firstSelectedToken)],
                    dbInputs, floatNoPad);
        SyncMte2ToV();
        Mul(product, dkLeftAcc, kCache, KDA_GROUPED_SELECTED_ELEMENTS);
        PipeBarrier<PIPE_V>();
        ReduceDbProductRows<KDA_GROUPED_SELECTED_ROWS>(
            dbCompact, dbAcc, product);
        Add(dbLocal, dbLocal, dbCompact, KDA_GROUPED_SELECTED_ROWS);
        ScaleRowsByBeta<KDA_GROUPED_SELECTED_ROWS>(
            dkLeftAcc, dkLeftAcc, betaBrcb);

        // q/dk inputs use retired lower-UB regions.  The final dg input is
        // loaded directly into dkRight only after both expressions that need
        // the pre-update dkRight value have completed, avoiding an extra UB
        // copy and leaving all lower regions free for the next task.
        LocalTensor<float> dqInput =
            UbTensor<float, KDA_GROUPED_Q_TYPED_UB>();
        LocalTensor<float> dkInput =
            UbTensor<float, KDA_GROUPED_K_BETA_CACHE_UB>();
        LocalTensor<float> rowTmp =
            UbTensor<float, KDA_GROUPED_BATCH_ROW_TMP_UB>();
        SyncVToMte2();
        DataCopyPad(dqInput,
                    dq_[VOffset(b, hv, firstSelectedToken)],
                    featureInputs, floatNoPad);
        DataCopyPad(dkInput,
                    dk_[VOffset(b, hv, firstSelectedToken)],
                    featureInputs, floatNoPad);
        SyncMte2ToV();

        BeginReusableFp32Mask();
        ContiguousSub<KDA_GROUPED_SELECTED_ELEMENTS>(
            rowTmp, dkLeftAcc, dkRightAcc);
        ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
            dkLeftAcc, dkInput, dkLeftAcc);
        PipeBarrier<PIPE_V>();
        ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
            dkLeftAcc, dkLeftAcc, dkRightAcc);
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();

        SyncVToMte2();
        DataCopyPad(dkRightAcc,
                    dg_[VOffset(b, hv, firstSelectedToken)],
                    featureInputs, floatNoPad);
        SyncMte2ToV();
        BeginReusableFp32Mask();
        ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>(
            dkRightAcc, qCache, dqAcc);
        PipeBarrier<PIPE_V>();
        // dqAcc must remain the pre-update value until the q*dq contribution
        // above has consumed it.  The following Add is bit-for-bit the same
        // input+dqAcc expression as the serial output buffer path.
        ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
            dqAcc, dqInput, dqAcc);
        ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>(
            dkRightAcc, rowTmp, kCache);
        PipeBarrier<PIPE_V>();
        EndReusableFp32Mask();
    }

    __aicore__ inline void IssueStagedTaskOutputs(
        uint64_t b, uint64_t hv, uint64_t chunkStart,
        uint32_t rowStart)
    {
        LocalTensor<float> dbLocal =
            UbTensor<float, KDA_GROUPED_DB_LOCAL_UB>();
        LocalTensor<float> dqAcc =
            UbTensor<float, KDA_GROUPED_DQ_ACC_UB>();
        LocalTensor<float> dkLeftAcc =
            UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>();
        LocalTensor<float> dkRightAcc =
            UbTensor<float, KDA_GROUPED_DK_RIGHT_ACC_UB>();
        const uint64_t firstSelectedToken = chunkStart + rowStart;

        constexpr uint32_t featureBlockBytes =
            KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
        constexpr uint32_t featureBlockGapBytes =
            (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) *
            KDA_GROUPED_K * sizeof(float);
        constexpr uint32_t dbBlockBytes =
            KDA_GROUPED_ROWS_PER_AIV * sizeof(float);
        constexpr uint32_t dbBlockGapBytes =
            (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) * sizeof(float);
        DataCopyExtParams featureOutputs{
            KDA_GROUPED_BLOCKS, featureBlockBytes,
            0, featureBlockGapBytes, 0};
        DataCopyExtParams dbOutputs{
            KDA_GROUPED_BLOCKS, dbBlockBytes,
            0, dbBlockGapBytes, 0};

        // Both workspace stages for the next task are submitted before this
        // helper.  Publishing ready first lets the AIC start stage 0/1; these
        // output transfers then drain concurrently from the three accumulator
        // banks, which no stage preparation aliases.
        SyncVToMte3();
        DataCopyPad(dqOut_[VOffset(b, hv, firstSelectedToken)],
                    dqAcc, featureOutputs);
        DataCopyPad(dkOut_[VOffset(b, hv, firstSelectedToken)],
                    dkLeftAcc, featureOutputs);
        DataCopyPad(dgOut_[VOffset(b, hv, firstSelectedToken)],
                    dkRightAcc, featureOutputs);
        DataCopyPad(dbOut_[BetaOffset(b, hv, firstSelectedToken)],
                    dbLocal, dbOutputs);
    }

    __aicore__ inline void StoreTaskOutputs(uint64_t b, uint64_t hv,
                                             uint64_t chunkStart,
                                             uint32_t rowStart)
    {
        LocalTensor<float> dbLocal =
            UbTensor<float, KDA_GROUPED_DB_LOCAL_UB>();
        LocalTensor<float> dqAcc = UbTensor<float, KDA_GROUPED_DQ_ACC_UB>();
        LocalTensor<float> dkLeftAcc =
            UbTensor<float, KDA_GROUPED_DK_LEFT_ACC_UB>();
        LocalTensor<float> dkRightAcc =
            UbTensor<float, KDA_GROUPED_DK_RIGHT_ACC_UB>();
        LocalTensor<float> qCache = UbTensor<float, KDA_GROUPED_Q_CACHE_UB>();
        LocalTensor<float> kCache = UbTensor<float, KDA_GROUPED_K_CACHE_UB>();
        if constexpr (!KDA_GROUPED_OVERLAP_STAGE_EPILOGUE) {
            // Rollback path: keep the original one-shot 32-row tail.  All
            // stage gates and references are dead after ConsumeStageAiv<3>,
            // so their storage can be reused for reduction scratch.
            LocalTensor<float> product =
                UbTensor<float, KDA_GROUPED_Q_TYPED_UB>();
            LocalTensor<float> dbAcc =
                dbLocal[KDA_GROUPED_SELECTED_ROWS];
            LocalTensor<float> dbCompact =
                dbAcc[KDA_GROUPED_SELECTED_ROWS * 8];
            LocalTensor<float> betaBrcb =
                UbTensor<float, KDA_GROUPED_BETA_BRCB_UB>();

            SyncVToMte2();
            if constexpr (KDA_GROUPED_BATCH_TAIL_BLOCKS) {
                constexpr uint32_t dbBlockBytes =
                    KDA_GROUPED_ROWS_PER_AIV * sizeof(float);
                constexpr uint32_t dbBlockGapBytes =
                    (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) *
                    sizeof(float);
                DataCopyExtParams dbBlocks{
                    KDA_GROUPED_BLOCKS, dbBlockBytes,
                    dbBlockGapBytes, 0, 0};
                DataCopyPadExtParams<float> floatNoPad{false, 0, 0, 0.0f};
                const uint64_t firstSelectedToken = chunkStart + rowStart;
                DataCopyPad(dbLocal,
                            db_[BetaOffset(b, hv, firstSelectedToken)],
                            dbBlocks, floatNoPad);
            } else {
                for (uint32_t block = 0; block < KDA_GROUPED_BLOCKS; ++block) {
                    const uint64_t token =
                        chunkStart + block * KDA_GROUPED_BC + rowStart;
                    DataCopy(dbLocal[block * KDA_GROUPED_ROWS_PER_AIV],
                             db_[BetaOffset(b, hv, token)],
                             KDA_GROUPED_ROWS_PER_AIV);
                }
            }
            SyncMte2ToV();

            Mul(product, dkLeftAcc, kCache, KDA_GROUPED_SELECTED_ELEMENTS);
            PipeBarrier<PIPE_V>();
            ReduceDbProductRows<KDA_GROUPED_SELECTED_ROWS>(
                dbCompact, dbAcc, product);
            Add(dbLocal, dbLocal, dbCompact, KDA_GROUPED_SELECTED_ROWS);
            ScaleRowsByBeta<KDA_GROUPED_SELECTED_ROWS>(
                dkLeftAcc, dkLeftAcc, betaBrcb);
        } else if constexpr (!KDA_GROUPED_BATCH_TAIL_BLOCKS) {
            // Stage 3 has just retired ScratchBank<0>; order that V read before
            // the first output DMA overwrites the same bank.
            SyncVToMte2();
        }

        if constexpr (KDA_GROUPED_BATCH_TAIL_BLOCKS) {
            LocalTensor<float> batchOutQ =
                UbTensor<float, KDA_GROUPED_BATCH_OUT_Q_UB>();
            LocalTensor<float> batchOutK =
                UbTensor<float, KDA_GROUPED_BATCH_OUT_K_UB>();
            LocalTensor<float> batchOutG =
                UbTensor<float, KDA_GROUPED_BATCH_OUT_G_UB>();
            LocalTensor<float> batchRowTmp =
                UbTensor<float, KDA_GROUPED_BATCH_ROW_TMP_UB>();
            const uint64_t firstSelectedToken = chunkStart + rowStart;
            constexpr uint32_t featureBlockBytes =
                KDA_GROUPED_ROW_BLOCK_ELEMENTS * sizeof(float);
            constexpr uint32_t dbBlockBytes =
                KDA_GROUPED_ROWS_PER_AIV * sizeof(float);
            constexpr uint32_t featureBlockGapBytes =
                (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) *
                KDA_GROUPED_K * sizeof(float);
            constexpr uint32_t dbBlockGapBytes =
                (KDA_GROUPED_BC - KDA_GROUPED_ROWS_PER_AIV) * sizeof(float);
            DataCopyExtParams featureInputs{
                KDA_GROUPED_BLOCKS, featureBlockBytes,
                featureBlockGapBytes, 0, 0};
            DataCopyExtParams featureOutputs{
                KDA_GROUPED_BLOCKS, featureBlockBytes,
                0, featureBlockGapBytes, 0};
            DataCopyExtParams dbOutputs{
                KDA_GROUPED_BLOCKS, dbBlockBytes,
                0, dbBlockGapBytes, 0};
            DataCopyPadExtParams<float> floatNoPad{false, 0, 0, 0.0f};

            // product aliases batchOutQ in the rollback epilogue, so retire
            // all preceding vector reads before the gathered inputs replace
            // the four output banks.
            SyncVToMte2();
            DataCopyPad(batchOutQ,
                        dq_[VOffset(b, hv, firstSelectedToken)],
                        featureInputs, floatNoPad);
            DataCopyPad(batchOutK,
                        dk_[VOffset(b, hv, firstSelectedToken)],
                        featureInputs, floatNoPad);
            DataCopyPad(batchOutG,
                        dg_[VOffset(b, hv, firstSelectedToken)],
                        featureInputs, floatNoPad);
            SyncMte2ToV();

            BeginReusableFp32Mask();
            ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchOutQ, batchOutQ, dqAcc);
            ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchOutK, batchOutK, dkLeftAcc);
            PipeBarrier<PIPE_V>();
            ContiguousAdd<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchOutK, batchOutK, dkRightAcc);
            ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchOutG, qCache, dqAcc);
            ContiguousSub<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchRowTmp, dkLeftAcc, dkRightAcc);
            PipeBarrier<PIPE_V>();
            ContiguousMulAddDst<KDA_GROUPED_SELECTED_ELEMENTS>(
                batchOutG, batchRowTmp, kCache);
            PipeBarrier<PIPE_V>();
            EndReusableFp32Mask();

            SyncVToMte3();
            DataCopyPad(dqOut_[VOffset(b, hv, firstSelectedToken)],
                        batchOutQ, featureOutputs);
            DataCopyPad(dkOut_[VOffset(b, hv, firstSelectedToken)],
                        batchOutK, featureOutputs);
            DataCopyPad(dgOut_[VOffset(b, hv, firstSelectedToken)],
                        batchOutG, featureOutputs);
            DataCopyPad(dbOut_[BetaOffset(b, hv, firstSelectedToken)],
                        dbLocal, dbOutputs);
            SyncMte3ToMte2();
            SyncMte3ToV();
        } else {
            LocalTensor<float> outQ = ScratchBank<0>();
            LocalTensor<float> outK = ScratchBank<1>();
            LocalTensor<float> outG =
                UbTensor<float, KDA_GROUPED_OPERAND0_UB>();
            LocalTensor<float> rowTmp =
                UbTensor<float, KDA_GROUPED_RAW_DA_UB>();
            for (uint32_t block = 0; block < KDA_GROUPED_BLOCKS; ++block) {
                const uint64_t token =
                    chunkStart + block * KDA_GROUPED_BC + rowStart;
                const uint32_t cacheOffset = CacheBlockOffset(block);
                DataCopy(outQ, dq_[VOffset(b, hv, token)],
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                DataCopy(outK, dk_[VOffset(b, hv, token)],
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                DataCopy(outG, dg_[VOffset(b, hv, token)],
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                SyncMte2ToV();

                BeginReusableFp32Mask();
                ContiguousAdd<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    outQ, outQ, dqAcc[cacheOffset]);
                ContiguousAdd<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    outK, outK, dkLeftAcc[cacheOffset]);
                PipeBarrier<PIPE_V>();
                ContiguousAdd<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    outK, outK, dkRightAcc[cacheOffset]);
                ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    outG, qCache[cacheOffset], dqAcc[cacheOffset]);
                ContiguousSub<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    rowTmp, dkLeftAcc[cacheOffset], dkRightAcc[cacheOffset]);
                PipeBarrier<PIPE_V>();
                ContiguousMulAddDst<KDA_GROUPED_ROW_BLOCK_ELEMENTS>(
                    outG, rowTmp, kCache[cacheOffset]);
                PipeBarrier<PIPE_V>();
                EndReusableFp32Mask();

                SyncVToMte3();
                DataCopy(dqOut_[VOffset(b, hv, token)], outQ,
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                DataCopy(dkOut_[VOffset(b, hv, token)], outK,
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                DataCopy(dgOut_[VOffset(b, hv, token)], outG,
                         KDA_GROUPED_ROW_BLOCK_ELEMENTS);
                DataCopy(dbOut_[BetaOffset(b, hv, token)],
                         dbLocal[block * KDA_GROUPED_ROWS_PER_AIV],
                         KDA_GROUPED_ROWS_PER_AIV);
                SyncMte3ToMte2();
                SyncMte3ToV();
            }
        }
    }

    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<float> g_;
    GlobalTensor<float> beta_;
    GlobalTensor<float> dAqk_;
    GlobalTensor<float> dAkk_;
    GlobalTensor<float> dq_;
    GlobalTensor<float> dk_;
    GlobalTensor<float> db_;
    GlobalTensor<float> dg_;
    GlobalTensor<float> dqOut_;
    GlobalTensor<float> dkOut_;
    GlobalTensor<float> dbOut_;
    GlobalTensor<float> dgOut_;
    GlobalTensor<float> workspace_;
    TBuf<TPosition::VECCALC> ubBuf_;
    TSCM<TPosition::VECIN, 1, KDA_GROUPED_TSCM_BLOCK_GROUP_MASK>
        stageAbQueue_;
    LocalTensor<float> stageAbSlots_[KDA_GROUPED_QUEUE_DEPTH];
    TPipe *pipe_ = nullptr;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_GROUPED_QUEUE_DEPTH> readyFlag_{
        KDA_GROUPED_READY_FLAG0, KDA_GROUPED_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_GROUPED_QUEUE_DEPTH> doneFlag_{
        KDA_GROUPED_DONE_FLAG0, KDA_GROUPED_DONE_FLAG1};
    TEventID pairScratchDone0_ = 0;
    TEventID pairScratchDone1_ = 0;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID sToVEvent_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToMte2Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    uint64_t batch_ = 0;
    uint64_t heads_ = 0;
    uint64_t seqlen_ = 0;
    uint64_t chunks_ = 0;
    uint64_t usedCoreNum_ = 1;
    uint64_t logicalCoreIdx_ = 0;
};

#endif // CHUNK_KDA_BWD_INTRA_GROUPED_HPP
