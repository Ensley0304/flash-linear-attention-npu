/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_COMMON_H

#include "../arch35/chunk_kda_bwd_finalize_common.h"

namespace KDA {

static_assert(KDA_FINALIZE_A2_RESIDENT_SOURCE_REVISION == 155, "resident source revision mismatch");

// One Atlas A2 MIX block owns one head/chunk. Both AIV sub-blocks split the
// same logical work; neither sub-block owns an independent head.
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_CG = 1;
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_AIVS = 2;
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH = 2;
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS = 32;
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_STATE_ROWS = 64;
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_BRINGUP = 1;
#ifndef KDA_FINALIZE_A2_SPLIT_INTRA
#define KDA_FINALIZE_A2_SPLIT_INTRA 1
#endif
constexpr bool KDA_FINALIZE_A2_RESIDENT_SPLIT_INTRA = KDA_FINALIZE_A2_SPLIT_INTRA != 0;
#ifndef KDA_FINALIZE_A2_GATE_SIGMOID_PREFETCH
#define KDA_FINALIZE_A2_GATE_SIGMOID_PREFETCH 1
#endif
constexpr bool KDA_FINALIZE_A2_RESIDENT_GATE_SIGMOID_PREFETCH = KDA_FINALIZE_A2_GATE_SIGMOID_PREFETCH != 0;
#ifndef KDA_FINALIZE_A2_STAGE1B_L1_RESIDENT
#define KDA_FINALIZE_A2_STAGE1B_L1_RESIDENT 1
#endif
constexpr bool KDA_FINALIZE_A2_RESIDENT_STAGE1B_L1 = KDA_FINALIZE_A2_STAGE1B_L1_RESIDENT != 0;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_DW = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_ZV = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_DKGB = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_ZW = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3 = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3_COMPUTE = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3_DAKK = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE4 = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5 = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5_COMPUTE = true;
constexpr bool KDA_FINALIZE_A2_RESIDENT_RUN_STAGE6 = true;

// A slot is private to one physical AIC. WORK_DONE closes every producer /
// consumer lifetime before the next grid-stride work reuses these addresses.
constexpr uint32_t KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES = 372 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES = 32;
constexpr uint32_t KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES = 512;
constexpr uint32_t KDA_FINALIZE_A2_WS_DK_STATE = 0;
constexpr uint32_t KDA_FINALIZE_A2_WS_DVB = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_DKGB = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_DW_RELAY = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_FINAL_GATE = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_FINAL_RH = KDA_FINALIZE_A2_WS_FINAL_GATE + 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_FINAL_EXPG = KDA_FINALIZE_A2_WS_FINAL_RH + 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_ZV = 96 * 1024;
// KE occupies [112, 128) until it has been loaded into L1. Zw then reuses the
// same range, so both Zv and Zw can stay FP32 without increasing slot depth.
constexpr uint32_t KDA_FINALIZE_A2_WS_ZW = 112 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_KE = 112 * 1024;
// Zb cannot overwrite FP32 Zw in place: the two AIVs load disjoint halves but
// write packed BF16 halves that overlap the peer's source range. Keep Zb in
// the future DQ range; Stage 3 consumes it before Stage 5 produces DQ.
constexpr uint32_t KDA_FINALIZE_A2_WS_ZB = 148 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_BASE_RH = 112 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_BASE_GATE = 113 * 1024;

// Intra operands use FP32. Each output band has its own centered factors.
// Stage 4 now publishes A/Kneg before Base-gradient completion. DkState is
// still live at [0, 32), so A uses the dead DW relay instead of [24, 32).
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_DAQK = 128 * 1024;
// Stage 3 reuses the dead Zw range for FP32 dAkk while base gradients
// still read DkState. Zb has a separate relay until Tza reaches L1.
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_DAKK = 112 * 1024;
// Both 32-row output bands need a complete 64x128 Kneg matrix.
// Separate full NZ matrices preserve the reduction stride in L1.
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_K_NEG0 = 180 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_K_NEG1 = 212 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_Q_POS = 244 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_BK_POS = 308 * 1024;

// After AIC has loaded all Intra operands into L1, it reuses the slot for
// three FP32 outputs. DQ and DK each store two consecutive 32-row bands.
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_RIGHT = 0;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_DK = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_DQ_LEFT = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_WS_INTRA_DQ_EARLY = 148 * 1024;

// Each AIV owns a complete 192-KiB UB. K is converted once and remains FP32;
// six 16-KiB planes remain available for phase-local scratch.
constexpr uint32_t KDA_FINALIZE_A2_UB_EXP = 0;
constexpr uint32_t KDA_FINALIZE_A2_UB_Q = 16 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_K = 24 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_DQ_BASE = 40 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_DK_BASE = 56 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_DG_BASE = 72 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_SCRATCH_PAD = 256;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK0 = 88 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK1 = 104 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK2 = 120 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK3 = 136 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK4 = 152 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_WORK5 = 168 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_IO = 184 * 1024 + KDA_FINALIZE_A2_UB_SCRATCH_PAD;
constexpr uint32_t KDA_FINALIZE_A2_UB_CENTER_EXP = KDA_FINALIZE_A2_UB_IO + 2 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_CORRECTION = KDA_FINALIZE_A2_UB_IO + 2560;
constexpr uint32_t KDA_FINALIZE_A2_UB_FINAL_RH = KDA_FINALIZE_A2_UB_IO + 3584;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_PA_ACC = KDA_FINALIZE_A2_UB_IO + 4608;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_PB_ACC = KDA_FINALIZE_A2_UB_IO + 5 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_BIAS = KDA_FINALIZE_A2_UB_IO + 5632;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_EXPA = KDA_FINALIZE_A2_UB_IO + 6 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_GATE_ONE = KDA_FINALIZE_A2_UB_IO + 7 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_UB_END = 192 * 1024;

// Early Cube phases use [0,128) KiB for inputs. Resident intermediates
// and Intra operands occupy fixed addresses in the remaining L1 space.
constexpr uint32_t KDA_FINALIZE_A2_L1_DW = 240 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_KE = 256 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_ZB = 272 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_TZA = 280 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_DAQK = 288 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_DAKK = 304 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_K_NEG = 320 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_Q_POS = 384 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_BK_POS = 416 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_DAQK_T = 448 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_INTRA_DAKK_T = 464 * 1024;
constexpr uint32_t KDA_FINALIZE_A2_L1_END = 480 * 1024;

// Atlas A2 mode 0x2 is a collective 1C2V rendezvous. Both AIVs participate
// in all eight generations, including stages where one slice has no payload.
constexpr uint32_t KDA_FINALIZE_A2_FLAG_KE_READY = 0;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_BASE0_READY = 1;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_BASE1_READY = 2;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_ZB_READY = 3;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_DAKK_READY = 4;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_INPUT_READY = 5;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_DQ_READY = 6;
// Stage 5 consumes these generations before their original events are reused.
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_DK_INPUT_READY = KDA_FINALIZE_A2_FLAG_ZB_READY;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_INPUT_READY = KDA_FINALIZE_A2_FLAG_KE_READY;
// BASE0/BASE1 have already been consumed when Intra starts. Reuse those
// event IDs for DK/Right so three outstanding results never share one flag.
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_DK_READY = KDA_FINALIZE_A2_FLAG_BASE0_READY;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_READY = KDA_FINALIZE_A2_FLAG_BASE1_READY;
constexpr uint32_t KDA_FINALIZE_A2_FLAG_WORK_DONE = 7;

__aicore__ inline bool FinalizeA2CanUseFusedGate(const ChunkKdaBwdFinalizeTilingData &tiling)
{
    return KDA_FINALIZE_A2_RESIDENT_FUSED_GATE != 0 && KDA_FINALIZE_A2_DIAG_SKIP_GATE == 0 && tiling.safeGate != 0U &&
           tiling.a2.useGateInKernel != 0 && tiling.a2.rawGateIsBf16 == 0 && tiling.a2.isVarLen == 0 &&
           tiling.a2.seqlen % KDA_FINALIZE_CHUNK == 0 && tiling.NV <= 32;
}

struct FinalizeA2ResidentWork {
    int64_t chunkTask;
    int64_t head;
};

__aicore__ inline FinalizeA2ResidentWork ResolveFinalizeA2ResidentWork(int64_t work,
                                                                       const ChunkKdaBwdFinalizeTilingData &tiling)
{
    FinalizeA2ResidentWork result{};
    result.chunkTask = work / tiling.NV;
    result.head = work - result.chunkTask * tiling.NV;
    return result;
}

struct FinalizeA2ResidentChunkRange {
    int64_t begin;
    int64_t count;
};

__aicore__ inline FinalizeA2ResidentChunkRange
ResolveFinalizeA2ResidentChunkRange(int64_t logicalCore, int64_t head, const ChunkKdaBwdFinalizeTilingData &tiling)
{
    const int64_t coreNum = tiling.a2.usedCoreNum;
    const int64_t base = tiling.chunkTaskNum / coreNum;
    const int64_t remainder = tiling.chunkTaskNum - base * coreNum;
    const int64_t rotation = head % coreNum;
    const int64_t rank = (logicalCore + coreNum - rotation) % coreNum;
    FinalizeA2ResidentChunkRange range{};
    range.begin = rank * base + (rank < remainder ? rank : remainder);
    range.count = base + (rank < remainder ? 1 : 0);
    return range;
}

__aicore__ inline uint64_t FinalizeA2ResidentSlotBase(int64_t logicalCore, uint32_t slot)
{
    return (static_cast<uint64_t>(logicalCore) * KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH + slot) *
           KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES;
}

__aicore__ inline uint64_t FinalizeA2GateCarryOffset(int64_t logicalCore, const ChunkKdaBwdFinalizeTilingData &tiling)
{
    constexpr uint64_t gateScratchPerCore = 8 * 1024;
    return static_cast<uint64_t>(tiling.a2.usedCoreNum) * KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH *
               KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES +
           static_cast<uint64_t>(logicalCore) * gateScratchPerCore;
}

static_assert(KDA_FINALIZE_A2_RESIDENT_CG == 1, "The first resident implementation is one head per MIX block.");
static_assert(KDA_FINALIZE_A2_RESIDENT_AIVS * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS == KDA_FINALIZE_CHUNK,
              "A2 token row split must cover one complete chunk.");
static_assert(KDA_FINALIZE_A2_RESIDENT_AIVS * KDA_FINALIZE_A2_RESIDENT_STATE_ROWS == KDA_FINALIZE_DIM,
              "A2 state row split must cover K/V=128.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_DAQK == KDA_FINALIZE_A2_WS_DW_RELAY,
              "A2 prepacked dAqk must reuse the dead DW relay.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_DAQK + KDA_FINALIZE_MATRIX_FP32_BYTES <= KDA_FINALIZE_A2_WS_FINAL_GATE,
              "A2 prepacked dAqk overlaps the final gate records.");
static_assert(KDA_FINALIZE_A2_WS_KE + KDA_FINALIZE_VECTOR_BF16_BYTES <= KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 base/KE relay exceeds the resident slot.");
static_assert(KDA_FINALIZE_A2_WS_ZV + KDA_FINALIZE_MATRIX_FP32_BYTES <= KDA_FINALIZE_A2_WS_ZW,
              "A2 FP32 Zv relay overlaps the FP32 Zw relay.");
static_assert(KDA_FINALIZE_A2_WS_ZW + KDA_FINALIZE_MATRIX_FP32_BYTES <= KDA_FINALIZE_A2_WS_DW_RELAY,
              "A2 FP32 Zw relay overlaps the DW relay.");
static_assert(KDA_FINALIZE_A2_WS_ZB + KDA_FINALIZE_MATRIX_BF16_BYTES <= KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 Zb relay exceeds the resident slot.");
static_assert(KDA_FINALIZE_A2_WS_BASE_RH + 512 <= KDA_FINALIZE_A2_WS_BASE_GATE,
              "A2 base-rh relay overlaps the gate partial relay.");
static_assert(KDA_FINALIZE_A2_WS_BASE_GATE + 1024 <= KDA_FINALIZE_A2_WS_DW_RELAY,
              "A2 base scratch exceeds the relay phase boundary.");
static_assert(KDA_FINALIZE_A2_WS_FINAL_GATE + 1024 <= KDA_FINALIZE_A2_WS_FINAL_RH,
              "A2 final gate records exceed the relay slot.");
static_assert(KDA_FINALIZE_A2_WS_FINAL_EXPG + 512 <= KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 final gate inputs exceed the relay slot.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_BK_POS + 2 * KDA_FINALIZE_VECTOR_BF16_BYTES <=
                  KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 Intra high/residual operands exceed the slot.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_K_NEG0 + KDA_FINALIZE_VECTOR_FP32_BYTES == KDA_FINALIZE_A2_WS_INTRA_K_NEG1,
              "Each centered band needs its own complete FP32 Kneg matrix.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_BK_POS + 2 * KDA_FINALIZE_VECTOR_FP32_BYTES <=
                  KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "FP32 band operands exceed the private slot.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_DQ_LEFT + 2 * KDA_FINALIZE_VECTOR_FP32_BYTES <=
                  KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 Intra Cube outputs exceed the slot.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_RIGHT + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_A2_WS_INTRA_DQ_LEFT,
              "A2 FP32 Right relay overlaps the DQ/Left relay.");
static_assert(KDA_FINALIZE_A2_WS_INTRA_DQ_EARLY + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_A2_RESIDENT_SLOT_BYTES,
              "A2 early Intra dQ output exceeds the slot.");
static_assert(KDA_FINALIZE_A2_UB_END <= 192 * 1024, "A2 resident Vector exceeds the 192-KiB UB budget.");
static_assert(KDA_FINALIZE_A2_L1_END <= 512 * 1024, "A2 resident Cube exceeds the 512-KiB L1 budget.");

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_COMMON_H
