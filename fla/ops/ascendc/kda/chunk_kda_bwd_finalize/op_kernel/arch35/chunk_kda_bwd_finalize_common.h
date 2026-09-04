/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "../chunk_kda_bwd_finalize_struct.h"
#include "kernel_operator.h"

namespace KDA {

constexpr uint32_t KDA_FINALIZE_CHUNK = 64;
constexpr uint32_t KDA_FINALIZE_DIM = 128;
constexpr uint32_t KDA_FINALIZE_HEADS_PER_WINDOW = 4;
constexpr uint32_t KDA_FINALIZE_AIV_COUNT = 2;
constexpr uint32_t KDA_FINALIZE_AIV_SLOTS = 2;
constexpr uint32_t KDA_FINALIZE_WORKSPACE_SLOTS = 8;
constexpr uint32_t KDA_FINALIZE_SLOT_BYTES = 160 * 1024;

constexpr uint32_t KDA_FINALIZE_MATRIX_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_CHUNK;
constexpr uint32_t KDA_FINALIZE_VECTOR_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_STATE_ELEMS = KDA_FINALIZE_DIM * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_MATRIX_BF16_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_MATRIX_FP32_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_VECTOR_BF16_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_VECTOR_FP32_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_STATE_BF16_BYTES = KDA_FINALIZE_STATE_ELEMS * sizeof(bfloat16_t);

// Per-slot GM layout frozen by the design document.
constexpr uint32_t KDA_FINALIZE_WS_DK_STATE_RAW = 0;
constexpr uint32_t KDA_FINALIZE_WS_DVB = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DKGB_RAW = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_EXP2_GK = 96 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_KE = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_GK_LAST = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_RH = 144 * 1024 + 512;
constexpr uint32_t KDA_FINALIZE_WS_GATE_STATE = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DB_V = 145 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DK_BASE = KDA_FINALIZE_WS_DK_STATE_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DQ_BASE = KDA_FINALIZE_WS_DVB;
constexpr uint32_t KDA_FINALIZE_WS_DG_BASE = KDA_FINALIZE_WS_DKGB_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DB_BASE = KDA_FINALIZE_WS_DB_V;

// Per-AIV BuildZ fixed UB layout.
constexpr uint32_t KDA_FINALIZE_UB_ZV = 0;
constexpr uint32_t KDA_FINALIZE_UB_ZW = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_ZB = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BETA = 80 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_WORK = 81 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BYTES = 248 * 1024;

// Per-AIV Stage4 layout.  dAkk_raw occupies two 8-KiB BF16 slots while
// BaseFinalize uses the disjoint [16, 248)-KiB range.
constexpr uint32_t KDA_FINALIZE_UB_DAKK_RAW = 0;
constexpr uint32_t KDA_FINALIZE_UB_STAGE4_WORK = 16 * 1024;

// Stage5 IntraPre keeps one BF16 NZ egress slot per local head generation.
// dAkk_raw remains in [0,16) KiB until VF has consumed it; the other three
// outputs use independent ping/pong ranges.  [112,248) KiB is phase-local
// input/scratch storage and does not overlap any live handoff.
constexpr uint32_t KDA_FINALIZE_UB_K_NEG = 16 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_Q_POS = 48 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BK_POS = 80 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_STAGE5_WORK = 112 * 1024;

// Four 64-KiB owner slots occupy [64,320) KiB of L1.  This range is disjoint
// from the only Stage4-live operands, Akk [0,32) and Tza [416,448), so each
// Stage5 head may publish immediately after its own dAkk becomes ready.
// dAqk's first 8 KiB is
// reserved for Stage6's direct GM->L1 load; Stage5 publishes the remaining
// 56 KiB without a GM round trip.
constexpr uint32_t KDA_FINALIZE_LOCAL_BASE = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BYTES = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAQK = 0;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAKK = 8 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_K_NEG = 16 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_Q_POS = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BK_POS = 48 * 1024;

// KernelA-compatible directed 1C2V handshake.  AIV1's hardware flag bank is
// selected with the fixed +16 sub-block stride.
constexpr uint8_t KDA_FINALIZE_CROSS_MODE = 0x4;
constexpr uint64_t KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint64_t KDA_FINALIZE_ZV_FREE_BASE = 0;
constexpr uint64_t KDA_FINALIZE_ZW_FREE_BASE = 2;
constexpr uint64_t KDA_FINALIZE_ZV_READY_BASE = 4;
constexpr uint64_t KDA_FINALIZE_ZW_READY_BASE = 6;
constexpr uint64_t KDA_FINALIZE_KE_READY_BASE = 8;
constexpr uint64_t KDA_FINALIZE_ZB_READY_BASE = 10;
constexpr uint64_t KDA_FINALIZE_ZB_FREE_BASE = 12;
// Stage4 executes only after Stage1 KE_READY has been consumed, so its pair
// can safely carry the later AIV->AIC dAkk credit.  dAkk ready must not reuse
// ZB_READY in the reverse direction: an early AIV can consume its own old ZB
// token.  Reuse ZV_FREE instead; Stage2 cannot finish until AIC has consumed
// that Stage0 AIC->AIV credit and returned ZV_READY.
constexpr uint64_t KDA_FINALIZE_DAKK_FREE_BASE = KDA_FINALIZE_KE_READY_BASE;
constexpr uint64_t KDA_FINALIZE_DAKK_READY_BASE = KDA_FINALIZE_ZV_FREE_BASE;
// Keep the final Stage5 AIV->AIC publication on a dedicated pair.  This
// avoids aliasing earlier per-head handoffs without requiring a group-wide
// barrier inside the uneven multi-core work-task loop.
constexpr uint64_t KDA_FINALIZE_LOCAL_READY_BASE = 14;

struct FinalizeChunkInfo {
    int64_t b = 0;
    int64_t seq = 0;
    int64_t localChunk = 0;
    int64_t stateIndex = 0;
    int64_t tokenStart = 0;
    int64_t validRows = 0;
    bool valid = false;
};

__aicore__ inline int64_t FinalizeMin(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void ResolveFinalizeChunk(
    int64_t task, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdFinalizeTilingData &tiling, FinalizeChunkInfo &info)
{
    info.valid = false;
    if (task < 0 || task >= tiling.chunkTaskNum) {
        return;
    }
    if (tiling.isVariable == 0) {
        info.b = task / tiling.denseChunkNum;
        info.seq = info.b;
        info.localChunk = task - info.b * tiling.denseChunkNum;
        info.stateIndex = info.localChunk;
        info.tokenStart = info.localChunk * tiling.chunkSize;
        info.validRows = FinalizeMin(tiling.chunkSize, tiling.T - info.tokenStart);
        info.valid = info.b >= 0 && info.b < tiling.B && info.validRows > 0;
        return;
    }
    if (cuSeqlens == nullptr || chunkIndices == nullptr) {
        return;
    }
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> indices;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    info.seq = indices.GetValue(2 * task);
    info.localChunk = indices.GetValue(2 * task + 1);
    if (info.seq < 0 || info.seq >= tiling.seqNum || info.localChunk < 0) {
        return;
    }
    const int64_t seqBegin = cu.GetValue(info.seq);
    const int64_t seqEnd = cu.GetValue(info.seq + 1);
    info.b = 0;
    info.stateIndex = task;
    info.tokenStart = seqBegin + info.localChunk * tiling.chunkSize;
    info.validRows = FinalizeMin(tiling.chunkSize, seqEnd - info.tokenStart);
    info.valid = seqBegin >= 0 && seqEnd >= seqBegin && seqEnd <= tiling.T && info.validRows > 0;
}

__aicore__ inline int64_t FinalizeTokenOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.T + chunk.tokenStart) * width;
    }
    return ((chunk.b * tiling.NV + head) * tiling.T + chunk.tokenStart) * width;
}

__aicore__ inline int64_t FinalizeStateOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.totalChunkNum + chunk.stateIndex) * tiling.K * tiling.V;
    }
    return ((chunk.b * tiling.NV + head) * tiling.denseChunkNum + chunk.stateIndex) *
        tiling.K * tiling.V;
}

__aicore__ inline uint64_t FinalizeWorkspaceSlotBase(
    int64_t coreIdx, uint64_t groupGeneration, uint32_t owner)
{
    const uint64_t window = groupGeneration & 1U;
    const uint64_t slot = window * KDA_FINALIZE_HEADS_PER_WINDOW + owner;
    return (static_cast<uint64_t>(coreIdx) * KDA_FINALIZE_WORKSPACE_SLOTS + slot) *
        KDA_FINALIZE_SLOT_BYTES;
}

static_assert(KDA_FINALIZE_WS_DB_V + 512 <= KDA_FINALIZE_SLOT_BYTES,
              "Stage0--3 workspace slot exceeds 160 KiB.");
static_assert(KDA_FINALIZE_UB_WORK < KDA_FINALIZE_UB_BYTES,
              "BuildZ fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE4_WORK < KDA_FINALIZE_UB_BYTES,
              "Stage4 fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE5_WORK < KDA_FINALIZE_UB_BYTES,
              "Stage5 fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <= 256 * 1024,
              "Stage5 four-head LocalOperand window exceeds 256 KiB.");
static_assert(KDA_FINALIZE_LOCAL_BASE +
                  KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <=
              512 * 1024,
              "Stage5 LocalOperand window exceeds A5 L1.");

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
