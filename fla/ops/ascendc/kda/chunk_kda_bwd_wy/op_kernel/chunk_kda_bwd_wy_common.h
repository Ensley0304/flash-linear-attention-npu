#ifndef CHUNK_KDA_BWD_WY_COMMON_H
#define CHUNK_KDA_BWD_WY_COMMON_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_wy_struct.h"

namespace KDA {

constexpr uint32_t kWyChunkSize = 64;
constexpr uint32_t kWyKeyDim = 128;
constexpr uint32_t kWyValueDim = 128;
constexpr uint32_t kWyHeadsPerWindow = 2;
constexpr uint32_t kWyFusedHeadsPerWindow = 12;
constexpr uint32_t kWyWorkspaceSlotCount = 12;
// Stages [0, 7] are the original diagnostic path in which every stage is a
// separate device launch.  Stage 8 executes the same dependency graph inside
// one MIX kernel and is the production/default path.
constexpr int64_t kWyFusedStage = 8;
// Match the proven two-slot flag arrays used by the fused GDN forward
// schedulers: each head lane owns a distinct synchronization channel.  The
// dependent Cube result can reuse that lane's Cube->Vector flag after the
// base-ready event has been consumed.
__aicore__ inline uint32_t WyVectorToCubeFlag(uint32_t headInWindow)
{
    return 2U + headInWindow;
}

__aicore__ inline uint32_t WyCubeToVectorFlag(uint32_t headInWindow)
{
    return 4U + headInWindow;
}

// Per-generation free flags copied from the mature
// prepare_wy_repr_bwd_full ping-pong workspace protocol.
__aicore__ inline uint32_t WyWorkspaceFreeFlag(uint32_t generation)
{
    return 6U + (generation & 1U);
}

struct WyChunkTask {
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline WyChunkTask GetWyChunkTask(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t taskIdx)
{
    WyChunkTask task{};
    task.batchIdx = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.chunkIdx = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.begin = task.chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
    task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
    if (task.end > static_cast<uint32_t>(tiling.seqlen)) {
        task.end = static_cast<uint32_t>(tiling.seqlen);
    }
    return task;
}

__aicore__ inline uint64_t WyTokenOffset(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t width)
{
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.seqlen + tokenIdx) * width;
}

// Stage 3 stores BF16 Zb and stage 6 stores BF16 T_za in the two halves of
// this task's final FP32 dAkk allocation.  Both temporaries therefore remain
// inside the task's own byte range until stage 7 writes the final FP32 tile.
__aicore__ inline uint64_t WyDAkkBf16TaskBase(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    return 2U * WyTokenOffset(tiling, batchIdx, headIdx, tokenIdx, 64);
}

// h is the persistent forward output [B, NT, HV, K, V], whereas dh is
// internal [B, HV, NT, K, V].  Keeping the two address functions separate
// prevents a silent full-tensor transpose from entering the backward path.
__aicore__ inline uint64_t WySavedHOffset(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    return ((static_cast<uint64_t>(batchIdx) * tiling.chunkNumPerBatch + chunkIdx) *
            tiling.headNum + headIdx) * kWyKeyDim * kWyValueDim;
}

__aicore__ inline uint64_t WyDhOffset(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.chunkNumPerBatch + chunkIdx) * kWyKeyDim * kWyValueDim;
}

__aicore__ inline uint64_t WyWorkspaceSlotBase(
    const ChunkKdaBwdWyTilingData &tiling, uint32_t logicalCore,
    uint32_t generation, uint32_t headInWindow)
{
    // The fused path is task-serial and devotes one slot to every head in an
    // 12-head window.  Diagnostic split stages retain the original
    // two-generation x two-head ping-pong mapping.
    const uint32_t slot = tiling.stage == kWyFusedStage ?
        headInWindow :
        ((generation & 1U) * kWyHeadsPerWindow) + headInWindow;
    return (static_cast<uint64_t>(logicalCore) * tiling.workspaceSlotCount + slot) *
           tiling.workspaceSlotSize;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_WY_COMMON_H
