#ifndef CHUNK_KDA_BWD_C_COMMON_H
#define CHUNK_KDA_BWD_C_COMMON_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_c_struct.h"

namespace KDA {

constexpr uint32_t kWyChunkSize = 64;
constexpr uint32_t kWyKeyDim = 128;
constexpr uint32_t kWyValueDim = 128;
constexpr uint32_t kWyHeadsPerWindow = 2;
// Keep WY and Intra on the same two-head owner grid.  Besides simplifying
// the MIX handshake, this is a correctness requirement: the kernel has no
// grid-wide barrier between WY, Intra and Gate, so a later phase may only
// consume data produced by the same physical core.
constexpr uint32_t kWyFusedHeadsPerWindow = 2;
constexpr uint32_t kWyWorkspaceSlotCount = 4;
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
    uint32_t sequence;
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline WyChunkTask GetWyChunkTask(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdCTilingData &tiling, uint32_t taskIdx)
{
    WyChunkTask task{};
    if (tiling.isVarLen != 0) {
        AscendC::GlobalTensor<int64_t> cu;
        AscendC::GlobalTensor<int64_t> chunks;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        chunks.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        task.sequence = static_cast<uint32_t>(chunks.GetValue(2 * taskIdx));
        task.batchIdx = 0;
        const uint32_t localChunk =
            static_cast<uint32_t>(chunks.GetValue(2 * taskIdx + 1));
        task.chunkIdx = taskIdx;
        const uint32_t seqBegin =
            static_cast<uint32_t>(cu.GetValue(task.sequence));
        const uint32_t seqEnd =
            static_cast<uint32_t>(cu.GetValue(task.sequence + 1));
        task.begin = localChunk * static_cast<uint32_t>(tiling.chunkSize) +
                     seqBegin;
        task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
        if (task.end > seqEnd) {
            task.end = seqEnd;
        }
        return task;
    }
    task.batchIdx = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.sequence = task.batchIdx;
    task.chunkIdx = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.begin = task.chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
    task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
    if (task.end > static_cast<uint32_t>(tiling.seqlen)) {
        task.end = static_cast<uint32_t>(tiling.seqlen);
    }
    return task;
}

__aicore__ inline uint64_t WyTokenOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t width)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
               width;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.seqlen + tokenIdx) * width;
}

// Stage 3 stores BF16 Zb and stage 6 stores BF16 T_za in the two halves of
// this task's final FP32 dAkk allocation.  Both temporaries therefore remain
// inside the task's own byte range until stage 7 writes the final FP32 tile.
__aicore__ inline uint64_t WyDAkkBf16TaskBase(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    return 2U * WyTokenOffset(tiling, batchIdx, headIdx, tokenIdx, 64);
}

// h is the public forward output and dh is Kernel B's output.  Both use the
// fixed sequence/chunk-major state layout [B, NT, H, K, V] (or [NT,H,K,V]
// for varlen), so C consumes them without an extra transpose or launch.
__aicore__ inline uint64_t WySavedHOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(chunkIdx) * tiling.headNum + headIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.chunkNumPerBatch +
             chunkIdx) * tiling.headNum + headIdx) *
           tiling.keyDim * tiling.valueDim;
}

__aicore__ inline uint64_t WyDhOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(chunkIdx) * tiling.headNum + headIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    if (tiling.dhHeadMajor != 0) {
        return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                    tiling.chunkNumPerBatch +
                chunkIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.chunkNumPerBatch +
             chunkIdx) * tiling.headNum + headIdx) *
           tiling.keyDim * tiling.valueDim;
}

__aicore__ inline uint64_t WyWorkspaceSlotBase(
    const ChunkKdaBwdCTilingData &tiling, uint32_t logicalCore,
    uint32_t generation, uint32_t headInWindow)
{
    (void)generation;
    // The production path is task-serial and devotes one slot to every head
    // in the current head window.  No diagnostic split-stage mapping exists.
    const uint32_t slot = headInWindow;
    return (static_cast<uint64_t>(logicalCore) * tiling.workspaceSlotCount + slot) *
           tiling.workspaceSlotSize;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_COMMON_H
