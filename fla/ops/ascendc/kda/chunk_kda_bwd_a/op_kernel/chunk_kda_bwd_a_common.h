#ifndef CHUNK_KDA_BWD_A_COMMON_H
#define CHUNK_KDA_BWD_A_COMMON_H

#include "chunk_kda_bwd_a_struct.h"

namespace KDA {

constexpr uint32_t KDA_BWD_A_C = 64;
constexpr uint32_t KDA_BWD_A_K = 128;
constexpr uint32_t KDA_BWD_A_POST_SLOTS = 2;
constexpr uint32_t KDA_BWD_A_READY_FLAG0 = 0;
constexpr uint32_t KDA_BWD_A_READY_FLAG1 = 1;
constexpr uint32_t KDA_BWD_A_FREE_FLAG0 = 2;
constexpr uint32_t KDA_BWD_A_FREE_FLAG1 = 3;

struct ChunkKdaBwdATask {
    uint32_t sequence = 0;
    uint32_t batch = 0;
    uint32_t localChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validC = 0;
};

__aicore__ inline uint64_t KdaBwdACeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}

__aicore__ inline void GetChunkKdaBwdATask(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdATilingData &tiling, uint32_t taskIdx,
    ChunkKdaBwdATask &task)
{
    if (tiling.isVarLen == 0) {
        task.batch = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
        task.sequence = task.batch;
        task.localChunk = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
        task.tokenBegin = task.localChunk * static_cast<uint32_t>(tiling.chunkSize);
        const uint32_t remain = static_cast<uint32_t>(tiling.seqlen) - task.tokenBegin;
        task.validC = remain < static_cast<uint32_t>(tiling.chunkSize) ?
                          remain :
                          static_cast<uint32_t>(tiling.chunkSize);
        return;
    }

    // Metadata is scalar control data.  It is consumed inside the one kernel
    // launch; the Host never loops over cu_seqlens to enqueue per-sequence work.
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> chunks;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    chunks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    task.sequence = static_cast<uint32_t>(chunks.GetValue(2 * taskIdx));
    task.localChunk = static_cast<uint32_t>(chunks.GetValue(2 * taskIdx + 1));
    const uint32_t seqBegin = static_cast<uint32_t>(cu.GetValue(task.sequence));
    const uint32_t seqEnd = static_cast<uint32_t>(cu.GetValue(task.sequence + 1));
    task.tokenBegin =
        seqBegin + task.localChunk * static_cast<uint32_t>(tiling.chunkSize);
    const uint32_t remain = seqEnd - task.tokenBegin;
    task.validC = remain < static_cast<uint32_t>(tiling.chunkSize) ?
                      remain :
                      static_cast<uint32_t>(tiling.chunkSize);
}

__aicore__ inline uint64_t KdaBwdAHeadTokenOffset(
    const ChunkKdaBwdATilingData &tiling, const ChunkKdaBwdATask &task,
    uint32_t head, uint32_t dim)
{
    if (tiling.isVarLen == 0) {
        return ((static_cast<uint64_t>(task.batch) * tiling.headNum + head) *
                    tiling.seqlen +
                task.tokenBegin) *
               dim;
    }
    return (static_cast<uint64_t>(head) * tiling.seqlen + task.tokenBegin) * dim;
}

__aicore__ inline uint64_t KdaBwdAChunkHeadOffset(
    const ChunkKdaBwdATilingData &tiling, const ChunkKdaBwdATask &task,
    uint32_t taskIdx, uint32_t head, uint32_t elementsPerChunk)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(taskIdx) * tiling.headNum + head) *
               elementsPerChunk;
    }
    return ((static_cast<uint64_t>(task.batch) *
                 tiling.chunkNumPerBatch + task.localChunk) *
                tiling.headNum + head) *
           elementsPerChunk;
}

__aicore__ inline GM_ADDR KdaBwdAPostSlot(
    GM_ADDR workspace, uint32_t logicalCore, uint32_t slot,
    const ChunkKdaBwdATilingData &tiling)
{
    return workspace + static_cast<uint64_t>(logicalCore) *
                           static_cast<uint64_t>(tiling.workspaceCoreSize) +
           static_cast<uint64_t>(slot) *
               static_cast<uint64_t>(tiling.postSlotSize);
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_COMMON_H
