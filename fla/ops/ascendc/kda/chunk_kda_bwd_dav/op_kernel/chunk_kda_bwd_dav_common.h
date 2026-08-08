#ifndef CHUNK_KDA_BWD_DAV_COMMON_H
#define CHUNK_KDA_BWD_DAV_COMMON_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_dav_struct.h"

namespace KDA {

constexpr uint32_t kDavChunkSize = 64;
constexpr uint32_t kDavValueDim = 128;
constexpr uint32_t kDavHeadsPerWindow = 2;
constexpr uint32_t kDavVectorToCubeStartFlag = 2;
constexpr uint32_t kDavCubeToVectorReadyFlag = 4;

struct DavChunkTask {
    uint32_t batchIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline DavChunkTask GetDavChunkTask(
    const ChunkKdaBwdDAvTilingData &tiling, uint32_t taskIdx)
{
    DavChunkTask task{};
    task.batchIdx = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
    const uint32_t chunkIdx = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.begin = chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
    task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
    const uint32_t seqlen = static_cast<uint32_t>(tiling.seqlen);
    if (task.end > seqlen) {
        task.end = seqlen;
    }
    return task;
}

__aicore__ inline uint64_t DavMatrixOffset(
    const ChunkKdaBwdDAvTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.seqlen + tokenIdx) * tiling.chunkSize;
}

__aicore__ inline uint64_t DavValueOffset(
    const ChunkKdaBwdDAvTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.seqlen + tokenIdx) * tiling.valueDim;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_DAV_COMMON_H
