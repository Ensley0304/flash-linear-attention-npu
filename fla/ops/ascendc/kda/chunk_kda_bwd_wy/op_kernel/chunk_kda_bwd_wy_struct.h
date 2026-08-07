#ifndef CHUNK_KDA_BWD_WY_STRUCT_H
#define CHUNK_KDA_BWD_WY_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdWyTilingData {
    int64_t batch;
    int64_t headNum;
    int64_t seqlen;
    int64_t chunkNum;
    int64_t chunkNumPerBatch;
    int64_t chunkSize;
    int64_t keyDim;
    int64_t valueDim;
    int64_t workspaceSlotSize;
    int64_t workspaceSlotCount;
    int64_t usedCoreNum;
    int64_t stage;
    int64_t kEOffset;
    int64_t dqRawOffset;
    int64_t dkRawOffset;
    int64_t dWOffset;
    int64_t zVOffset;
    int64_t dVbOffset;
    int64_t zWOffset;
    int64_t dKgbOffset;
    int64_t zaInputOffset;
    int64_t zaOutputOffset;
    float scale;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_WY_STRUCT_H
