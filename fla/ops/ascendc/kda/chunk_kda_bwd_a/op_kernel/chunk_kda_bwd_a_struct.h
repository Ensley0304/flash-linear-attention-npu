#ifndef CHUNK_KDA_BWD_A_STRUCT_H
#define CHUNK_KDA_BWD_A_STRUCT_H

#include <cstdint>

namespace KDA {

// Private L0 contract for the first fused backward kernel.  The public L2
// wrapper normalizes BSND/TND inputs to the head-major storage used here.
struct ChunkKdaBwdATilingData {
    int64_t batch;
    int64_t seqNum;
    int64_t headNum;
    int64_t seqlen;
    int64_t keyDim;
    int64_t valueDim;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerBatch;
    int64_t isVarLen;
    uint32_t usedCoreNum;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_STRUCT_H
