#ifndef CHUNK_KDA_BWD_GATE_POST_STRUCT_H
#define CHUNK_KDA_BWD_GATE_POST_STRUCT_H

#include <cstdint>

namespace KDA {
struct ChunkKdaBwdGatePostTilingData {
    uint64_t batch;
    uint64_t headNum;
    uint64_t seqlen;
    uint64_t keyDim;
    uint64_t chunkSize;
    uint64_t chunkNumPerBatch;
    uint64_t taskNum;
    uint32_t usedCoreNum;
};
} // namespace KDA

#endif // CHUNK_KDA_BWD_GATE_POST_STRUCT_H
