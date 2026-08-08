#ifndef CHUNK_KDA_BWD_DAV_STRUCT_H
#define CHUNK_KDA_BWD_DAV_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdDAvTilingData {
    int64_t batch;
    int64_t headNum;
    int64_t seqlen;
    int64_t valueDim;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerBatch;
    uint32_t usedCoreNum;
    int64_t stage;
    float scale;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_DAV_STRUCT_H
