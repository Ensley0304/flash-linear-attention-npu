#ifndef CHUNK_KDA_BWD_C_STRUCT_H
#define CHUNK_KDA_BWD_C_STRUCT_H

#include <cstdint>

namespace KDA {

struct ChunkKdaBwdCTilingData {
    // P0 bounds (C=64, K=128, V<=256 and practical B/H/T) keep every
    // scalar and per-core byte offset below INT32_MAX.  Using int32_t keeps
    // the complete MIX kernel argument block below the 512-byte device ABI
    // boundary; int64_t fields made the otherwise no-op kernel exceed it.
    int32_t batch;
    int32_t seqNum;
    int32_t headNum;
    int32_t seqlen;
    int32_t chunkNum;
    int32_t chunkNumPerBatch;
    int32_t chunkSize;
    int32_t keyDim;
    int32_t valueDim;
    int32_t workspaceSlotSize;
    int32_t workspaceSlotCount;
    int32_t workspaceCoreSize;
    int32_t usedCoreNum;
    int32_t isVarLen;
    // PR291 Kernel B writes dh as [B,H,NT,K,V] for dense and
    // [1,H,totalChunks,K,V] for varlen. Saved h remains chunk-major.
    int32_t dhHeadMajor;
    int32_t safeGate;
    int32_t useGateInKernel;
    int32_t hasDtBias;
    int32_t intraRowBlock;
    int32_t kEOffset;
    int32_t dqRawOffset;
    int32_t dkRawOffset;
    int32_t dWOffset;
    int32_t zVOffset;
    int32_t dVbOffset;
    int32_t zWOffset;
    int32_t dKgbOffset;
    int32_t zaInputOffset;
    int32_t zaOutputOffset;
    int32_t intraALowerOffset;
    int32_t intraBLowerOffset;
    int32_t intraAUpperOffset;
    int32_t intraBUpperOffset;
    int32_t intraResultRegionOffset;
    int32_t intraResultDqOffset;
    int32_t intraResultDkLowerOffset;
    int32_t intraResultDkUpperOffset;
    float scale;
    float lowerBound;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_STRUCT_H
