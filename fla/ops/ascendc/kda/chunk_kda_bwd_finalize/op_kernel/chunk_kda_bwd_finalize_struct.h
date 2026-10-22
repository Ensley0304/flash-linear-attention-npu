/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_STRUCT_H
#define CHUNK_KDA_BWD_FINALIZE_STRUCT_H

#include <cstdint>

#include "../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_common.h"

namespace KDA {

// The public contract is already complete, while this development branch only
// executes Stage0--3.  Keeping the final tiling schema from the first patch
// prevents a later ABI change when Stage3--12 are appended.
struct ChunkKdaBwdFinalizeTilingData {
    int64_t B;
    int64_t NQ;
    int64_t NV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t denseChunkNum;
    int64_t totalChunkNum;
    int64_t chunkTaskNum;
    int64_t headWindowNum;
    int64_t workTaskNum;
    int64_t seqNum;
    int64_t chunkSize;
    uint32_t isVariable;
    uint32_t hasQkL2Norm;
    uint32_t safeGate;
    uint32_t useGateInKernel;
    uint32_t useExp2;
    uint32_t stateVFirst;
    float scale;
    float lowerBound;
    uint64_t slotWorkspaceOffset;
    uint64_t gatePartialOffset;
    uint64_t dtBiasPartialOffset;
    uint64_t a2DqFp32Offset;
    uint64_t a2DkFp32Offset;
    uint64_t a2DBetaFp32Offset;
    uint64_t a2DAkkFp32Offset;
    uint64_t a2WorkspaceBytes;
    uint32_t isAscend950;
    uint32_t reserved;
    ChunkKdaBwdCTilingData a2;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_STRUCT_H
