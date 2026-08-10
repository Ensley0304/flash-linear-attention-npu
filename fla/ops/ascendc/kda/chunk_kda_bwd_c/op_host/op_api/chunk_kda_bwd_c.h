#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_C_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_C_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 8> ChunkKdaBwdC(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *akk, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dqRaw,
    const aclTensor *dAqk, const aclTensor *rawG,
    const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *cuSeqlens, const aclTensor *chunkIndices,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dAkkOut,
    const aclTensor *dAOut, const aclTensor *dBiasOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_C_H
