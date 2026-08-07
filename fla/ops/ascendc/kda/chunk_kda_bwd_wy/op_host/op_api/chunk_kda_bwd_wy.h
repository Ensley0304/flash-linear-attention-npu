#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_WY_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_WY_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 6> ChunkKdaBwdWy(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *a, const aclTensor *h, const aclTensor *dO,
    const aclTensor *dh, const aclTensor *dvScan, float scale,
    int64_t chunkSize, int64_t stage,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut, const aclTensor *dgOut,
    const aclTensor *dAkkOut, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_WY_H
