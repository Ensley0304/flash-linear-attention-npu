#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_A_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_A_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 3> ChunkKdaBwdA(
    const aclTensor *aqk, const aclTensor *vNew,
    const aclTensor *h, const aclTensor *dO,
    const aclTensor *cuSeqlens, const aclTensor *chunkIndices,
    int64_t chunkSize, const aclTensor *dv0Out,
    const aclTensor *dqRawOut, const aclTensor *dAqkOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_A_H
