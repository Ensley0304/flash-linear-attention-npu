#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_DAV_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_DAV_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> ChunkKdaBwdDav(
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *dO,
    float scale,
    int64_t chunkSize,
    const aclTensor *dAqkOut,
    const aclTensor *dvOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_DAV_H
