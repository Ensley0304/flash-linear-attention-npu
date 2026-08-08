#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_GATE_POST_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_GATE_POST_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *ChunkKdaBwdGatePost(
    const aclTensor *dgHv, int64_t chunkSize, const aclTensor *dgOut, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_GATE_POST_H
