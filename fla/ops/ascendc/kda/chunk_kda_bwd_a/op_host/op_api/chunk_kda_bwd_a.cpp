#include "chunk_kda_bwd_a.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdA);

const std::array<const aclTensor *, 4> ChunkKdaBwdA(
    const aclTensor *aqk, const aclTensor *qg, const aclTensor *vNew,
    const aclTensor *h, const aclTensor *dO,
    const aclTensor *cuSeqlens, const aclTensor *chunkIndices,
    float scale, int64_t chunkSize,
    const aclTensor *dv0Out, const aclTensor *q0Out,
    const aclTensor *dqRawOut, const aclTensor *dAqkOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdA, aqk, qg, vNew, h, dO, cuSeqlens,
           chunkIndices, scale, chunkSize, dv0Out, q0Out, dqRawOut,
           dAqkOut);
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdA,
        OP_INPUT(aqk, qg, vNew, h, dO, cuSeqlens, chunkIndices),
        OP_OUTPUT(dv0Out, q0Out, dqRawOut, dAqkOut),
        OP_ATTR(scale, chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdA failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {dv0Out, q0Out, dqRawOut, dAqkOut};
}

} // namespace l0op
