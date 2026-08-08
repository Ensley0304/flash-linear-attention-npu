#include "chunk_kda_bwd_gate_post.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdGatePost);

const aclTensor *ChunkKdaBwdGatePost(
    const aclTensor *dgHv, int64_t chunkSize, const aclTensor *dgOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdGatePost, dgHv, chunkSize, dgOut);
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdGatePost, OP_INPUT(dgHv), OP_OUTPUT(dgOut), OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdGatePost failed.");
        return nullptr;
    }
    return dgOut;
}
} // namespace l0op
