#include "chunk_kda_bwd_wy.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdWy);

const std::array<const aclTensor *, 6> ChunkKdaBwdWy(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *a, const aclTensor *h, const aclTensor *dO,
    const aclTensor *dh, const aclTensor *dvScan, float scale,
    int64_t chunkSize, int64_t stage,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut, const aclTensor *dgOut,
    const aclTensor *dAkkOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdWy, q, k, v, vNew, gk, beta, a, h, dO, dh,
           dvScan, scale, chunkSize, stage, dqOut, dkOut, dvOut, dbOut, dgOut,
           dAkkOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdWy,
        OP_INPUT(q, k, v, vNew, gk, beta, a, h, dO, dh, dvScan),
        OP_OUTPUT(dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut),
        OP_ATTR(scale, chunkSize, stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdWy failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut};
}

} // namespace l0op
