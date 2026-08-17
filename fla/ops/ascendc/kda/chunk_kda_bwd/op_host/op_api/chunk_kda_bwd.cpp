#include "chunk_kda_bwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwd);

ChunkKdaBwdOutputs KdaChunkBackward(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *w, const aclTensor *qg, const aclTensor *kg,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool disableRecompute, bool useExp2,
    const aclTensor *dq,
    const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg,
    const aclTensor *dAOptional,
    const aclTensor *dBiasOptional, aclOpExecutor *executor)
{
    L0_DFX(KdaChunkBackward, q, k, v, beta, gk, aqk, akk, w, qg, kg,
           vNew, h, dO, rawGOptional, aLogOptional, dtBiasOptional,
           cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize,
           safeGate, useGateInKernel, lowerBound, disableRecompute, useExp2,
           dq, dk, dv, db, dg,
           dAOptional, dBiasOptional);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwd,
        OP_INPUT(q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h, dO,
                 rawGOptional, aLogOptional, dtBiasOptional,
                 cuSeqlensOptional, chunkIndicesOptional),
        OP_OUTPUT(dq, dk, dv, db, dg, dAOptional, dBiasOptional),
        OP_ATTR(scale, chunkSize, safeGate, useGateInKernel, lowerBound,
                disableRecompute, useExp2));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwd failed.");
        return {};
    }
    return {dq, dk, dv, db, dg, dAOptional, dBiasOptional};
}

} // namespace l0op
