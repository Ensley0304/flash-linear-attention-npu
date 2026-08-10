#include "chunk_kda_bwd_c.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdC);

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
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdC, q, k, v, vNew, gk, beta, akk, h, dh,
           dvScan, dqRaw, dAqk, rawG, aLog, dtBias, cuSeqlens,
           chunkIndices, scale, chunkSize, safeGate, useGateInKernel,
           lowerBound, dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut,
           dAOut, dBiasOut);
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdC,
        OP_INPUT(q, k, v, vNew, gk, beta, akk, h, dh, dvScan,
                 dqRaw, dAqk, rawG, aLog, dtBias, cuSeqlens,
                 chunkIndices),
        OP_OUTPUT(dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut,
                  dAOut, dBiasOut),
        OP_ATTR(scale, chunkSize, safeGate, useGateInKernel,
                lowerBound));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdC failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut, dAOut,
            dBiasOut};
}

} // namespace l0op
