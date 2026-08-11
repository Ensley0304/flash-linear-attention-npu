#include "aclnn_chunk_kda_bwd_c.h"
#include "chunk_kda_bwd_c.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

extern "C" aclnnStatus aclnnChunkKdaBwdCGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *akk, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dqRaw,
    const aclTensor *dAqk, const aclTensor *rawGOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool dhHeadMajor,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dAkkOut,
    const aclTensor *dAOutOptional, const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnChunkKdaBwdC,
        DFX_IN(q, k, v, vNew, gk, beta, akk, h, dh, dvScan, dqRaw,
               dAqk, rawGOptional, aLogOptional, dtBiasOptional,
               cuSeqlensOptional, chunkIndicesOptional, scale,
               chunkSize, safeGate, useGateInKernel, lowerBound,
               dhHeadMajor),
        DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut,
                dAOutOptional, dBiasOutOptional));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    const aclTensor *required[] = {
        q, k, v, vNew, gk, beta, akk, h, dh, dvScan, dqRaw, dAqk,
        dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdC required tensor is nullptr.");
    }
    CHECK_COND((cuSeqlensOptional == nullptr) ==
                   (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be supplied together.");
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdC requires chunk_size=64.");
    CHECK_COND(!useGateInKernel ||
                   (rawGOptional != nullptr && aLogOptional != nullptr &&
                    dAOutOptional != nullptr &&
                    (dtBiasOptional == nullptr ||
                     dBiasOutOptional != nullptr)),
               ACLNN_ERR_PARAM_NULLPTR,
               "raw gate mode requires raw_g/a_log/dA and dbias when dt_bias is present.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    const auto result = l0op::ChunkKdaBwdC(
        q, k, v, vNew, gk, beta, akk, h, dh, dvScan, dqRaw, dAqk,
        rawGOptional, aLogOptional, dtBiasOptional,
        cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, safeGate,
        useGateInKernel, lowerBound, dhHeadMajor, dqOut, dkOut, dvOut, dbOut,
        dgOut, dAkkOut, dAOutOptional, dBiasOutOptional, executorPtr);
    for (size_t i = 0; i < 6; ++i) {
        CHECK_RET(result[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (useGateInKernel) {
        CHECK_RET(result[6] != nullptr &&
                      (dtBiasOptional == nullptr || result[7] != nullptr),
                  ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdC(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdC);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwdC launch failed.");
    return ACLNN_SUCCESS;
}
