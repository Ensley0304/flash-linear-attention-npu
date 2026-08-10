#include "aclnn_chunk_kda_bwd_a.h"
#include "chunk_kda_bwd_a.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

extern "C" aclnnStatus aclnnChunkKdaBwdAGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *qg, const aclTensor *vNew,
    const aclTensor *h, const aclTensor *dO,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize,
    const aclTensor *dv0Out, const aclTensor *q0Out,
    const aclTensor *dqRawOut, const aclTensor *dAqkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdA,
                   DFX_IN(aqk, qg, vNew, h, dO, cuSeqlensOptional,
                          chunkIndicesOptional, scale, chunkSize),
                   DFX_OUT(dv0Out, q0Out, dqRawOut, dAqkOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    const aclTensor *required[] = {
        aqk, qg, vNew, h, dO, dv0Out, q0Out, dqRawOut, dAqkOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdA required tensor is nullptr.");
    }
    CHECK_COND((cuSeqlensOptional == nullptr) ==
                   (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be supplied together.");
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdA requires chunk_size=64.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    const auto result = l0op::ChunkKdaBwdA(
        aqk, qg, vNew, h, dO, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, dv0Out, q0Out, dqRawOut, dAqkOut,
        executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdA(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdA);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwdA launch failed.");
    return ACLNN_SUCCESS;
}
