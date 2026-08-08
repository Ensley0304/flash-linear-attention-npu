#include "aclnn_chunk_kda_bwd_gate_post.h"
#include "chunk_kda_bwd_gate_post.h"

#include <cstddef>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
bool SameShape(const aclTensor *a, const aclTensor *b)
{
    const auto lhs = a->GetViewShape();
    const auto rhs = b->GetViewShape();
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t dim = 0; dim < lhs.GetDimNum(); ++dim) {
        if (lhs.GetDim(dim) != rhs.GetDim(dim)) {
            return false;
        }
    }
    return true;
}
} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdGatePostGetWorkspaceSize(
    const aclTensor *dgHv, int64_t chunkSize, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdGatePost, DFX_IN(dgHv, chunkSize), DFX_OUT(dgOut));
    CHECK_COND(dgHv != nullptr && dgOut != nullptr && workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "ChunkKdaBwdGatePost arguments must not be nullptr.");
    CHECK_COND(IsContiguous(dgHv) && IsContiguous(dgOut), ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdGatePost canary requires contiguous tensors.");
    CHECK_COND(dgHv->GetDataType() == DataType::DT_FLOAT &&
                   dgOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "dg_hv and dg must be FP32.");
    CHECK_COND(SameShape(dgHv, dgOut) && dgHv->GetViewShape().GetDimNum() == 4 &&
                   dgHv->GetViewShape().GetDim(3) == 128 && chunkSize == 64,
               ACLNN_ERR_PARAM_INVALID, "P0 requires equal [B,HV,T,128] shapes and chunk_size=64.");
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    CHECK_RET(l0op::ChunkKdaBwdGatePost(dgHv, chunkSize, dgOut, executorPtr) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdGatePost(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdGatePost);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaBwdGatePost launch failed.");
    return ACLNN_SUCCESS;
}
