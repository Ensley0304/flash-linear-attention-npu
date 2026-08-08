#include "aclnn_chunk_kda_bwd_dav.h"
#include "chunk_kda_bwd_dav.h"

#include <cstddef>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    const auto a = lhs->GetViewShape();
    const auto b = rhs->GetViewShape();
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t dim = 0; dim < a.GetDimNum(); ++dim) {
        if (a.GetDim(dim) != b.GetDim(dim)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckParams(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO,
    int64_t chunkSize, const aclTensor *dAqkOut, const aclTensor *dvOut)
{
    const aclTensor *required[] = {aqk, vNew, dO, dAqkOut, dvOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdDAv tensor arguments must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "ChunkKdaBwdDAv canary requires contiguous tensors.");
    }
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdDAv P0 requires chunk_size=64.");
    CHECK_COND(aqk->GetDataType() == DataType::DT_BF16 &&
                   vNew->GetDataType() == DataType::DT_BF16 &&
                   dO->GetDataType() == DataType::DT_BF16 &&
                   dvOut->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID,
               "Aqk/v_new/d_o/dv must be BF16 for the P0 key.");
    CHECK_COND(dAqkOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "dAqk must be FP32.");
    const auto aqkShape = aqk->GetViewShape();
    const auto valueShape = vNew->GetViewShape();
    CHECK_COND(aqkShape.GetDimNum() == 4 && valueShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdDAv P0 requires rank-4 BNSD tensors.");
    CHECK_COND(aqkShape.GetDim(3) == 64 && valueShape.GetDim(3) == 128,
               ACLNN_ERR_PARAM_INVALID, "ChunkKdaBwdDAv P0 requires C=64,V=128.");
    CHECK_COND(SameShape(vNew, dO) && SameShape(vNew, dvOut) &&
                   SameShape(aqk, dAqkOut),
               ACLNN_ERR_PARAM_INVALID,
               "v_new/d_o/dv and Aqk/dAqk shape pairs must match.");
    for (size_t dim = 0; dim < 3; ++dim) {
        CHECK_COND(aqkShape.GetDim(dim) == valueShape.GetDim(dim),
                   ACLNN_ERR_PARAM_INVALID,
                   "Aqk and value tensors must share B,H,T dimensions.");
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdDavGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO,
    float scale, int64_t chunkSize, const aclTensor *dAqkOut,
    const aclTensor *dvOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdDav,
                   DFX_IN(aqk, vNew, dO, scale, chunkSize),
                   DFX_OUT(dAqkOut, dvOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_RET(CheckParams(aqk, vNew, dO, chunkSize, dAqkOut, dvOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    const auto result = l0op::ChunkKdaBwdDav(
        aqk, vNew, dO, scale, chunkSize, dAqkOut, dvOut, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdDav(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdDav);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwdDAv launch failed.");
    return ACLNN_SUCCESS;
}
