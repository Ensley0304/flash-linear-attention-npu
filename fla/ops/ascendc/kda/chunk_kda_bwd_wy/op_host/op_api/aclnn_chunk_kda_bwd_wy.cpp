#include "aclnn_chunk_kda_bwd_wy.h"
#include "chunk_kda_bwd_wy.h"

#include <cstddef>
#include <cstdlib>
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

bool SameShape(const aclTensor *a, const aclTensor *b)
{
    const auto lhs = a->GetViewShape();
    const auto rhs = b->GetViewShape();
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckParams(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *a, const aclTensor *h, const aclTensor *dO,
    const aclTensor *dh, const aclTensor *dvScan, int64_t chunkSize,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbOut, const aclTensor *dgOut, const aclTensor *dAkkOut)
{
    const aclTensor *required[] = {
        q, k, v, vNew, gk, beta, a, h, dO, dh, dvScan,
        dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut
    };
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdWy tensor arguments must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "ChunkKdaBwdWy canary requires contiguous tensors.");
    }
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdWy P0 requires chunk_size=64.");
    const aclTensor *bf16[] = {q, k, v, vNew, a, h, dO, dh, dvScan, dvOut};
    for (const aclTensor *tensor : bf16) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "P0 matrix tensors must be BF16.");
    }
    CHECK_COND(gk->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "gk must be FP32.");
    CHECK_COND(beta->GetDataType() == DataType::DT_BF16 ||
                   beta->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta must be BF16 or FP32.");
    const aclTensor *fp32[] = {dqOut, dkOut, dbOut, dgOut, dAkkOut};
    for (const aclTensor *tensor : fp32) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "base gradients and dAkk must be FP32.");
    }
    const auto qs = q->GetViewShape();
    const auto hs = h->GetViewShape();
    CHECK_COND(qs.GetDimNum() == 4 && hs.GetDimNum() == 5,
               ACLNN_ERR_PARAM_INVALID, "P0 requires BNSD q and [B,NT,H,K,V] h.");
    CHECK_COND(qs.GetDim(3) == 128 && hs.GetDim(3) == 128 && hs.GetDim(4) == 128,
               ACLNN_ERR_PARAM_INVALID, "P0 requires K=V=128.");
    CHECK_COND(SameShape(q, k) && SameShape(q, v) && SameShape(q, vNew) &&
                   SameShape(q, gk) && SameShape(q, dO) && SameShape(q, dvScan) &&
                   SameShape(q, dqOut) && SameShape(q, dkOut) &&
                   SameShape(q, dvOut) && SameShape(q, dgOut),
               ACLNN_ERR_PARAM_INVALID,
               "P0 q/k/v/v_new/gk/d_o/dv_scan and vector gradients must match.");
    CHECK_COND(SameShape(beta, dbOut), ACLNN_ERR_PARAM_INVALID,
               "beta and db_base shapes must match.");
    CHECK_COND(SameShape(a, dAkkOut), ACLNN_ERR_PARAM_INVALID,
               "Akk and dAkk shapes must match.");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdWyGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *a, const aclTensor *h, const aclTensor *dO,
    const aclTensor *dh, const aclTensor *dvScan, float scale,
    int64_t chunkSize, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut, const aclTensor *dgOut,
    const aclTensor *dAkkOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdWy,
                   DFX_IN(q, k, v, vNew, gk, beta, a, h, dO, dh, dvScan,
                          scale, chunkSize),
                   DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_RET(CheckParams(q, k, v, vNew, gk, beta, a, h, dO, dh, dvScan,
                          chunkSize, dqOut, dkOut, dvOut, dbOut, dgOut,
                          dAkkOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    const bool splitFallback = []() {
        const char *value = std::getenv("KDA_BWD_WY_SPLIT_FALLBACK");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    int64_t launchCount = splitFallback ? 8 : 1;
    if (splitFallback) {
        if (const char *value = std::getenv("KDA_BWD_WY_STAGE_COUNT")) {
            const int64_t requested = std::atoll(value);
            if (requested >= 1 && requested <= 8) {
                launchCount = requested;
            }
        }
    }
    for (int64_t launchIdx = 0; launchIdx < launchCount; ++launchIdx) {
        const int64_t stage = splitFallback ? launchIdx : 8;
        const auto result = l0op::ChunkKdaBwdWy(
            q, k, v, vNew, gk, beta, a, h, dO, dh, dvScan, scale,
            chunkSize, stage, dqOut, dkOut, dvOut, dbOut, dgOut,
            dAkkOut, executorPtr);
        for (const aclTensor *tensor : result) {
            CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdWy(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdWy);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwdWy launch failed.");
    return ACLNN_SUCCESS;
}
