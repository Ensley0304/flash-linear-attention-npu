#include "aclnn_chunk_kda_bwd.h"

#include "chunk_kda_bwd_a.h"
#include "../../../chunk_kda_bwd_c/op_host/op_api/chunk_kda_bwd_c.h"
#include "../../../../gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"

#include <initializer_list>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

namespace {

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

const aclTensor *AllocTensor(
    aclOpExecutor *executor, const op::Shape &shape, DataType dtype)
{
    return executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
}

const aclTensor *AsRank4(
    const aclTensor *tensor, const op::Shape &shape,
    aclOpExecutor *executor)
{
    return l0op::Reshape(tensor, shape, executor);
}

const aclTensor *ConvertIntArrayToTensor(
    const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor =
        executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *w, const aclTensor *qg, const aclTensor *kg,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, double lowerBound,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dh0OutOptional,
    const aclTensor *dAOutOptional, const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnChunkKdaBwd,
        DFX_IN(q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h, dO,
               rawGOptional, aLogOptional, dtBiasOptional,
               initialStateOptional, dhtOptional, cuSeqlensOptional,
               chunkIndicesOptional, scale, chunkSize, safeGate,
               useGateInKernel, lowerBound),
        DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dh0OutOptional,
                dAOutOptional, dBiasOutOptional));

    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    const aclTensor *required[] = {
        q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h, dO,
        dqOut, dkOut, dvOut, dbOut, dgOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwd required tensor is nullptr.");
    }
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwd requires chunk_size=64.");
    CHECK_COND((cuSeqlensOptional == nullptr) ==
                   (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be supplied together.");
    // PR291 has not completed the reviewed dht/dh0 semantics. Keep the
    // public composition honest until that implementation is repaired.
    CHECK_COND(initialStateOptional == nullptr && dhtOptional == nullptr &&
                   dh0OutOptional == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "initial_state, dht and dh0 are not supported by this PR291 integration.");
    CHECK_COND(!useGateInKernel ||
                   (rawGOptional != nullptr && aLogOptional != nullptr &&
                    dAOutOptional != nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "raw_g, a_log and dA are required for raw-gate backward.");
    CHECK_COND(dtBiasOptional == nullptr || dBiasOutOptional != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "dbias output is required when dt_bias is present.");

    const op::Shape &qShape = q->GetViewShape();
    const op::Shape &hShape = h->GetViewShape();
    const bool isVarLen = cuSeqlensOptional != nullptr;
    CHECK_COND(qShape.GetDimNum() == (isVarLen ? 3U : 4U) &&
                   hShape.GetDimNum() == (isVarLen ? 4U : 5U),
               ACLNN_ERR_PARAM_INVALID,
               "canonical dense q/h ranks are 4/5 and varlen ranks are 3/4.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor *cuTensor =
        ConvertIntArrayToTensor(cuSeqlensOptional, executorPtr);
    const aclTensor *chunkTensor =
        ConvertIntArrayToTensor(chunkIndicesOptional, executorPtr);
    CHECK_COND(!isVarLen || (cuTensor != nullptr && chunkTensor != nullptr),
               ACLNN_ERR_INNER_NULLPTR,
               "convert varlen metadata to tensors failed.");

    const aclTensor *dqRaw =
        AllocTensor(executorPtr, qShape, DataType::DT_FLOAT);
    const aclTensor *dAqk =
        AllocTensor(executorPtr, aqk->GetViewShape(), DataType::DT_FLOAT);
    const aclTensor *dAkk =
        AllocTensor(executorPtr, akk->GetViewShape(), DataType::DT_FLOAT);
    const aclTensor *dv0 =
        AllocTensor(executorPtr, dO->GetViewShape(), dO->GetDataType());
    // PR291 treats dv and dv2 as distinct producer/consumer tensors. Its
    // AIV writes dv2 while AIC may still read it for the recurrent update;
    // aliasing dv2 back onto the dv0 input is not a supported contract.
    const aclTensor *dvScan =
        AllocTensor(executorPtr, dO->GetViewShape(), dO->GetDataType());

    op::Shape dhShape;
    if (isVarLen) {
        dhShape = MakeShape({
            1, qShape.GetDim(0), hShape.GetDim(0),
            qShape.GetDim(2), v->GetViewShape().GetDim(2)});
    } else {
        dhShape = MakeShape({
            qShape.GetDim(0), qShape.GetDim(1), hShape.GetDim(1),
            qShape.GetDim(3), v->GetViewShape().GetDim(3)});
    }
    const aclTensor *dh =
        AllocTensor(executorPtr, dhShape, q->GetDataType());
    CHECK_RET(dqRaw != nullptr && dAqk != nullptr && dAkk != nullptr &&
                  dv0 != nullptr && dvScan != nullptr && dh != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    // PR291 keeps its varlen token tensors in the dense-compatible
    // [1,H,T,D] ABI.  A and C use the forward operator's canonical [H,T,D]
    // ABI.  Reshape creates metadata-only views, so the composition still
    // launches exactly A, B and C once each and never splits cu_seqlens.
    const aclTensor *qgB = qg;
    const aclTensor *kgB = kg;
    const aclTensor *wB = w;
    const aclTensor *dOB = dO;
    const aclTensor *dv0B = dv0;
    const aclTensor *gkB = gk;
    const aclTensor *dvScanB = dvScan;
    if (isVarLen) {
        const int64_t headNum = qShape.GetDim(0);
        const int64_t tokenNum = qShape.GetDim(1);
        const int64_t keyDim = qShape.GetDim(2);
        const int64_t valueDim = v->GetViewShape().GetDim(2);
        const op::Shape keyShape4 =
            MakeShape({1, headNum, tokenNum, keyDim});
        const op::Shape valueShape4 =
            MakeShape({1, headNum, tokenNum, valueDim});
        qgB = AsRank4(qg, keyShape4, executorPtr);
        kgB = AsRank4(kg, keyShape4, executorPtr);
        wB = AsRank4(w, keyShape4, executorPtr);
        gkB = AsRank4(gk, keyShape4, executorPtr);
        dOB = AsRank4(dO, valueShape4, executorPtr);
        dv0B = AsRank4(dv0, valueShape4, executorPtr);
        dvScanB = AsRank4(dvScan, valueShape4, executorPtr);
        CHECK_RET(qgB != nullptr && kgB != nullptr && wB != nullptr &&
                      gkB != nullptr && dOB != nullptr && dv0B != nullptr &&
                      dvScanB != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    const auto resultA = l0op::ChunkKdaBwdA(
        aqk, vNew, h, dO, cuTensor, chunkTensor, chunkSize,
        dv0, dqRaw, dAqk, executorPtr);
    CHECK_RET(resultA[0] != nullptr && resultA[1] != nullptr &&
                  resultA[2] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const auto resultB = l0op::ChunkGatedDeltaRuleBwdDhu(
        qgB, kgB, wB, dOB, dv0B, nullptr, gkB, nullptr, nullptr,
        cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize,
        dh, nullptr, dvScanB, executorPtr);
    CHECK_RET(resultB[0] != nullptr && resultB[1] != nullptr &&
                  resultB[2] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const auto resultC = l0op::ChunkKdaBwdC(
        q, k, v, vNew, gk, beta, akk, h, dh, dvScan, dqRaw, dAqk,
        rawGOptional, aLogOptional, dtBiasOptional, cuTensor, chunkTensor,
        static_cast<float>(scale), chunkSize, safeGate, useGateInKernel,
        static_cast<float>(lowerBound), true, dqOut, dkOut, dvOut, dbOut, dgOut,
        dAkk, dAOutOptional, dBiasOutOptional, executorPtr);
    for (size_t i = 0; i < 6; ++i) {
        CHECK_RET(resultC[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(!useGateInKernel || resultC[6] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dtBiasOptional == nullptr || resultC[7] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwd);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwd three-kernel launch failed.");
    return ACLNN_SUCCESS;
}
