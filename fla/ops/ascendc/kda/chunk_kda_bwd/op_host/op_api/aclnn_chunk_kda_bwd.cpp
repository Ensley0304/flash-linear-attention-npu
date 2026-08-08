#include "aclnn_chunk_kda_bwd.h"
#include "chunk_kda_bwd.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

enum class Layout { BSND, BNSD };

struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *v;
    const aclTensor *beta;
    const aclTensor *gk;
    const aclTensor *aqk;
    const aclTensor *akk;
    const aclTensor *w;
    const aclTensor *qg;
    const aclTensor *kg;
    const aclTensor *vNew;
    const aclTensor *h;
    const aclTensor *dO;
    const aclTensor *rawG;
    const aclTensor *aLog;
    const aclTensor *dtBias;
    const aclTensor *initialState;
    const aclTensor *dht;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    const char *layoutText;
    double scale;
    int64_t chunkSize;
    bool safeGate;
    double lowerBound;
    bool useGateInKernel;
    bool stateVFirst;
    const char *recomputePolicy;
    const aclTensor *dAqkScratch;
    const aclTensor *dv0Scratch;
    const aclTensor *dhScratch;
    const aclTensor *dvScanScratch;
    const aclTensor *dqBaseScratch;
    const aclTensor *dkBaseScratch;
    const aclTensor *dbBaseScratch;
    const aclTensor *dgBaseScratch;
    const aclTensor *dAkkScratch;
    const aclTensor *dqOut;
    const aclTensor *dkOut;
    const aclTensor *dvOut;
    const aclTensor *dbOut;
    const aclTensor *dgOut;
    const aclTensor *dh0Out;
    const aclTensor *dAOut;
    const aclTensor *dbiasOut;
};

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

bool SameShape(const aclTensor *a, const aclTensor *b)
{
    if (a == nullptr || b == nullptr) {
        return false;
    }
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

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : dims) {
        if (tensor->GetViewShape().GetDim(idx++) != dim) {
            return false;
        }
    }
    return true;
}

aclnnStatus ParseLayout(const char *text, Layout &layout)
{
    CHECK_COND(text != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must be BSND or BNSD for the P0 key.");
    if (std::strcmp(text, "BSND") == 0) {
        layout = Layout::BSND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(text, "BNSD") == 0) {
        layout = Layout::BNSD;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "P0 ChunkKdaBwd supports dense BSND/BNSD only.");
}

aclnnStatus Check(const Params &p, Layout &layout)
{
    const aclTensor *required[] = {
        p.q, p.k, p.v, p.beta, p.gk, p.aqk, p.akk, p.w, p.qg,
        p.kg, p.vNew, p.h, p.dO, p.dqOut, p.dkOut, p.dvOut,
        p.dbOut, p.dgOut, p.dAqkScratch, p.dv0Scratch, p.dhScratch,
        p.dvScanScratch, p.dqBaseScratch, p.dkBaseScratch,
        p.dbBaseScratch, p.dgBaseScratch, p.dAkkScratch
    };
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "P0 ChunkKdaBwd required tensors must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "P0 ChunkKdaBwd requires contiguous tensors.");
    }
    CHECK_RET(ParseLayout(p.layoutText, layout) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.rawG == nullptr && p.aLog == nullptr && p.dtBias == nullptr &&
                   p.initialState == nullptr && p.dht == nullptr &&
                   p.cuSeqlens == nullptr && p.chunkIndices == nullptr &&
                   p.dh0Out == nullptr && p.dAOut == nullptr && p.dbiasOut == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "P0 is no-raw, no-state and dense; optional P1 tensors must be nullptr.");
    CHECK_COND(p.recomputePolicy != nullptr &&
                   std::strcmp(p.recomputePolicy, "NONE") == 0,
               ACLNN_ERR_PARAM_INVALID,
               "P0 requires recompute_policy=NONE.");
    CHECK_COND(p.chunkSize == 64 && p.safeGate && !p.useGateInKernel &&
                   !p.stateVFirst,
               ACLNN_ERR_PARAM_INVALID,
               "P0 requires chunk_size=64, safe_gate=true, no raw gate, state_v_first=false.");
    (void)p.lowerBound;

    const aclTensor *bf16[] = {
        p.q, p.k, p.v, p.aqk, p.akk, p.w, p.qg, p.kg,
        p.vNew, p.h, p.dO, p.dvOut, p.dv0Scratch, p.dhScratch,
        p.dvScanScratch
    };
    for (const aclTensor *tensor : bf16) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "P0 data tensors must be BF16.");
    }
    CHECK_COND(p.beta->GetDataType() == DataType::DT_BF16 ||
                   p.beta->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta must be BF16 or FP32.");
    CHECK_COND(p.gk->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "gk must be FP32.");
    const aclTensor *fp32[] = {
        p.dqOut, p.dkOut, p.dbOut, p.dgOut, p.dAqkScratch,
        p.dqBaseScratch, p.dkBaseScratch, p.dbBaseScratch,
        p.dgBaseScratch, p.dAkkScratch
    };
    for (const aclTensor *tensor : fp32) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "dq/dk/db/dg must be FP32.");
    }
    CHECK_COND(SameShape(p.q, p.k) && SameShape(p.q, p.dqOut) &&
                   SameShape(p.k, p.dkOut) && SameShape(p.v, p.dvOut) &&
                   SameShape(p.beta, p.dbOut),
               ACLNN_ERR_PARAM_INVALID, "public gradient shapes must match inputs.");

    const auto q = p.q->GetViewShape();
    CHECK_COND(q.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "P0 q/k/v must be dense rank-4.");
    const int64_t batch = q.GetDim(0);
    const int64_t seqlen = q.GetDim(layout == Layout::BSND ? 1 : 2);
    const int64_t heads = q.GetDim(layout == Layout::BSND ? 2 : 1);
    const int64_t keyDim = q.GetDim(3);
    CHECK_COND(batch > 0 && seqlen > 0 && heads > 0 && keyDim == 128,
               ACLNN_ERR_PARAM_INVALID, "P0 requires positive B/T/H and K=128.");
    const bool publicShapeOk = layout == Layout::BSND
                                   ? HasShape(p.v, {batch, seqlen, heads, 128}) &&
                                         HasShape(p.beta, {batch, seqlen, heads})
                                   : HasShape(p.v, {batch, heads, seqlen, 128}) &&
                                         HasShape(p.beta, {batch, heads, seqlen});
    CHECK_COND(publicShapeOk,
               ACLNN_ERR_PARAM_INVALID, "P0 requires H=HV and V=128.");
    CHECK_COND(HasShape(p.dO, {batch, seqlen, heads, 128}),
               ACLNN_ERR_PARAM_INVALID, "d_o is always sequence-major BSND.");
    CHECK_COND(HasShape(p.gk, {batch, heads, seqlen, 128}) &&
                   HasShape(p.w, {batch, heads, seqlen, 128}) &&
                   HasShape(p.qg, {batch, heads, seqlen, 128}) &&
                   HasShape(p.kg, {batch, heads, seqlen, 128}) &&
                   HasShape(p.vNew, {batch, heads, seqlen, 128}),
               ACLNN_ERR_PARAM_INVALID,
               "saved gk/w/qg/kg/v_new must use dense head-major layout.");
    CHECK_COND(HasShape(p.aqk, {batch, heads, seqlen, 64}) &&
                   HasShape(p.akk, {batch, heads, seqlen, 64}),
               ACLNN_ERR_PARAM_INVALID, "saved Aqk/Akk must be [B,H,T,64].");
    const int64_t chunks = (seqlen + 63) / 64;
    CHECK_COND(HasShape(p.h, {batch, chunks, heads, 128, 128}),
               ACLNN_ERR_PARAM_INVALID,
               "saved h must be sequence-major [B,NT,H,128,128].");
    CHECK_COND(
        HasShape(p.dAqkScratch, {batch, heads, seqlen, 64}) &&
            HasShape(p.dv0Scratch, {batch, heads, seqlen, 128}) &&
            HasShape(p.dhScratch, {batch, heads, chunks, 128, 128}) &&
            HasShape(p.dvScanScratch, {batch, heads, seqlen, 128}) &&
            HasShape(p.dqBaseScratch, {batch, heads, seqlen, 128}) &&
            HasShape(p.dkBaseScratch, {batch, heads, seqlen, 128}) &&
            HasShape(p.dbBaseScratch, {batch, heads, seqlen}) &&
            HasShape(p.dgBaseScratch, {batch, heads, seqlen, 128}) &&
            HasShape(p.dAkkScratch, {batch, heads, seqlen, 64}),
        ACLNN_ERR_PARAM_INVALID,
        "P0 scratch tensors must use the fixed dense BNSD shapes.");
    const bool dgShapeOk = layout == Layout::BSND
                               ? HasShape(p.dgOut, {batch, seqlen, heads, 128})
                               : HasShape(p.dgOut, {batch, heads, seqlen, 128});
    CHECK_COND(dgShapeOk, ACLNN_ERR_PARAM_INVALID,
               "dg output must follow the public gate layout.");
    return ACLNN_SUCCESS;
}

const aclTensor *TransposeSwap12Contiguous(
    const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t i = 0; i < rank; ++i) {
        perm[i] = static_cast<int64_t>(i);
    }
    std::swap(perm[1], perm[2]);
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    CHECK_RET(permArray != nullptr, nullptr);
    const aclTensor *view = l0op::Transpose(input, permArray, executor);
    CHECK_RET(view != nullptr, nullptr);
    const aclTensor *materialized = l0op::Contiguous(view, executor);
    CHECK_RET(materialized != nullptr, nullptr);
    const aclTensor *reshaped = l0op::Reshape(
        materialized, view->GetViewShape(), executor);
    CHECK_RET(reshaped != nullptr, nullptr);
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk, const aclTensor *aqk,
    const aclTensor *akk, const aclTensor *w, const aclTensor *qg,
    const aclTensor *kg, const aclTensor *vNew, const aclTensor *h,
    const aclTensor *dO, const aclTensor *rawGOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional, const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout,
    double scale, int64_t chunkSize, bool safeGate, double lowerBound,
    bool useGateInKernel, bool stateVFirst, const char *recomputePolicy,
    const aclTensor *dAqkScratch, const aclTensor *dv0Scratch,
    const aclTensor *dhScratch, const aclTensor *dvScanScratch,
    const aclTensor *dqBaseScratch, const aclTensor *dkBaseScratch,
    const aclTensor *dbBaseScratch, const aclTensor *dgBaseScratch,
    const aclTensor *dAkkScratch,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dh0Out,
    const aclTensor *dAOut, const aclTensor *dbiasOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    Params p{
        q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h, dO,
        rawGOptional, aLogOptional, dtBiasOptional, initialStateOptional,
        dhtOptional, cuSeqlensOptional, chunkIndicesOptional, layout, scale,
        chunkSize, safeGate, lowerBound, useGateInKernel, stateVFirst,
        recomputePolicy, dAqkScratch, dv0Scratch, dhScratch, dvScanScratch,
        dqBaseScratch, dkBaseScratch, dbBaseScratch, dgBaseScratch,
        dAkkScratch, dqOut, dkOut, dvOut, dbOut, dgOut, dh0Out,
        dAOut, dbiasOut
    };
    L2_DFX_PHASE_1(aclnnChunkKdaBwd,
                   DFX_IN(q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h,
                          dO, rawGOptional, aLogOptional, dtBiasOptional,
                          initialStateOptional, dhtOptional, cuSeqlensOptional,
                          chunkIndicesOptional, layout, scale, chunkSize,
                          safeGate, lowerBound, useGateInKernel,
                          stateVFirst, recomputePolicy, dAqkScratch,
                          dv0Scratch, dhScratch, dvScanScratch,
                          dqBaseScratch, dkBaseScratch, dbBaseScratch,
                          dgBaseScratch, dAkkScratch),
                   DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dh0Out,
                           dAOut, dbiasOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    Layout parsed{};
    CHECK_RET(Check(p, parsed) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor *qHead = q;
    const aclTensor *kHead = k;
    const aclTensor *vHead = v;
    const aclTensor *betaHead = beta;
    if (parsed == Layout::BSND) {
        qHead = TransposeSwap12Contiguous(q, executorPtr);
        kHead = TransposeSwap12Contiguous(k, executorPtr);
        vHead = TransposeSwap12Contiguous(v, executorPtr);
        betaHead = TransposeSwap12Contiguous(beta, executorPtr);
    }
    const aclTensor *dOHead = TransposeSwap12Contiguous(dO, executorPtr);
    CHECK_RET(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                  betaHead != nullptr && dOHead != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *dqBaseCompute = dqBaseScratch;
    const aclTensor *dkBaseCompute = dkBaseScratch;
    const aclTensor *dbBaseCompute = dbBaseScratch;
    const aclTensor *dgBaseCompute = dgBaseScratch;
    const aclTensor *dqHead = dqOut;
    const aclTensor *dkHead = dkOut;
    const aclTensor *dbHead = dbOut;
    const aclTensor *dvHead = dvOut;
    const aclTensor *dgHead = dgOut;
    if (parsed == Layout::BSND) {
        const auto shape = qHead->GetViewShape();
        const int64_t batch = shape.GetDim(0);
        const int64_t heads = shape.GetDim(1);
        const int64_t seqlen = shape.GetDim(2);
        // K3's base-gradient buffers and the public outputs have identical
        // capacities.  Use the public allocations as temporary BNSD bases,
        // then let K4 write out-of-place into the old base scratch.  The final
        // transpose therefore has distinct source/destination storage without
        // allocating another H*T*D-sized tensor.
        dqBaseCompute = l0op::Reshape(
            dqOut, MakeShape({batch, heads, seqlen, 128}), executorPtr);
        dkBaseCompute = l0op::Reshape(
            dkOut, MakeShape({batch, heads, seqlen, 128}), executorPtr);
        dbBaseCompute = l0op::Reshape(
            dbOut, MakeShape({batch, heads, seqlen}), executorPtr);
        dgBaseCompute = l0op::Reshape(
            dgOut, MakeShape({batch, heads, seqlen, 128}), executorPtr);
        dqHead = dqBaseScratch;
        dkHead = dkBaseScratch;
        dbHead = dbBaseScratch;
        dvHead = dv0Scratch;
        dgHead = dgBaseScratch;
        for (const aclTensor *tensor : {
                 dqBaseCompute, dkBaseCompute, dbBaseCompute, dgBaseCompute}) {
            CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensor->SetStorageShape(tensor->GetViewShape());
            tensor->SetOriginalShape(tensor->GetViewShape());
        }
        CHECK_RET(dqHead != nullptr && dkHead != nullptr && dbHead != nullptr &&
                      dvHead != nullptr && dgHead != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(dqHead != nullptr && dkHead != nullptr &&
                  dvHead != nullptr && dbHead != nullptr && dgHead != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const auto result = l0op::KdaChunkBackward(
        qHead, kHead, vHead, betaHead, gk, aqk, akk, w, qg, kg,
        vNew, h, dOHead, scale, chunkSize,
        dAqkScratch, dv0Scratch, dhScratch, dvScanScratch,
        dqBaseCompute, dkBaseCompute, dbBaseCompute, dgBaseCompute,
        dAkkScratch, dqHead, dkHead, dvHead,
        dbHead, dgHead, executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (parsed == Layout::BSND) {
        const aclTensor *dqSequence =
            TransposeSwap12Contiguous(result[0], executorPtr);
        CHECK_RET(dqSequence != nullptr &&
                      l0op::ViewCopy(dqSequence, dqOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *dkSequence =
            TransposeSwap12Contiguous(result[1], executorPtr);
        CHECK_RET(dkSequence != nullptr &&
                      l0op::ViewCopy(dkSequence, dkOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        // K3 must write BF16 dv to an external tensor on the current CANN
        // runtime.  Its physical contents are head-major here although the
        // public descriptor is BSND.  Reinterpret that storage as BNSD, then
        // use the same proven materialized transpose path as the inputs.
        const aclTensor *dvSequence =
            TransposeSwap12Contiguous(result[2], executorPtr);
        CHECK_RET(dvSequence != nullptr &&
                      l0op::ViewCopy(dvSequence, dvOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *dbSequence =
            TransposeSwap12Contiguous(result[3], executorPtr);
        CHECK_RET(dbSequence != nullptr &&
                      l0op::ViewCopy(dbSequence, dbOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *dgSequence =
            TransposeSwap12Contiguous(result[4], executorPtr);
        CHECK_RET(dgSequence != nullptr &&
                      l0op::ViewCopy(dgSequence, dgOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

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
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwd launch failed.");
    return ACLNN_SUCCESS;
}
