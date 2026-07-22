/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_chunk_kda_bwd_intra.h"
#include "chunk_kda_bwd_intra.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
struct Params {
    const aclTensor *q, *k, *g, *beta, *dAqk, *dAkk, *dq, *dk, *db, *dg;
    const aclIntArray *cu, *chunks;
    int64_t chunkSize;
    bool safeGate;
    int64_t totalChunks;
    const aclTensor *dqOut, *dkOut, *dbOut, *dgOut;
};

aclnnStatus Check(const Params &p)
{
    for (const aclTensor *tensor : {p.q, p.k, p.g, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
                                    p.dqOut, p.dkOut, p.dbOut, p.dgOut}) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "required tensor must not be nullptr.");
    }
    CHECK_COND(p.chunkSize == 64 || p.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunk_size must be 64 or 128.");
    CHECK_COND(p.totalChunks > 0, ACLNN_ERR_PARAM_INVALID, "total_chunks must be positive.");
    const auto qs = p.q->GetViewShape();
    const auto gs = p.g->GetViewShape();
    CHECK_COND(qs.GetDimNum() == 4 && gs.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "q/k and g must use internal BNSD rank-4 layout.");
    const int64_t b = qs.GetDim(0), h = qs.GetDim(1), t = qs.GetDim(2), k = qs.GetDim(3);
    const int64_t hv = gs.GetDim(1);
    CHECK_COND(k >= 16 && k <= 256 && (k % 16) == 0, ACLNN_ERR_PARAM_INVALID,
               "K must be a multiple of 16 in [16, 256].");
    CHECK_COND(h > 0 && hv >= h && (hv % h) == 0, ACLNN_ERR_PARAM_INVALID,
               "HV must be divisible by H.");
    CHECK_COND(h <= 128 && hv <= 128, ACLNN_ERR_PARAM_INVALID, "H and HV must be <= 128.");
    auto same4 = [](const auto &s, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
        return s.GetDimNum() == 4 && s.GetDim(0) == d0 && s.GetDim(1) == d1 &&
               s.GetDim(2) == d2 && s.GetDim(3) == d3;
    };
    auto same3 = [](const auto &s, int64_t d0, int64_t d1, int64_t d2) {
        return s.GetDimNum() == 3 && s.GetDim(0) == d0 && s.GetDim(1) == d1 && s.GetDim(2) == d2;
    };
    CHECK_COND(same4(p.k->GetViewShape(), b, h, t, k), ACLNN_ERR_PARAM_INVALID, "k must match q.");
    for (const aclTensor *tensor : {p.g, p.dq, p.dk, p.dg, p.dqOut, p.dkOut, p.dgOut}) {
        CHECK_COND(same4(tensor->GetViewShape(), b, hv, t, k), ACLNN_ERR_PARAM_INVALID,
                   "g and feature-gradient tensors must be [B,HV,T,K].");
    }
    for (const aclTensor *tensor : {p.beta, p.db, p.dbOut}) {
        CHECK_COND(same3(tensor->GetViewShape(), b, hv, t), ACLNN_ERR_PARAM_INVALID,
                   "beta/db tensors must be [B,HV,T].");
    }
    for (const aclTensor *tensor : {p.dAqk, p.dAkk}) {
        CHECK_COND(same4(tensor->GetViewShape(), b, hv, t, p.chunkSize), ACLNN_ERR_PARAM_INVALID,
                   "dA tensors must be [B,HV,T,chunk_size].");
    }
    CHECK_COND((p.q->GetDataType() == DataType::DT_FLOAT16 || p.q->GetDataType() == DataType::DT_BF16) &&
               p.k->GetDataType() == p.q->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "q/k must have the same float16 or bfloat16 dtype.");
    for (const aclTensor *tensor : {p.g, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
                                    p.dqOut, p.dkOut, p.dbOut, p.dgOut}) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "gate, dA, accumulated gradients and outputs must be float32.");
    }
    CHECK_COND(p.chunks == nullptr || p.cu != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunk_indices requires cu_seqlens.");
    if (p.cu == nullptr) {
        CHECK_COND(p.totalChunks == (t + p.chunkSize - 1) / p.chunkSize, ACLNN_ERR_PARAM_INVALID,
                   "dense total_chunks does not match T and chunk_size.");
    } else {
        CHECK_COND(b == 1, ACLNN_ERR_PARAM_INVALID,
                   "varlen mode requires flattened internal batch B=1.");
        CHECK_COND(p.cu->Size() >= 2 && (*p.cu)[0] == 0 && (*p.cu)[p.cu->Size() - 1] == t,
                   ACLNN_ERR_PARAM_INVALID, "cu_seqlens must start at 0 and end at T.");
        int64_t expectedChunks = 0;
        for (size_t i = 0; i + 1 < p.cu->Size(); ++i) {
            CHECK_COND((*p.cu)[i + 1] >= (*p.cu)[i], ACLNN_ERR_PARAM_INVALID,
                       "cu_seqlens must be nondecreasing.");
            expectedChunks += ((*p.cu)[i + 1] - (*p.cu)[i] + p.chunkSize - 1) / p.chunkSize;
        }
        CHECK_COND(expectedChunks == p.totalChunks, ACLNN_ERR_PARAM_INVALID,
                   "varlen total_chunks does not match cu_seqlens.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    return tensor == nullptr ? ACLNN_ERR_INNER_NULLPTR : ACLNN_SUCCESS;
}

bool UseRow3MixedFastPath(const Params &p)
{
    const auto qs = p.q->GetViewShape();
    const auto gs = p.g->GetViewShape();
    constexpr int64_t ROW3_SCRATCH_BYTES_PER_SLOT = (32 * 48 + 48 * 128 + 32 * 128) * 4;
    constexpr int64_t ROW3_MAX_SCRATCH_BYTES = 256LL * 1024 * 1024;
    constexpr int64_t ROW3_MAX_SLOTS = ROW3_MAX_SCRATCH_BYTES / ROW3_SCRATCH_BYTES_PER_SLOT;
    // Keep the first Cube integration deliberately narrow.  Every other
    // supported shape continues to use the proven single-stage key7 path.
    return p.safeGate && p.cu == nullptr && p.q->GetDataType() == DataType::DT_BF16 &&
           p.chunkSize == 64 && qs.GetDim(0) == 1 && qs.GetDim(1) == gs.GetDim(1) &&
           qs.GetDim(2) > 0 && (qs.GetDim(2) % p.chunkSize) == 0 && qs.GetDim(3) == 128 &&
           p.totalChunks <= ROW3_MAX_SLOTS / gs.GetDim(1);
}
} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclTensor *dAqk, const aclTensor *dAkk, const aclTensor *dq, const aclTensor *dk,
    const aclTensor *db, const aclTensor *dg, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, bool safeGate, int64_t totalChunks,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dbOut, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdIntra,
                   DFX_IN(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(dqOut, dkOut, dbOut, dgOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    Params p{q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cuSeqlensOptional, chunkIndicesOptional,
             chunkSize, safeGate, totalChunks, dqOut, dkOut, dbOut, dgOut};
    CHECK_RET(Check(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    for (const aclTensor **tensor : {&p.q, &p.k, &p.g, &p.beta, &p.dAqk, &p.dAkk, &p.dq, &p.dk, &p.db, &p.dg}) {
        CHECK_RET(MakeContiguous(*tensor, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    std::array<const aclTensor *, 4> result{};
    if (UseRow3MixedFastPath(p)) {
        // The BF16/safe target now uses one MIX launch.  AIC and the two AIV
        // lanes exchange workspace readiness inside the kernel; no executor
        // tensor or intermediate launcher entry is needed.
        result = l0op::ChunkKdaBwdIntra(
            p.q, p.k, p.g, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg, p.cu, p.chunks,
            p.chunkSize, p.safeGate, p.totalChunks, nullptr, nullptr, nullptr, 4,
            p.dqOut, p.dkOut, p.dbOut, p.dgOut, executorPtr);
    } else {
        result = l0op::ChunkKdaBwdIntra(
            p.q, p.k, p.g, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg, p.cu, p.chunks,
            p.chunkSize, p.safeGate, p.totalChunks, nullptr, nullptr, nullptr, 0,
            p.dqOut, p.dkOut, p.dbOut, p.dgOut, executorPtr);
    }
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(result[0], p.dqOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], p.dkOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[2], p.dbOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[3], p.dgOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdIntra(void *workspace, uint64_t workspaceSize,
                                               aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdIntra);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "failed to launch ChunkKdaBwdIntra.");
    return ACLNN_SUCCESS;
}
