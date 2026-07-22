/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chunk_kda_bwd_intra.h"

#include <algorithm>
#include <vector>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdIntra);

namespace {
const aclIntArray *BuildPackedChunkMetadata(const aclIntArray *cuSeqlens, const aclIntArray *chunkIndices,
                                            int64_t chunkSize, int64_t totalChunks, aclOpExecutor *executor)
{
    if (cuSeqlens == nullptr || cuSeqlens->Size() < 2 || chunkSize <= 0 || totalChunks <= 0) {
        return nullptr;
    }
    std::vector<int64_t> packed;
    packed.reserve(static_cast<size_t>(totalChunks) * 4);
    auto appendChunk = [&](int64_t seq, int64_t localChunk) -> bool {
        if (seq < 0 || static_cast<size_t>(seq + 1) >= cuSeqlens->Size() || localChunk < 0) {
            return false;
        }
        const int64_t seqStart = (*cuSeqlens)[static_cast<size_t>(seq)];
        const int64_t seqEnd = (*cuSeqlens)[static_cast<size_t>(seq + 1)];
        const int64_t start = seqStart + localChunk * chunkSize;
        if (start < seqStart || start >= seqEnd) {
            return false;
        }
        packed.insert(packed.end(), {seq, start, std::min(start + chunkSize, seqEnd), 0});
        return true;
    };
    if (chunkIndices != nullptr && chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
        return nullptr;
    }
    size_t suppliedIndex = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t len = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        for (int64_t chunk = 0; chunk < (len + chunkSize - 1) / chunkSize; ++chunk) {
            if (chunkIndices != nullptr &&
                ((*chunkIndices)[suppliedIndex] != static_cast<int64_t>(seq) ||
                 (*chunkIndices)[suppliedIndex + 1] != chunk)) {
                return nullptr;
            }
            suppliedIndex += 2;
            if (!appendChunk(static_cast<int64_t>(seq), chunk)) {
                return nullptr;
            }
        }
    }
    if (packed.size() != static_cast<size_t>(totalChunks) * 4) {
        return nullptr;
    }
    return executor->AllocIntArray(packed.data(), packed.size());
}
} // namespace

const std::array<const aclTensor *, 4> ChunkKdaBwdIntra(
    const aclTensor *q, const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclTensor *dAqk, const aclTensor *dAkk, const aclTensor *dq, const aclTensor *dk,
    const aclTensor *db, const aclTensor *dg, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, bool safeGate, int64_t totalChunks,
    const aclTensor *stageAOptional, const aclTensor *stageBOptional, const aclTensor *stageCOptional,
    int64_t stage,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dbOut, const aclTensor *dgOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdIntra, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cuSeqlensOptional,
           chunkIndicesOptional, chunkSize, safeGate, totalChunks, stageAOptional, stageBOptional,
           stageCOptional, stage, dqOut, dkOut, dbOut, dgOut);
    const aclTensor *actualCu = nullptr;
    const aclTensor *actualChunks = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCu = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const aclIntArray *packed = BuildPackedChunkMetadata(cuSeqlensOptional, chunkIndicesOptional,
                                                             chunkSize, totalChunks, executor);
        if (actualCu == nullptr || packed == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "failed to build KDA backward varlen metadata.");
            return {nullptr, nullptr, nullptr, nullptr};
        }
        actualChunks = executor->ConvertToTensor(packed, DataType::DT_INT64);
        if (actualChunks == nullptr) {
            return {nullptr, nullptr, nullptr, nullptr};
        }
        for (const aclTensor *tensor : {actualCu, actualChunks}) {
            auto mutableTensor = const_cast<aclTensor *>(tensor);
            mutableTensor->SetStorageFormat(Format::FORMAT_ND);
            mutableTensor->SetViewFormat(Format::FORMAT_ND);
            mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        }
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdIntra,
        OP_INPUT(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, actualCu, actualChunks,
                 stageAOptional, stageBOptional, stageCOptional),
        OP_OUTPUT(dqOut, dkOut, dbOut, dgOut), OP_ATTR(chunkSize, safeGate, totalChunks, stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdIntra failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dbOut, dgOut};
}
} // namespace l0op
