/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chunk_kda_bwd_intra_tiling.h"

#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_G = 2;
constexpr size_t INPUT_CU = 10;
constexpr size_t INPUT_CHUNK_INDICES = 11;
constexpr size_t ATTR_CHUNK_SIZE = 0;
constexpr size_t ATTR_SAFE_GATE = 1;
constexpr size_t ATTR_TOTAL_CHUNKS = 2;
constexpr int64_t BC = 16;
constexpr bool ENABLE_BLOCKWISE_SAFE = true;
} // namespace

ge::graphStatus Tiling4ChunkKdaBwdIntra(gert::TilingContext *context)
{
    auto qShapePtr = context->GetOptionalInputShape(INPUT_Q);
    auto gShapePtr = context->GetOptionalInputShape(INPUT_G);
    auto qDesc = context->GetInputDesc(INPUT_Q);
    auto attrs = context->GetAttrs();
    if (qShapePtr == nullptr || gShapePtr == nullptr || qDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto qShape = qShapePtr->GetStorageShape();
    auto gShape = gShapePtr->GetStorageShape();
    if (qShape.GetDimNum() != 4 || gShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = qShape.GetDim(0);
    const int64_t h = qShape.GetDim(1);
    const int64_t t = qShape.GetDim(2);
    const int64_t k = qShape.GetDim(3);
    const int64_t hv = gShape.GetDim(1);
    const int64_t chunkSize = *attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const bool safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const int64_t totalChunks = *attrs->GetAttrPointer<int64_t>(ATTR_TOTAL_CHUNKS);
    if ((chunkSize != 64 && chunkSize != 128) || k < 16 || k > 256 || (k % 16) != 0 ||
        h <= 0 || hv < h || (hv % h) != 0 || h > 128 || hv > 128 || totalChunks <= 0) {
        return ge::GRAPH_FAILED;
    }

    const bool isVarLen = context->GetOptionalInputTensor(INPUT_CU) != nullptr;
    if (isVarLen) {
        auto cuTensor = context->GetOptionalInputTensor(INPUT_CU);
        auto chunkTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
        const int64_t seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
        if (batch != 1 || seqNum <= 0 || chunkTensor == nullptr ||
            chunkTensor->GetStorageShape().GetShapeSize() != totalChunks * 4) {
            return ge::GRAPH_FAILED;
        }
        const int64_t *cu = cuTensor->GetData<int64_t>();
        if (cu == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int64_t offset = 0;
        for (int64_t seq = 0; seq < seqNum; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
                return ge::GRAPH_FAILED;
            }
            offset += (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
        }
        if (cu[seqNum] != t || offset != totalChunks) {
            return ge::GRAPH_FAILED;
        }
    } else if (totalChunks != (t + chunkSize - 1) / chunkSize) {
        return ge::GRAPH_FAILED;
    }

    const int64_t blockCount = (chunkSize + BC - 1) / BC;
    const int64_t chunks = isVarLen ? totalChunks : batch * totalChunks;
    const int64_t taskCount = chunks * hv * blockCount;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskCount, aivNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();

    ChunkKdaBwdIntraTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_qHeadNum(h);
    tiling.set_vHeadNum(hv);
    tiling.set_seqlen(t);
    tiling.set_headDim(k);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : blockDim);
    tiling.set_isVarLen(isVarLen ? 1 : 0);
    tiling.set_dataType(qDesc->GetDataType() == ge::DT_BF16 ? 1 : 0);
    tiling.set_safeGate(safeGate ? 1 : 0);
    const uint64_t baseTilingKey = (qDesc->GetDataType() == ge::DT_BF16 ? 2 : 0) + (safeGate ? 1 : 0);
    // Keys 0..3 retain the proven row-wise implementation.  The optimized
    // safe-gate implementation uses keys 5/7 so the legacy instances remain
    // available as an immediate compile-time rollback while profiling.
    context->SetTilingKey(baseTilingKey + (safeGate && ENABLE_BLOCKWISE_SAFE ? 4 : 0));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdIntra(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdIntra)
    .Tiling(Tiling4ChunkKdaBwdIntra)
    .TilingParse<ChunkKdaBwdIntraCompileInfo>(TilingPrepare4ChunkKdaBwdIntra);
} // namespace optiling
