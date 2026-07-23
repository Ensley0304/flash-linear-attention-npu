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
constexpr size_t INPUT_STAGE_A = 12;
constexpr size_t INPUT_STAGE_B = 13;
constexpr size_t INPUT_STAGE_C = 14;
constexpr size_t OUTPUT_0 = 0;
constexpr size_t OUTPUT_1 = 1;
constexpr size_t ATTR_CHUNK_SIZE = 0;
constexpr size_t ATTR_SAFE_GATE = 1;
constexpr size_t ATTR_TOTAL_CHUNKS = 2;
constexpr size_t ATTR_STAGE = 3;
constexpr int64_t BC = 16;
constexpr bool ENABLE_BLOCKWISE_SAFE = true;
constexpr uint64_t KDA_ROW3_PREP_TILING_KEY = 9;
constexpr uint64_t KDA_ROW3_CUBE_TILING_KEY = 10;
constexpr uint64_t KDA_ROW3_CONSUME_TILING_KEY = 11;
constexpr uint64_t KDA_ROW3_MIXED_TILING_KEY = 12;
constexpr uint64_t KDA_ROW3_BATCHED_GATE_TILING_KEY = 13;
constexpr uint64_t KDA_ROW3_BATCHED_POST_GATE_TILING_KEY = 14;
constexpr uint64_t KDA_FULL_CUBE_TILING_KEY = 15;
constexpr uint64_t KDA_STAGE4_TILING_KEY = KDA_FULL_CUBE_TILING_KEY;
static_assert(KDA_ROW3_MIXED_TILING_KEY != KDA_ROW3_BATCHED_GATE_TILING_KEY,
              "ChunkKdaBwdIntra MIX fallback and experiment require distinct tiling keys");
static_assert(KDA_ROW3_BATCHED_GATE_TILING_KEY != KDA_ROW3_BATCHED_POST_GATE_TILING_KEY,
              "ChunkKdaBwdIntra row gate experiments require distinct tiling keys");
static_assert(KDA_ROW3_BATCHED_POST_GATE_TILING_KEY != KDA_FULL_CUBE_TILING_KEY,
              "ChunkKdaBwdIntra full-Cube path requires a distinct tiling key");
constexpr int64_t KDA_ROW3_BYTES_PER_SLOT = (32 * 48 + 48 * 128 + 32 * 128) * 4;
constexpr int64_t KDA_ROW3_MAX_SLOTS = (256LL * 1024 * 1024) / KDA_ROW3_BYTES_PER_SLOT;
// key15 stores six block-diagonal GEMM A/B/C groups in one 600-KiB slot per
// logical AIC core.  AIV0/AIV1 cannot advance to the next task until both have
// consumed the current C groups, so task-count-sized scratch is unnecessary.
constexpr uint64_t KDA_FULL_CUBE_BYTES_PER_CORE = 614400;

bool MatchScratchShape(const gert::StorageShape *shape, int64_t dim0, int64_t dim1, int64_t dim2)
{
    if (shape == nullptr) {
        return false;
    }
    const gert::Shape storage = shape->GetStorageShape();
    return storage.GetDimNum() == 3 && storage.GetDim(0) == dim0 &&
           storage.GetDim(1) == dim1 && storage.GetDim(2) == dim2;
}
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
    const int64_t stage = *attrs->GetAttrPointer<int64_t>(ATTR_STAGE);
    if ((chunkSize != 64 && chunkSize != 128) || k < 16 || k > 256 || (k % 16) != 0 ||
        h <= 0 || hv < h || (hv % h) != 0 || h > 128 || hv > 128 || totalChunks <= 0 ||
        stage < 0 || stage > 4) {
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

    const int64_t chunks = isVarLen ? totalChunks : batch * totalChunks;
    const int64_t scratchSlots = chunks * hv;
    if (stage != 0) {
        const bool fastPathShape = safeGate && qDesc->GetDataType() == ge::DT_BF16 && !isVarLen &&
                                   batch == 1 && h == hv && chunkSize == 64 && k == 128 &&
                                   t > 0 && (t % chunkSize) == 0 &&
                                   ((stage == 4 &&
                                     KDA_STAGE4_TILING_KEY == KDA_FULL_CUBE_TILING_KEY) ||
                                    scratchSlots <= KDA_ROW3_MAX_SLOTS);
        if (!fastPathShape) {
            return ge::GRAPH_FAILED;
        }
        if (stage == 1) {
            auto aOutputShape = context->GetOutputShape(OUTPUT_0);
            auto bOutputShape = context->GetOutputShape(OUTPUT_1);
            if (!MatchScratchShape(aOutputShape, scratchSlots, 32, 48) ||
                !MatchScratchShape(bOutputShape, scratchSlots, 48, 128)) {
                return ge::GRAPH_FAILED;
            }
        } else if (stage == 2) {
            auto aShape = context->GetOptionalInputShape(INPUT_STAGE_A);
            auto bShape = context->GetOptionalInputShape(INPUT_STAGE_B);
            auto cOutputShape = context->GetOutputShape(OUTPUT_0);
            if (aShape == nullptr || bShape == nullptr ||
                !MatchScratchShape(aShape, scratchSlots, 32, 48) ||
                !MatchScratchShape(bShape, scratchSlots, 48, 128) ||
                !MatchScratchShape(cOutputShape, scratchSlots, 32, 128)) {
                return ge::GRAPH_FAILED;
            }
        } else if (stage == 3) {
            auto cShape = context->GetOptionalInputShape(INPUT_STAGE_C);
            if (cShape == nullptr ||
                !MatchScratchShape(cShape, scratchSlots, 32, 128)) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    const int64_t blockCount = (chunkSize + BC - 1) / BC;
    int64_t taskCount = chunks * hv * blockCount;
    if (stage == 1) {
        taskCount = scratchSlots;
    } else if (stage == 2) {
        taskCount = scratchSlots * 2;
    } else if (stage == 4) {
        taskCount = scratchSlots;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    const uint32_t aicNum = platform.GetCoreNumAic();
    uint32_t usedCoreNum = static_cast<uint32_t>(std::min<int64_t>(taskCount, aivNum));
    uint32_t blockDim = usedCoreNum;
    if (stage == 2 || stage == 4) {
        usedCoreNum = static_cast<uint32_t>(std::min<int64_t>(taskCount, aicNum));
        if (usedCoreNum == 0) {
            return ge::GRAPH_FAILED;
        }
        blockDim = usedCoreNum;
    }
    if (blockDim == 0 || usedCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(blockDim);
    if (stage == 4) {
        context->SetScheduleMode(1);
    }
    const bool useFullCube = stage == 4 && KDA_STAGE4_TILING_KEY == KDA_FULL_CUBE_TILING_KEY;
    const uint64_t row3WorkspaceBytes = stage == 4
        ? (useFullCube
               ? static_cast<uint64_t>(usedCoreNum) * KDA_FULL_CUBE_BYTES_PER_CORE
               : static_cast<uint64_t>(scratchSlots) *
                     static_cast<uint64_t>(KDA_ROW3_BYTES_PER_SLOT))
        : 0;
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + row3WorkspaceBytes;

    ChunkKdaBwdIntraTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_qHeadNum(h);
    tiling.set_vHeadNum(hv);
    tiling.set_seqlen(t);
    tiling.set_headDim(k);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_isVarLen(isVarLen ? 1 : 0);
    tiling.set_dataType(qDesc->GetDataType() == ge::DT_BF16 ? 1 : 0);
    tiling.set_safeGate(safeGate ? 1 : 0);
    tiling.set_stage(stage);
    const uint64_t baseTilingKey = (qDesc->GetDataType() == ge::DT_BF16 ? 2 : 0) + (safeGate ? 1 : 0);
    // Keys 0..3 retain the proven row-wise implementation.  The optimized
    // safe-gate implementation uses keys 5/7 so the legacy instances remain
    // available as an immediate compile-time rollback while profiling.
    if (stage == 1) {
        context->SetTilingKey(KDA_ROW3_PREP_TILING_KEY);
    } else if (stage == 2) {
        context->SetTilingKey(KDA_ROW3_CUBE_TILING_KEY);
    } else if (stage == 3) {
        context->SetTilingKey(KDA_ROW3_CONSUME_TILING_KEY);
    } else if (stage == 4) {
        // key13 is the proven 31-ms BK64 fallback.  key14 remains compiled as
        // the no-gain post-scale experiment.  Switching KDA_STAGE4_TILING_KEY
        // back to key13 also restores the task-count-sized legacy workspace
        // formula above.
        context->SetTilingKey(KDA_STAGE4_TILING_KEY);
    } else {
        context->SetTilingKey(baseTilingKey + (safeGate && ENABLE_BLOCKWISE_SAFE ? 4 : 0));
    }
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
