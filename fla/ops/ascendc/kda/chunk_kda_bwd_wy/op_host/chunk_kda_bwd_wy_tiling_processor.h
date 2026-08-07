#ifndef CHUNK_KDA_BWD_WY_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_WY_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_wy_struct.h"

namespace optiling {

enum ChunkKdaBwdWyInputIndex : size_t {
    KDA_WY_Q = 0, KDA_WY_K, KDA_WY_V, KDA_WY_V_NEW, KDA_WY_GK,
    KDA_WY_BETA, KDA_WY_A, KDA_WY_H, KDA_WY_DO, KDA_WY_DH, KDA_WY_DV_SCAN
};
constexpr size_t KDA_WY_SCALE_ATTR = 0;
constexpr size_t KDA_WY_CHUNK_SIZE_ATTR = 1;
constexpr int64_t KDA_WY_SLOT_BYTES = 256 * 1024;
constexpr int64_t KDA_WY_SLOT_COUNT = 12;

struct ChunkKdaBwdWyTilingContext {
    const char *nodeName;
    const gert::StorageShape *shapes[11];
    ge::DataType types[11];
    float scale;
    int64_t chunkSize;
    int64_t stage;
    uint32_t aicCoreNum;
    size_t systemWorkspaceSize;
};

class ChunkKdaBwdWyTilingProcessor {
public:
    ChunkKdaBwdWyTilingProcessor(
        ChunkKdaBwdWyTilingContext &ctx, KDA::ChunkKdaBwdWyTilingData &tiling)
        : ctx_(ctx), tiling_(tiling) {}

    ge::graphStatus Process()
    {
        for (size_t i = 0; i < 11; ++i) {
            OP_CHECK_IF(ctx_.shapes[i] == nullptr,
                        OP_LOGE(ctx_.nodeName, "all ChunkKdaBwdWy inputs are required"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdWy requires chunk_size=64"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.stage < 0 || ctx_.stage > 8,
                    OP_LOGE(ctx_.nodeName, "ChunkKdaBwdWy stage must be in [0, 8]"),
                    return ge::GRAPH_FAILED);
        for (size_t i : {KDA_WY_Q, KDA_WY_K, KDA_WY_V, KDA_WY_V_NEW,
                         KDA_WY_A, KDA_WY_H, KDA_WY_DO, KDA_WY_DH, KDA_WY_DV_SCAN}) {
            OP_CHECK_IF(ctx_.types[i] != ge::DT_BF16,
                        OP_LOGE(ctx_.nodeName, "P0 matrix inputs must be BF16"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(ctx_.types[KDA_WY_GK] != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "gk must be FP32"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.types[KDA_WY_BETA] != ge::DT_BF16 &&
                        ctx_.types[KDA_WY_BETA] != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "beta must be BF16 or FP32"),
                    return ge::GRAPH_FAILED);

        const gert::Shape q = ctx_.shapes[KDA_WY_Q]->GetStorageShape();
        const gert::Shape k = ctx_.shapes[KDA_WY_K]->GetStorageShape();
        const gert::Shape v = ctx_.shapes[KDA_WY_V]->GetStorageShape();
        const gert::Shape a = ctx_.shapes[KDA_WY_A]->GetStorageShape();
        const gert::Shape h = ctx_.shapes[KDA_WY_H]->GetStorageShape();
        const gert::Shape beta = ctx_.shapes[KDA_WY_BETA]->GetStorageShape();
        const gert::Shape dh = ctx_.shapes[KDA_WY_DH]->GetStorageShape();
        OP_CHECK_IF(q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4 ||
                        a.GetDimNum() != 4 || h.GetDimNum() != 5,
                    OP_LOGE(ctx_.nodeName, "P0 requires BNSD rank-4 tensors and rank-5 h"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(q.GetDim(3) != 128 || k.GetDim(3) != 128 || v.GetDim(3) != 128 ||
                        a.GetDim(3) != 64 || h.GetDim(3) != 128 || h.GetDim(4) != 128,
                    OP_LOGE(ctx_.nodeName, "P0 requires C=64 and K=V=128"),
                    return ge::GRAPH_FAILED);
        tiling_.batch = q.GetDim(0);
        tiling_.headNum = q.GetDim(1);
        tiling_.seqlen = q.GetDim(2);
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.keyDim = 128;
        tiling_.valueDim = 128;
        tiling_.chunkNumPerBatch = (tiling_.seqlen + 63) / 64;
        tiling_.chunkNum = tiling_.batch * tiling_.chunkNumPerBatch;
        OP_CHECK_IF(tiling_.batch <= 0 || tiling_.headNum <= 0 || tiling_.seqlen <= 0,
                    OP_LOGE(ctx_.nodeName, "B, H and T must be positive"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(h.GetDim(0) != tiling_.batch || h.GetDim(1) != tiling_.chunkNumPerBatch ||
                        h.GetDim(2) != tiling_.headNum,
                    OP_LOGE(ctx_.nodeName, "h must be [B,NT,H,128,128]"),
                    return ge::GRAPH_FAILED);
        auto sameQ = [&q](const gert::Shape &shape) {
            if (shape.GetDimNum() != 4) {
                return false;
            }
            for (size_t dim = 0; dim < 4; ++dim) {
                if (shape.GetDim(dim) != q.GetDim(dim)) {
                    return false;
                }
            }
            return true;
        };
        for (size_t idx : {KDA_WY_K, KDA_WY_V, KDA_WY_V_NEW, KDA_WY_GK,
                           KDA_WY_DO, KDA_WY_DV_SCAN}) {
            OP_CHECK_IF(!sameQ(ctx_.shapes[idx]->GetStorageShape()),
                        OP_LOGE(ctx_.nodeName, "q/k/v/v_new/gk/d_o/dv_scan shapes must match"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(a.GetDim(0) != tiling_.batch || a.GetDim(1) != tiling_.headNum ||
                        a.GetDim(2) != tiling_.seqlen,
                    OP_LOGE(ctx_.nodeName, "Akk must be [B,H,T,64]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(beta.GetDimNum() != 3 || beta.GetDim(0) != tiling_.batch ||
                        beta.GetDim(1) != tiling_.headNum || beta.GetDim(2) != tiling_.seqlen,
                    OP_LOGE(ctx_.nodeName, "beta must be [B,H,T]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dh.GetDimNum() != 5 || dh.GetDim(0) != tiling_.batch ||
                        dh.GetDim(1) != tiling_.headNum ||
                        dh.GetDim(2) != tiling_.chunkNumPerBatch ||
                        dh.GetDim(3) != 128 || dh.GetDim(4) != 128,
                    OP_LOGE(ctx_.nodeName, "dh must be [B,H,NT,128,128]"),
                    return ge::GRAPH_FAILED);

        const uint64_t headWindows = (static_cast<uint64_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t taskGroups = static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        blockDim_ = static_cast<uint32_t>(std::min<uint64_t>(taskGroups, ctx_.aicCoreNum));
        blockDim_ = blockDim_ == 0 ? 1 : blockDim_;
        tiling_.workspaceSlotSize = KDA_WY_SLOT_BYTES;
        tiling_.workspaceSlotCount = KDA_WY_SLOT_COUNT;
        tiling_.usedCoreNum = blockDim_;
        tiling_.stage = ctx_.stage;
        tiling_.kEOffset = 0;
        tiling_.dqRawOffset = 16 * 1024;
        tiling_.dkRawOffset = 48 * 1024;
        tiling_.dWOffset = 80 * 1024;
        tiling_.zVOffset = 96 * 1024;
        tiling_.dVbOffset = 112 * 1024;
        tiling_.zWOffset = 144 * 1024;
        tiling_.dKgbOffset = 160 * 1024;
        tiling_.zaInputOffset = 192 * 1024;
        tiling_.zaOutputOffset = 224 * 1024;
        tiling_.scale = ctx_.scale;
        workspaceSize_ = static_cast<size_t>(blockDim_) * KDA_WY_SLOT_COUNT *
                         KDA_WY_SLOT_BYTES + ctx_.systemWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }

private:
    ChunkKdaBwdWyTilingContext &ctx_;
    KDA::ChunkKdaBwdWyTilingData &tiling_;
    uint32_t blockDim_ = 1;
    size_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_WY_TILING_PROCESSOR_H
