#ifndef CHUNK_KDA_BWD_DAV_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_DAV_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_dav_struct.h"

namespace optiling {

constexpr size_t KDA_DAV_AQK_IDX = 0;
constexpr size_t KDA_DAV_V_NEW_IDX = 1;
constexpr size_t KDA_DAV_DO_IDX = 2;
constexpr size_t KDA_DAV_SCALE_ATTR_IDX = 0;
constexpr size_t KDA_DAV_CHUNK_SIZE_ATTR_IDX = 1;
constexpr size_t KDA_DAV_STAGE_ATTR_IDX = 2;

struct ChunkKdaBwdDAvTilingContext {
    const char *nodeName;
    const gert::StorageShape *aqkShape;
    const gert::StorageShape *vNewShape;
    const gert::StorageShape *doShape;
    ge::DataType aqkType;
    ge::DataType vNewType;
    ge::DataType doType;
    float scale;
    int64_t chunkSize;
    int64_t stage;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    size_t systemWorkspaceSize;
};

class ChunkKdaBwdDAvTilingProcessor {
public:
    ChunkKdaBwdDAvTilingProcessor(
        ChunkKdaBwdDAvTilingContext &ctx, KDA::ChunkKdaBwdDAvTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(ctx_.aqkShape == nullptr || ctx_.vNewShape == nullptr || ctx_.doShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Aqk, v_new and d_o are required."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.aqkType != ge::DT_BF16 || ctx_.vNewType != ge::DT_BF16 ||
                        ctx_.doType != ge::DT_BF16,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdDAv requires BF16 inputs."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdDAv requires chunk_size=64."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.stage < 0 || ctx_.stage > 2,
                    OP_LOGE(ctx_.nodeName, "ChunkKdaBwdDAv stage must be in [0, 2]."),
                    return ge::GRAPH_FAILED);
        // Shape checks and B/H/T addressing are defined by the tensor's
        // logical dimensions.  The decoupled ctypes runtime deliberately
        // describes the backing allocation as a flat storage capacity, so
        // GetStorageShape() is rank-1 even for a contiguous BNSD tensor.
        const gert::Shape aqk = ctx_.aqkShape->GetOriginShape();
        const gert::Shape vNew = ctx_.vNewShape->GetOriginShape();
        const gert::Shape dO = ctx_.doShape->GetOriginShape();
        OP_CHECK_IF(aqk.GetDimNum() != 4 || vNew.GetDimNum() != 4 || dO.GetDimNum() != 4,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdDAv requires dense rank-4 BNSD inputs."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(aqk.GetDim(3) != 64,
                    OP_LOGE(ctx_.nodeName, "Aqk last dimension must be 64."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vNew.GetDim(3) != 128,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdDAv requires V=128."),
                    return ge::GRAPH_FAILED);
        for (size_t dim = 0; dim < 3; ++dim) {
            OP_CHECK_IF(aqk.GetDim(dim) != vNew.GetDim(dim) ||
                            vNew.GetDim(dim) != dO.GetDim(dim),
                        OP_LOGE(ctx_.nodeName, "Aqk/v_new/d_o B,H,T dimensions must match."),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(vNew.GetDim(3) != dO.GetDim(3),
                    OP_LOGE(ctx_.nodeName, "v_new and d_o shapes must match."),
                    return ge::GRAPH_FAILED);

        tiling_.batch = static_cast<int64_t>(aqk.GetDim(0));
        tiling_.headNum = static_cast<int64_t>(aqk.GetDim(1));
        tiling_.seqlen = static_cast<int64_t>(aqk.GetDim(2));
        tiling_.valueDim = static_cast<int64_t>(vNew.GetDim(3));
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.chunkNumPerBatch =
            (tiling_.seqlen + tiling_.chunkSize - 1) / tiling_.chunkSize;
        tiling_.chunkNum = tiling_.batch * tiling_.chunkNumPerBatch;
        tiling_.scale = ctx_.scale;
        tiling_.stage = ctx_.stage;
        OP_CHECK_IF(tiling_.batch <= 0 || tiling_.headNum <= 0 || tiling_.seqlen <= 0,
                    OP_LOGE(ctx_.nodeName, "B, H and T must be positive."),
                    return ge::GRAPH_FAILED);

        const uint64_t headWindows =
            (static_cast<uint64_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t taskGroups = static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        const uint32_t availableCoreNum =
            ctx_.stage == 1 ? ctx_.aivCoreNum : ctx_.aicCoreNum;
        blockDim_ = static_cast<uint32_t>(
            std::min<uint64_t>(taskGroups, static_cast<uint64_t>(availableCoreNum)));
        if (blockDim_ == 0) {
            blockDim_ = 1;
        }
        tiling_.usedCoreNum = blockDim_;
        workspaceSize_ = ctx_.systemWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }

private:
    ChunkKdaBwdDAvTilingContext &ctx_;
    KDA::ChunkKdaBwdDAvTilingData &tiling_;
    uint32_t blockDim_ = 1;
    size_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_DAV_TILING_PROCESSOR_H
