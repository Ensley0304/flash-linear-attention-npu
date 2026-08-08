#ifndef CHUNK_KDA_BWD_GATE_POST_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_GATE_POST_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/chunk_kda_bwd_gate_post_struct.h"

namespace optiling {
struct ChunkKdaBwdGatePostTilingContext {
    const char *nodeName;
    const gert::StorageShape *dgShape;
    ge::DataType dgType;
    int64_t chunkSize;
    uint32_t aivCoreNum;
    size_t systemWorkspaceSize;
};

class ChunkKdaBwdGatePostTilingProcessor {
public:
    ChunkKdaBwdGatePostTilingProcessor(
        ChunkKdaBwdGatePostTilingContext &ctx, KDA::ChunkKdaBwdGatePostTilingData &tiling)
        : ctx_(ctx), tiling_(tiling) {}

    ge::graphStatus Process()
    {
        OP_CHECK_IF(ctx_.dgShape == nullptr, OP_LOGE(ctx_.nodeName, "dg_hv is required."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.dgType != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "ChunkKdaBwdGatePost requires FP32 dg_hv."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "P0 ChunkKdaBwdGatePost requires chunk_size=64."),
                    return ge::GRAPH_FAILED);
        const gert::Shape shape = ctx_.dgShape->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 4 || shape.GetDim(3) != 128,
                    OP_LOGE(ctx_.nodeName, "P0 dg_hv must be [B,HV,T,128]."),
                    return ge::GRAPH_FAILED);
        tiling_.batch = shape.GetDim(0);
        tiling_.headNum = shape.GetDim(1);
        tiling_.seqlen = shape.GetDim(2);
        tiling_.keyDim = shape.GetDim(3);
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.chunkNumPerBatch = (tiling_.seqlen + tiling_.chunkSize - 1) / tiling_.chunkSize;
        tiling_.taskNum = tiling_.batch * tiling_.headNum * tiling_.chunkNumPerBatch;
        OP_CHECK_IF(tiling_.batch == 0 || tiling_.headNum == 0 || tiling_.seqlen == 0,
                    OP_LOGE(ctx_.nodeName, "B, HV and T must be positive."), return ge::GRAPH_FAILED);
        blockDim_ = static_cast<uint32_t>(
            std::min<uint64_t>(tiling_.taskNum, static_cast<uint64_t>(ctx_.aivCoreNum)));
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
    ChunkKdaBwdGatePostTilingContext &ctx_;
    KDA::ChunkKdaBwdGatePostTilingData &tiling_;
    uint32_t blockDim_ = 1;
    size_t workspaceSize_ = 0;
};
} // namespace optiling

#endif // CHUNK_KDA_BWD_GATE_POST_TILING_PROCESSOR_H
