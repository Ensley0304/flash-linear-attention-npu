#include "chunk_kda_bwd_gate_post_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus Tiling4ChunkKdaBwdGatePost(gert::TilingContext *context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdGatePostTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    const auto *desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    ChunkKdaBwdGatePostTilingContext ctx{
        context->GetNodeName(), context->GetRequiredInputShape(0), desc->GetDataType(), *chunkSize,
        static_cast<uint32_t>(platform.GetCoreNumAiv()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize())};
    ChunkKdaBwdGatePostTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // The generic AIV kernel is compiled as function entry 0.  Keep the
    // runtime tiling key aligned with that single generated entry.
    context->SetTilingKey(0);
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdGatePost(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdGatePost)
    .Tiling(Tiling4ChunkKdaBwdGatePost)
    .TilingParse<ChunkKdaBwdGatePostCompileInfo>(TilingPrepare4ChunkKdaBwdGatePost);
} // namespace optiling
