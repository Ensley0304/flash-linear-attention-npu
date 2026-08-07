#include "chunk_kda_bwd_wy_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdWy(gert::TilingContext *context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdWyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<float>(KDA_WY_SCALE_ATTR);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(KDA_WY_CHUNK_SIZE_ATTR);
    const auto *stage = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, stage);
    ChunkKdaBwdWyTilingContext ctx{};
    ctx.nodeName = context->GetNodeName();
    for (size_t i = 0; i < 11; ++i) {
        ctx.shapes[i] = context->GetRequiredInputShape(i);
        const auto *desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        ctx.types[i] = desc->GetDataType();
    }
    ctx.scale = *scale;
    ctx.chunkSize = *chunkSize;
    ctx.stage = *stage;
    ctx.aicCoreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    ctx.systemWorkspaceSize = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    ChunkKdaBwdWyTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    context->SetTilingKey(1);
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdWy(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdWy)
    .Tiling(Tiling4ChunkKdaBwdWy)
    .TilingParse<ChunkKdaBwdWyCompileInfo>(TilingPrepare4ChunkKdaBwdWy);

} // namespace optiling
