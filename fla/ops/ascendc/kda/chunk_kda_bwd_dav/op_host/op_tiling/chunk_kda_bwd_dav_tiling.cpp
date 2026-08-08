#include "chunk_kda_bwd_dav_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdDAv(gert::TilingContext *context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdDAvTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<float>(KDA_DAV_SCALE_ATTR_IDX);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(KDA_DAV_CHUNK_SIZE_ATTR_IDX);
    const auto *stage = attrs->GetAttrPointer<int64_t>(KDA_DAV_STAGE_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, stage);
    const auto *aqkDesc = context->GetInputDesc(KDA_DAV_AQK_IDX);
    const auto *vNewDesc = context->GetInputDesc(KDA_DAV_V_NEW_IDX);
    const auto *doDesc = context->GetInputDesc(KDA_DAV_DO_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, aqkDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, doDesc);

    ChunkKdaBwdDAvTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_DAV_AQK_IDX),
        context->GetRequiredInputShape(KDA_DAV_V_NEW_IDX),
        context->GetRequiredInputShape(KDA_DAV_DO_IDX),
        aqkDesc->GetDataType(), vNewDesc->GetDataType(), doDesc->GetDataType(),
        *scale, *chunkSize, *stage,
        static_cast<uint32_t>(platform.GetCoreNumAic()),
        static_cast<uint32_t>(platform.GetCoreNumAiv()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()),
    };
    ChunkKdaBwdDAvTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // stage 0/1 retain the proven split diagnostic; stage 2 is the default
    // single-launch MIX path.
    context->SetTilingKey(static_cast<uint64_t>(ctx.stage + 1));
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdDAv(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdDav)
    .Tiling(Tiling4ChunkKdaBwdDAv)
    .TilingParse<ChunkKdaBwdDAvCompileInfo>(TilingPrepare4ChunkKdaBwdDAv);

} // namespace optiling
