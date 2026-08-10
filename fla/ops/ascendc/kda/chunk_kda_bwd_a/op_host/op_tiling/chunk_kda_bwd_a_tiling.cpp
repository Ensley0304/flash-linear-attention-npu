#include "chunk_kda_bwd_a_tiling.h"

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdA(gert::TilingContext *context)
{
    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdATilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<float>(KDA_BWD_A_SCALE_ATTR_IDX);
    const auto *chunkSize =
        attrs->GetAttrPointer<int64_t>(KDA_BWD_A_CHUNK_SIZE_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);

    const auto *aqkDesc = context->GetInputDesc(KDA_BWD_A_AQK_IDX);
    const auto *qgDesc = context->GetInputDesc(KDA_BWD_A_QG_IDX);
    const auto *vNewDesc = context->GetInputDesc(KDA_BWD_A_V_NEW_IDX);
    const auto *hDesc = context->GetInputDesc(KDA_BWD_A_H_IDX);
    const auto *doDesc = context->GetInputDesc(KDA_BWD_A_DO_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, aqkDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, qgDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, doDesc);

    ChunkKdaBwdATilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_BWD_A_AQK_IDX),
        context->GetRequiredInputShape(KDA_BWD_A_QG_IDX),
        context->GetRequiredInputShape(KDA_BWD_A_V_NEW_IDX),
        context->GetRequiredInputShape(KDA_BWD_A_H_IDX),
        context->GetRequiredInputShape(KDA_BWD_A_DO_IDX),
        context->GetOptionalInputShape(KDA_BWD_A_CU_SEQLENS_IDX),
        context->GetOptionalInputShape(KDA_BWD_A_CHUNK_INDICES_IDX),
        aqkDesc->GetDataType(), qgDesc->GetDataType(),
        vNewDesc->GetDataType(), hDesc->GetDataType(), doDesc->GetDataType(),
        *scale, *chunkSize,
        static_cast<uint32_t>(platform.GetCoreNumAic()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()),
    };
    ChunkKdaBwdATilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdA(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdA)
    .Tiling(Tiling4ChunkKdaBwdA)
    .TilingParse<ChunkKdaBwdACompileInfo>(TilingPrepare4ChunkKdaBwdA);

} // namespace optiling
