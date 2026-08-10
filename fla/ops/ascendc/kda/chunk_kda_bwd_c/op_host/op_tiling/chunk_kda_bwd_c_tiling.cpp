#include "chunk_kda_bwd_c_tiling.h"

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdC(gert::TilingContext *context)
{
    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdCTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<float>(KDA_C_SCALE_ATTR);
    const auto *chunkSize =
        attrs->GetAttrPointer<int64_t>(KDA_C_CHUNK_SIZE_ATTR);
    const auto *safeGate =
        attrs->GetAttrPointer<bool>(KDA_C_SAFE_GATE_ATTR);
    const auto *useGate =
        attrs->GetAttrPointer<bool>(KDA_C_USE_GATE_ATTR);
    const auto *lowerBound =
        attrs->GetAttrPointer<float>(KDA_C_LOWER_BOUND_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, lowerBound);

    ChunkKdaBwdCTilingContext ctx{};
    ctx.nodeName = context->GetNodeName();
    for (size_t i = 0; i < KDA_C_INPUT_COUNT; ++i) {
        ctx.shapes[i] = i < KDA_C_REQUIRED_INPUT_COUNT ?
            context->GetRequiredInputShape(i) :
            context->GetOptionalInputShape(i);
        const auto *desc = context->GetInputDesc(i);
        if (desc != nullptr) {
            ctx.types[i] = desc->GetDataType();
        }
    }
    ctx.scale = *scale;
    ctx.chunkSize = *chunkSize;
    ctx.safeGate = *safeGate;
    ctx.useGateInKernel = *useGate;
    ctx.lowerBound = *lowerBound;
    ctx.aicCoreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    ctx.systemWorkspaceSize =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    ChunkKdaBwdCTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdC(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdC)
    .Tiling(Tiling4ChunkKdaBwdC)
    .TilingParse<ChunkKdaBwdCCompileInfo>(TilingPrepare4ChunkKdaBwdC);

} // namespace optiling
