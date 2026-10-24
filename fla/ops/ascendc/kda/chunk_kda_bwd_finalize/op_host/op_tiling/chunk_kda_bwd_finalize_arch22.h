#ifndef KDA_FINALIZE_ARCH22_HOST_H
#define KDA_FINALIZE_ARCH22_HOST_H

#include <cstring>
#include <limits>
#include "../../op_kernel/arch22/chunk_kda_bwd_finalize_common.h"

namespace optiling {
inline ge::graphStatus TilingFinalizeArch22(
    gert::TilingContext *context, const ChunkKdaBwdFinalizeTilingContext &ctx,
    bool useBetaSigmoid, bool allowNegEigval, const char *outputLayout)
{
    using namespace KDA::Arch22;
    auto fail = [&]() {
        OP_LOGE(ctx.nodeName, "invalid A2/A3 finalize shape, dtype, layout or metadata");
        return ge::GRAPH_FAILED;
    };
    if (!outputLayout || !ctx.safeGate || !ctx.useGateInKernel ||
        ctx.lowerBound < -5.0 || static_cast<float>(ctx.lowerBound) >= 0.0f) return fail();
    // Check dimensions before the shared shape processor multiplies them.
    if (!ctx.k || !ctx.h) return fail();
    const auto &kShape = ctx.k->GetStorageShape();
    const auto &hShape = ctx.h->GetStorageShape();
    uint64_t elements = 1;
    for (size_t i = 0; i < kShape.GetDimNum(); ++i) {
        const auto dim = kShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    elements = 1;
    for (size_t i = 0; i < hShape.GetDimNum(); ++i) {
        const auto dim = hShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    auto shapeContext = ctx;
    KDA::ChunkKdaBwdFinalizeTilingData shape{};
    ChunkKdaBwdFinalizeTilingProcessor shapeProcessor(shapeContext, shape);
    if (shapeProcessor.Process() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    auto *t = context->GetTilingData<FinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, t);
    *t = {};
    t->batch = shape.B;
    t->heads = shape.NV;
    t->tokens = shape.T;
    t->chunks = shape.totalChunkNum;
    const bool variable = shape.isVariable != 0;
    const size_t rank = variable ? 3 : 4;
    const size_t base = variable ? 0 : 1;
    if (t->heads > UINT32_MAX / (HEAD_DIM * sizeof(float)) ||
        static_cast<uint64_t>(t->batch) > INT64_MAX / HEAD_DIM / sizeof(float) /
            static_cast<uint64_t>(t->heads) / static_cast<uint64_t>(t->tokens)) return fail();
    const bool headMajor = std::strcmp(outputLayout, variable ? "NTD" : "BNSD") == 0;
    const bool tokenMajor = std::strcmp(outputLayout, variable ? "TND" : "BSND") == 0;
    if (!headMajor && !tokenMajor) return fail();
    auto tokenShape = [&](const gert::StorageShape *storage, int64_t width, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank) return false;
        const auto &s = storage->GetStorageShape();
        return (variable || s.GetDim(0) == t->batch) &&
            s.GetDim(base) == (tm ? t->tokens : t->heads) &&
            s.GetDim(base + 1) == (tm ? t->heads : t->tokens) && s.GetDim(base + 2) == width;
    };
    auto scalarShape = [&](const gert::StorageShape *storage, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank - 1) return false;
        const auto &s = storage->GetStorageShape();
        return (variable || s.GetDim(0) == t->batch) && s.GetDim(base) == (tm ? t->tokens : t->heads) &&
            s.GetDim(base + 1) == (tm ? t->heads : t->tokens);
    };
    for (size_t i : {INPUT_Q, INPUT_K, INPUT_V, INPUT_GK, INPUT_V_NEW, INPUT_DV_SCAN, INPUT_DQ_RAW}) {
        if (!tokenShape(context->GetRequiredInputShape(i), HEAD_DIM, false)) return fail();
    }
    for (size_t i : {INPUT_AKK, INPUT_D_AQK}) {
        if (!tokenShape(context->GetRequiredInputShape(i), CHUNK_SIZE, false)) return fail();
    }
    if (!scalarShape(ctx.beta, false) ||
        (ctx.qRstd && (!scalarShape(ctx.qRstd, false) || !scalarShape(ctx.kRstd, false)))) return fail();
    if (variable) {
        const auto *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
        const auto *indicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
        if (!cuTensor || !indicesTensor) return fail();
        const auto *cu = cuTensor->GetData<int64_t>();
        const auto *indices = indicesTensor->GetData<int64_t>();
        const int64_t sequences = shape.seqNum;
        if (!cu || !indices || cu[0] != 0 || cu[sequences] != t->tokens) return fail();
        int64_t task = 0;
        for (int64_t seq = 0; seq < sequences; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq] || cu[seq + 1] > t->tokens) return fail();
            const int64_t count = (cu[seq + 1] - cu[seq] + CHUNK_SIZE - 1) / CHUNK_SIZE;
            for (int64_t chunk = 0; chunk < count; ++chunk, ++task) {
                if (task >= t->chunks || indices[2 * task] != seq || indices[2 * task + 1] != chunk) return fail();
            }
        }
        if (task != t->chunks) return fail();
    }
    for (size_t i : {INPUT_Q, INPUT_K, INPUT_V, INPUT_AKK, INPUT_V_NEW, INPUT_H, INPUT_DH, INPUT_DV_SCAN}) {
        const auto *d = context->GetInputDesc(i);
        if (!d || d->GetDataType() != ge::DT_BF16) return fail();
    }
    for (size_t i : {INPUT_GK, INPUT_BETA, INPUT_D_AQK, INPUT_DQ_RAW, INPUT_Q_RSTD, INPUT_K_RSTD}) {
        const auto *d = context->GetOptionalInputDesc(i);
        if (d && d->GetDataType() != ge::DT_FLOAT) return fail();
    }
    if (ctx.useGateInKernel) {
        for (size_t i : {INPUT_RAW_G, INPUT_A_LOG}) {
            const auto *d = context->GetOptionalInputDesc(i);
            if (!d || d->GetDataType() != ge::DT_FLOAT) return fail();
        }
        const auto *a = context->GetOptionalInputShape(INPUT_A_LOG);
        const auto *bias = context->GetOptionalInputShape(INPUT_DT_BIAS);
        if (!tokenShape(context->GetOptionalInputShape(INPUT_RAW_G), HEAD_DIM, false) ||
            !a || a->GetStorageShape().GetShapeSize() != t->heads ||
            (!bias || (bias->GetStorageShape().GetShapeSize() != t->heads * HEAD_DIM ||
                      context->GetOptionalInputDesc(INPUT_DT_BIAS)->GetDataType() != ge::DT_FLOAT))) return fail();
    }
    for (size_t i = 0; i < 3; ++i) {
        if (!tokenShape(context->GetOutputShape(i), HEAD_DIM, tokenMajor)) return fail();
    }
    if (!scalarShape(context->GetOutputShape(3), true) ||
        !tokenShape(context->GetOutputShape(4), HEAD_DIM, true)) return fail();
    const auto qType = context->GetOutputDesc(0)->GetDataType();
    if (qType != ge::DT_BF16 || context->GetOutputDesc(1)->GetDataType() != qType ||
        context->GetOutputDesc(2)->GetDataType() != ge::DT_BF16) return fail();
    for (size_t i = 3; i < 7; ++i) {
        if (context->GetOutputDesc(i)->GetDataType() != ge::DT_FLOAT) return fail();
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto &input = context->GetRequiredInputShape(INPUT_A_LOG + i)->GetStorageShape();
        const auto &output = context->GetOutputShape(5 + i)->GetStorageShape();
        if (input.GetDimNum() != output.GetDimNum()) return fail();
        for (size_t axis = 0; axis < input.GetDimNum(); ++axis) {
            if (input.GetDim(axis) != output.GetDim(axis)) return fail();
        }
    }
    const uint64_t maxBytes = static_cast<uint64_t>(INT64_MAX) - ctx.sysWorkspaceSize;
    if (static_cast<uint64_t>(t->chunks) > maxBytes / t->heads / (PARTIAL_FLOATS * sizeof(float))) return fail();
    const uint64_t work = static_cast<uint64_t>(t->chunks) * ((t->heads + 1) / HEADS_PER_WORK);
    t->usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(ctx.aicCoreNum, work));
    if (!t->usedCoreNum) return fail();
    t->isVariable = variable;
    t->outputTokenMajor = tokenMajor;
    t->hasQkL2Norm = ctx.qRstd != nullptr;
    t->betaMode = useBetaSigmoid ? (allowNegEigval ? 2 : 1) : 0;
    t->scale = static_cast<float>(ctx.scale);
    t->lowerBound = static_cast<float>(ctx.lowerBound);
    t->partialOffset = static_cast<uint64_t>(t->usedCoreNum) * WS_CORE_BYTES;
    const uint64_t partialBytes = static_cast<uint64_t>(t->chunks) * t->heads * PARTIAL_FLOATS * sizeof(float);
    if (partialBytes > maxBytes - t->partialOffset) return fail();
    context->SetTilingKey(t->hasQkL2Norm ? 102 : 101);
    context->SetBlockDim(t->usedCoreNum);
    context->SetScheduleMode(1);
    context->GetWorkspaceSizes(1)[0] = ctx.sysWorkspaceSize + t->partialOffset + partialBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif
