#include "chunk_kda_bwd_finalize.h"
#include <cstring>
#include <vector>
#include "aclnn_kernels/transpose.h"
#include "opdev/tensor_view_utils.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdFinalize);

const std::array<const aclTensor *, 7> ChunkKdaBwdFinalize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *gk, const aclTensor *rawG, const aclTensor *beta,
    const aclTensor *aLog, const aclTensor *dtBias, const aclTensor *akk,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dAqk, const aclTensor *dqRaw,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, double lowerBound, int64_t chunkSize,
    bool safeGate, bool useGateInKernel, bool useExp2, bool stateVFirst,
    bool useBetaSigmoidInKernel, bool allowNegEigval, const char *outputLayout,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdFinalize,
           q, k, v, gk, rawG, beta, aLog, dtBias, akk,
           vNew, h, dh, dvScan, dAqk, dqRaw,
           qRstdOptional, kRstdOptional,
           cuSeqlensOptional, chunkIndicesOptional,
           scale, lowerBound, chunkSize,
           safeGate, useGateInKernel, useExp2, stateVFirst,
           useBetaSigmoidInKernel, allowNegEigval, outputLayout,
           dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut);
    const std::array<const aclTensor *, 7> empty{};
    if (!q || !k || !v || !gk || !beta || !h || !outputLayout ||
        (cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) return empty;
    const bool packed = cuSeqlensOptional != nullptr;
    const bool tokenMajor = std::strcmp(outputLayout, packed ? "TND" : "BSND") == 0;
    if (!tokenMajor && std::strcmp(outputLayout, packed ? "NTD" : "BNSD") != 0) return empty;
    const std::array<const aclTensor *, 7> requested{dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut};
    for (auto *output : requested) if (!output) return empty;
#ifndef KDA_BWD_STAGES_ARCH22
    if (useBetaSigmoidInKernel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A5 beta sigmoid backward is not implemented");
        return empty;
    }
    // A5 keeps its accepted head-major kernel. Convert only public outputs
    // whose contract differs; A2/A3 writes the requested layout directly.
    auto alloc = [&](const aclTensor *like, DataType dtype) {
        return executor->AllocTensor(like->GetViewShape(), dtype, Format::FORMAT_ND);
    };
    if (tokenMajor) {
        dqOut = alloc(q, dqOut->GetDataType());
        dkOut = alloc(k, dkOut->GetDataType());
        dvOut = alloc(v, dvOut->GetDataType());
    }
    dBetaOut = alloc(beta, dBetaOut->GetDataType());
    dGOut = alloc(gk, dGOut->GetDataType());
    if (!dqOut || !dkOut || !dvOut || !dBetaOut || !dGOut) return empty;
    outputLayout = packed ? "NTD" : "BNSD";
#endif
    const aclTensor *cuTensor = nullptr;
    const aclTensor *indicesTensor = nullptr;
    if (cuSeqlensOptional != nullptr) {
        cuTensor = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        indicesTensor = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (cuTensor == nullptr || indicesTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert KernelC metadata");
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        }
        for (const aclTensor *tensor : {cuTensor, indicesTensor}) {
            auto *mutableTensor = const_cast<aclTensor *>(tensor);
            mutableTensor->SetStorageFormat(Format::FORMAT_ND);
            mutableTensor->SetViewFormat(Format::FORMAT_ND);
            mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        }
    }
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdFinalize,
        OP_INPUT(q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh,
                 dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
                 cuTensor, indicesTensor),
        OP_OUTPUT(dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut),
        OP_ATTR(scale, lowerBound, chunkSize, safeGate,
                useGateInKernel, useExp2, stateVFirst,
                useBetaSigmoidInKernel, allowNegEigval, outputLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdFinalize failed");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
#ifndef KDA_BWD_STAGES_ARCH22
    const std::array<const aclTensor *, 5> internal{dqOut, dkOut, dvOut, dBetaOut, dGOut};
    for (size_t i = 0; i < internal.size(); ++i) {
        if (internal[i] == requested[i]) continue;
        const auto rank = internal[i]->GetViewShape().GetDimNum();
        std::vector<int64_t> perm(rank);
        for (size_t axis = 0; axis < rank; ++axis) perm[axis] = axis;
        std::swap(perm[packed ? 0 : 1], perm[packed ? 1 : 2]);
        const auto *order = executor->AllocIntArray(perm.data(), perm.size());
        const auto *transposed = order ? l0op::Transpose(internal[i], order, executor) : nullptr;
        if (!transposed || !l0op::ViewCopy(transposed, requested[i], executor)) return empty;
    }
#endif
    return requested;
}

} // namespace l0op
