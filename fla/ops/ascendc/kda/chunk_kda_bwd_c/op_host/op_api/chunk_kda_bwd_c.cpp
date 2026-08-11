#include "chunk_kda_bwd_c.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdC);

namespace {

const aclTensor *EmptyTensor(aclOpExecutor *executor, DataType dtype)
{
    op::Shape shape;
    shape.AppendDim(0);
    return executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
}

} // namespace

const std::array<const aclTensor *, 8> ChunkKdaBwdC(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *akk, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dqRaw,
    const aclTensor *dAqk, const aclTensor *rawG,
    const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *cuSeqlens, const aclTensor *chunkIndices,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool dhHeadMajor,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dAkkOut,
    const aclTensor *dAOut, const aclTensor *dBiasOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdC, q, k, v, vNew, gk, beta, akk, h, dh,
           dvScan, dqRaw, dAqk, rawG, aLog, dtBias, cuSeqlens,
           chunkIndices, scale, chunkSize, safeGate, useGateInKernel,
           lowerBound, dhHeadMajor, dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut,
           dAOut, dBiasOut);
    // On A5, nullable tensor entries in this mixed-kernel launch can leave
    // invalid device arguments even when the selected branch does not read
    // them. Preserve the public optional semantics while keeping a stable
    // kernel ABI with zero-element placeholders.
    const aclTensor *actualRawG = rawG != nullptr ? rawG :
        EmptyTensor(executor, DataType::DT_FLOAT);
    const aclTensor *actualALog = aLog != nullptr ? aLog :
        EmptyTensor(executor, DataType::DT_FLOAT);
    const aclTensor *actualDtBias = dtBias != nullptr ? dtBias :
        EmptyTensor(executor, DataType::DT_FLOAT);
    const aclTensor *actualDAOut = dAOut != nullptr ? dAOut :
        EmptyTensor(executor, DataType::DT_FLOAT);
    const aclTensor *actualDBiasOut = dBiasOut != nullptr ? dBiasOut :
        EmptyTensor(executor, DataType::DT_FLOAT);
    if (actualRawG == nullptr || actualALog == nullptr ||
        actualDtBias == nullptr || actualDAOut == nullptr ||
        actualDBiasOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "Allocate optional ChunkKdaBwdC placeholders failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr};
    }

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdC,
        OP_INPUT(q, k, v, vNew, gk, beta, akk, h, dh, dvScan,
                 dqRaw, dAqk, actualRawG, actualALog, actualDtBias, cuSeqlens,
                 chunkIndices),
        OP_OUTPUT(dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut,
                  actualDAOut, actualDBiasOut),
        OP_ATTR(scale, chunkSize, safeGate, useGateInKernel,
                lowerBound, dhHeadMajor));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdC failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dbOut, dgOut, dAkkOut, dAOut,
            dBiasOut};
}

} // namespace l0op
