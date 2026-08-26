#include "chunk_kda_bwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "acl/acl_base_rt.h"
#include <string>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwd);
OP_TYPE_REGISTER(KdaGateBwdPost);
OP_TYPE_REGISTER(KdaGateBwdPostVarlen);

ChunkKdaBwdOutputs KdaChunkBackward(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *dq,
    const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg,
    const aclTensor *dAOptional,
    const aclTensor *dBiasOptional, aclOpExecutor *executor)
{
    L0_DFX(KdaChunkBackward, q, k, v, beta, gk, aqk, akk, wOptional,
           qgOptional, kgOptional, vNewOptional, hOptional, dO,
           rawGOptional, aLogOptional, dtBiasOptional,
           cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize,
           safeGate, useGateInKernel, lowerBound, disableRecompute, useExp2,
           stateVFirst, dq, dk, dv, db, dg,
           dAOptional, dBiasOptional);

    // Kernel C writes the chunk-local, unscanned gate gradient into the
    // final dg storage.  The following AIV gate kernel consumes and updates
    // that storage in place, avoiding a full-size cross-kernel workspace.
    const aclTensor *dgBase = dg;

    // The A5 chunk-wide shared-anchor path can produce 0 * Inf when a valid
    // 64-token cumulative gate spans more than the FP32 exp2 range.  Keep the
    // numerically stable local-anchor Intra path and apply the already
    // supported raw-gate chain in the standalone post kernel.
    const bool fuseGateInC = false;
    const bool deferGatePost = useGateInKernel && !fuseGateInC;

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwd,
        OP_INPUT(q, k, v, beta, gk, aqk, akk, wOptional, qgOptional,
                 kgOptional, vNewOptional, hOptional, dO,
                 rawGOptional, aLogOptional, dtBiasOptional,
                 cuSeqlensOptional, chunkIndicesOptional),
        OP_OUTPUT(dq, dk, dv, db, dgBase, dAOptional, dBiasOptional),
        OP_ATTR(scale, chunkSize, safeGate, fuseGateInC, lowerBound,
                disableRecompute, useExp2, stateVFirst,
                deferGatePost));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwd failed.");
        return {};
    }
    if (deferGatePost) {
        if (cuSeqlensOptional != nullptr) {
            ret = ADD_TO_LAUNCHER_LIST_AICORE(
                KdaGateBwdPostVarlen,
                OP_INPUT(dgBase, rawGOptional, aLogOptional, dtBiasOptional,
                         cuSeqlensOptional, chunkIndicesOptional),
                OP_OUTPUT(dg, dAOptional, dBiasOptional),
                OP_ATTR(chunkSize, safeGate, lowerBound, false));
        } else {
            ret = ADD_TO_LAUNCHER_LIST_AICORE(
                KdaGateBwdPost,
                OP_INPUT(dgBase, rawGOptional, aLogOptional, dtBiasOptional,
                         cuSeqlensOptional, chunkIndicesOptional),
                OP_OUTPUT(dg, dAOptional, dBiasOptional),
                OP_ATTR(chunkSize, safeGate, lowerBound, false));
        }
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "ADD_TO_LAUNCHER_LIST_AICORE GatePost failed.");
            return {};
        }
    }
    return {dq, dk, dv, db, dg, dAOptional, dBiasOptional};
}

} // namespace l0op
