#include "chunk_kda_bwd.h"

#include "../../../chunk_kda_bwd_dav/op_host/op_api/chunk_kda_bwd_dav.h"
#include "../../../chunk_kda_bwd_wy/op_host/op_api/chunk_kda_bwd_wy.h"
#include "../../../chunk_kda_bwd_intra/op_host/op_api/chunk_kda_bwd_intra.h"
#include "../../../chunk_kda_bwd_gate_post/op_host/op_api/chunk_kda_bwd_gate_post.h"
#include "../../../../gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstdlib>

using namespace op;

namespace l0op {

namespace {
KdaBackwardOutputs Failure()
{
    return {nullptr, nullptr, nullptr, nullptr, nullptr};
}
} // namespace

KdaBackwardOutputs KdaChunkBackward(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk, const aclTensor *aqk,
    const aclTensor *akk, const aclTensor *w, const aclTensor *qg,
    const aclTensor *kg, const aclTensor *vNew, const aclTensor *h,
    const aclTensor *dO, double scale, int64_t chunkSize,
    const aclTensor *dAqk, const aclTensor *dv0,
    const aclTensor *dh, const aclTensor *dvScan,
    const aclTensor *dqBase, const aclTensor *dkBase,
    const aclTensor *dbBase, const aclTensor *dgBase,
    const aclTensor *dAkk,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbOut, const aclTensor *dgOut,
    aclOpExecutor *executor)
{
    L0_DFX(KdaChunkBackward, q, k, v, beta, gk, aqk, akk, w, qg, kg,
           vNew, h, dO, scale, chunkSize, dqOut, dkOut, dvOut, dbOut, dgOut);
    if (dAqk == nullptr || dv0 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "K1 scratch tensors are required");
        return Failure();
    }
    const auto k1 = ChunkKdaBwdDav(
        aqk, vNew, dO, static_cast<float>(scale), chunkSize, dAqk, dv0, executor);
    if (k1[0] == nullptr || k1[1] == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkKdaBwdDav launch registration failed");
        return Failure();
    }
    if (dh == nullptr || dvScan == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "K2 scratch tensors are required");
        return Failure();
    }
    const auto k2 = ChunkGatedDeltaRuleBwdDhu(
        qg, kg, w, dO, dv0, nullptr, gk, nullptr, nullptr, nullptr, nullptr,
        scale, chunkSize, dh, nullptr, dvScan, executor);
    if (k2[0] == nullptr || k2[2] == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "key-wise ChunkGatedDeltaRuleBwdDhu registration failed");
        return Failure();
    }
    if (dqBase == nullptr || dkBase == nullptr ||
        dbBase == nullptr || dgBase == nullptr || dAkk == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "K3 scratch tensors are required");
        return Failure();
    }
    const bool splitWyFallback = []() {
        const char *value = std::getenv("KDA_BWD_WY_SPLIT_FALLBACK");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    const int64_t wyLaunchCount = splitWyFallback ? 8 : 1;
    for (int64_t launchIdx = 0; launchIdx < wyLaunchCount; ++launchIdx) {
        const int64_t stage = splitWyFallback ? launchIdx : 8;
        const auto k3 = ChunkKdaBwdWy(
            q, k, v, vNew, gk, beta, akk, h, dO, dh, dvScan,
            static_cast<float>(scale), chunkSize, stage, dqBase, dkBase,
            dvOut, dbBase, dgBase, dAkk, executor);
        for (const aclTensor *tensor : k3) {
            if (tensor == nullptr) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "ChunkKdaBwdWy registration failed");
                return Failure();
            }
        }
    }

    const auto k4 = ChunkKdaBwdIntra(
        q, k, gk, beta, dAqk, dAkk, dqBase, dkBase, dbBase, dgBase,
        nullptr, nullptr, chunkSize, true, 0, 0,
        dqOut, dkOut, dbOut, dgOut, executor);
    for (const aclTensor *tensor : k4) {
        if (tensor == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkKdaBwdIntra registration failed");
            return Failure();
        }
    }
    // GatePost loads a complete chunk into UB before writing it back, so the
    // K4 dg_hv output can be finalized in place.  This also avoids an internal
    // custom-op output whose lifetime was unreliable on the current runtime.
    const aclTensor *gate = ChunkKdaBwdGatePost(dgOut, chunkSize, dgOut, executor);
    if (gate == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkKdaBwdGatePost registration failed");
        return Failure();
    }
    return {dqOut, dkOut, dvOut, dbOut, dgOut};
}

} // namespace l0op
