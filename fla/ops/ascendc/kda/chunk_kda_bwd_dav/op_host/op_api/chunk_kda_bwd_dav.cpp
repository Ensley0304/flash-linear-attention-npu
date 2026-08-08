#include "chunk_kda_bwd_dav.h"
#include <cstdlib>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdDav);

const std::array<const aclTensor *, 2> ChunkKdaBwdDav(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO,
    float scale, int64_t chunkSize, const aclTensor *dAqkOut,
    const aclTensor *dvOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdDav, aqk, vNew, dO, scale, chunkSize, dAqkOut, dvOut);
    const char *splitFallback = std::getenv("KDA_BWD_DAV_SPLIT_FALLBACK");
    const bool useSplitFallback =
        splitFallback != nullptr && splitFallback[0] == '1';
    const int64_t firstStage = useSplitFallback ? 0 : 2;
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdDav,
        OP_INPUT(aqk, vNew, dO),
        OP_OUTPUT(dAqkOut, dvOut),
        OP_ATTR(scale, chunkSize, firstStage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdDav failed.");
        return {nullptr, nullptr};
    }
    if (useSplitFallback) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(
            ChunkKdaBwdDav,
            OP_INPUT(aqk, vNew, dO),
            OP_OUTPUT(dAqkOut, dvOut),
            OP_ATTR(scale, chunkSize, static_cast<int64_t>(1)));
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdDav fallback post failed.");
            return {nullptr, nullptr};
        }
    }
    return {dAqkOut, dvOut};
}

} // namespace l0op
