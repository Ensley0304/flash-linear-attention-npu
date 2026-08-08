#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
using KdaBackwardOutputs = std::array<const aclTensor *, 5>;

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
    aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_H
