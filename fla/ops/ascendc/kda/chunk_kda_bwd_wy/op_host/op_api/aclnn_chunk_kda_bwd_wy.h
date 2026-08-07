#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_WY_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_WY_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal canary interface.  The final public API is ChunkKdaBwd; this L0
// entry exists so the fused WY stage can be precision-tested independently.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdWyGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *a, const aclTensor *h, const aclTensor *dO,
    const aclTensor *dh, const aclTensor *dvScan, float scale,
    int64_t chunkSize, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut, const aclTensor *dgOut,
    const aclTensor *dAkkOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdWy(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_WY_H
