#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_A_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_A_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal canary API.  The final three-kernel backward wrapper will own the
// stable public ABI after Kernel B is connected.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdAGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *vNew,
    const aclTensor *h, const aclTensor *dO,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    int64_t chunkSize, const aclTensor *dv0Out,
    const aclTensor *dqRawOut, const aclTensor *dAqkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdA(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_A_H
