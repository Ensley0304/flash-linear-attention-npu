#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_DAV_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_DAV_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal canary interface.  ChunkKdaBwd remains the only stable public
// backward API; this entry is kept so K1 can be validated independently.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdDavGetWorkspaceSize(
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *dO,
    float scale,
    int64_t chunkSize,
    const aclTensor *dAqkOut,
    const aclTensor *dvOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdDav(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_DAV_H
