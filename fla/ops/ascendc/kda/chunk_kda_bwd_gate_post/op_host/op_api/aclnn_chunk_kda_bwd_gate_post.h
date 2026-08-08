#ifndef ACLNN_CHUNK_KDA_BWD_GATE_POST_H
#define ACLNN_CHUNK_KDA_BWD_GATE_POST_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnChunkKdaBwdGatePostGetWorkspaceSize(
    const aclTensor *dgHv, int64_t chunkSize, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnChunkKdaBwdGatePost(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CHUNK_KDA_BWD_GATE_POST_H
