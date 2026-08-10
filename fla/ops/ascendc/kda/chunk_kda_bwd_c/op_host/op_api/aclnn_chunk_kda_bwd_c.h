#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_C_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_C_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal canary API. Kernel B will be connected by the final public wrapper.
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdCGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *akk, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dqRaw,
    const aclTensor *dAqk, const aclTensor *rawGOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dAkkOut,
    const aclTensor *dAOutOptional, const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdC(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_C_H
