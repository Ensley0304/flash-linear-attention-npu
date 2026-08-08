#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk, const aclTensor *aqk,
    const aclTensor *akk, const aclTensor *w, const aclTensor *qg,
    const aclTensor *kg, const aclTensor *vNew, const aclTensor *h,
    const aclTensor *dO, const aclTensor *rawGOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional, const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout,
    double scale, int64_t chunkSize, bool safeGate, double lowerBound,
    bool useGateInKernel, bool stateVFirst, const char *recomputePolicy,
    const aclTensor *dAqkScratch, const aclTensor *dv0Scratch,
    const aclTensor *dhScratch, const aclTensor *dvScanScratch,
    const aclTensor *dqBaseScratch, const aclTensor *dkBaseScratch,
    const aclTensor *dbBaseScratch, const aclTensor *dgBaseScratch,
    const aclTensor *dAkkScratch,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dh0Out,
    const aclTensor *dAOut, const aclTensor *dbiasOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_BWD_H
