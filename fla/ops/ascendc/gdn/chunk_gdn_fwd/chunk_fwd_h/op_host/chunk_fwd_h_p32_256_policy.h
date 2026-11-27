#pragma once
#include <cstdint>

namespace optiling {
// Qualification only. Public validity remains owned by existing ACLNN/Host checks.
struct FwdHP32Qualification {
    bool arch22=false, keyGateOnly=false, fp32Caches=false, ndCaches=false;
    bool denseBnsd=false, noSequenceMetadata=false, fp32State=false;
    bool stateVFirst=false, saveNewValue=false;
    int64_t batch=0, headsK=0, headsV=0, tokens=0, chunk=0, k=0, v=0;
};
inline bool UseFwdHP32K256(const FwdHP32Qualification &q) {
    return q.arch22 && q.keyGateOnly && q.fp32Caches && q.ndCaches &&
        q.denseBnsd && q.noSequenceMetadata && q.fp32State && !q.stateVFirst &&
        q.saveNewValue && q.batch==1 && (q.headsK==1 || q.headsK==2) &&
        q.headsK==q.headsV && q.tokens>=1 && q.tokens<=4096 &&
        q.chunk==64 && q.k==256 && q.v==256;
}
}
