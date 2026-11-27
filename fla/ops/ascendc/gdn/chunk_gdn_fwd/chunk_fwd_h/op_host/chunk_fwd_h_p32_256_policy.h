#pragma once
#include <cstdint>
#include <limits>

namespace optiling {
// Qualification only. Public validity remains owned by existing ACLNN/Host checks.
struct FwdHP32Qualification {
    bool arch22=false, keyGateOnly=false, fp32Caches=false, ndCaches=false;
    bool denseBnsd=false, noSequenceMetadata=false, fp32State=false;
    bool stateVFirst=false, saveNewValue=false;
    int64_t batch=0, headsK=0, headsV=0, tokens=0, chunk=0, k=0, v=0;
};
inline bool UseFwdHP32K256(const FwdHP32Qualification &q) {
    // Leave room for the final task += blocks, and for ceil(tokens / 64).
    constexpr int64_t maxTasks = std::numeric_limits<uint32_t>::max() -
        std::numeric_limits<uint16_t>::max();
    return q.arch22 && q.keyGateOnly && q.fp32Caches && q.ndCaches &&
        q.denseBnsd && q.noSequenceMetadata && q.fp32State && !q.stateVFirst &&
        q.saveNewValue && q.batch>0 && q.headsK>0 &&
        q.headsK<=maxTasks/4 && q.batch<=maxTasks/(q.headsK*4) &&
        q.headsK==q.headsV && q.tokens>=1 &&
        q.tokens<=std::numeric_limits<uint32_t>::max()-63 &&
        q.chunk==64 && q.k==256 && q.v==256;
}
}
