#include <cassert>
#include <iostream>
#include "../../op_host/chunk_fwd_h_p32_256_policy.h"
int main() {
    using namespace optiling;
    FwdHP32Qualification base;
    base.arch22=base.keyGateOnly=base.fp32Caches=base.ndCaches=true;
    base.denseBnsd=base.noSequenceMetadata=base.fp32State=base.saveNewValue=true;
    base.batch=base.headsK=base.headsV=1;base.tokens=65;base.chunk=64;base.k=base.v=256;
    assert(UseFwdHP32K256(base));
    unsigned count=1;
    for (auto member : {&FwdHP32Qualification::arch22,&FwdHP32Qualification::keyGateOnly,
        &FwdHP32Qualification::fp32Caches,&FwdHP32Qualification::ndCaches,
        &FwdHP32Qualification::denseBnsd,&FwdHP32Qualification::noSequenceMetadata,
        &FwdHP32Qualification::fp32State,&FwdHP32Qualification::saveNewValue}) {
        auto q=base;q.*member=false;assert(!UseFwdHP32K256(q));++count;
    }
    auto q=base;q.stateVFirst=true;assert(!UseFwdHP32K256(q));++count;
    for(auto member : {&FwdHP32Qualification::k,&FwdHP32Qualification::v}) {
        q=base;q.*member=128;assert(!UseFwdHP32K256(q));++count;
    }
    for(int64_t t : {1,7,63,64,65,129,1024,4096}) {
        q=base;q.tokens=t;q.headsK=q.headsV=2;assert(UseFwdHP32K256(q));++count;
    }
    for(int64_t t : {0LL,4294967233LL}) {q=base;q.tokens=t;assert(!UseFwdHP32K256(q));++count;}
    for(int64_t t : {4097,8193}) {q=base;q.tokens=t;assert(UseFwdHP32K256(q));++count;}
    for(int64_t b : {1,2,4,8}) for(int64_t h : {1,2,3,4,8,16,32}) {
        q=base;q.batch=b;q.headsK=q.headsV=h;assert(UseFwdHP32K256(q));++count;
    }
    for(int64_t b : {0LL,-1LL,1073741824LL}) {
        q=base;q.batch=b;assert(!UseFwdHP32K256(q));++count;
    }
    for(int64_t h : {0LL,-1LL,1073741824LL,9223372036854775807LL}) {
        q=base;q.headsK=q.headsV=h;assert(!UseFwdHP32K256(q));++count;
    }
    q=base;q.chunk=32;assert(!UseFwdHP32K256(q));++count;
    q=base;q.headsV=2;assert(!UseFwdHP32K256(q));++count;
    std::cout << "Host qualification checks passed: " << count << '\n';
}
