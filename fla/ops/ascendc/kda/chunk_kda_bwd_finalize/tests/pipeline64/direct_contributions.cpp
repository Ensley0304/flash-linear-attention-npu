#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/direct_contributions.h"
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_direct_contributions(GM_ADDR workspace,Layout l,ContributionInputs in,uint32_t dtype) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_READY);
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer,DirectContributions::FLOATS*4);
        DirectContributions op;op.Init(buffer.Get<float>());
        if(dtype==1)op.Run<half>(l,reinterpret_cast<uint64_t>(workspace),in);
        else if(dtype==2)op.Run<bfloat16_t>(l,reinterpret_cast<uint64_t>(workspace),in);
        else op.Run<float>(l,reinterpret_cast<uint64_t>(workspace),in);
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
}
extern "C" void run_c64_direct_contributions(void *workspace,void *q,void *k,void *g,void *qr,
    void *value,void *h,void *dh,void *beta,void *dv,void *bp,uint32_t m,uint32_t keys,uint32_t v,
    uint32_t vf,uint32_t exp2,uint32_t dtype,float scale,void *stream) {
    Layout l{};if(!MakeLayout(keys,v,1,1,1,1,0,l))return;
    ContributionInputs in{{reinterpret_cast<uint64_t>(q),keys,1},{reinterpret_cast<uint64_t>(k),keys,1},
        {reinterpret_cast<uint64_t>(g),keys,1},{reinterpret_cast<uint64_t>(qr),keys,1},
        {reinterpret_cast<uint64_t>(value),v,1},{reinterpret_cast<uint64_t>(h),vf?1U:v,vf?keys:1U},
        {reinterpret_cast<uint64_t>(dh),vf?1U:v,vf?keys:1U},{reinterpret_cast<uint64_t>(dv),v,1},
        reinterpret_cast<uint64_t>(beta),reinterpret_cast<uint64_t>(bp),m,scale,exp2?0.6931471805599453f:1.f};
    c64_direct_contributions<<<1,nullptr,stream>>>((GM_ADDR)workspace,l,in,dtype);
}
