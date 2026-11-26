#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/direct_score.h"
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_direct_score(DirectScoreInputs input) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_READY);
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;DirectScore op;op.Init(pipe);
        const uint32_t width=input.columns;
        for(uint32_t col=0;col<width;col+=64) {
            input.columnStart=col;input.columns=width-col<64?width-col:64;
            input.accumulateBeta=col!=0;op.Run(input);
        }
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
}
extern "C" void run_c64_direct_score(void *q,void *k,void *g,void *ds,void *dl,
    void *beta,void *dq,void *dk,void *dg,void *db,uint32_t m,uint32_t d,float base,void *stream) {
    DirectScoreInputs in{{reinterpret_cast<uint64_t>(q),d,1},{reinterpret_cast<uint64_t>(k),d,1},
        {reinterpret_cast<uint64_t>(g),d,1},{reinterpret_cast<uint64_t>(ds),m,1},
        {reinterpret_cast<uint64_t>(dl),m,1},{reinterpret_cast<uint64_t>(dq),d,1},
        {reinterpret_cast<uint64_t>(dk),d,1},{reinterpret_cast<uint64_t>(dg),d,1},
        reinterpret_cast<uint64_t>(beta),reinterpret_cast<uint64_t>(db),m,d,0,base};
    c64_direct_score<<<1,nullptr,stream>>>(in);
}
