#include "acl/acl.h"
#include "../../op_kernel/arch22/fwdh256/probe.h"
using namespace GDN::Arch22::FwdH256;
__global__ __aicore__ void fwdh256_mix_probe(Args args) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    uint32_t core=AscendC::GetBlockIdx();
    uint32_t tasks=args.batch*((args.heads+args.group-1)/args.group)*4;
    if ASCEND_IS_AIC {
        Cube op(pipe);op.Init();
        for(uint32_t task=core;task<tasks;task+=args.blocks) {Work w(args,task);CubeWork(op,args,w,core);}
        op.End();
    }
    if ASCEND_IS_AIV {
        core/=2;uint32_t lane=AscendC::GetSubBlockIdx();Vector op(pipe);op.Init();
        for(uint32_t task=core;task<tasks;task+=args.blocks) {Work w(args,task);op.Run(args,w,core,lane);}
    }
}
extern "C" void run_mix(void **ptr,uint32_t batch,uint32_t heads,uint32_t tokens,
    uint32_t group,uint32_t exp2,uint32_t blocks,void *stream) {
    Args a{(GM_ADDR)ptr[0],(GM_ADDR)ptr[1],(GM_ADDR)ptr[2],(GM_ADDR)ptr[3],(GM_ADDR)ptr[4],
           (GM_ADDR)ptr[5],(GM_ADDR)ptr[6],(GM_ADDR)ptr[7],(GM_ADDR)ptr[8],batch,heads,tokens,group,exp2,blocks};
    fwdh256_mix_probe<<<blocks,nullptr,stream>>>(a);
}
