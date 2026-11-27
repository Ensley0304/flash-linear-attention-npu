#pragma once
// Production configuration of the validated 557667b9 P32 work body.
#define FWDH_RIGHT_RESIDENT 1
#define FWDH_D_BLOCK_PIPELINE 1
#include "probe.h"

namespace GDN::Arch22::FwdH256 {
__aicore__ inline void Launch(const Args &args) {
    AscendC::TPipe pipe;
    uint32_t core=AscendC::GetBlockIdx();
    const uint32_t tasks=args.batch*args.heads*4;
    if ASCEND_IS_AIC {
        Cube op(pipe);op.Init();
        for(uint32_t task=core;task<tasks;task+=args.blocks) {
            Work work(args,task);CubeWork(op,args,work,core);
        }
        op.End();
    }
    if ASCEND_IS_AIV {
        core/=2;const uint32_t lane=AscendC::GetSubBlockIdx();
        Vector op(pipe);op.Init();
        for(uint32_t task=core;task<tasks;task+=args.blocks) {
            Work work(args,task);op.Run(args,work,core,lane);
        }
    }
}
}
