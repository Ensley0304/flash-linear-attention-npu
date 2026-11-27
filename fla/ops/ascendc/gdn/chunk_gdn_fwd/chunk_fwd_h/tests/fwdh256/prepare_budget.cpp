// Read-only budget probe of the existing producer. No producer implementation changes.
#include "acl/acl.h"
#include "../../../../../kda/chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_general.h"
using namespace KdaPrepare::Arch22;
struct PrepareBudgetArgs {GeneralPrepareBuffers buffers;uint32_t tokens,heads,blocks;};
__global__ __aicore__ void fwdh256_existing_prepare_budget(PrepareBudgetArgs a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;GeneralPrepare<bfloat16_t,float,bfloat16_t> op;op.Init(pipe);
    uint32_t chunks=(a.tokens+63)/64;
    for(uint32_t work=AscendC::GetBlockIdx();work<chunks*a.heads;work+=a.blocks) {
        uint32_t head=work%a.heads,begin=(work/a.heads)*64;
        GeneralPrepareChunk c{};c.rows=a.tokens-begin<64?a.tokens-begin:64;c.keyDim=c.valueDim=256;
        c.keyOffset=c.valueOffset=(uint64_t(begin)*a.heads+head)*256;c.betaOffset=uint64_t(begin)*a.heads+head;
        c.keyStride=c.valueStride=a.heads*256;c.betaStride=a.heads;
        c.cacheKeyOffset=c.cacheValueOffset=(uint64_t(head)*a.tokens+begin)*256;
        c.cacheScalarOffset=uint64_t(head)*a.tokens+begin;c.scoreOffset=c.cacheScalarOffset*64;
        c.scale=.37f;c.epsilon=1e-6f;c.lowerBound=-5.f;c.normalize=false;c.useExp2=false;c.gateMode=2;c.betaMode=0;
        auto buffers=a.buffers;buffers.aLog+=head*4;buffers.bias+=head*256*4;op.Process(buffers,c);
    }
}
extern "C" void run_prepare_budget(void **ptr,uint32_t tokens,uint32_t heads,uint32_t blocks,void *stream) {
    PrepareBudgetArgs a{};
    a.buffers={(GM_ADDR)ptr[0],(GM_ADDR)ptr[1],(GM_ADDR)ptr[2],(GM_ADDR)ptr[3],(GM_ADDR)ptr[4],
        (GM_ADDR)ptr[5],(GM_ADDR)ptr[6],(GM_ADDR)ptr[7],(GM_ADDR)ptr[8],(GM_ADDR)ptr[9],(GM_ADDR)ptr[10],
        (GM_ADDR)ptr[11],(GM_ADDR)ptr[12],(GM_ADDR)ptr[13],(GM_ADDR)ptr[14],(GM_ADDR)ptr[15],
        (GM_ADDR)ptr[16],(GM_ADDR)ptr[17],(GM_ADDR)ptr[18],(GM_ADDR)ptr[19]};
    a.tokens=tokens;a.heads=heads;a.blocks=blocks;
    fwdh256_existing_prepare_budget<<<blocks,nullptr,stream>>>(a);
}
