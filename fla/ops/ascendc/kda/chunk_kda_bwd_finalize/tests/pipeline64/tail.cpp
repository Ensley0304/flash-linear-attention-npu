#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/tail.h"
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_tail(GM_ADDR workspace,Layout l,TailInputs in,uint32_t dtype) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { TailExchangeCube(in.rows,in.normalize); }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> b;
        pipe.InitBuffer(b,Tail::FLOATS*4);Tail op;op.Init(b.Get<float>());
        if(dtype==1)op.Run<half,float,half>(l,reinterpret_cast<uint64_t>(workspace),in);
        else if(dtype==2)op.Run<bfloat16_t,float,bfloat16_t>(l,reinterpret_cast<uint64_t>(workspace),in);
        else op.Run<float,float,float>(l,reinterpret_cast<uint64_t>(workspace),in);
    }
}
// Pointer table is host memory, consumed only by the host launcher.
extern "C" void run_c64_tail(void **p,uint32_t m,uint32_t k,uint32_t norm,uint32_t gate,
    uint32_t betaMode,uint32_t exp2,uint32_t dtype,void *stream) {
    Layout l{};if(!MakeLayout(k,k,1,1,1,1,0,l))return;
    TailInputs in{{reinterpret_cast<uint64_t>(p[1]),k,1},{reinterpret_cast<uint64_t>(p[2]),k,1},{0,k,1},
        {reinterpret_cast<uint64_t>(p[3]),k,1},{reinterpret_cast<uint64_t>(p[4]),1,1},
        {reinterpret_cast<uint64_t>(p[5]),1,1},{reinterpret_cast<uint64_t>(p[6]),1,1},
        {reinterpret_cast<uint64_t>(p[7]),k,1},{reinterpret_cast<uint64_t>(p[8]),k,1},
        {reinterpret_cast<uint64_t>(p[9]),k,1},{reinterpret_cast<uint64_t>(p[10]),1,1},
        reinterpret_cast<uint64_t>(p[11]),reinterpret_cast<uint64_t>(p[12]),reinterpret_cast<uint64_t>(p[13]),
        reinterpret_cast<uint64_t>(p[14]),reinterpret_cast<uint64_t>(p[15]),m,norm,gate,betaMode,exp2,-5.f};
    c64_tail<<<1,nullptr,stream>>>((GM_ADDR)p[0],l,in,dtype);
}
