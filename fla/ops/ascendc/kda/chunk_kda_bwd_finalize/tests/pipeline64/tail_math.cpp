#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/tail_math.h"
using namespace AscendC;
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_tail_math(GM_ADDR input,GM_ADDR output,uint32_t n,
    uint32_t gateMode,uint32_t betaMode,uint32_t norm,float a,float lower,float projection,float rstd) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,16*256*4);
    auto b=buffer.Get<float>();
    GlobalTensor<float> in,out;in.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    for(uint32_t i=0;i<6;++i)DataCopy(b[i*256],in[i*n],n);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    BetaDerivative(b[6*256],b[3*256],b[2*256],b[10*256],betaMode,n);
    PipeBarrier<PIPE_V>();
    NormDerivative(b[7*256],b[5*256],b[4*256],b[10*256],projection,rstd,norm,n);
    PipeBarrier<PIPE_V>();
    GateDerivative(b[8*256],b[9*256],b[256],b,b[10*256],b[11*256],b[12*256],a,lower,gateMode,n);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for(uint32_t i=0;i<4;++i)DataCopy(out[i*n],b[(6+i)*256],n);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
extern "C" void run_c64_tail_math(void *input,void *output,uint32_t n,uint32_t gateMode,
    uint32_t betaMode,uint32_t norm,float a,float lower,float projection,float rstd,void *stream) {
    c64_tail_math<<<1,nullptr,stream>>>((GM_ADDR)input,(GM_ADDR)output,n,gateMode,betaMode,norm,a,lower,projection,rstd);
}
