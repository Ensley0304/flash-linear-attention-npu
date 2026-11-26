#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/dhu_work.h"
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_dhu_work(GM_ADDR w,Layout l,DhuSequence x,uint32_t dtype) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint32_t block=AscendC::GetBlockIdx();if ASCEND_IS_AIV {block/=2;}
    x.valueStart=block*64;x.valueCount=l.v-x.valueStart<64?l.v-x.valueStart:64;
    if(dtype==1)RunDhuSequence<half>(w+uint64_t(block)*l.groupBytes,l,x);
    else RunDhuSequence<bfloat16_t>(w+uint64_t(block)*l.groupBytes,l,x);
}
extern "C" uint64_t c64_dhu_workspace(uint32_t k,uint32_t v) {
    Layout l{};return MakeLayout(k,v,1,1,1,1,0,l)?l.groupBytes*((v+63)/64):0;
}
extern "C" void run_c64_dhu_work(void **p,uint32_t t,uint32_t k,uint32_t v,uint32_t exp2,uint32_t dtype,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return;
    auto mat=[&](uint32_t i,uint32_t pitch){return Matrix{reinterpret_cast<uint64_t>(p[i]),pitch,1};};
    DhuSequence x{mat(1,k),mat(2,k),mat(3,k),mat(4,k),mat(5,v),mat(6,v),mat(7,v),
        reinterpret_cast<uint64_t>(p[8]),reinterpret_cast<uint64_t>(p[9]),reinterpret_cast<uint64_t>(p[10]),
        uint64_t(k)*v*4,t,0,0,.37f,bool(exp2)};
    c64_dhu_work<<<(v+63)/64,nullptr,stream>>>((GM_ADDR)p[0],l,x,dtype);
}
