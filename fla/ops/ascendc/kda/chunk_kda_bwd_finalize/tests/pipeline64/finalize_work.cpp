#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/finalize_work.h"
#include "../../op_kernel/arch22/pipeline64/finalize_pair.h"
using namespace KDA::Arch22::Pipeline64;
__global__ __aicore__ void c64_finalize_work(GM_ADDR workspace,Layout l,FinalizeWorkInputs x,uint32_t dtype,uint32_t cubeScore) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if(cubeScore==2) {
        if(dtype==1)RunFinalizeHeads<half,float,half>(workspace,l,&x,1);
        else RunFinalizeHeads<bfloat16_t,float,bfloat16_t>(workspace,l,&x,1);
    } else if(cubeScore) {
        if(dtype==1)RunProtectedFinalizeWork<half,float,half,true>(workspace,l,x);
        else RunProtectedFinalizeWork<bfloat16_t,float,bfloat16_t,true>(workspace,l,x);
    } else {
        if(dtype==1)RunProtectedFinalizeWork<half,float,half>(workspace,l,x);
        else RunProtectedFinalizeWork<bfloat16_t,float,bfloat16_t>(workspace,l,x);
    }
}
// Pointer table: ws,du,h,dh,vnew,akk,qhat,khat,G,value,beta,dqRaw,ds,
// raw,bias,Alog,rstdq,rstdk,dq,dk,dv,dg,db,betaPartial,exchange,params.
static FinalizeWorkInputs MakeInputs(void **p,uint32_t m,uint32_t k,uint32_t v,
    uint32_t norm,uint32_t gate,uint32_t bm,uint32_t exp2) {
    auto addr=[&](int i){return reinterpret_cast<uint64_t>(p[i]);};
    auto mat=[&](int i,uint32_t pitch){return Matrix{addr(i),pitch,1};};
    FinalizeWorkInputs x{};
    x.products={mat(1,v),mat(2,v),mat(3,v),mat(4,v),mat(5,64)};
    x.direct={mat(6,k),mat(7,k),mat(8,k),mat(11,k),mat(9,v),mat(2,v),mat(3,v),mat(20,v),
              addr(10),addr(23),m,.37f,exp2?.6931471805599453f:1.f};
    x.tail={mat(6,k),mat(7,k),mat(8,k),mat(13,k),mat(10,1),mat(16,1),mat(17,1),
            mat(18,k),mat(19,k),mat(21,k),mat(22,1),addr(14),addr(15),addr(23),addr(24),addr(25),
            m,norm,gate,bm,exp2,-5.f};
    x.ds=mat(12,m);
    return x;
}
extern "C" void run_c64_finalize_work(void **p,uint32_t m,uint32_t k,uint32_t v,
    uint32_t norm,uint32_t gate,uint32_t bm,uint32_t exp2,uint32_t dtype,uint32_t cubeScore,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return;
    auto x=MakeInputs(p,m,k,v,norm,gate,bm,exp2);
    c64_finalize_work<<<1,nullptr,stream>>>((GM_ADDR)p[0],l,x,dtype,cubeScore);
}
struct HeadPair {FinalizeWorkInputs x[2];};
__global__ __aicore__ void c64_finalize_pair(GM_ADDR ws,Layout l,HeadPair pair,uint32_t dtype) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if(dtype==1)RunFinalizeHeads<half,float,half>(ws,l,pair.x,2);
    else RunFinalizeHeads<bfloat16_t,float,bfloat16_t>(ws,l,pair.x,2);
}
extern "C" uint64_t c64_pair_workspace_bytes(uint32_t k,uint32_t v) {
    Layout l{};return MakeLayout(k,v,1,2,2,1,0,l)?l.bytes:0;
}
extern "C" void run_c64_finalize_pair(void **p,uint32_t m,uint32_t k,uint32_t v,
    uint32_t norm,uint32_t gate,uint32_t bm,uint32_t exp2,uint32_t dtype,uint32_t unused,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,2,2,1,0,l))return;
    HeadPair pair{{MakeInputs(p,m,k,v,norm,gate,bm,exp2),MakeInputs(p+26,m,k,v,norm,gate,bm,exp2)}};
    c64_finalize_pair<<<1,nullptr,stream>>>((GM_ADDR)p[0],l,pair,dtype);
}
