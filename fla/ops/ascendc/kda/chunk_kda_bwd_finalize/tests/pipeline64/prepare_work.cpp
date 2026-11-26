#include "acl/acl.h"
#include "../../../chunk_kda_bwd_prepare/op_kernel/chunk_kda_bwd_prepare_struct.h"
#include "../../../chunk_kda_bwd_prepare/op_kernel/arch22/chunk_kda_bwd_prepare_general.h"
#include "../../../chunk_kda_bwd_prepare/op_kernel/arch22/chunk_kda_bwd_prepare_pipeline.h"
using namespace KDA::Arch22::Pipeline64;
struct PrepareWork {Matrix aqk,vn,grad,h,ds,du,dq;uint32_t rows,dtype;};
__global__ __aicore__ void c64_prepare_work(GM_ADDR w,Layout l,PrepareWork x) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if(x.dtype==1)KDA::Arch22::PreparePipeline::RunWork<half>(w,l,x.rows,x.aqk,x.vn,x.grad,x.h,x.ds,x.du,x.dq,.37f);
    else KDA::Arch22::PreparePipeline::RunWork<bfloat16_t>(w,l,x.rows,x.aqk,x.vn,x.grad,x.h,x.ds,x.du,x.dq,.37f);
}
extern "C" void run_c64_prepare_work(void **p,uint32_t m,uint32_t k,uint32_t v,uint32_t vf,uint32_t dtype,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return;
    auto mat=[&](uint32_t i,uint32_t pitch){return Matrix{reinterpret_cast<uint64_t>(p[i]),pitch,1};};
    auto h=mat(4,v);if(vf){h.rowStride=1;h.colStride=k;}
    PrepareWork x{mat(1,64),mat(2,v),mat(3,v),h,mat(5,64),mat(6,v),mat(7,k),m,dtype};
    c64_prepare_work<<<1,nullptr,stream>>>((GM_ADDR)p[0],l,x);
}
