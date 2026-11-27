#include "acl/acl.h"
#include "../../op_kernel/arch22/fwdh256/cube.h"
using namespace GDN::Arch22::FwdH256;
__global__ __aicore__ void fwdh256_cube_probe(GM_ADDR a,GM_ADDR b,GM_ADDR c,
    uint32_t m,uint32_t k,uint32_t as,uint32_t bs,uint32_t ta,uint32_t repeat) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;Cube op(pipe);op.Init();
    for(uint32_t i=0;i<repeat;++i)op.Multiply(a,b,c+i*64*64*4,as,bs,64,m,k,ta);
    op.End();
}
extern "C" void run_cube(void *a,void *b,void *c,uint32_t m,uint32_t k,
    uint32_t as,uint32_t bs,uint32_t ta,uint32_t repeat,void *stream) {
    fwdh256_cube_probe<<<1,nullptr,stream>>>((GM_ADDR)a,(GM_ADDR)b,(GM_ADDR)c,m,k,as,bs,ta,repeat);
}
