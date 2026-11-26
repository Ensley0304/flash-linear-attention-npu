#include "acl/acl.h"
#include "fused32/owner.h"
using namespace KDA::Arch22::Fused32;

template<class T>
__global__ __aicore__ void bounded_owner(Buffers b,Config c,Workspace w,int mode){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Owner<T,float,float> owner;owner.Init(b,c,w);
    if(mode==0)owner.template Process<false>();else owner.template Process<true>();
}
static Config config(uint32_t t,uint32_t k,uint32_t v,int mode){
    return {1,1,t,k,v,1,(t+31)/32,0,1,0,0,0,2,1,uint32_t(mode==2),.37F,1e-6F,-2.F};
}
extern "C" uint64_t owner_workspace_size(uint32_t t,uint32_t k,uint32_t v){
    return MakeWorkspace(config(t,k,v,2),true).bytes;
}
extern "C" void run_owner_probe(void** pointers,uint32_t t,uint32_t k,uint32_t v,int mode,int bf16,void* stream){
    Address p[47];for(int i=0;i<47;++i)p[i]=static_cast<uint8_t*>(pointers[i]);
    Buffers b{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[20],p[28],p[24],
        p[7],p[8],p[9],p[10],p[12],p[13],p[22],p[21],p[15],p[16],p[17],p[18],p[19],
        p[44],p[23],p[37],p[38],p[39],p[40],p[41],p[31],p[42],p[43],p[45],p[46]};
    auto c=config(t,k,v,mode);auto w=MakeWorkspace(c,mode!=0);
    if(bf16)bounded_owner<bfloat16_t><<<1,nullptr,stream>>>(b,c,w,mode);
    else bounded_owner<half><<<1,nullptr,stream>>>(b,c,w,mode);
}
extern "C" uint64_t bounded_workspace_size(const Config* c,int backward){return MakeWorkspace(*c,backward!=0).bytes;}
extern "C" void run_bounded(void** pointers,const Config* cfg,int backward,int bf16,uint32_t blocks,void* stream){
    Address p[47];for(int i=0;i<47;++i)p[i]=static_cast<uint8_t*>(pointers[i]);
    Buffers b{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[20],p[28],p[24],
        p[7],p[8],p[9],p[10],p[12],p[13],p[22],p[21],p[15],p[16],p[17],p[18],p[19],
        p[44],p[23],p[37],p[38],p[39],p[40],p[41],p[31],p[42],p[43],p[45],p[46]};
    auto c=*cfg;auto w=MakeWorkspace(c,backward!=0);
    if(bf16)bounded_owner<bfloat16_t><<<blocks,nullptr,stream>>>(b,c,w,backward);
    else bounded_owner<half><<<blocks,nullptr,stream>>>(b,c,w,backward);
}
