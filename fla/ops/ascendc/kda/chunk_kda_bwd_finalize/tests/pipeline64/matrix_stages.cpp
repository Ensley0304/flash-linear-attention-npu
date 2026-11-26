#include "acl/acl.h"
#include "../../op_kernel/arch22/pipeline64/matrix_stages.h"
#include "../../op_kernel/arch22/pipeline64/score_products.h"
#include "../../op_kernel/arch22/pipeline64/score_merge.h"
using namespace KDA::Arch22::Pipeline64;
struct PrepareInputs { uint64_t khat,gate,value,beta;uint32_t exp2,dtype; };
__global__ __aicore__ void c64_matrix_stages(GM_ADDR workspace,Layout l,Inputs inputs,uint32_t rows,PrepareInputs prep) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint64_t packets=0,outputs=0;
    for(uint32_t i=0;i<8;++i) {
        auto p=Describe(static_cast<ProductKind>(i),l,reinterpret_cast<uint64_t>(workspace),rows,inputs);
        outputs+=Tiles(p);packets+=Tiles(p)*((Pad8(p.k)+127)/128);
    }
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs);
        if(prep.khat)cube.RetireRecord();
        for(uint32_t i=0;i<8;++i) {
            auto p=Describe(static_cast<ProductKind>(i),l,reinterpret_cast<uint64_t>(workspace),rows,inputs);
            for(uint32_t j=0;j<Tiles(p);++j) {
                auto t=GetTile(p,j);
                cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1);
            }
            cube.RetireRecord();
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vector;vector.Init(workspace+l.inputs,workspace+l.results,outputs);
        if(prep.khat) {
            if(prep.dtype==1)vector.Prepare<half>(l,reinterpret_cast<uint64_t>(workspace),rows,
                {prep.khat,l.k,1},{prep.gate,l.k,1},{prep.value,l.v,1},prep.beta,prep.exp2);
            else if(prep.dtype==2)vector.Prepare<bfloat16_t>(l,reinterpret_cast<uint64_t>(workspace),rows,
                {prep.khat,l.k,1},{prep.gate,l.k,1},{prep.value,l.v,1},prep.beta,prep.exp2);
            else vector.Prepare<float>(l,reinterpret_cast<uint64_t>(workspace),rows,
                {prep.khat,l.k,1},{prep.gate,l.k,1},{prep.value,l.v,1},prep.beta,prep.exp2);
        }
        for(uint32_t i=0;i<8;++i)
            vector.Run(Describe(static_cast<ProductKind>(i),l,reinterpret_cast<uint64_t>(workspace),rows,inputs));
    }
}
extern "C" uint64_t c64_workspace_bytes(uint32_t k,uint32_t v) {
    Layout l{};return MakeLayout(k,v,1,1,1,1,0,l)?l.bytes:0;
}
extern "C" uint64_t c64_offset(uint32_t k,uint32_t v,uint32_t kind,uint32_t index) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return UINT64_MAX;
    return kind==0?index*l.scoreBytes:kind==1?l.keys+index*l.keyBytes:l.values+index*l.valueBytes;
}
extern "C" void run_c64_matrix_stages(void *workspace,void *du,void *h,void *dh,
    void *vn,void *akk,uint32_t m,uint32_t k,uint32_t v,uint32_t vf,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return;
    Inputs x{{reinterpret_cast<uint64_t>(du),v,1},
        {reinterpret_cast<uint64_t>(h),vf?1U:v,vf?k:1U},
        {reinterpret_cast<uint64_t>(dh),vf?1U:v,vf?k:1U},
        {reinterpret_cast<uint64_t>(vn),v,1},{reinterpret_cast<uint64_t>(akk),64,1}};
    c64_matrix_stages<<<1,nullptr,stream>>>((GM_ADDR)workspace,l,x,m,PrepareInputs{});
}
extern "C" void run_c64_prepared_stages(void *workspace,void *du,void *h,void *dh,
    void *vn,void *akk,void *khat,void *gate,void *value,void *beta,
    uint32_t m,uint32_t k,uint32_t v,uint32_t vf,uint32_t exp2,uint32_t dtype,void *stream) {
    Layout l{};if(!MakeLayout(k,v,1,1,1,1,0,l))return;
    Inputs x{{reinterpret_cast<uint64_t>(du),v,1},
        {reinterpret_cast<uint64_t>(h),vf?1U:v,vf?k:1U},
        {reinterpret_cast<uint64_t>(dh),vf?1U:v,vf?k:1U},
        {reinterpret_cast<uint64_t>(vn),v,1},{reinterpret_cast<uint64_t>(akk),64,1}};
    PrepareInputs p{reinterpret_cast<uint64_t>(khat),reinterpret_cast<uint64_t>(gate),
                    reinterpret_cast<uint64_t>(value),reinterpret_cast<uint64_t>(beta),exp2,dtype};
    c64_matrix_stages<<<1,nullptr,stream>>>((GM_ADDR)workspace,l,x,m,p);
}

__global__ __aicore__ void c64_score_stages(GM_ADDR workspace,Layout l,ScoreInputs x,ScoreMergeInputs merge) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint64_t packets=0,outputs=0;
    for(uint32_t i=0;i<5;++i) {
        auto p=DescribeScore(static_cast<ScoreProduct>(i),l,reinterpret_cast<uint64_t>(workspace),x);
        outputs+=Tiles(p);packets+=Tiles(p)*((Pad8(p.k)+127)/128);
    }
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs);
        for(uint32_t i=0;i<5;++i) {
            auto p=DescribeScore(static_cast<ScoreProduct>(i),l,reinterpret_cast<uint64_t>(workspace),x);
            for(uint32_t j=0;j<Tiles(p);++j) {
                auto t=GetTile(p,j);
                cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1);
            }
            cube.RetireRecord();
        }
        if(merge.dq.address)cube.RetireRecord();
        cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vec;vec.Init(workspace+l.inputs,workspace+l.results,outputs);
        for(uint32_t i=0;i<5;++i)
            vec.Run(DescribeScore(static_cast<ScoreProduct>(i),l,reinterpret_cast<uint64_t>(workspace),x));
        if(merge.dq.address) {
            ScoreMerge tail;tail.Init(vec.Pipe());tail.Run(l,reinterpret_cast<uint64_t>(workspace),merge);
            AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
            AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        }
    }
}
extern "C" void run_c64_score_stages(void *workspace,void *ds,void *dlBeta,
    void *qp,void *kp,void *km,uint32_t m,uint32_t n,uint32_t d,void *stream) {
    Layout l{};if(!MakeLayout(256,256,1,1,1,1,0,l))return;
    ScoreInputs x{{reinterpret_cast<uint64_t>(ds),n,1},{reinterpret_cast<uint64_t>(dlBeta),n,1},
        {reinterpret_cast<uint64_t>(qp),d,1},{reinterpret_cast<uint64_t>(kp),d,1},
        {reinterpret_cast<uint64_t>(km),d,1},m,n,d};
    c64_score_stages<<<1,nullptr,stream>>>((GM_ADDR)workspace,l,x,ScoreMergeInputs{});
}
extern "C" void run_c64_score_merged(void *workspace,void *ds,void *dlBeta,
    void *qp,void *kp,void *km,void *ep,void *em,void *dq,void *dk,void *dg,
    uint32_t m,uint32_t n,uint32_t d,uint32_t queryStart,float base,void *stream) {
    Layout l{};if(!MakeLayout(256,256,1,1,1,1,0,l))return;
    ScoreInputs x{{reinterpret_cast<uint64_t>(ds),n,1},{reinterpret_cast<uint64_t>(dlBeta),n,1},
        {reinterpret_cast<uint64_t>(qp),d,1},{reinterpret_cast<uint64_t>(kp),d,1},
        {reinterpret_cast<uint64_t>(km),d,1},m,n,d};
    ScoreMergeInputs s{x.qPlus,x.kPlus,x.kMinus,
        {reinterpret_cast<uint64_t>(ep),d,1},{reinterpret_cast<uint64_t>(em),d,1},
        {reinterpret_cast<uint64_t>(dq),d,1},{reinterpret_cast<uint64_t>(dk),d,1},
        {reinterpret_cast<uint64_t>(dg),d,1},queryStart,0,0,m,n,d,base};
    c64_score_stages<<<1,nullptr,stream>>>((GM_ADDR)workspace,l,x,s);
}
