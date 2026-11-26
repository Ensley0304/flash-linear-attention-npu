#include "acl/acl.h"
#include "probe_prepare.h"
#include "probe_state.h"
#include "probe_bp.h"
#include "probe_dhu.h"
#include "probe_bf.h"
using namespace AscendC;
using namespace KdaPrepare::Arch22;
using namespace GDN::Arch22;
using namespace KDA::Arch22;
struct OwnerPointers { GM_ADDR p[44]; };

template<class T>
__global__ __aicore__ void owner_probe(OwnerPointers a,uint32_t tokens,uint32_t K,uint32_t V,int mode){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TQue<QuePosition::VECIN,1> in;TQue<QuePosition::VECOUT,1> out;TBuf<TPosition::VECCALC> arena;
    pipe.InitBuffer(in,1,1024);pipe.InitBuffer(out,1,1024);pipe.InitBuffer(arena,112640);
    auto mem=arena.Get<float>();auto p=a.p;
    GeneralPrepareBuffers b{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19]};
    if(mode==0 || mode==2){
        GeneralPrepare<T,float,float> prep;prep.Init(pipe,mem,in,out);
        for(uint32_t start=0;start<tokens;start+=32){
            const uint32_t M=tokens-start<32?tokens-start:32;
            GeneralPrepareChunk c{M,K,V,uint64_t(start)*K,uint64_t(start)*V,start,K,V,1,uint64_t(start)*K,uint64_t(start)*V,start,uint64_t(start)*32,.37F,1e-6F,-2.0F,false,true,2,0};
            if(mode==0)prep.Process(b,c);else prep.Recompute(b,c);
        }
        PipeBarrier<PIPE_ALL>();
        GeneralFwdH<float> state;state.Init(pipe,mem,in,out);
        GeneralFwdHBuffers sb{p[10],p[11],p[13],p[7],p[20],p[21],p[22],p[23]};
        for(uint32_t vb=0;vb<V;vb+=32){
            GeneralFwdHSequence c{tokens,K,V,vb,V-vb<32?V-vb:32,0,0,0,0,uint64_t(K)*V,true,false};
            state.Process(sb,c);
        }
    }
    if(mode==0)return;
    GeneralBwdPrepare<T> bp;bp.Init(pipe,mem,in,out);
    GeneralBwdPrepareBuffers bb{p[8],p[22],p[24],p[21],p[25],p[26],p[27]};
    for(uint32_t start=0,n=0;start<tokens;start+=32,++n){
        GeneralBwdPrepareChunk c{tokens-start<32?tokens-start:32,K,V,uint64_t(start)*32,uint64_t(start)*V,uint64_t(start)*V,V,uint64_t(n)*K*V,uint64_t(start)*K,.37F,false};bp.Process(bb,c);
    }
    GeneralDhu<T> scan;scan.Init(pipe,mem,in,out);
    GeneralDhuBuffers db{p[12],p[13],p[10],p[7],p[24],p[26],p[28],p[29],p[30],p[31]};
    for(uint32_t vb=0;vb<V;vb+=32){
        GeneralDhuSequence c{tokens,K,V,vb,V-vb<32?V-vb:32,0,0,0,V,0,0,uint64_t(K)*V,.37F,true,false};scan.Process(db,c);
    }
    GeneralBwdFinalize<T,float,float> fin;fin.Init(pipe,mem,in,out);
    for(uint32_t start=0,n=0;start<tokens;start+=32,++n){
        GeneralBwdFinalizeBuffers fb{p[15],p[16],p[2],p[7],p[19],p[9],p[22],p[21],p[29],p[30],p[25],p[27],p[17],p[18],p[3],p[5],p[6],p[32],p[33],p[34],p[35],p[36],p[37],p[38],p[39],p[40],p[41],p[42]+n*4,p[43]+n*K*4};
        GeneralBwdFinalizeChunk c{tokens-start<32?tokens-start:32,K,V,uint64_t(start)*K,uint64_t(start)*V,start,uint64_t(start)*32,uint64_t(n)*K*V,uint64_t(start)*K,uint64_t(start)*V,start,K,V,1,.37F,-2.0F,false,true,false,2,0};fin.Process(fb,c);
    }
    fin.ReduceParameters((tokens+31)/32,1,0,K,p[42],p[43],p[42],p[43]);
    PipeBarrier<PIPE_ALL>();
}
extern "C" void run_owner_probe(void** pointers,uint32_t T,uint32_t K,uint32_t V,int mode,int bf16,void* stream){
    OwnerPointers a;for(int i=0;i<44;++i)a.p[i]=(uint8_t*)pointers[i];
    if(bf16)owner_probe<bfloat16_t><<<1,nullptr,stream>>>(a,T,K,V,mode);
    else owner_probe<half><<<1,nullptr,stream>>>(a,T,K,V,mode);
}
