// Development gate: same-kernel private GM write/read and owner reuse, not production KDA.
#include "acl/acl.h"
#include "kernel_operator.h"
using namespace AscendC;

class RoundTrip {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> in;
    TQue<QuePosition::VECOUT, 1> out;
    GlobalTensor<float> gm, result;
    uint32_t width, steps;
    __aicore__ inline void Fence() {
        auto id = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
    }
    __aicore__ inline void Seed(uint64_t offset, float value) {
        auto x = out.AllocTensor<float>();
        Duplicate(x,value,width);
        out.EnQue(x); x=out.DeQue<float>();
        DataCopyPad(gm[offset],x,DataCopyExtParams{1,width*4,0,0,0});
        out.FreeTensor(x); Fence();
    }
    __aicore__ inline void Step(uint64_t src,uint64_t dst,float add) {
        auto x=in.AllocTensor<float>();
        DataCopyPad(x,gm[src],DataCopyExtParams{1,width*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        in.EnQue(x); x=in.DeQue<float>();
        auto y=out.AllocTensor<float>();
        Adds(y,x,add,width);
        out.EnQue(y); y=out.DeQue<float>();
        DataCopyPad(gm[dst],y,DataCopyExtParams{1,width*4,0,0,0});
        in.FreeTensor(x);out.FreeTensor(y);Fence();
    }
public:
    __aicore__ inline void Run(GM_ADDR scratch,GM_ADDR output,uint32_t heads,uint32_t count,uint32_t loops,bool noCache){
        width=count;steps=loops;
        gm.SetGlobalBuffer((__gm__ float*)scratch);
        result.SetGlobalBuffer((__gm__ float*)output);
        if(noCache)gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe.InitBuffer(in,1,1024);pipe.InitBuffer(out,1,1024);
        // Each block reuses its scratch for multiple logically different heads.
        const uint64_t base=GetBlockIdx()*(steps+2)*width;
        for(uint32_t h=GetBlockIdx();h<heads;h+=GetBlockNum()){
            Seed(base, float(h*100+1));Seed(base+width,-777.0F);
            uint32_t slot=0;
            for(uint32_t c=0;c<steps;++c){
                Step(base+slot*width,base+(1-slot)*width,float(c+1));slot=1-slot;
                Step(base+slot*width,base+slot*width,1.0F); // same-address read/modify/write
                Step(base+slot*width,base+(c+2)*width,0.0F); // tape forward write
            }
            // Tape reverse reads repeatedly overwrite the same state slot.
            for(uint32_t c=steps;c>0;--c)Step(base+(c+1)*width,base,0.0F);
            auto x=in.AllocTensor<float>();
            DataCopyPad(x,gm[base],DataCopyExtParams{1,width*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            in.EnQue(x);x=in.DeQue<float>();
            auto y=out.AllocTensor<float>();Adds(y,x,0.0F,width);
            out.EnQue(y);y=out.DeQue<float>();
            DataCopyPad(result[h*width],y,DataCopyExtParams{1,width*4,0,0,0});
            in.FreeTensor(x);out.FreeTensor(y);Fence();
        }
        PipeBarrier<PIPE_ALL>();
    }
};

__global__ __aicore__ void gm_roundtrip(GM_ADDR scratch,GM_ADDR output,uint32_t heads,uint32_t width,uint32_t loops,bool noCache){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RoundTrip op;op.Run(scratch,output,heads,width,loops,noCache);
}
extern "C" void run_gm_roundtrip(void* scratch,void* output,uint32_t heads,uint32_t width,uint32_t loops,uint32_t blocks,int noCache,void* stream){
    gm_roundtrip<<<blocks,nullptr,stream>>>((uint8_t*)scratch,(uint8_t*)output,heads,width,loops,noCache!=0);
}
