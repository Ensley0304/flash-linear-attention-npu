// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "cube.h"
#include "products.h"
namespace KDA::Arch22::Pipeline64 {
__aicore__ inline uint32_t Pad16(uint32_t x) { return (x+15)&~15U; }
__aicore__ inline uint32_t Pad8(uint32_t x) { return (x+7)&~7U; }
struct Tile { uint32_t row,col,m,n,mp,np; };
__aicore__ inline uint32_t Tiles(const Product &p) { return ((p.m+63)/64)*((p.n+63)/64); }
__aicore__ inline Tile GetTile(const Product &p,uint32_t i) {
    const uint32_t cols=(p.n+63)/64, row=i/cols*64,col=i%cols*64;
    const uint32_t m=p.m-row<64?p.m-row:64,n=p.n-col<64?p.n-col:64;
    return {row,col,m,n,Pad16(m),Pad16(n)};
}

// F1--F4 component. Both AIVs process every output tile, each its contiguous
// column half. Product boundaries retire their GM writes before the next
// dependent product. Cross-work scheduling is layered above this component.
class MatrixVector {
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_, result_, previous_;
    AscendC::LocalTensor<float> prepare_;
    AscendC::GlobalTensor<float> inputs_,results_;
    uint64_t packet_=0,output_=0,outputs_=0;
    uint32_t lane_=0;
    bool dynamic_=false;
    struct CachedOperand {
        Matrix view;uint32_t row,col,rows,cols,pr,pc;bool transpose,valid;
    } cached_[4]{};
    uint32_t cacheVictim_=0;
    __aicore__ inline void LoadCached(AscendC::LocalTensor<float> dst,Matrix view,
        uint32_t row,uint32_t col,uint32_t rows,uint32_t cols,uint32_t pr,uint32_t pc,bool transpose) {
        using namespace AscendC;
        const uint32_t size=pr*pc/2;
        if(size>2048) {Load(dst,view,row,col,rows,cols,pr,pc,transpose);return;}
        for(uint32_t i=0;i<4;++i) {
            const auto &c=cached_[i];
            if(c.valid && c.view.address==view.address && c.view.rowStride==view.rowStride &&
               c.view.colStride==view.colStride && c.row==row && c.col==col &&
               c.rows==rows && c.cols==cols && c.pr==pr && c.pc==pc && c.transpose==transpose) {
                Adds(dst,prepare_[i*2048],0.f,size);PipeBarrier<PIPE_V>();return;
            }
        }
        Load(dst,view,row,col,rows,cols,pr,pc,transpose);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t slot=cacheVictim_++%4;
        Adds(prepare_[slot*2048],dst,0.f,size);PipeBarrier<PIPE_V>();
        cached_[slot]={view,row,col,rows,cols,pr,pc,transpose,true};
    }
    __aicore__ inline void Load(AscendC::LocalTensor<float> dst,Matrix view,
            uint32_t row,uint32_t col,uint32_t rows,uint32_t cols,
            uint32_t pr,uint32_t pc,bool transpose) {
        using namespace AscendC;
        if(transpose) {
            uint32_t t=row;row=col;col=t;t=rows;rows=cols;cols=t;t=pr;pr=pc;pc=t;
        }
        uint32_t half=pr/2,begin=lane_*half;
        rows=rows>begin?(rows-begin<half?rows-begin:half):0;
        const uint32_t stride=transpose?view.colStride:view.rowStride;
        Duplicate(dst,0.0f,half*pc);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if(rows && cols) {
            GlobalTensor<float> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.address));
            src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            DataCopyExtParams cp{uint16_t(rows),cols*4,(stride-cols)*4,(pc-Pad8(cols))/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(cols)-cols),0.0f};
            DataCopyPad(dst,src[uint64_t(row+begin)*stride+col],cp,pad);
        }
    }
public:
    __aicore__ inline AscendC::TPipe &Pipe() { return pipe_; }
    // F6 factors are published before matrix transport can reuse this arena.
    __aicore__ inline AscendC::LocalTensor<float> OperandScratch() {return input_.Get<float>();}
    __aicore__ inline void Init(GM_ADDR input,GM_ADDR result,uint64_t outputs,bool dynamic=false) {
        pipe_.InitBuffer(input_,64*1024+1024);pipe_.InitBuffer(result_,16*1024);
        pipe_.InitBuffer(previous_,16*1024);
        prepare_=input_.Get<float>()[8192];
        inputs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        results_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result));
        inputs_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        results_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        lane_=AscendC::GetSubBlockIdx();outputs_=outputs;
        dynamic_=dynamic;
    }
    // Saved FP32 khat/G and effective beta are already activated. F0 must not
    // apply norm/sigmoid/cumsum a second time. v is promoted by the input load.
    template<class InputT>
    __aicore__ inline void Prepare(const Layout &l,uint64_t record,uint32_t rows,
                                  Matrix khat,Matrix gate,Matrix value,uint64_t beta,bool exp2) {
        using namespace AscendC;
        auto buf=prepare_;
        auto x=buf, g=buf[2048], transfer=buf[4096], effectiveBeta=buf[8192];
        GlobalTensor<float> bg;bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        DataCopyExtParams bc{1,rows*4,0,0,0};
        DataCopyPadExtParams<float> bp{true,0,uint8_t(Pad8(rows)-rows),0.0f};
        DataCopyPad(effectiveBeta,bg,bc,bp);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto betaBroadcast=buf[6144];
        Brcb(betaBroadcast,effectiveBeta,uint8_t((rows+7)/8),{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t which=0;which<2;++which) {
            const uint32_t width=which?l.v:l.k,stride=which?l.vp:l.kp;
            GlobalTensor<float> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                record+(which?l.Offset(Value::Rv):l.Offset(Key::Rk))));
            for(uint32_t tile=0;tile<width;tile+=64) {
                const uint32_t begin=tile+lane_*32;
                const uint32_t valid=width>begin?(width-begin<32?width-begin:32):0;
                if(!valid)continue;
                Duplicate(x,0.0f,rows*32);Duplicate(g,0.0f,rows*32);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyPadExtParams<float> fp{true,0,uint8_t(Pad8(valid)-valid),0.0f};
                if(!which) {
                    GlobalTensor<float> kg,gg;
                    kg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(khat.address));
                    gg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gate.address));
                    DataCopyExtParams kc{uint16_t(rows),valid*4,(khat.rowStride-valid)*4,(32-Pad8(valid))/8,0};
                    DataCopyPad(x,kg[begin],kc,fp);
                    kc.srcStride=(gate.rowStride-valid)*4;
                    DataCopyPad(g,gg[begin],kc,fp);
                } else {
                    GlobalTensor<InputT> vg;vg.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(value.address));
                    auto raw=transfer.ReinterpretCast<InputT>();
                    constexpr uint32_t block=32/sizeof(InputT);
                    const uint32_t aligned=(valid+block-1)/block*block;
                    DataCopyExtParams vc{uint16_t(rows),uint32_t(valid*sizeof(InputT)),
                        uint32_t((value.rowStride-valid)*sizeof(InputT)),uint32_t((32-aligned)/block),0};
                    DataCopyPadExtParams<InputT> vp{true,0,uint8_t(aligned-valid),InputT(0)};
                    DataCopyPad(raw,vg[begin],vc,vp);
                }
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if(!which) {
                    if(exp2){Muls(g,g,0.6931471805599453f,rows*32);PipeBarrier<PIPE_V>();}
                    Exp(g,g,rows*32);PipeBarrier<PIPE_V>();
                    Mul(x,x,g,rows*32);
                } else {
                    if constexpr(sizeof(InputT)==4)Adds(x,transfer,0.0f,rows*32);
                    else Cast(x,transfer.ReinterpretCast<InputT>(),RoundMode::CAST_NONE,rows*32);
                }
                PipeBarrier<PIPE_V>();
                Mul(x,x,betaBroadcast,uint64_t(32),uint8_t(rows),{1,1,0,4,4,1});
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams wr{uint16_t(rows),valid*4,(32-Pad8(valid))/8,(stride-valid)*4,0};
                DataCopyPad(dst[begin],x,wr);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
    __aicore__ inline void Consume(const Product &p,Tile t,uint64_t index) {
        using namespace AscendC;
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(ResultReady(index));
        const uint32_t half=t.np/2,start=lane_*half;
        auto x=result_.Get<float>();
        DataCopyExtParams cp{uint16_t(t.mp),half*4,(t.np-half)*4,0,0};
        DataCopyPadExtParams<float> pad{false,0,0,0.0f};
        DataCopyPad(x,results_[(index%2)*RESULT_BANK_BYTES/4+start],cp,pad);
        if(dynamic_ || index+2<outputs_)CrossCoreSetFlag<0x2,PIPE_MTE2>(ResultFree(index));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t valid=t.n>start?(t.n-start<half?t.n-start:half):0;
        if(p.epilogue==Epilogue::Negate || p.epilogue==Epilogue::NegateStrictLower)
            Muls(x,x,-1.0f,t.mp*half);
        if(p.epilogue==Epilogue::ScaleLower)Muls(x,x,p.scale,t.mp*half);
        PipeBarrier<PIPE_V>();
        GlobalTensor<float> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.out.address));
        dst.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if(p.epilogue==Epilogue::Add && valid) {
            GlobalTensor<float> priorSource;
            priorSource.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.addSource?p.addSource:p.out.address));
            priorSource.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            auto old=previous_.Get<float>();
            DataCopyExtParams rd{uint16_t(t.m),valid*4,(p.out.rowStride-valid)*4,(half-Pad8(valid))/8,0};
            DataCopyPadExtParams<float> tail{true,0,uint8_t(Pad8(valid)-valid),0.0f};
            DataCopyPad(old,priorSource[uint64_t(t.row)*p.out.rowStride+t.col+start],rd,tail);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(x,x,old,t.m*half);PipeBarrier<PIPE_V>();
        }
        if(p.epilogue==Epilogue::NegateStrictLower || p.epilogue==Epilogue::ScaleLower) {
            for(uint32_t row=0;row<t.m;++row) {
                const uint32_t globalRow=t.row+row+(p.epilogue==Epilogue::ScaleLower?1:0),globalCol=t.col+start;
                const uint32_t lower=globalRow>globalCol?(globalRow-globalCol<half?globalRow-globalCol:half):0;
                if(lower<half) {
                    uint64_t mask[2]={((1ULL<<half)-1)^((1ULL<<lower)-1),0};
                    Duplicate(x[row*half],0.0f,mask,1,1,8);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if(valid) {
            DataCopyExtParams wr{uint16_t(t.m),valid*4,(half-Pad8(valid))/8,(p.out.rowStride-valid)*4,0};
            DataCopyPad(dst[uint64_t(t.row)*p.out.rowStride+t.col+start],x,wr);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    // Independent products share the two outstanding result slots. Drain at
    // the stage dependency boundary, not after every matrix multiplication.
    __aicore__ inline void RunStage(const Product *products,uint32_t count,bool direct=false) {
        using namespace AscendC;
        Product pending[2];Tile pendingTile[2];
        for(uint32_t i=0;i<4;++i)cached_[i].valid=false;
        cacheVictim_=0; // F0 arena is now reused; never cache across producers.
        const uint64_t first=output_;
        for(uint32_t product=0;product<count;++product) {
        const Product &p=products[product];const uint32_t nt=Tiles(p);
        for(uint32_t i=0;i<nt;++i) {
            if(output_>=first+2)Consume(pending[output_%2],pendingTile[output_%2],output_-2);
            const auto t=GetTile(p,i);
            pending[output_%2]=p;pendingTile[output_%2]=t;
            for(uint32_t begin=0;!(direct && DirectOperand(p)) && begin<p.k;begin+=128) {
                const uint32_t k=p.k-begin<128?p.k-begin:128,kp=Pad8(k);
                if(packet_>=2)CrossCoreWaitFlag<0x2,PIPE_MTE3>(InputFree(packet_));
                auto x=input_.Get<float>();
                if(count>1) {
                    LoadCached(x,p.a,t.row,begin,t.m,k,t.mp,kp,p.a.rowStride==1);
                    LoadCached(x[4096],p.b,begin,t.col,k,t.n,kp,t.np,p.b.rowStride==1);
                } else {
                    Load(x,p.a,t.row,begin,t.m,k,t.mp,kp,p.a.rowStride==1);
                    Load(x[4096],p.b,begin,t.col,k,t.n,kp,t.np,p.b.rowStride==1);
                }
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t base=(packet_%2)*INPUT_BANK_BYTES/4;
                DataCopy(inputs_[base+lane_*t.mp*kp/2],x,t.mp*kp/2);
                DataCopy(inputs_[base+8192+lane_*kp*t.np/2],x[4096],kp*t.np/2);
                CrossCoreSetFlag<0x2,PIPE_MTE3>(InputReady(packet_));
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ++packet_;
            }
            ++output_;
        }
        }
        for(uint64_t i=output_>first+2?output_-2:first;i<output_;++i)
            Consume(pending[i%2],pendingTile[i%2],i);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
    __aicore__ inline void Run(const Product &p) { RunStage(&p,1); }
    __aicore__ inline void End() {
        if(dynamic_)for(uint64_t i=packet_>2?packet_-2:0;i<packet_;++i)
            AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE3>(InputFree(i));
    }
};
} // namespace KDA::Arch22::Pipeline64
