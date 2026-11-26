// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "tail_math.h"
#include "matrix_stages.h"
namespace KDA::Arch22::Pipeline64 {
struct TailInputs {
    Matrix qhat,khat,g,raw,beta,rstdQ,rstdK,dq,dk,dg,dbeta;
    uint64_t bias,aLog,betaPartial,exchange,parameterPartial;
    uint32_t rows,normalize,gateMode,betaMode,useExp2;
    float lowerBound;
};
// The owning AIC services one exchange per normalized row, plus one final
// parameter exchange. No consumer reads the other AIV's partial before ACK.
__aicore__ inline void TailExchangeCube(uint32_t rows,bool normalize) {
    for(uint32_t i=0;i<(normalize?rows:0)+1;++i) {
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(10);
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE2>(11);
    }
}
class Tail {
    using F=AscendC::LocalTensor<float>;
    F s_,transfer_,small_;
    uint32_t lane_;
    template<class T>
    __aicore__ inline void Read(F dst,uint64_t address,uint64_t offset,uint32_t n) {
        using namespace AscendC;
        GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        auto raw=transfer_.ReinterpretCast<T>();
        DataCopyExtParams cp{1,uint32_t(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,0,T(0)};
        DataCopyPad(raw,gm[offset],cp,pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(sizeof(T)==4)Adds(dst,transfer_,0.0f,n);
        else Cast(dst,raw,RoundMode::CAST_NONE,n);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    template<class T>
    __aicore__ inline void Write(F src,uint64_t address,uint64_t offset,uint32_t n) {
        using namespace AscendC;
        auto raw=transfer_.ReinterpretCast<T>();
        if constexpr(sizeof(T)==4)Adds(transfer_,src,0.0f,n);
        else Cast(raw,src,RoundMode::CAST_RINT,n);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
        DataCopyExtParams cp{1,uint32_t(n*sizeof(T)),0,0,0};DataCopyPad(gm[offset],raw,cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void ScalarReady() {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void Exchange(uint64_t address,float first,float second) {
        using namespace AscendC;
        Duplicate(small_[32],0.0f,8);ScalarReady();
        small_.SetValue(32,first);small_.SetValue(33,second);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopy(gm[lane_*8],small_[32],8);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(10);CrossCoreWaitFlag<0x2,PIPE_MTE2>(11);
        DataCopy(small_[48],gm,16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
public:
    static constexpr uint32_t FLOATS=44*256;
    __aicore__ inline void Init(F scratch) {
        s_=scratch;transfer_=s_[15*256];small_=s_[16*256];lane_=AscendC::GetSubBlockIdx();
    }
    template<class T>
    __aicore__ inline void ReadBlock(F dst,uint64_t address,uint32_t stride,
                                     uint32_t rows,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        constexpr uint32_t block=32/sizeof(T);
        const uint32_t aligned=(n+block-1)/block*block;
        // DMA fills the complete pitch when its block padding matches it.
        if(aligned!=pitch)Duplicate(transfer_,0.f,1024);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        auto raw=transfer_.ReinterpretCast<T>();
        DataCopyExtParams cp{uint16_t(rows),uint32_t(n*sizeof(T)),uint32_t((stride-n)*sizeof(T)),
            uint32_t((pitch-aligned)/block),0};
        DataCopyPadExtParams<T> pad{true,0,uint8_t(aligned-n),T(0)};
        DataCopyPad(raw,gm,cp,pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(sizeof(T)==4)Adds(dst,transfer_,0.f,rows*pitch);
        else Cast(dst,raw,RoundMode::CAST_NONE,rows*pitch);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    template<class T>
    __aicore__ inline void WriteBlock(F src,uint64_t address,uint32_t stride,
                                      uint32_t rows,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        constexpr uint32_t block=32/sizeof(T);
        const uint32_t aligned=(n+block-1)/block*block;
        auto raw=transfer_.ReinterpretCast<T>();
        if constexpr(sizeof(T)==4)Adds(transfer_,src,0.f,rows*pitch);
        else Cast(raw,src,RoundMode::CAST_RINT,rows*pitch);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),uint32_t(n*sizeof(T)),uint32_t((pitch-aligned)/block),
            uint32_t((stride-n)*sizeof(T)),0};
        DataCopyPad(gm,raw,cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    // Norm-off stage: eight rows share each DMA/cast/gate operation. The
    // recurrence itself retains the original reverse-token FP32 add order.
    template<class InputT,class GateT,class BetaT>
    __aicore__ inline void RunBlocked(const Layout &l,uint64_t record,const TailInputs &in) {
        using namespace AscendC;
        const uint32_t start=lane_*(l.kp/2),n=l.k>start?(l.k-start<l.kp/2?l.k-start:l.kp/2):0;
        const uint32_t pitch=Pad16(n);
        auto dq=s_,dk=s_[1024],gradient=s_[2048],raw=s_[3072],dg=s_[4096],da=s_[5120];
        auto temp=s_[6144],denom=s_[7168],aux=s_[8192];
        auto suffix=s_[10240],bias=s_[10368],totalA=s_[10496],totalBias=s_[10624];
        transfer_=s_[9216];small_=s_[10752];auto db=s_[10880],beta=s_[11008];
        Duplicate(suffix,0.f,512);PipeBarrier<PIPE_V>();
        if(in.bias && in.gateMode)ReadBlock<float>(bias,in.bias+start*4,n,1,n,pitch);
        float a=1.f;
        if(in.aLog && in.gateMode) {
            Read<float>(small_,in.aLog,0,1);Exp(small_,small_,1);ScalarReady();a=small_.GetValue(0);
        }
        for(uint32_t end=in.rows;end>0;) {
            const uint32_t count=end<8?end:8,first=end-count,elements=count*pitch;
            const uint64_t pos=(uint64_t(first)*l.kp+start)*4;
            ReadBlock<float>(dq,record+l.Offset(Key::Dq)+pos,l.kp,count,n,pitch);
            ReadBlock<float>(dk,record+l.Offset(Key::Dk)+pos,l.kp,count,n,pitch);
            ReadBlock<float>(gradient,record+l.Offset(Key::Dg)+pos,l.kp,count,n,pitch);
            WriteBlock<InputT>(dq,in.dq.address+(uint64_t(first)*in.dq.rowStride+start)*sizeof(InputT),in.dq.rowStride,count,n,pitch);
            WriteBlock<InputT>(dk,in.dk.address+(uint64_t(first)*in.dk.rowStride+start)*sizeof(InputT),in.dk.rowStride,count,n,pitch);
            for(uint32_t r=count;r>0;--r) {
                Add(suffix,suffix,gradient[(r-1)*pitch],pitch);PipeBarrier<PIPE_V>();
                Muls(gradient[(r-1)*pitch],suffix,in.useExp2?1.4426950408889634f:1.f,pitch);PipeBarrier<PIPE_V>();
            }
            if(in.gateMode) {
                ReadBlock<GateT>(raw,in.raw.address+(uint64_t(first)*in.raw.rowStride+start)*sizeof(GateT),in.raw.rowStride,count,n,pitch);
                for(uint32_t c=0;c<pitch;c+=64) {
                    const uint32_t width=pitch-c<64?pitch-c:64;
                    Add(raw[c],raw[c],bias[c],uint64_t(width),uint8_t(count),
                        {1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),0});
                }
                PipeBarrier<PIPE_V>();
            }
            GateDerivative(dg,da,gradient,raw,temp,denom,aux,a,in.lowerBound,in.gateMode,elements);PipeBarrier<PIPE_V>();
            for(uint32_t r=count;r>0;--r) {
                Add(totalA,totalA,da[(r-1)*pitch],pitch);
                if(in.gateMode)Add(totalBias,totalBias,dg[(r-1)*pitch],pitch);
                PipeBarrier<PIPE_V>();
            }
            WriteBlock<GateT>(dg,in.dg.address+(uint64_t(first)*in.dg.rowStride+start)*sizeof(GateT),in.dg.rowStride,count,n,pitch);
            end=first;
        }
        ReadBlock<float>(db,in.betaPartial,64,2,64,64);
        ReadBlock<float>(beta,in.beta.address,in.rows,1,in.rows,Pad16(in.rows));
        Add(db,db,db[64],in.rows);PipeBarrier<PIPE_V>();
        if(in.betaMode) {
            Muls(temp,beta,in.betaMode==1?-1.f:-.5f,in.rows);PipeBarrier<PIPE_V>();
            Adds(temp,temp,1.f,in.rows);PipeBarrier<PIPE_V>();
            Mul(temp,temp,beta,in.rows);PipeBarrier<PIPE_V>();Mul(db,db,temp,in.rows);PipeBarrier<PIPE_V>();
        }
        // Cast/Vector source addresses must be 32-byte aligned. A balanced
        // split of a 52-row tail starts lane 1 at float 26 (104 bytes).
        const uint32_t half=Pad8((in.rows+1)/2),begin=lane_*half,valid=in.rows>begin?in.rows-begin:0;
        const uint32_t count=valid<half?valid:half;
        if(count)WriteBlock<BetaT>(db[begin],in.dbeta.address+begin*sizeof(BetaT),count,1,count,Pad16(count));
        ReduceSum(small_,totalA,aux,n);ScalarReady();Exchange(in.exchange,small_.GetValue(0),0.f);
        const float sumA=small_.GetValue(48)+small_.GetValue(56);
        Write<float>(totalBias,in.parameterPartial,start,l.kp/2);
        if(lane_==0) {Duplicate(small_,sumA,8);PipeBarrier<PIPE_V>();Write<float>(small_,in.parameterPartial,l.kp,1);}
    }
    template<class InputT,class GateT,class BetaT>
    __aicore__ inline void Run(const Layout &l,uint64_t record,const TailInputs &in) {
        if(!in.normalize) {RunBlocked<InputT,GateT,BetaT>(l,record,in);return;}
        transfer_=s_[15*256];small_=s_[16*256];
        using namespace AscendC;
        const uint32_t start=lane_*(l.kp/2),n=l.k>start?(l.k-start<l.kp/2?l.k-start:l.kp/2):0;
        auto dq=s_,dk=s_[256],q=s_[512],k=s_[768],gradient=s_[1024],suffix=s_[1280];
        auto x=s_[1536],bias=s_[1792],dg=s_[2048],da=s_[2304],totalA=s_[2560],totalBias=s_[2816];
        auto t=s_[3072],denom=s_[3328],aux=s_[3584],db=s_[17*256];
        Duplicate(suffix,0.0f,256);Duplicate(totalA,0.0f,256);Duplicate(totalBias,0.0f,256);
        Duplicate(bias,0.0f,256);PipeBarrier<PIPE_V>();
        Read<float>(db,in.betaPartial,0,128);
        if(in.bias && in.gateMode)Read<float>(bias,in.bias,start,n);
        float a=1.f;
        if(in.aLog && in.gateMode) { Read<float>(small_,in.aLog,0,1);Exp(small_,small_,1);ScalarReady();a=small_.GetValue(0); }
        for(uint32_t r=in.rows;r>0;--r) {
            const uint32_t row=r-1;const uint64_t pos=uint64_t(row)*l.kp+start;
            Read<float>(dq,record+l.Offset(Key::Dq),pos,n);Read<float>(dk,record+l.Offset(Key::Dk),pos,n);
            if(in.normalize) {
                Read<float>(q,in.qhat.address,uint64_t(row)*in.qhat.rowStride+start,n);
                Read<float>(k,in.khat.address,uint64_t(row)*in.khat.rowStride+start,n);
                Mul(t,q,dq,n);Mul(denom,k,dk,n);PipeBarrier<PIPE_V>();
                ReduceSum(small_,t,aux,n);ReduceSum(small_[8],denom,aux,n);ScalarReady();
                Exchange(in.exchange,small_.GetValue(0),small_.GetValue(8));
                const float pq=small_.GetValue(48)+small_.GetValue(56),pk=small_.GetValue(49)+small_.GetValue(57);
                Read<float>(small_,in.rstdQ.address,uint64_t(row)*in.rstdQ.rowStride,1);ScalarReady();float rq=small_.GetValue(0);
                Read<float>(small_,in.rstdK.address,uint64_t(row)*in.rstdK.rowStride,1);ScalarReady();float rk=small_.GetValue(0);
                NormDerivative(dq,dq,q,t,pq,rq,true,n);PipeBarrier<PIPE_V>();NormDerivative(dk,dk,k,t,pk,rk,true,n);
                PipeBarrier<PIPE_V>();
            }
            Write<InputT>(dq,in.dq.address,uint64_t(row)*in.dq.rowStride+start,n);
            Write<InputT>(dk,in.dk.address,uint64_t(row)*in.dk.rowStride+start,n);
            Read<float>(gradient,record+l.Offset(Key::Dg),pos,n);
            Add(suffix,suffix,gradient,n);PipeBarrier<PIPE_V>();
            Muls(gradient,suffix,in.useExp2?1.4426950408889634f:1.f,n);PipeBarrier<PIPE_V>();
            if(in.gateMode) {
                Read<GateT>(x,in.raw.address,uint64_t(row)*in.raw.rowStride+start,n);
                Add(x,x,bias,n);PipeBarrier<PIPE_V>();
            }
            GateDerivative(dg,da,gradient,x,t,denom,aux,a,in.lowerBound,in.gateMode,n);PipeBarrier<PIPE_V>();
            Add(totalA,totalA,da,n);
            if(in.gateMode)Add(totalBias,totalBias,dg,n);
            PipeBarrier<PIPE_V>();Write<GateT>(dg,in.dg.address,uint64_t(row)*in.dg.rowStride+start,n);
            // Scalar beta rows are split independently of the key-column split.
            if((row<(in.rows+1)/2)==(lane_==0)) {
                Read<float>(small_,in.beta.address,uint64_t(row)*in.beta.rowStride,1);ScalarReady();
                float be=small_.GetValue(0),grad=db.GetValue(row)+db.GetValue(64+row);
                if(in.betaMode==1)grad*=be*(1.f-be);
                else if(in.betaMode==2)grad*=be*(1.f-.5f*be);
                Duplicate(small_,grad,8);PipeBarrier<PIPE_V>();
                Write<BetaT>(small_,in.dbeta.address,uint64_t(row)*in.dbeta.rowStride,1);
            }
        }
        ReduceSum(small_,totalA,aux,n);ScalarReady();
        Exchange(in.exchange,small_.GetValue(0),0.0f);
        const float sumA=small_.GetValue(48)+small_.GetValue(56);
        Write<float>(totalBias,in.parameterPartial,start,l.kp/2);
        if(lane_==0) { Duplicate(small_,sumA,8);PipeBarrier<PIPE_V>();Write<float>(small_,in.parameterPartial,l.kp,1); }
    }
};
} // namespace KDA::Arch22::Pipeline64
