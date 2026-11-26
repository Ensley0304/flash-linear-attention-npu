// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "matrix_stages.h"
namespace KDA::Arch22::Pipeline64 {
struct ContributionInputs {
    Matrix q,k,g,dqRaw,value,h,dnext,dv;
    uint64_t beta,betaPartial;
    uint32_t rows;
    float scale,exponentDerivative;
};
// F5 direct branches. FP32 scratch is shared with the later direct-score/tail
// phases: only one of those Vector phases may own this arena at a time.
class DirectContributions {
    using F=AscendC::LocalTensor<float>;
    F scratch_;
    uint32_t lane_;
    __aicore__ inline void Read(F dst,Matrix m,uint32_t row,uint32_t col,uint32_t rows,uint32_t cols,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),cols*4,(m.rowStride-cols)*4,(pitch-Pad8(cols))/8,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(cols)-cols),0.0f};
        DataCopyPad(dst,gm[uint64_t(row)*m.rowStride+col],cp,pad);
    }
    __aicore__ inline void Write(F src,Matrix m,uint32_t col,uint32_t rows,uint32_t cols,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),cols*4,(pitch-Pad8(cols))/8,(m.rowStride-cols)*4,0};
        DataCopyPad(gm[col],src,cp);
    }
    __aicore__ inline void Ready() {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    __aicore__ inline void Free() {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void RowBeta(F product,F db,F reduce,F sum,uint32_t rows,uint32_t n) {
        using namespace AscendC;
        WholeReduceSum(sum,product,n,rows,1,1,4);PipeBarrier<PIPE_V>();
        Add(db,db,sum,rows);PipeBarrier<PIPE_V>();
    }
public:
    static constexpr uint32_t FLOATS=21568;
    __aicore__ inline void Init(F scratch) { scratch_=scratch;lane_=AscendC::GetSubBlockIdx(); }
    template<class InputT>
    __aicore__ inline void Run(const Layout &l,uint64_t record,const ContributionInputs &in) {
        using namespace AscendC;
        auto q=scratch_,k=q[2048],g=q[4096],qr=q[6144],rk=q[8192],kg=q[10240];
        auto dq=q[12288],dk=q[14336],dg=q[16384],ex=q[18432];
        auto last=q[20480],gl=last[64],state=last[128],z=last[192],reduce=last[256],sum=last[320];
        auto beta=last[384],db=last[448];
        auto betaBroadcast=scratch_[21056];
        Duplicate(db,0.0f,64);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        Read(beta,{in.beta,64,1},0,0,1,in.rows,64);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        Ready();Brcb(betaBroadcast,beta,uint8_t((in.rows+7)/8),{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t tile=0;tile<l.k;tile+=64) {
            const uint32_t col=tile+lane_*32,n=l.k>col?(l.k-col<32?l.k-col:32):0;
            if(!n)continue;
            if(n<32)Duplicate(q,0.0f,20480);
            Duplicate(last,0.0f,256);Free();
            Read(q,in.q,0,col,in.rows,n,32);Read(k,in.k,0,col,in.rows,n,32);Read(g,in.g,0,col,in.rows,n,32);
            Read(qr,in.dqRaw,0,col,in.rows,n,32);
            Read(rk,{record+l.Offset(Key::Drk),l.kp,1},0,col,in.rows,n,32);
            Read(kg,{record+l.Offset(Key::Dkg),l.kp,1},0,col,in.rows,n,32);Ready();
            Adds(gl,g[(in.rows-1)*32],0.0f,32);
            Muls(ex,g,in.exponentDerivative,in.rows*32);PipeBarrier<PIPE_V>();Exp(ex,ex,in.rows*32);PipeBarrier<PIPE_V>();
            Mul(dq,qr,ex,in.rows*32);Mul(dk,rk,ex,in.rows*32);PipeBarrier<PIPE_V>();
            Muls(dq,dq,in.scale,in.rows*32);
            // drk * khat * exp(G) must retain its beta derivative at beta=0.
            Mul(qr,dk,k,in.rows*32);PipeBarrier<PIPE_V>();RowBeta(qr,db,reduce,sum,in.rows,n);
            Mul(dk,dk,betaBroadcast,uint64_t(32),uint8_t(in.rows),{1,1,0,4,4,1});
            PipeBarrier<PIPE_V>();Mul(dg,dq,q,in.rows*32);Mul(qr,dk,k,in.rows*32);PipeBarrier<PIPE_V>();
            Add(dg,dg,qr,in.rows*32);PipeBarrier<PIPE_V>();Muls(dg,dg,in.exponentDerivative,in.rows*32);
            Sub(g,gl,g,uint64_t(32),uint8_t(in.rows),{1,1,1,4,0,4});
            PipeBarrier<PIPE_V>();Muls(g,g,in.exponentDerivative,in.rows*32);PipeBarrier<PIPE_V>();
            Exp(g,g,in.rows*32);PipeBarrier<PIPE_V>();Mul(kg,kg,g,in.rows*32);PipeBarrier<PIPE_V>();
            Add(dk,dk,kg,in.rows*32);Mul(kg,kg,k,in.rows*32);PipeBarrier<PIPE_V>();
            Muls(kg,kg,in.exponentDerivative,in.rows*32);PipeBarrier<PIPE_V>();Sub(dg,dg,kg,in.rows*32);
            PipeBarrier<PIPE_V>();
            // kg's token rows are dead after the subtraction above. Reduce
            // the full row plane in a fixed FP32 tree instead of 64 barriers.
            for(uint32_t active=in.rows;active>1;) {
                const uint32_t half=active/2;
                Add(kg,kg,kg[half*32],half*32);PipeBarrier<PIPE_V>();
                if(active%2) {Adds(kg[half*32],kg[(active-1)*32],0.f,32);PipeBarrier<PIPE_V>();}
                active=half+active%2;
            }
            Adds(last,kg,0.f,32);PipeBarrier<PIPE_V>();
            // State decay contribution. Reuse dead input planes for H*Dnext.
            for(uint32_t v=0;v<l.v;v+=64) {
                uint32_t nv=l.v-v<64?l.v-v:64;
                if(n<32 || nv<64)Duplicate(q,0.0f,4096);
                Free();
                if(in.h.rowStride==1) {
                    Read(q,in.h.T(),v,col,nv,n,32);Read(k,in.dnext.T(),v,col,nv,n,32);
                } else {
                    Read(q,in.h,col,v,n,nv,64);Read(k,in.dnext,col,v,n,nv,64);
                }
                Ready();Mul(q,q,k,2048);PipeBarrier<PIPE_V>();
                if(in.h.rowStride==1) {
                    for(uint32_t j=0;j<nv;++j){Add(state,state,q[j*32],32);PipeBarrier<PIPE_V>();}
                } else {
                    WholeReduceSum(z,q,64,n,1,1,8);PipeBarrier<PIPE_V>();Add(state,state,z,n);PipeBarrier<PIPE_V>();
                }
            }
            Muls(gl,gl,in.exponentDerivative,32);PipeBarrier<PIPE_V>();Exp(gl,gl,32);PipeBarrier<PIPE_V>();
            Mul(state,state,gl,32);PipeBarrier<PIPE_V>();Muls(state,state,in.exponentDerivative,32);PipeBarrier<PIPE_V>();
            Add(last,last,state,32);PipeBarrier<PIPE_V>();Add(dg[(in.rows-1)*32],dg[(in.rows-1)*32],last,32);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(dq,{record+l.Offset(Key::Dq),l.kp,1},col,in.rows,n,32);
            Write(dk,{record+l.Offset(Key::Dk),l.kp,1},col,in.rows,n,32);
            Write(dg,{record+l.Offset(Key::Dg),l.kp,1},col,in.rows,n,32);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for(uint32_t tile=0;tile<l.v;tile+=64) {
            const uint32_t col=tile+lane_*32,n=l.v>col?(l.v-col<32?l.v-col:32):0;
            if(!n)continue;
            if(n<32)Duplicate(q,0.0f,4096);
            Free();
            Read(q,{record+l.Offset(Value::Drv),l.vp,1},0,col,in.rows,n,32);
            GlobalTensor<InputT> raw;raw.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(in.value.address));
            auto transfer=g.ReinterpretCast<InputT>();
            constexpr uint32_t block=32/sizeof(InputT);
            uint32_t aligned=(n+block-1)/block*block;
            DataCopyExtParams cp{uint16_t(in.rows),uint32_t(n*sizeof(InputT)),
                uint32_t((in.value.rowStride-n)*sizeof(InputT)),uint32_t((32-aligned)/block),0};
            DataCopyPadExtParams<InputT> pad{true,0,uint8_t(aligned-n),InputT(0)};
            DataCopyPad(transfer,raw[col],cp,pad);Ready();
            if constexpr(sizeof(InputT)==4)Adds(k,g,0.0f,in.rows*32);
            else Cast(k,transfer,RoundMode::CAST_NONE,in.rows*32);
            PipeBarrier<PIPE_V>();Mul(qr,q,k,in.rows*32);PipeBarrier<PIPE_V>();RowBeta(qr,db,reduce,sum,in.rows,n);
            Mul(q,q,betaBroadcast,uint64_t(32),uint8_t(in.rows),{1,1,0,4,4,1});
            PipeBarrier<PIPE_V>();
            if constexpr(sizeof(InputT)==4)Adds(g,q,0.0f,in.rows*32);
            else Cast(transfer,q,RoundMode::CAST_RINT,in.rows*32);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<InputT> dv;dv.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(in.dv.address));
            DataCopyExtParams wr{uint16_t(in.rows),uint32_t(n*sizeof(InputT)),uint32_t((32-aligned)/block),
                uint32_t((in.dv.rowStride-n)*sizeof(InputT)),0};
            DataCopyPad(dv[col],transfer,wr);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Write(db,{in.betaPartial+lane_*64*4,64,1},0,1,in.rows,64);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
};
} // namespace KDA::Arch22::Pipeline64
