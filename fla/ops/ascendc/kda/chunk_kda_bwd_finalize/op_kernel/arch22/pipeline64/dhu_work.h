// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "matrix_stages.h"
namespace KDA::Arch22::Pipeline64 {
struct DhuSequence {
    Matrix qg,kg,w,g,grad,local,du;
    uint64_t dht,dh0,dh,snapshotStride;
    uint32_t tokens,valueStart,valueCount;
    float scale;bool exp2;
};
__aicore__ inline Matrix AtRow(Matrix x,uint32_t row,uint32_t bytes=4) {
    x.address+=uint64_t(row)*x.rowStride*bytes;return x;
}
__aicore__ inline Product DhuProduct(uint32_t i,const Layout &l,uint64_t record,
    const DhuSequence &x,uint32_t begin,uint32_t rows) {
    const Matrix state{record+l.Offset(Key::Rk),64,1},grad{record+l.Offset(Value::Do),64,1};
    Matrix du=AtRow(x.du,begin);du.address+=x.valueStart*4;
    if(i==0)return {AtRow(x.kg,begin),state,du,rows,x.valueCount,l.k,Epilogue::Add,1.f,
        AtRow(x.local,begin).address+x.valueStart*4};
    if(i==1)return {AtRow(x.qg,begin).T(),grad,{record+l.Offset(Key::Dw),64,1},
        l.k,x.valueCount,rows,Epilogue::Assign};
    return {AtRow(x.w,begin).T(),du,{record+l.Offset(Key::Drk),64,1},
        l.k,x.valueCount,rows,Epilogue::Assign};
}
class DhuVector {
    using F=AscendC::LocalTensor<float>;
    F state_,tmp_,raw_,decay_;
    uint32_t lane_,n_,start_;
    __aicore__ inline void Read(F dst,Matrix src,uint32_t col,uint32_t rows,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(src.rowStride-n)*4,(pitch-Pad8(n))/8,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(n)-n),0.f};DataCopyPad(dst,gm[col],cp,pad);
    }
    __aicore__ inline void Write(F src,Matrix dst,uint32_t col,uint32_t rows) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n_*4,(32-Pad8(n_))/8,(dst.rowStride-n_)*4,0};DataCopyPad(gm[col],src,cp);
    }
public:
    static constexpr uint32_t FLOATS=19008;
    __aicore__ inline void Init(F s,const Layout &l,const DhuSequence &x) {
        using namespace AscendC;
        state_=s;tmp_=s[8192];raw_=s[16384];decay_=s[18432];lane_=GetSubBlockIdx();start_=lane_*32;
        n_=x.valueCount>start_?(x.valueCount-start_<32?x.valueCount-start_:32):0;
        Duplicate(state_,0.f,l.k*32);
        if(n_ && x.dht) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Read(state_,{x.dht,l.v,1},x.valueStart+start_,l.k,n_,32);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
    template<class GradT>
    __aicore__ inline void Begin(const Layout &l,uint64_t record,const DhuSequence &x,uint32_t chunk,uint32_t begin,uint32_t rows) {
        using namespace AscendC;
        if(n_) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(state_,{record+l.Offset(Key::Rk),64,1},start_,l.k);
            Write(state_,{x.dh+chunk*x.snapshotStride,l.v,1},x.valueStart+start_,l.k);
            GlobalTensor<GradT> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ GradT*>(x.grad.address));
            constexpr uint32_t block=32/sizeof(GradT);const uint32_t aligned=(n_+block-1)/block*block;
            auto raw=raw_.ReinterpretCast<GradT>();
            DataCopyExtParams cp{uint16_t(rows),uint32_t(n_*sizeof(GradT)),uint32_t((x.grad.rowStride-n_)*sizeof(GradT)),(32-aligned)/block,0};
            DataCopyPadExtParams<GradT> pad{true,0,uint8_t(aligned-n_),GradT(0)};
            DataCopyPad(raw,gm[uint64_t(begin)*x.grad.rowStride+x.valueStart+start_],cp,pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(GradT)==4)Adds(tmp_,raw_,0.f,rows*32);
            else Cast(tmp_,raw,RoundMode::CAST_NONE,rows*32);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(tmp_,{record+l.Offset(Value::Do),64,1},start_,rows);
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    __aicore__ inline void Update(const Layout &l,uint64_t record,const DhuSequence &x,uint32_t last) {
        using namespace AscendC;
        if(n_) {
            Read(decay_,AtRow(x.g,last),0,1,l.k,256);
            Read(tmp_,{record+l.Offset(Key::Dw),64,1},start_,l.k,n_,32);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if(x.exp2){Muls(decay_,decay_,.6931471805599453f,l.k);PipeBarrier<PIPE_V>();}
            Exp(decay_,decay_,l.k);Muls(tmp_,tmp_,x.scale,l.k*32);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for(uint32_t i=0;i<l.k;++i)Muls(state_[i*32],state_[i*32],decay_.GetValue(i),32);
            PipeBarrier<PIPE_V>();Add(state_,state_,tmp_,l.k*32);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Read(tmp_,{record+l.Offset(Key::Drk),64,1},start_,l.k,n_,32);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Sub(state_,state_,tmp_,l.k*32);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Finish(const Layout &l,const DhuSequence &x) {
        using namespace AscendC;
        if(n_ && x.dh0) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(state_,{x.dh0,l.v,1},x.valueStart+start_,l.k);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
};
template<class GradT>
__aicore__ inline void RunDhuSequence(GM_ADDR workspace,const Layout &l,const DhuSequence &x) {
    using namespace AscendC;
    const uint64_t record=reinterpret_cast<uint64_t>(workspace);const uint32_t chunks=(x.tokens+63)/64;
    uint64_t packets=0,outputs=0;
    for(uint32_t chunk=0;chunk<chunks;++chunk) {
        const uint32_t begin=chunk*64,rows=x.tokens-begin<64?x.tokens-begin:64;
        for(uint32_t i=0;i<3;++i) {auto p=DhuProduct(i,l,record,x,begin,rows);outputs+=Tiles(p);packets+=Tiles(p)*((Pad8(p.k)+127)/128);}
    }
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs);
        for(uint32_t c=chunks;c>0;--c) {
            const uint32_t begin=(c-1)*64,rows=x.tokens-begin<64?x.tokens-begin:64;cube.RetireRecord();
            for(uint32_t i=0;i<3;++i) {
                auto p=DhuProduct(i,l,record,x,begin,rows);
                for(uint32_t j=0;j<Tiles(p);++j) {auto t=GetTile(p,j);cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1);}
                cube.RetireRecord();
            }
        }
        cube.RetireRecord();cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vec;vec.Init(workspace+l.inputs,workspace+l.results,outputs);
        TBuf<TPosition::VECCALC> arena;vec.Pipe().InitBuffer(arena,DhuVector::FLOATS*4);
        DhuVector state;state.Init(arena.Get<float>(),l,x);
        for(uint32_t c=chunks;c>0;--c) {
            const uint32_t begin=(c-1)*64,rows=x.tokens-begin<64?x.tokens-begin:64;
            state.Begin<GradT>(l,record,x,c-1,begin,rows);
            for(uint32_t i=0;i<3;++i)vec.Run(DhuProduct(i,l,record,x,begin,rows));
            state.Update(l,record,x,begin+rows-1);
        }
        state.Finish(l,x);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
}
} // namespace KDA::Arch22::Pipeline64
