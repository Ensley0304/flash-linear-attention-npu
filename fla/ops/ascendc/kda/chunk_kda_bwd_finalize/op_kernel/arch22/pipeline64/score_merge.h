// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "matrix_stages.h"
namespace KDA::Arch22::Pipeline64 {
struct ScoreMergeInputs {
    Matrix qPlus,kPlus,kMinus,expPlus,expMinus;
    Matrix dq,dk,dg;
    uint32_t queryStart,keyStart,keyColumn,queryRows,keyRows,columns;
    float exponentDerivative;
};
// Safe-band F7: explicit gradient targets permit overlapping diagonal bands.
// Query contribution is committed before key contribution reads the same row.
// Every column has exactly one AIV writer; no atomic or cross-AIV vector read.
class ScoreMerge {
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<float> storage_;
    uint32_t lane_;
    using F=AscendC::LocalTensor<float>;
    __aicore__ inline void Read(F x,Matrix m,uint32_t row,uint32_t col,uint32_t n) {
        using namespace AscendC;
        GlobalTensor<float> g;g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        g.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{1,n*4,0,0,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(n)-n),0.0f};
        DataCopyPad(x,g[uint64_t(row)*m.rowStride+col],cp,pad);
    }
    __aicore__ inline void Commit(Matrix m,uint32_t row,uint32_t col,F delta,F prior,uint32_t n) {
        using namespace AscendC;
        Read(prior,m,row,col,n);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Add(prior,prior,delta,n);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<float> g;g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        g.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{1,n*4,0,0,0};DataCopyPad(g[uint64_t(row)*m.rowStride+col],prior,cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
public:
    __aicore__ inline void Column(const Layout &l,uint64_t record,uint32_t rows,uint32_t column,uint32_t columns,bool write) {
        using namespace AscendC;
        const uint32_t half=Pad16(columns)/2,start=lane_*half;
        const uint32_t n=columns>start?(columns-start<half?columns-start:half):0;
        if(!n)return;
        if(!write) {
            if(n<32 || rows<64)Duplicate(storage_,0.f,6144);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            ReadRows(storage_,{record+l.Offset(Key::Dq),l.kp,1},rows,column+start,n);
            ReadRows(storage_[2048],{record+l.Offset(Key::Dk),l.kp,1},rows,column+start,n);
            ReadRows(storage_[4096],{record+l.Offset(Key::Dg),l.kp,1},rows,column+start,n);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WriteRows(storage_,{record+l.Offset(Key::Dq),l.kp,1},rows,column+start,n);
            WriteRows(storage_[2048],{record+l.Offset(Key::Dk),l.kp,1},rows,column+start,n);
            WriteRows(storage_[4096],{record+l.Offset(Key::Dg),l.kp,1},rows,column+start,n);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    // F6 uses <=64-row bands. Keep all three destination planes resident
    // while both sides of the band contribute, then issue three 2-D stores.
    // Factors/score gradients stream in 32-row strips. Uses 15552 floats.
    __aicore__ inline void RunBand(const Layout &l,uint64_t record,const ScoreMergeInputs &in,bool resident=false,bool combined=false) {
        using namespace AscendC;
        const uint32_t half=Pad16(in.columns)/2,start=lane_*half;
        const uint32_t n=in.columns>start?(in.columns-start<half?in.columns-start:half):0;
        if(!n)return;
        const uint32_t qend=in.queryStart+in.queryRows,kend=in.keyStart+in.keyRows;
        const uint32_t rows=qend>kend?qend:kend;
        auto dq=storage_,dk=dq[2048],dg=dq[4096];
        auto qp=dq[6144],kp=dq[7168],km=dq[8192],ep=dq[9216],em=dq[10240];
        auto qa=dq[11264],ka=dq[12288],qb=dq[13312],kb=dq[14336];
        auto tmp=dq[15360],delta=dq[15424],gate=dq[15488];
        if(n<32) {
            if(resident)Duplicate(storage_[6144],0.f,15552-6144);
            else Duplicate(storage_,0.f,15552);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if(!resident) {
            ReadRows(dq,in.dq,rows,in.keyColumn+start,n);
            ReadRows(dk,in.dk,rows,in.keyColumn+start,n);
            ReadRows(dg,in.dg,rows,in.keyColumn+start,n);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for(uint32_t begin=0;begin<in.queryRows;begin+=32) {
          const uint32_t count=in.queryRows-begin<32?in.queryRows-begin:32;
          SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
          ReadRows(qp,in.qPlus,count,start,n,begin);ReadRows(kp,in.kPlus,count,start,n,begin);
          ReadRows(ep,in.expPlus,count,start,n,begin);
          ReadRows(qa,{record+l.Offset(Score::Query),64,1},count,start,n,begin);
          ReadRows(ka,{record+l.Offset(Score::Da),64,1},count,start,n,begin);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          const uint32_t r=(in.queryStart+begin)*32,elements=count*32;
          // The key-side result buffers are dead during the query-side stage.
          // Operate on the complete contiguous strip, not one row at a time.
          Mul(qb,qa,ep,elements);Mul(kb,ka,ep,elements);PipeBarrier<PIPE_V>();
          Add(dq[r],dq[r],qb,elements);Add(dk[r],dk[r],kb,elements);PipeBarrier<PIPE_V>();
          Mul(qb,qa,qp,elements);Mul(kb,ka,kp,elements);PipeBarrier<PIPE_V>();
          Add(qb,qb,kb,elements);PipeBarrier<PIPE_V>();
          Muls(qb,qb,in.exponentDerivative,elements);PipeBarrier<PIPE_V>();
          Add(dg[r],dg[r],qb,elements);PipeBarrier<PIPE_V>();
        }
        for(uint32_t begin=0;begin<in.keyRows;begin+=32) {
          const uint32_t count=in.keyRows-begin<32?in.keyRows-begin:32;
          SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
          ReadRows(km,in.kMinus,count,start,n,begin);ReadRows(em,in.expMinus,count,start,n,begin);
          ReadRows(qb,{record+l.Offset(Score::Key),64,1},count,start,n,begin);
          if(!combined)ReadRows(kb,{record+l.Offset(Score::Middle),64,1},count,start,n,begin);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          const uint32_t r=(in.keyStart+begin)*32,elements=count*32;
          if(!combined)Add(qb,qb,kb,elements);
          PipeBarrier<PIPE_V>();
          Mul(qa,qb,em,elements);Mul(ka,qb,km,elements);PipeBarrier<PIPE_V>();
          Muls(ka,ka,-in.exponentDerivative,elements);Add(dk[r],dk[r],qa,elements);PipeBarrier<PIPE_V>();
          Add(dg[r],dg[r],ka,elements);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if(!resident) {
            WriteRows(dq,in.dq,rows,in.keyColumn+start,n);
            WriteRows(dk,in.dk,rows,in.keyColumn+start,n);
            WriteRows(dg,in.dg,rows,in.keyColumn+start,n);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void Init(AscendC::TPipe &pipe) {
        pipe.InitBuffer(buffer_,8*256);Init(buffer_.Get<float>());
    }
    __aicore__ inline void Init(F scratch) { storage_=scratch;lane_=AscendC::GetSubBlockIdx(); }
    __aicore__ inline void Run(const Layout &l,uint64_t record,const ScoreMergeInputs &in) {
        using namespace AscendC;
        const uint32_t half=Pad16(in.columns)/2,start=lane_*half;
        const uint32_t n=in.columns>start?(in.columns-start<half?in.columns-start:half):0;
        if(!n)return;
        auto a=storage_,b=a[64],q=a[128],k=a[192],e=a[256];
        auto delta=a[320],gate=a[384],prior=a[448];
        const Matrix qgrad{record+l.Offset(Score::Query),64,1};
        const Matrix kpgrad{record+l.Offset(Score::Da),64,1};
        const Matrix qkmgrad{record+l.Offset(Score::Key),64,1};
        const Matrix kkmgrad{record+l.Offset(Score::Middle),64,1};
        for(uint32_t i=0;i<in.queryRows;++i) {
            Read(a,qgrad,i,start,n);Read(b,kpgrad,i,start,n);
            Read(q,in.qPlus,i,start,n);Read(k,in.kPlus,i,start,n);Read(e,in.expPlus,i,start,n);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(gate,a,q,n);Mul(delta,b,k,n);PipeBarrier<PIPE_V>();
            Add(gate,gate,delta,n);PipeBarrier<PIPE_V>();
            Muls(gate,gate,in.exponentDerivative,n);
            Mul(delta,a,e,n);PipeBarrier<PIPE_V>();
            Commit(in.dq,in.queryStart+i,in.keyColumn+start,delta,prior,n);
            Mul(delta,b,e,n);PipeBarrier<PIPE_V>();
            Commit(in.dk,in.queryStart+i,in.keyColumn+start,delta,prior,n);
            Commit(in.dg,in.queryStart+i,in.keyColumn+start,gate,prior,n);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        for(uint32_t j=0;j<in.keyRows;++j) {
            Read(a,qkmgrad,j,start,n);Read(b,kkmgrad,j,start,n);
            Read(k,in.kMinus,j,start,n);Read(e,in.expMinus,j,start,n);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(a,a,b,n);PipeBarrier<PIPE_V>();
            Mul(gate,a,k,n);PipeBarrier<PIPE_V>();Muls(gate,gate,-in.exponentDerivative,n);
            Mul(delta,a,e,n);PipeBarrier<PIPE_V>();
            Commit(in.dk,in.keyStart+j,in.keyColumn+start,delta,prior,n);
            Commit(in.dg,in.keyStart+j,in.keyColumn+start,gate,prior,n);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }
private:
    __aicore__ inline void ReadRows(F dst,Matrix m,uint32_t rows,uint32_t col,uint32_t n,uint32_t begin=0) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(m.rowStride-n)*4,(32-Pad8(n))/8,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(n)-n),0.f};
        DataCopyPad(dst,gm[uint64_t(begin)*m.rowStride+col],cp,pad);
    }
    __aicore__ inline void WriteRows(F src,Matrix m,uint32_t rows,uint32_t col,uint32_t n) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(32-Pad8(n))/8,(m.rowStride-n)*4,0};DataCopyPad(gm[col],src,cp);
    }
};
} // namespace KDA::Arch22::Pipeline64
