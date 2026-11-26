// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "score_products.h"
#include "score_merge.h"
namespace KDA::Arch22::Pipeline64 {
struct Band {
    uint32_t query,key,column,m,n,d;
};
__aicore__ inline ScoreInputs BandProducts(const Layout &l,uint64_t record,const Band &b) {
    const uint32_t pitch=Pad16(b.n);
    return {{record+l.Offset(Value::Rv),pitch,1},{record+l.Offset(Value::Rv)+Pad16(b.m)*pitch*4,pitch,1},
            {record+l.Offset(Key::Rk),64,1},{record+l.Offset(Key::Dw),64,1},
            {record+l.Offset(Key::Dkg),64,1},b.m,b.n,b.d};
}
__aicore__ inline ScoreMergeInputs BandMerge(const Layout &l,uint64_t record,const Band &b,float base) {
    auto x=BandProducts(l,record,b);
    return {x.qPlus,x.kPlus,x.kMinus,{record+l.Offset(Key::Drk),64,1},
            {record+l.Offset(Key::Band),64,1},{record+l.Offset(Key::Dq),l.kp,1},
            {record+l.Offset(Key::Dk),l.kp,1},{record+l.Offset(Key::Dg),l.kp,1},
            b.query,b.key,b.column,b.m,b.n,b.d,base};
}
// F5 has consumed Rk/Dw/Drk/Dkg and Rv/Drv before these addresses are reused.
// Scalar planes share retired Rv/Drv. Full C64 bands require Vp>=64;
// smaller V and raw-gate paths use C16 bands. Factors fit each key plane.
class ScoreBand {
    using F=AscendC::LocalTensor<float>;
    F s_;
    uint32_t lane_;
    __aicore__ inline void Read(F dst,Matrix src,uint32_t row,uint32_t col,uint32_t rows,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(src.rowStride-n)*4,(pitch-Pad8(n))/8,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(n)-n),0.f};
        DataCopyPad(dst,gm[uint64_t(row)*src.rowStride+col],cp,pad);
    }
    __aicore__ inline void Write(F src,Matrix dst,uint32_t row,uint32_t col,uint32_t rows,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(pitch-Pad8(n))/8,(dst.rowStride-n)*4,0};
        DataCopyPad(gm[uint64_t(row)*dst.rowStride+col],src,cp);
    }
    __aicore__ inline void Ready() {
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
public:
    __aicore__ inline void Init(F scratch) {s_=scratch;lane_=AscendC::GetSubBlockIdx();}
    __aicore__ inline bool DiagonalSafe(const Band &b,Matrix g,float base) {
        using namespace AscendC;
        auto values=s_,ref=s_[4096],maximum=s_[4160],reduce=s_[4224];
        if(b.d<64)Duplicate(s_,0.f,4160);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        Read(values,g,b.query,b.column,b.m,b.d,64);
        Read(ref,g,b.query+b.m/2,b.column,1,b.d,64);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Sub(values,values,ref,uint64_t(64),uint8_t(b.m),{1,1,1,8,8,0});PipeBarrier<PIPE_V>();
        Muls(values,values,base,b.m*64);PipeBarrier<PIPE_V>();
        auto absolute=s_[6144];
        Abs(absolute,values,b.m*64);PipeBarrier<PIPE_V>();
        ReduceMax(maximum,absolute,reduce,b.m*64,false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float largest=maximum.GetValue(0);
        const uint32_t half=Pad16(b.d)/2,start=lane_*half;
        const uint32_t n=b.d>start?(b.d-start<half?b.d-start:half):0;
        auto ep=s_[6144],em=s_[8192];
        if(n<32) {Duplicate(ep,0.f,2048);PipeBarrier<PIPE_V>();}
        if(n)Adds(ep,values[start],0.f,uint64_t(n),uint8_t(b.m),{1,1,4,8});
        PipeBarrier<PIPE_V>();Muls(em,ep,-1.f,b.m*32);PipeBarrier<PIPE_V>();
        return largest==largest && largest<=32.f;
    }
    __aicore__ inline void PrepareScalars(const Layout &l,uint64_t record,uint32_t rows,Matrix ds,uint64_t beta) {
        using namespace AscendC;
        const uint32_t half=(rows+1)/2,start=lane_*half,count=rows>start?(rows-start<half?rows-start:half):0;
        if(!count)return;
        auto sr=s_,lr=s_[2048],br=s_[4096],bc=s_[4160];
        Duplicate(s_,0.f,4672);SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        Read(sr,ds,start,0,count,rows,64);
        Read(lr,{record+l.Offset(Score::Dl),64,1},start,0,count,rows,64);
        Read(br,{beta,64,1},0,start,1,count,64);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Brcb(bc,br,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();
        Mul(lr,lr,bc,uint64_t(64),uint8_t(count),{1,1,0,8,8,1});PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<count;++r) {
            const uint32_t valid=start+r+1;
            if(valid<64) {uint64_t mask[2]={~((1ULL<<valid)-1),0};Duplicate(sr[r*64],0.f,mask,1,1,8);}
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Write(sr,{record+l.Offset(Value::Do),64,1},start,0,count,64,64);
        Write(lr,{record+l.Offset(Value::Do)+C*C*4,64,1},start,0,count,64,64);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    __aicore__ inline bool Prepare(const Layout &l,uint64_t record,const Band &b,
        Matrix q,Matrix k,Matrix g,Matrix ds,uint64_t beta,uint64_t exchange,float base,uint64_t sourceRecord=0,
        bool knownSafe=false,bool cachedScalars=false,bool replicatePredicate=false) {
        using namespace AscendC;
        auto qv=s_,kv=s_[2048],km=s_[4096],ep=s_[6144],em=s_[8192];
        auto ref=s_[10240],tmp=s_[10304],reduce=s_[10368],small=s_[10432];
        const uint32_t half=Pad16(b.d)/2,start=lane_*half;
        const uint32_t n=b.d>start?(b.d-start<half?b.d-start:half):0;
        bool localSafe=replicatePredicate?DiagonalSafe(b,g,base):true;
        // Full 32-column lanes overwrite every live operand before use.
        if(n<32)Duplicate(s_,0.f,replicatePredicate?6144:10496);
        PipeBarrier<PIPE_V>();
        if(n) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Read(qv,q,b.query,b.column+start,b.m,n,32);
            Read(kv,k,b.query,b.column+start,b.m,n,32);
            Read(km,k,b.key,b.column+start,b.n,n,32);
            if(!replicatePredicate) {
            Read(ep,g,b.query,b.column+start,b.m,n,32);
            Read(em,g,b.key,b.column+start,b.n,n,32);
            // Center diagonal bands to avoid one-sided exponential amplification.
            // For disjoint causal bands the first query remains between the bands.
            const uint32_t referenceRow=b.query==b.key?b.query+b.m/2:b.query;
            Read(ref,g,referenceRow,b.column+start,1,n,32);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if(!replicatePredicate) {
            Sub(ep,ep,ref,uint64_t(32),uint8_t(b.m),{1,1,1,4,4,0});
            Sub(em,ref,em,uint64_t(32),uint8_t(b.n),{1,1,1,4,0,4});
            PipeBarrier<PIPE_V>();
            Muls(ep,ep,base,b.m*32);Muls(em,em,base,b.n*32);
            PipeBarrier<PIPE_V>();
            }
            if(!knownSafe && !replicatePredicate) {
            ReduceMax(tmp,ep,reduce,b.m*32,false);PipeBarrier<PIPE_V>();
            ReduceMax(tmp[8],em,reduce,b.n*32,false);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float a=tmp.GetValue(0),c=tmp.GetValue(8);
            localSafe=(a==a && c==c && a<=32.f && c<=32.f);
            }
        }
        bool safe=knownSafe || (replicatePredicate && localSafe);
        if(!knownSafe) {
        small.SetValue(0,localSafe?1.f:0.f);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Write(small,{exchange,16,1},0,lane_*8,1,1,8);
        if(!replicatePredicate) {
        Ready();
        Read(small,{exchange,16,1},0,0,1,16,16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        safe=small.GetValue(0)!=0.f && small.GetValue(8)!=0.f;
        }
        }
        if(n && safe) {
                Exp(ep,ep,b.m*32);Exp(em,em,b.n*32);PipeBarrier<PIPE_V>();
                Mul(qv,qv,ep,b.m*32);Mul(kv,kv,ep,b.m*32);Mul(km,km,em,b.n*32);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            auto x=BandProducts(l,record,b);auto merge=BandMerge(l,record,b,base);
            Write(qv,x.qPlus,0,start,b.m,n,32);Write(kv,x.kPlus,0,start,b.m,n,32);
            Write(km,x.kMinus,0,start,b.n,n,32);Write(ep,merge.expPlus,0,start,b.m,n,32);
            Write(em,merge.expMinus,0,start,b.n,n,32);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        if(cachedScalars) {Ready();return safe;}
        // Reuse the arena only after all factor stores have consumed it.
        const Matrix dl{(sourceRecord?sourceRecord:record)+l.Offset(Score::Dl),64,1};auto x=BandProducts(l,record,b);
        auto sr=s_,lr=s_[2048],br=s_[4096];
        const uint32_t rowHalf=(b.m+1)/2,rowStart=lane_*rowHalf;
        const uint32_t rowCount=b.m>rowStart?(b.m-rowStart<rowHalf?b.m-rowStart:rowHalf):0;
        if(rowCount) {
            Duplicate(sr,0.f,4160);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            if(safe) {
                Read(sr,ds,b.query+rowStart,b.key,rowCount,b.n,64);
                Read(lr,dl,b.query+rowStart,b.key,rowCount,b.n,64);
                Read(br,{beta,64,1},0,b.query+rowStart,1,rowCount,64);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                auto betaBroadcast=br[64];
                Brcb(betaBroadcast,br,uint8_t((rowCount+7)/8),{1,8});PipeBarrier<PIPE_V>();
                Mul(lr,lr,betaBroadcast,uint64_t(64),uint8_t(rowCount),{1,1,0,8,8,1});
                PipeBarrier<PIPE_V>();
                for(uint32_t r=0;r<rowCount;++r) {
                    const uint32_t i=rowStart+r;
                    const uint32_t valid=b.query+i>=b.key?(b.query+i-b.key+1<b.n?b.query+i-b.key+1:b.n):0;
                    if(valid<64) {
                        uint64_t mask[2]={~((1ULL<<valid)-1),0};
                        Duplicate(sr[r*64],0.f,mask,1,1,8);
                    }
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(sr,x.ds,rowStart,0,rowCount,x.ds.rowStride,64);
            Write(lr,x.dlBeta,rowStart,0,rowCount,x.dlBeta.rowStride,64);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Ready();return safe;
    }
    __aicore__ inline void Beta(const Layout &l,uint64_t record,const Band &b,uint64_t partial,uint64_t sourceRecord=0) {
        using namespace AscendC;
        auto a=s_,d=s_[2048],old=s_[4096],sum=s_[4160];
        const uint32_t half=(b.m+1)/2,start=lane_*half;
        const uint32_t count=b.m>start?(b.m-start<half?b.m-start:half):0;
        if(count) {
            // Read/reduce only the valid key columns; row padding is dead.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Read(a,{record+l.Offset(Score::Rkk),64,1},start,0,count,b.n,64);
            Read(d,{(sourceRecord?sourceRecord:record)+l.Offset(Score::Dl),64,1},b.query+start,b.key,count,b.n,64);
            Read(old,{partial+lane_*64*4,64,1},0,b.query+start,1,count,64);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(a,a,d,uint64_t(b.n),uint8_t(count),{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,a,b.n,count,1,1,8);PipeBarrier<PIPE_V>();
            Add(old,old,sum,count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(old,{partial+lane_*64*4,64,1},0,b.query+start,1,count,64);
            // The next head reuses this arena on PIPE_V before its first load.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
};
} // namespace KDA::Arch22::Pipeline64
