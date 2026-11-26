// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "finalize_work.h"
namespace KDA::Arch22::Pipeline64 {
constexpr uint16_t SCORE_NATIVE_READY=12;
__aicore__ inline bool CacheScoreScalars(const Layout &l,uint32_t rows) {return l.vp>=128 && rows>=16;}
__aicore__ inline ScoreInputs PipelineBandProducts(const Layout &l,uint64_t record,const Band &b,uint64_t base,uint32_t rows) {
    auto x=BandProducts(l,record,b);
    if(CacheScoreScalars(l,rows)) {
        const uint64_t offset=4*(uint64_t(b.query)*C+b.key);
        x.ds={base+l.Offset(Value::Do)+offset,C,1};
        x.dlBeta={base+l.Offset(Value::Do)+C*C*4+offset,C,1};
    }
    return x;
}
__aicore__ inline bool NativeScoreBand(const Band &b) {
    return b.m>=16 && b.n>=16 && b.d>=16 && b.m%8==0 && b.n%8==0 && b.d%8==0;
}
__aicore__ inline bool MonotoneRectangle(const FinalizeWorkInputs &x,const Band &b) {
    // Both raw activations produce non-positive log steps. With all keys
    // before the first query, both centered exponential arguments are <=0.
    return b.query>=b.key+b.n &&
        (x.tail.gateMode==1 || (x.tail.gateMode==2 && x.tail.lowerBound<=0.f));
}
__aicore__ inline uint32_t ColumnBands(Band *bands,uint32_t rows,uint32_t bandRows,uint32_t col,uint32_t width) {
    uint32_t count=0;
    for(uint32_t qi=0;qi<rows;qi+=bandRows) {
        const uint32_t m=rows-qi<bandRows?rows-qi:bandRows;
        // The strictly earlier keys share the same first-query gate
        // reference. Coalesce that rectangle; keep diagonal safety unchanged.
        if(qi)bands[count++]={qi,0,col,m,qi,width};
        bands[count++]={qi,qi,col,m,m,width};
    }
    return count;
}
// Each record is free after its F7 last read. Preparing i+1 precedes F7(i):
// the AIC can execute the next band's two result credits during that F7.
// dL and the gradient accumulator remain in record zero, never in the ring.
__aicore__ inline void ScorePipelineCube(Cube &cube,const Layout &l,uint64_t base,
                                         const FinalizeWorkInputs &x,uint32_t bandRows) {
    for(uint32_t col=0;col<l.k;col+=64) {
        Band bands[6];const uint32_t count=ColumnBands(bands,x.direct.rows,bandRows,col,l.k-col<64?l.k-col:64);
        for(uint32_t index=0;index<count;++index) {
            const uint64_t record=base+l.Record(index%2,0);
            bool safe=MonotoneRectangle(x,bands[index]);
            if(!safe) {
                cube.RetireRecord();safe=cube.BandSafe(x.tail.exchange);
                if(bands[index].query!=bands[index].key)cube.RetireRecord();
            } else cube.RetireRecord();
            cube.InvalidateScratch(base,l.inputs);
            const bool native=safe && NativeScoreBand(bands[index]);
            if(safe)for(uint32_t order=0;order<5;++order) {
                const uint32_t i=native?(order==1?2:order==2?1:order):order;
                auto p=DescribeScore(static_cast<ScoreProduct>(i),l,record,PipelineBandProducts(l,record,bands[index],base,x.direct.rows));
                if(native && i==3)p.out={record+l.Offset(Score::Key),C,1};
                for(uint32_t j=0;j<Tiles(p);++j) {
                    auto t=GetTile(p,j);cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1,
                        DirectOperand(p)?&p:nullptr,t.row,t.col,native,native && i==3,!(native && i==1));
                }
            }
            if(native)AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(SCORE_NATIVE_READY);
            cube.RetireRecord();
        }
        cube.RetireRecord(); // final F7 and gradient column writeback
    }
}
__aicore__ inline void ScorePipelineVector(MatrixVector &vector,AscendC::LocalTensor<float> scratch,
    const Layout &l,uint64_t base,const FinalizeWorkInputs &x,uint32_t bandRows) {
    using namespace AscendC;
    ScoreBand band;band.Init(vector.OperandScratch());
    ScoreMerge merge;merge.Init(scratch);DirectScore score;score.Init(scratch);
    for(uint32_t col=0;col<l.k;col+=64) {
        const uint32_t width=l.k-col<64?l.k-col:64;
        Band bands[6];const uint32_t count=ColumnBands(bands,x.direct.rows,bandRows,col,width);
        merge.Column(l,base,x.direct.rows,col,width,false);
        bool safe=band.Prepare(l,base,bands[0],x.direct.q,x.direct.k,x.direct.g,x.ds,
            x.direct.beta,x.tail.exchange,x.direct.exponentDerivative,base,MonotoneRectangle(x,bands[0]),CacheScoreScalars(l,x.direct.rows),true);
        for(uint32_t index=0;index<count;++index) {
            const auto b=bands[index];const uint64_t record=base+l.Record(index%2,0);
            Product stage[5];
            for(uint32_t i=0;i<5;++i)stage[i]=DescribeScore(static_cast<ScoreProduct>(i),l,record,PipelineBandProducts(l,record,b,base,x.direct.rows));
            if(safe && NativeScoreBand(b)) {
                // Fixpipe publishes the final FP32 planes directly. No AIV
                // result repacking and no result-slot credit is needed here.
                CrossCoreWaitFlag<0x2,PIPE_MTE2>(SCORE_NATIVE_READY);
                CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
            } else vector.RunStage(stage,safe?5:0,true);
            bool nextSafe=false;
            if(index+1<count)nextSafe=band.Prepare(l,base+l.Record((index+1)%2,0),bands[index+1],
                x.direct.q,x.direct.k,x.direct.g,x.ds,x.direct.beta,x.tail.exchange,x.direct.exponentDerivative,base,
                MonotoneRectangle(x,bands[index+1]),CacheScoreScalars(l,x.direct.rows),bands[index+1].query==bands[index+1].key);
            if(safe) {
                auto args=BandMerge(l,record,b,x.direct.exponentDerivative);
                args.dq={base+l.Offset(Key::Dq),l.kp,1};args.dk={base+l.Offset(Key::Dk),l.kp,1};args.dg={base+l.Offset(Key::Dg),l.kp,1};
                merge.RunBand(l,record,args,true,NativeScoreBand(b));
                band.Beta(l,record,b,x.direct.betaPartial,base);
            } else {
                merge.Column(l,base,x.direct.rows,col,width,true);
                DirectScoreInputs s{x.direct.q,x.direct.k,x.direct.g,x.ds,
                    {base+l.Offset(Score::Dl),C,1},{base+l.Offset(Key::Dq),l.kp,1},
                    {base+l.Offset(Key::Dk),l.kp,1},{base+l.Offset(Key::Dg),l.kp,1},
                    x.direct.beta,x.direct.betaPartial,x.direct.rows,0,0,x.direct.exponentDerivative,true};
                s.columns=b.d;s.columnStart=col;s.queryStart=b.query;s.queryRows=b.m;s.keyStart=b.key;s.keyRows=b.n;
                score.Run(s);merge.Column(l,base,x.direct.rows,col,width,false);
            }
            safe=nextSafe;
        }
        merge.Column(l,base,x.direct.rows,col,width,true);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
}
} // namespace KDA::Arch22::Pipeline64
