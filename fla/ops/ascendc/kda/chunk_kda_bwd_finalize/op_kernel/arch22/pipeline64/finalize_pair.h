// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "finalize_work.h"
#include "score_pipeline.h"
namespace KDA::Arch22::Pipeline64 {
// Two independent head records advance stage-major. All producers of a stage
// finish before its dependent consumers, but independent GEMMs share credits.
template<class InputT,class GateT,class BetaT>
__aicore__ inline void RunFinalizeHeads(GM_ADDR workspace,const Layout &l,
                                      const FinalizeWorkInputs *x,uint32_t heads) {
    using namespace AscendC;
    const uint64_t base=reinterpret_cast<uint64_t>(workspace);
    const uint32_t rows=x[0].direct.rows;
    const uint32_t bandRows=x[0].tail.gateMode==0 && l.vp>=64?64:24;
    // Dynamic credit accounting drains the actual packets/results in End().
    constexpr uint64_t packets=0,outputs=0;
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs,true);
        for(uint32_t h=0;h<heads;++h)cube.RetireRecord();
        for(uint32_t i=0;i<8;++i) {
            cube.InvalidateScratch(base,l.inputs);
            bool native=true;
            for(uint32_t h=0;h<heads;++h) {
                const auto p=Describe(static_cast<ProductKind>(i),l,base+h*l.recordBytes,rows,x[h].products);
                native=native && DirectOperand(p) && p.epilogue==Epilogue::Assign;
            }
            for(uint32_t h=0;h<heads;++h) {
                const auto p=Describe(static_cast<ProductKind>(i),l,base+h*l.recordBytes,rows,x[h].products);
                for(uint32_t j=0;j<Tiles(p);++j) {
                    auto t=GetTile(p,j);cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1,
                        DirectOperand(p)?&p:nullptr,t.row,t.col,native);
                }
            }
            if(native)CrossCoreSetFlag<0x2,PIPE_FIX>(SCORE_NATIVE_READY);
            cube.RetireRecord();
        }
        cube.RetireRecord();
        if(heads==1)ScorePipelineCube(cube,l,base,x[0],bandRows);
        else {
        for(uint32_t col=0;col<l.k;col+=64)
        for(uint32_t qi=0;qi<rows;qi+=bandRows)for(uint32_t kj=0;kj<=qi;kj+=bandRows) {
            Band b{qi,kj,col,rows-qi<bandRows?rows-qi:bandRows,rows-kj<bandRows?rows-kj:bandRows,l.k-col<64?l.k-col:64};
            bool safe[2];
            cube.InvalidateScratch(base,l.inputs);
            for(uint32_t h=0;h<heads;++h) {
                cube.RetireRecord();safe[h]=cube.BandSafe(x[h].tail.exchange);cube.RetireRecord();
            }
            for(uint32_t i=0;i<5;++i)for(uint32_t h=0;h<heads;++h) {
                if(!safe[h])continue;
                const uint64_t record=base+h*l.recordBytes;
                const auto p=DescribeScore(static_cast<ScoreProduct>(i),l,record,BandProducts(l,record,b));
                for(uint32_t j=0;j<Tiles(p);++j) {
                    auto t=GetTile(p,j);cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1,
                        DirectOperand(p)?&p:nullptr,t.row,t.col);
                }
            }
            cube.RetireRecord();cube.RetireRecord();
        }
        }
        for(uint32_t h=0;h<heads;++h) {
            TailExchangeCube(rows,x[h].tail.normalize);cube.RetireRecord();
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vector;vector.Init(workspace+l.inputs,workspace+l.results,outputs,true);
        TBuf<TPosition::VECCALC> arena;vector.Pipe().InitBuffer(arena,DirectContributions::FLOATS*4);
        auto scratch=arena.Get<float>();
        for(uint32_t h=0;h<heads;++h)
            vector.Prepare<InputT>(l,base+h*l.recordBytes,rows,x[h].direct.k,x[h].direct.g,
                x[h].direct.value,x[h].direct.beta,x[h].tail.useExp2);
        for(uint32_t i=0;i<8;++i) {
            Product stage[2];bool native=true;
            for(uint32_t h=0;h<heads;++h) {
                stage[h]=Describe(static_cast<ProductKind>(i),l,base+h*l.recordBytes,rows,x[h].products);
                native=native && DirectOperand(stage[h]) && stage[h].epilogue==Epilogue::Assign;
            }
            if(native) {
                CrossCoreWaitFlag<0x2,PIPE_MTE2>(SCORE_NATIVE_READY);
                CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
            } else vector.RunStage(stage,heads,true);
        }
        DirectContributions direct;direct.Init(scratch);
        for(uint32_t h=0;h<heads;++h)direct.Run<InputT>(l,base+h*l.recordBytes,x[h].direct);
        if(heads==1 && CacheScoreScalars(l,rows)) {
            ScoreBand scalar;scalar.Init(vector.OperandScratch());
            scalar.PrepareScalars(l,base,rows,x[0].ds,x[0].direct.beta);
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        if(heads==1)ScorePipelineVector(vector,scratch,l,base,x[0],bandRows);
        else {
        ScoreBand band;band.Init(vector.OperandScratch());ScoreMerge merge;merge.Init(scratch);DirectScore score;score.Init(scratch);
        for(uint32_t col=0;col<l.k;col+=64) {
        const bool resident=heads==1;
        const uint32_t width=l.k-col<64?l.k-col:64;
        if(resident)merge.Column(l,base,rows,col,width,false);
        for(uint32_t qi=0;qi<rows;qi+=bandRows)for(uint32_t kj=0;kj<=qi;kj+=bandRows) {
            Band b{qi,kj,col,rows-qi<bandRows?rows-qi:bandRows,rows-kj<bandRows?rows-kj:bandRows,l.k-col<64?l.k-col:64};
            bool safe[2];Product stage[10];
            for(uint32_t h=0;h<heads;++h) {
                const uint64_t record=base+h*l.recordBytes;
                safe[h]=band.Prepare(l,record,b,x[h].direct.q,x[h].direct.k,x[h].direct.g,x[h].ds,
                    x[h].direct.beta,x[h].tail.exchange,x[h].direct.exponentDerivative);
                for(uint32_t i=0;i<5;++i)
                    stage[i*heads+h]=DescribeScore(static_cast<ScoreProduct>(i),l,record,BandProducts(l,record,b));
            }
            Product active[10];uint32_t count=0;
            for(uint32_t i=0;i<5;++i)for(uint32_t h=0;h<heads;++h)
                if(safe[h])active[count++]=stage[i*heads+h];
            vector.RunStage(active,count,true);
            for(uint32_t h=0;h<heads;++h) {
                const uint64_t record=base+h*l.recordBytes;
                if(safe[h]) {
                    merge.RunBand(l,record,BandMerge(l,record,b,x[h].direct.exponentDerivative),resident);
                    band.Beta(l,record,b,x[h].direct.betaPartial);
                } else {
                    if(resident)merge.Column(l,base,rows,col,width,true);
                    DirectScoreInputs s{x[h].direct.q,x[h].direct.k,x[h].direct.g,x[h].ds,
                        {record+l.Offset(Score::Dl),C,1},{record+l.Offset(Key::Dq),l.kp,1},
                        {record+l.Offset(Key::Dk),l.kp,1},{record+l.Offset(Key::Dg),l.kp,1},
                        x[h].direct.beta,x[h].direct.betaPartial,rows,0,0,x[h].direct.exponentDerivative,true};
                    s.columns=b.d;s.columnStart=col;
                    s.queryStart=qi;s.queryRows=b.m;s.keyStart=kj;s.keyRows=b.n;score.Run(s);
                    if(resident)merge.Column(l,base,rows,col,width,false);
                }
            }
            CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        }
        if(resident)merge.Column(l,base,rows,col,width,true);
        }
        }
        Tail tail;tail.Init(scratch);
        for(uint32_t h=0;h<heads;++h) {
            tail.Run<InputT,GateT,BetaT>(l,base+h*l.recordBytes,x[h].tail);
            CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        }
        vector.End();
    }
}
} // namespace KDA::Arch22::Pipeline64
