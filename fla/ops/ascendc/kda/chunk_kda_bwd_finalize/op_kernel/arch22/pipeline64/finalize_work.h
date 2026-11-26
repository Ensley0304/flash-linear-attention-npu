// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "direct_contributions.h"
#include "direct_score.h"
#include "tail.h"
#include "score_band.h"
namespace KDA::Arch22::Pipeline64 {
// One complete work record. The protected score implementation is retained
// for numerical validation and unsafe bands; it is not the fast-path policy.
struct FinalizeWorkInputs {
    Inputs products;
    ContributionInputs direct;
    TailInputs tail;
    Matrix ds;
};
template<class InputT,class GateT,class BetaT,bool CubeScore=false>
__aicore__ inline void RunProtectedFinalizeWork(GM_ADDR workspace,const Layout &l,
                                               const FinalizeWorkInputs &x) {
    using namespace AscendC;
    const uint64_t record=reinterpret_cast<uint64_t>(workspace);
    const uint32_t rows=x.direct.rows;
    const uint32_t bandRows=x.tail.gateMode==0 && l.vp>=64?64:24;
    uint64_t packets=0,outputs=0;
    for(uint32_t i=0;i<8;++i) {
        auto p=Describe(static_cast<ProductKind>(i),l,record,rows,x.products);
        outputs+=Tiles(p);packets+=Tiles(p)*((Pad8(p.k)+127)/128);
    }
    if constexpr(CubeScore) {
        for(uint32_t qi=0;qi<rows;qi+=bandRows)for(uint32_t kj=0;kj<=qi;kj+=bandRows)
        for(uint32_t col=0;col<l.k;col+=64) {
            Band b{qi,kj,col,rows-qi<bandRows?rows-qi:bandRows,rows-kj<bandRows?rows-kj:bandRows,l.k-col<64?l.k-col:64};
            auto s=BandProducts(l,record,b);
            for(uint32_t i=0;i<5;++i) {
                auto p=DescribeScore(static_cast<ScoreProduct>(i),l,record,s);
                outputs+=Tiles(p);packets+=Tiles(p)*((Pad8(p.k)+127)/128);
            }
        }
    }
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs);
        cube.RetireRecord(); // F0 publishes Rk/Rv before any dependent read.
        for(uint32_t i=0;i<8;++i) {
            auto p=Describe(static_cast<ProductKind>(i),l,record,rows,x.products);
            for(uint32_t j=0;j<Tiles(p);++j) {
                auto t=GetTile(p,j);
                cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1);
            }
            cube.RetireRecord();
        }
        cube.RetireRecord(); // Both F5 owners have initialized all gradient planes.
        if constexpr(CubeScore) {
            for(uint32_t qi=0;qi<rows;qi+=bandRows)for(uint32_t kj=0;kj<=qi;kj+=bandRows)
            for(uint32_t col=0;col<l.k;col+=64) {
                Band b{qi,kj,col,rows-qi<bandRows?rows-qi:bandRows,rows-kj<bandRows?rows-kj:bandRows,l.k-col<64?l.k-col:64};
                cube.RetireRecord();cube.RetireRecord(); // predicate, then operand publication
                auto s=BandProducts(l,record,b);
                for(uint32_t i=0;i<5;++i) {
                    auto p=DescribeScore(static_cast<ScoreProduct>(i),l,record,s);
                    for(uint32_t j=0;j<Tiles(p);++j) {
                        auto t=GetTile(p,j);cube.Multiply(t.mp,t.np,Pad8(p.k),p.a.rowStride==1,p.b.rowStride==1);
                    }
                }
                cube.RetireRecord(); // one completion for the independent score stage
                cube.RetireRecord(); // merge has retired the factors and score outputs
            }
        }
        cube.RetireRecord(); // Both score owners have committed their last panel.
        TailExchangeCube(rows,x.tail.normalize);
        cube.RetireRecord(); // Tail output and parameter partial last writers.
        cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vector;vector.Init(workspace+l.inputs,workspace+l.results,outputs);
        TBuf<TPosition::VECCALC> arena;
        vector.Pipe().InitBuffer(arena,DirectContributions::FLOATS*4);
        auto scratch=arena.Get<float>();
        vector.Prepare<InputT>(l,record,rows,x.direct.k,x.direct.g,x.direct.value,
                              x.direct.beta,x.tail.useExp2);
        for(uint32_t i=0;i<8;++i)
            vector.Run(Describe(static_cast<ProductKind>(i),l,record,rows,x.products));
        DirectContributions direct;direct.Init(scratch);direct.Run<InputT>(l,record,x.direct);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        DirectScore score;score.Init(scratch);
        DirectScoreInputs s{x.direct.q,x.direct.k,x.direct.g,x.ds,
            {record+l.Offset(Score::Dl),C,1},{record+l.Offset(Key::Dq),l.kp,1},
            {record+l.Offset(Key::Dk),l.kp,1},{record+l.Offset(Key::Dg),l.kp,1},
            x.direct.beta,x.direct.betaPartial,rows,0,0,x.direct.exponentDerivative,true};
        if constexpr(CubeScore) {
            ScoreBand band;band.Init(scratch);ScoreMerge merge;merge.Init(scratch);
            for(uint32_t qi=0;qi<rows;qi+=bandRows)for(uint32_t kj=0;kj<=qi;kj+=bandRows)
            for(uint32_t col=0;col<l.k;col+=64) {
                Band b{qi,kj,col,rows-qi<bandRows?rows-qi:bandRows,rows-kj<bandRows?rows-kj:bandRows,l.k-col<64?l.k-col:64};
                const bool safe=band.Prepare(l,record,b,x.direct.q,x.direct.k,x.direct.g,x.ds,
                    x.direct.beta,x.tail.exchange,x.direct.exponentDerivative);
                auto products=BandProducts(l,record,b);
                Product stage[5];
                for(uint32_t i=0;i<5;++i)
                    stage[i]=DescribeScore(static_cast<ScoreProduct>(i),l,record,products);
                vector.RunStage(stage,5);
                if(safe) {
                    merge.RunBand(l,record,BandMerge(l,record,b,x.direct.exponentDerivative));
                    band.Beta(l,record,b,x.direct.betaPartial);
                } else {
                    s.columnStart=col;s.columns=b.d;s.queryStart=qi;s.queryRows=b.m;s.keyStart=kj;s.keyRows=b.n;
                    score.Run(s);
                }
                CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
                CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
            }
        } else {
            for(uint32_t col=0;col<l.k;col+=64) {
                s.columnStart=col;s.columns=l.k-col<64?l.k-col:64;score.Run(s);
            }
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
        Tail tail;tail.Init(scratch);tail.Run<InputT,GateT,BetaT>(l,record,x.tail);
        CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
    }
}
} // namespace KDA::Arch22::Pipeline64
