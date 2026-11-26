// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "matrix_stages.h"
namespace KDA::Arch22::Pipeline64 {
struct DirectScoreInputs {
    Matrix q,k,g,ds,dl,dq,dk,dg;
    uint64_t beta,betaPartial;
    uint32_t rows,columns,columnStart;
    float exponentDerivative;
    bool accumulateBeta = false;
    uint32_t queryStart=0,queryRows=0,keyStart=0,keyRows=0;
};
// Numerically protected score-gradient path. The complete valid causal domain
// of one K64 panel is computed in UB; no per-(i,j) GM access. This component is
// reserved for unsafe exponent factorizations, not the common Cube path.
class DirectScore {
    using F=AscendC::LocalTensor<float>;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    F storage_;
    uint32_t lane_;
    __aicore__ inline void Read(F dst,Matrix src,uint32_t rows,uint32_t col,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(src.rowStride-n)*4,(pitch-Pad8(n))/8,0};
        DataCopyPadExtParams<float> pad{true,0,uint8_t(Pad8(n)-n),0.0f};
        DataCopyPad(dst,gm[col],cp,pad);
    }
    __aicore__ inline void Write(Matrix dst,F src,uint32_t rows,uint32_t col,uint32_t n,uint32_t pitch) {
        using namespace AscendC;
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst.address));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{uint16_t(rows),n*4,(pitch-Pad8(n))/8,(dst.rowStride-n)*4,0};
        DataCopyPad(gm[col],src,cp);
    }
public:
    static constexpr uint32_t FLOATS=21056;
    __aicore__ inline void Init(AscendC::TPipe &pipe) {
        pipe.InitBuffer(buffer_,FLOATS*4);Init(buffer_.Get<float>());
    }
    __aicore__ inline void Init(F scratch) { storage_=scratch;lane_=AscendC::GetSubBlockIdx(); }
    __aicore__ inline void Run(const DirectScoreInputs &in) {
        using namespace AscendC;
        // Match the safe-band/retained-gradient owner for a short K panel.
        const uint32_t half=Pad16(in.columns)/2,start=lane_*half;
        const uint32_t n=in.columns>start?(in.columns-start<half?in.columns-start:half):0;
        auto q=storage_,k=q[2048],g=q[4096],dq=q[6144],dk=q[8192],dg=q[10240];
        auto ds=q[12288],dl=q[16384],ex=q[20480],a=ex[64],b=ex[128],z=ex[192];
        auto betaRow=ex[256],reduce=ex[320],sum=ex[384],beta=ex[448],db=ex[512];
        Duplicate(db,0.0f,64);
        if(n) {
            Duplicate(q,0.0f,20480);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            const uint32_t col=in.columnStart+start;
            Read(q,in.q,in.rows,col,n,32);Read(k,in.k,in.rows,col,n,32);Read(g,in.g,in.rows,col,n,32);
            Read(dq,in.dq,in.rows,col,n,32);Read(dk,in.dk,in.rows,col,n,32);Read(dg,in.dg,in.rows,col,n,32);
            Read(ds,in.ds,in.rows,0,in.rows,64);Read(dl,in.dl,in.rows,0,in.rows,64);
            Read(beta,{in.beta,64,1},1,0,in.rows,64);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            const uint32_t queryEnd=in.queryRows?in.queryStart+in.queryRows:in.rows;
            const uint32_t keyEnd=in.keyRows?in.keyStart+in.keyRows:in.rows;
            for(uint32_t i=in.queryStart;i<queryEnd;++i) {
                Duplicate(betaRow,0.0f,32);PipeBarrier<PIPE_V>();
                const float be=beta.GetValue(i);
                for(uint32_t j=in.keyStart;j<=i && j<keyEnd;++j) {
                    const float s=ds.GetValue(i*64+j), l=j<i?dl.GetValue(i*64+j):0.0f;
                    Sub(ex,g[i*32],g[j*32],32);PipeBarrier<PIPE_V>();
                    Muls(ex,ex,in.exponentDerivative,32);PipeBarrier<PIPE_V>();Exp(ex,ex,32);PipeBarrier<PIPE_V>();
                    Mul(a,k[j*32],ex,32);PipeBarrier<PIPE_V>();
                    Muls(b,a,s,32);PipeBarrier<PIPE_V>();Add(dq[i*32],dq[i*32],b,32);
                    Muls(b,a,l*be,32);PipeBarrier<PIPE_V>();Add(dk[i*32],dk[i*32],b,32);
                    // beta derivative uses l directly, including beta==0.
                    Mul(b,a,k[i*32],32);PipeBarrier<PIPE_V>();Muls(b,b,l,32);PipeBarrier<PIPE_V>();
                    Add(betaRow,betaRow,b,32);
                    Muls(a,q[i*32],s,32);Muls(b,k[i*32],l*be,32);PipeBarrier<PIPE_V>();
                    Add(a,a,b,32);PipeBarrier<PIPE_V>();Mul(a,a,ex,32);PipeBarrier<PIPE_V>();
                    Add(dk[j*32],dk[j*32],a,32);
                    Mul(z,a,k[j*32],32);PipeBarrier<PIPE_V>();Muls(z,z,in.exponentDerivative,32);PipeBarrier<PIPE_V>();
                    Add(dg[i*32],dg[i*32],z,32);PipeBarrier<PIPE_V>();Sub(dg[j*32],dg[j*32],z,32);
                    PipeBarrier<PIPE_V>();
                }
                ReduceSum(sum,betaRow,reduce,n);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                db.SetValue(i,sum.GetValue(0));
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            Write(in.dq,dq,in.rows,col,n,32);Write(in.dk,dk,in.rows,col,n,32);Write(in.dg,dg,in.rows,col,n,32);
        }
        if(in.accumulateBeta) {
            // F5 initializes this lane's partial; every subsequent K panel
            // adds its strictly-lower score contribution exactly once.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Read(ex,{in.betaPartial+lane_*64*4,64,1},1,0,in.rows,64);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Add(db,db,ex,64);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Write({in.betaPartial+lane_*64*4,64,1},db,1,0,in.rows,64);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
};
} // namespace KDA::Arch22::Pipeline64
