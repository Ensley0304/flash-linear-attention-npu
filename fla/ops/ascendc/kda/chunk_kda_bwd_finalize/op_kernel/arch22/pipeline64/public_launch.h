// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "finalize_pair.h"
namespace KDA::Arch22::Pipeline64 {
__aicore__ inline Matrix View(GM_ADDR p,uint64_t elements,uint32_t pitch,uint32_t bytes=4) {
    return {reinterpret_cast<uint64_t>(p)+elements*bytes,pitch,1};
}
// Fixed-length canonical general inputs. Packed metadata remains on key201
// until its AIC/AIV metadata publication is integrated and tested.
template<class InputT,class GateT,class BetaT>
__aicore__ inline void RunPublicFixed(const GeneralBwdFinalizeBuffers &b,
    GM_ADDR paOut,GM_ADDR pbOut,GM_ADDR workspace,const FinalizeTilingData &t) {
    using namespace AscendC;
    const auto &l=t.pipeline;
    uint32_t block=GetBlockIdx();
    if ASCEND_IS_AIV {block/=2;}
    GM_ADDR group=workspace+uint64_t(block)*l.groupBytes;
    const uint64_t base=reinterpret_cast<uint64_t>(group);
    const uint64_t chunksPerBatch=(t.tokens+63)/64;
    const uint64_t headGroups=(t.heads+l.cg-1)/l.cg;
    for(uint64_t work=block;work<uint64_t(t.chunks)*headGroups;work+=l.blocks) {
      const uint64_t chunk=work/headGroups,firstHead=(work%headGroups)*l.cg;
      const uint32_t heads=t.heads-firstHead<l.cg?t.heads-firstHead:l.cg;
      FinalizeWorkInputs inputs[2];
      for(uint32_t owner=0;owner<heads;++owner) {
        const uint64_t head=firstHead+owner,task=chunk*t.heads+head,batch=chunk/chunksPerBatch;
        // K256 parameter partial has 257 floats: reserve the final scalar too.
        const uint64_t betaPartial=base+l.exchange+owner*2304,exchange=betaPartial+512,parameterPartial=betaPartial+1024;
        const uint64_t begin=(chunk%chunksPerBatch)*64,token=(batch*t.heads+head)*t.tokens+begin;
        const uint32_t rows=t.tokens-begin<64?t.tokens-begin:64;
        const uint64_t ko=token*l.k,vo=token*l.v,so=task*l.k*l.v;
        auto stateH=View(b.h,so,l.v),stateDh=View(b.dh,so,l.v);
        if(t.stateVFirst) {stateH.rowStride=stateDh.rowStride=1;stateH.colStride=stateDh.colStride=l.k;}
        FinalizeWorkInputs x{};
        x.products={View(b.dvScan,vo,l.v),stateH,stateDh,View(b.vNew,vo,l.v),View(b.akk,token*64,64)};
        x.direct={View(b.qHat,ko,l.k),View(b.kHat,ko,l.k),View(b.gk,ko,l.k),View(b.dqRaw,ko,l.k),
            View(b.v,vo,l.v,sizeof(InputT)),stateH,stateDh,View(b.dv,vo,l.v,sizeof(InputT)),
            View(b.betaEff,token,1).address,betaPartial,rows,t.scale,t.useExp2?.6931471805599453f:1.f};
        x.tail={x.direct.q,x.direct.k,x.direct.g,View(b.rawGate,ko,l.k,sizeof(GateT)),
            View(b.betaEff,token,1),View(b.qRstd,token,1),View(b.kRstd,token,1),
            View(b.dq,ko,l.k,sizeof(InputT)),View(b.dk,ko,l.k,sizeof(InputT)),
            View(b.dg,ko,l.k,sizeof(GateT)),View(b.dbeta,token,1,sizeof(BetaT)),
            b.bias?View(b.bias,head*l.k,l.k).address:0,b.aLog?View(b.aLog,head,1).address:0,
            betaPartial,exchange,parameterPartial,rows,t.hasQkL2Norm,t.gateMode,t.betaMode,t.useExp2,t.lowerBound};
        x.ds=View(b.dAqk,token*64,64);
        inputs[owner]=x;
      }
        RunFinalizeHeads<InputT,GateT,BetaT>(group,l,inputs,heads);
        if ASCEND_IS_AIV {
            // Publish the existing independent Finalize API's partial outputs.
            // The next work cannot consume this private arena before this AIV
            // reaches F0's two-participant publication barrier.
            TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,1024);
            auto local=buf.Get<float>();GlobalTensor<float> src,a,d;
          for(uint32_t owner=0;owner<heads;++owner) {
            const uint64_t task=chunk*t.heads+firstHead+owner;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputs[owner].tail.parameterPartial));
            src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(paOut));
            d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pbOut));
            const uint32_t start=GetSubBlockIdx()*(l.kp/2),n=l.k-start<l.kp/2?l.k-start:l.kp/2;
            DataCopyExtParams cp{1,n*4,0,0,0};DataCopyPadExtParams<float> pad{true,0,0,0.f};
            DataCopyPad(local,src[start],cp,pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(d[task*l.k+start],local,cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            if(GetSubBlockIdx()==0) {
                cp.blockLen=4;DataCopyPad(local,src[l.kp],cp,pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(a[task],local,cp);
            }
            PipeBarrier<PIPE_ALL>();
          }
        }
    }
}
} // namespace KDA::Arch22::Pipeline64
