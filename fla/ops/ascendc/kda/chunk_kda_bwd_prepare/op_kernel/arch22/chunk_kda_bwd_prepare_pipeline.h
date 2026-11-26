// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "../../../chunk_kda_bwd_finalize/op_kernel/arch22/pipeline64/matrix_stages.h"
namespace KDA::Arch22 {
namespace PreparePipeline {
using namespace Pipeline64;
template<class GradT>
__aicore__ inline void Promote(AscendC::LocalTensor<float> scratch,Matrix input,Matrix output,
                               Matrix score,uint32_t rows,uint32_t width) {
    using namespace AscendC;
    auto x=scratch;auto raw=scratch[4096].ReinterpretCast<GradT>();
    GlobalTensor<GradT> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ GradT*>(input.address));
    GlobalTensor<float> dst,ds;dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output.address));
    ds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score.address));
    const uint32_t lane=GetSubBlockIdx();
    for(uint32_t col=lane*32;col<width;col+=64) {
        const uint32_t n=width-col<32?width-col:32;
        constexpr uint32_t block=32/sizeof(GradT);const uint32_t aligned=(n+block-1)/block*block;
        DataCopyExtParams cp{uint16_t(rows),uint32_t(n*sizeof(GradT)),uint32_t((input.rowStride-n)*sizeof(GradT)),(32-aligned)/block,0};
        DataCopyPadExtParams<GradT> pad{true,0,uint8_t(aligned-n),GradT(0)};
        DataCopyPad(raw,src[col],cp,pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(sizeof(GradT)==4)Adds(x,scratch[4096],0.f,rows*32);
        else Cast(x,raw,RoundMode::CAST_NONE,rows*32);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        cp={uint16_t(rows),n*4,(32-Pad8(n))/8,(output.rowStride-n)*4,0};DataCopyPad(dst[col],x,cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    Duplicate(x,0.f,rows*32);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams clear{uint16_t(rows),32*4,0,32*4,0};DataCopyPad(ds[lane*32],x,clear);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    CrossCoreSetFlag<0x2,PIPE_MTE3>(RETIRE_READY);CrossCoreWaitFlag<0x2,PIPE_MTE2>(RETIRE_ACK);
}
template<class GradT>
__aicore__ inline void RunWork(GM_ADDR workspace,const Layout &l,uint32_t rows,
    Matrix aqk,Matrix vn,Matrix grad,Matrix h,Matrix ds,Matrix du,Matrix dq,float scale) {
    using namespace AscendC;
    const Matrix promoted{reinterpret_cast<uint64_t>(workspace)+l.Offset(Value::Do),l.vp,1};
    Product p[3]={{promoted,vn.T(),ds,rows,rows,l.v,Epilogue::ScaleLower,scale},
                  {promoted,h.T(),dq,rows,l.k,l.v,Epilogue::Assign},
                  {aqk.T(),promoted,du,rows,l.v,rows,Epilogue::Assign}};
    uint64_t packets=0,outputs=0;
    for(uint32_t i=0;i<3;++i){outputs+=Tiles(p[i]);packets+=Tiles(p[i])*((Pad8(p[i].k)+127)/128);}
    if ASCEND_IS_AIC {
        Cube cube;cube.Init(workspace+l.inputs,workspace+l.results,packets,outputs);cube.RetireRecord();
        for(uint32_t i=0;i<3;++i) {
            for(uint32_t j=0;j<Tiles(p[i]);++j) {
                auto t=GetTile(p[i],j);cube.Multiply(t.mp,t.np,Pad8(p[i].k),p[i].a.rowStride==1,p[i].b.rowStride==1);
            }
            cube.RetireRecord();
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        MatrixVector vector;vector.Init(workspace+l.inputs,workspace+l.results,outputs);
        TBuf<TPosition::VECCALC> promotion;vector.Pipe().InitBuffer(promotion,24*1024);
        Promote<GradT>(promotion.Get<float>(),grad,promoted,ds,rows,l.v);
        for(uint32_t i=0;i<3;++i)vector.Run(p[i]);
    }
}
template<class GradT>
__aicore__ inline void RunFixed(const GeneralBwdPrepareBuffers &b,GM_ADDR workspace,
                                const KDA::ChunkKdaBwdPrepareTilingData &t) {
    using namespace AscendC;
    const auto &l=t.pipeline;uint32_t block=GetBlockIdx();if ASCEND_IS_AIV {block/=2;}
    GM_ADDR group=workspace+uint64_t(block)*l.groupBytes;
    for(uint64_t task=block;task<uint64_t(t.totalChunkNum)*t.NV;task+=l.blocks) {
        const uint64_t chunk=task/t.NV,head=task%t.NV,batch=chunk/t.denseChunkNum,begin=chunk%t.denseChunkNum*64;
        const uint64_t token=(batch*t.NV+head)*t.T+begin,state=task*l.k*l.v;
        const uint32_t rows=t.T-begin<64?t.T-begin:64;
        Matrix h{reinterpret_cast<uint64_t>(b.h)+state*4,t.stateVFirst?1U:l.v,t.stateVFirst?l.k:1U};
        RunWork<GradT>(group,l,rows,{reinterpret_cast<uint64_t>(b.aqk)+token*64*4,64,1},
            {reinterpret_cast<uint64_t>(b.vNew)+token*l.v*4,l.v,1},
            {reinterpret_cast<uint64_t>(b.dO)+token*l.v*sizeof(GradT),l.v,1},h,
            {reinterpret_cast<uint64_t>(b.dAqk)+token*64*4,64,1},
            {reinterpret_cast<uint64_t>(b.dvLocal)+token*l.v*4,l.v,1},
            {reinterpret_cast<uint64_t>(b.dqRaw)+token*l.k*4,l.k,1},t.scale);
    }
}
} // namespace PreparePipeline
} // namespace KDA::Arch22
