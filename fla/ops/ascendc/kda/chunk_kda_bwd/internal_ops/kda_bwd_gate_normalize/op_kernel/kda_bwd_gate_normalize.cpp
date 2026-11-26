// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "tiling.h"
#include "../../chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_general.h"
extern "C" __global__ __aicore__ void kda_bwd_gate_normalize(GM_ADDR g,GM_ADDR a,GM_ADDR bias,
    GM_ADDR metadata,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KDA::Arch22::GateNormalizeTiling);
    GET_TILING_DATA_WITH_STRUCT(KDA::Arch22::GateNormalizeTiling,t,tiling);
    AscendC::TPipe pipe;
    KdaPrepare::Arch22::GeneralPrepare<float,DTYPE_G,float> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> queue;pipe.InitBuffer(queue,1,32);
    AscendC::GlobalTensor<int64_t> meta;meta.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(metadata));
    for(uint64_t work=AscendC::GetBlockIdx();work<uint64_t(t.chunks)*t.heads;work+=AscendC::GetBlockNum()){
        auto chunk=work/t.heads,head=work%t.heads;
        auto row=queue.AllocTensor<int64_t>();
        AscendC::DataCopyPad(row,meta[chunk*3],AscendC::DataCopyExtParams{1,24,0,0,0},AscendC::DataCopyPadExtParams<int64_t>{true,0,0,0});
        queue.EnQue(row);row=queue.DeQue<int64_t>();
        auto event=pipe.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
        uint64_t batch=row.GetValue(0),begin=row.GetValue(1);uint32_t rows=row.GetValue(2);queue.FreeTensor(row);
        KdaPrepare::Arch22::GeneralPrepareChunk c{};
        c.rows=rows;c.keyDim=t.keys;c.keyStride=t.keys;c.keyOffset=((batch*t.heads+head)*t.tokens+begin)*t.keys;
        c.cacheKeyOffset=c.keyOffset;c.gateMode=t.safeGate?2:1;c.useExp2=t.useExp2;c.lowerBound=t.lowerBound;
        KdaPrepare::Arch22::GeneralPrepareBuffers b{};b.gate=g;b.gk=out;
        b.aLog=a?a+head*4:nullptr;b.bias=bias?bias+head*t.keys*4:nullptr;
        op.GateOnly(b,c);
    }
}
