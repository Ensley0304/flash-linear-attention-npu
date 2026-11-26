// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#if __has_include("../../../../../kda/chunk_kda_bwd_finalize/op_kernel/arch22/pipeline64/dhu_work.h")
#include "../../../../../kda/chunk_kda_bwd_finalize/op_kernel/arch22/pipeline64/dhu_work.h"
#else
#include "../../../chunk_kda_bwd_finalize/op_kernel/arch22/pipeline64/dhu_work.h"
#endif
namespace GDN::Arch22 {
template<class GradT>
__aicore__ inline void RunPipelineDhu(const GeneralDhuBuffers &b,GM_ADDR workspace,
                                     const GDN::ChunkGatedDeltaRuleBwdDhuTilingData &t) {
    using namespace AscendC;using namespace KDA::Arch22::Pipeline64;
    const auto &l=t.pipeline;uint32_t block=GetBlockIdx();if ASCEND_IS_AIV {block/=2;}
    const uint32_t slices=(t.V+63)/64;
    for(uint64_t task=block;task<uint64_t(t.taskNum);task+=l.blocks) {
        const uint32_t tile=task%slices,head=(task/slices)%t.HV,seq=task/(slices*t.HV);
        const uint64_t token=(uint64_t(seq)*t.HV+head)*t.T,so=(uint64_t(seq)*t.HV+head)*t.K*t.V;
        const uint64_t snapshot=(uint64_t(seq)*t.chunkNumForT*t.HV+head)*t.K*t.V;
        const uint32_t start=tile*64,count=t.V-start<64?t.V-start:64;
        DhuSequence x{{reinterpret_cast<uint64_t>(b.qg)+token*t.K*4,l.k,1},
            {reinterpret_cast<uint64_t>(b.kg)+token*t.K*4,l.k,1},
            {reinterpret_cast<uint64_t>(b.w)+token*t.K*4,l.k,1},
            {reinterpret_cast<uint64_t>(b.gk)+token*t.K*4,l.k,1},
            {reinterpret_cast<uint64_t>(b.dO)+token*t.V*sizeof(GradT),l.v,1},
            {reinterpret_cast<uint64_t>(b.dvLocal)+token*t.V*4,l.v,1},
            {reinterpret_cast<uint64_t>(b.dv)+token*t.V*4,l.v,1},
            b.dht?reinterpret_cast<uint64_t>(b.dht)+so*4:0,
            b.dh0?reinterpret_cast<uint64_t>(b.dh0)+so*4:0,
            reinterpret_cast<uint64_t>(b.dh)+snapshot*4,uint64_t(t.HV)*t.K*t.V*4,
            uint32_t(t.T),start,count,t.scale,bool(t.useExp2)};
        RunDhuSequence<GradT>(workspace+uint64_t(block)*l.groupBytes,l,x);
    }
}
} // namespace GDN::Arch22
