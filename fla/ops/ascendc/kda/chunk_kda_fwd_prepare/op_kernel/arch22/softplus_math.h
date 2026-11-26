// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace KDA::Arch22::Math {
// log(1+t) loses small representable t when 1+t rounds to one. Keep the
// design's six-term log1p approximation for t<=1/16; the backward contract
// remains the ideal softplus derivative, not the polynomial derivative.
__aicore__ inline void CorrectSmallLog1p(AscendC::LocalTensor<float> out,
    AscendC::LocalTensor<float> t,uint32_t count) {
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for(uint32_t i=0;i<count;++i){
        float x=t.GetValue(i);
        if(x<=0.0625F)out.SetValue(i,x*(1.0F+x*(-0.5F+x*(1.0F/3.0F+x*(-0.25F+x*(0.2F-x/6.0F))))));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
}
}
