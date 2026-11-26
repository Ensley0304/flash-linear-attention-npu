// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#include "../../../../chunk_kda_fwd_prepare/op_kernel/arch22/softplus_math.h"
namespace KDA::Arch22::Pipeline64 {
using FloatVector=AscendC::LocalTensor<float>;
// These functions operate on already accumulated gradients. They deliberately
// do not own cumsum, parameter reduction or the two-AIV norm projection.
// The caller must finish those dependencies before invoking the epilogue.
__aicore__ inline void BetaDerivative(FloatVector out,FloatVector gradient,
                                     FloatVector beta,FloatVector tmp,uint32_t mode,uint32_t n) {
    using namespace AscendC;
    if(mode==0) { Adds(out,gradient,0.0f,n);return; }
    Muls(tmp,beta,mode==2?-0.5f:-1.0f,n);PipeBarrier<PIPE_V>();
    Adds(tmp,tmp,1.0f,n);PipeBarrier<PIPE_V>();
    Mul(tmp,tmp,beta,n);PipeBarrier<PIPE_V>();
    Mul(out,gradient,tmp,n);
}
__aicore__ inline void NormDerivative(FloatVector out,FloatVector gradient,
                                     FloatVector normalized,FloatVector tmp,
                                     float projection,float rstd,bool enabled,uint32_t n) {
    using namespace AscendC;
    if(!enabled) { Adds(out,gradient,0.0f,n);return; }
    Muls(tmp,normalized,projection,n);PipeBarrier<PIPE_V>();
    Sub(out,gradient,tmp,n);PipeBarrier<PIPE_V>();
    Muls(out,out,rstd,n);
}
// x = raw_gate + bias, a = exp(A_log) or 1. The gate gradient entering this
// function is in natural-log step units, after the per-chunk reverse cumsum.
// daTerm is per-element dA_log, reduced by the caller without an atomic.
__aicore__ inline void GateDerivative(FloatVector dg,FloatVector daTerm,
    FloatVector deltaGradient,FloatVector x,FloatVector t,FloatVector denom,FloatVector aux,
    float a,float lowerBound,uint32_t mode,uint32_t n) {
    using namespace AscendC;
    if(mode==0) {
        Adds(dg,deltaGradient,0.0f,n);Duplicate(daTerm,0.0f,n);return;
    }
    if(mode==2) {
        Muls(t,x,a,n);PipeBarrier<PIPE_V>();Abs(t,t,n);PipeBarrier<PIPE_V>();
        Muls(t,t,-1.0f,n);PipeBarrier<PIPE_V>();Exp(t,t,n);PipeBarrier<PIPE_V>();
        Adds(denom,t,1.0f,n);PipeBarrier<PIPE_V>();Mul(denom,denom,denom,n);PipeBarrier<PIPE_V>();
        Div(t,t,denom,n);PipeBarrier<PIPE_V>();Muls(t,t,lowerBound*a,n);PipeBarrier<PIPE_V>();
        Mul(dg,deltaGradient,t,n);PipeBarrier<PIPE_V>();Mul(daTerm,dg,x,n);
    } else {
        Abs(t,x,n);PipeBarrier<PIPE_V>();Muls(t,t,-1.0f,n);PipeBarrier<PIPE_V>();
        Exp(t,t,n);PipeBarrier<PIPE_V>();Adds(denom,t,1.0f,n);PipeBarrier<PIPE_V>();
        Ln(aux,denom,n);PipeBarrier<PIPE_V>();
        KDA::Arch22::Math::CorrectSmallLog1p(aux,t,n);
        Maxs(t,x,0.0f,n);PipeBarrier<PIPE_V>();Add(aux,aux,t,n);PipeBarrier<PIPE_V>();
        Muls(aux,aux,-a,n);PipeBarrier<PIPE_V>();Mul(daTerm,deltaGradient,aux,n);
        Mins(t,x,0.0f,n);PipeBarrier<PIPE_V>();Exp(t,t,n);PipeBarrier<PIPE_V>();
        Div(t,t,denom,n);PipeBarrier<PIPE_V>();Muls(t,t,-a,n);PipeBarrier<PIPE_V>();
        Mul(dg,deltaGradient,t,n);
    }
}
} // namespace KDA::Arch22::Pipeline64
