// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "contract.h"
#ifdef __aicore__
#define KDA_C64_FN __aicore__ inline
#else
#define KDA_C64_FN inline
#endif
namespace KDA::Arch22::Pipeline64 {
// Byte address and FP32 element strides. Both input state layouts are views;
// no full-state transpose or cast is inserted between stages.
struct Matrix {
    uint64_t address;
    uint32_t rowStride,colStride;
    KDA_C64_FN Matrix T() const { return {address,colStride,rowStride}; }
};
struct Inputs { Matrix du,h,dnext,vnew,akk; };
enum class ProductKind : uint32_t { Dw,Dkg,Drv,DaKey,DaValue,Middle,Dl,Drk };
enum class Epilogue : uint32_t { Assign,Negate,Add,NegateStrictLower,ScaleLower };
struct Product { Matrix a,b,out; uint32_t m,n,k; Epilogue epilogue; float scale=1.f; uint64_t addSource=0; };

KDA_C64_FN Product Describe(ProductKind kind,const Layout &l,uint64_t record,
                             uint32_t rows,const Inputs &x) {
    const Matrix rk{record+l.Offset(Key::Rk),l.kp,1};
    const Matrix dw{record+l.Offset(Key::Dw),l.kp,1};
    const Matrix dkg{record+l.Offset(Key::Dkg),l.kp,1};
    const Matrix drk{record+l.Offset(Key::Drk),l.kp,1};
    const Matrix rv{record+l.Offset(Value::Rv),l.vp,1};
    const Matrix drv{record+l.Offset(Value::Drv),l.vp,1};
    const Matrix da{record+l.Offset(Score::Da),C,1};
    const Matrix middle{record+l.Offset(Score::Middle),C,1};
    const Matrix dl{record+l.Offset(Score::Dl),C,1};
    switch(kind) {
    case ProductKind::Dw: return {x.du,x.h.T(),dw,rows,l.k,l.v,Epilogue::Negate};
    case ProductKind::Dkg: return {x.vnew,x.dnext.T(),dkg,rows,l.k,l.v,Epilogue::Assign};
    case ProductKind::Drv: return {x.akk.T(),x.du,drv,rows,l.v,rows,Epilogue::Assign};
    case ProductKind::DaKey: return {dw,rk.T(),da,rows,rows,l.k,Epilogue::Assign};
    case ProductKind::DaValue: return {x.du,rv.T(),da,rows,rows,l.v,Epilogue::Add};
    case ProductKind::Middle: return {x.akk.T(),da,middle,rows,rows,rows,Epilogue::Assign};
    case ProductKind::Dl: return {middle,x.akk.T(),dl,rows,rows,rows,Epilogue::NegateStrictLower};
    case ProductKind::Drk: return {x.akk.T(),dw,drk,rows,l.k,rows,Epilogue::Assign};
    }
    return {};
}
// Producer must apply the epilogue before publishing READY. In particular,
// dL cannot become a score-gradient input before strict-triangular masking.
KDA_C64_FN bool ValidOutput(Epilogue e,uint32_t row,uint32_t col) {
    return e==Epilogue::NegateStrictLower?col<row:e==Epilogue::ScaleLower?col<=row:true;
}
} // namespace KDA::Arch22::Pipeline64
#undef KDA_C64_FN
