// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "products.h"
#ifdef __aicore__
#define KDA_SCORE_FN __aicore__ inline
#else
#define KDA_SCORE_FN inline
#endif
namespace KDA::Arch22::Pipeline64 {
// One safe (query band, key band, K-column panel). Scalar beta is applied to
// dL rows, never recovered by division. Invalid causal pairs must be zeroed
// before these operands are published. Exponent factoring is an upstream
// numerical decision, not an unconditional assumption made by this component.
struct ScoreInputs {
    Matrix ds,dlBeta,qPlus,kPlus,kMinus;
    uint32_t queryRows,keyRows,columns;
};
enum class ScoreProduct : uint32_t { QPlusGrad,QkMinusGrad,KPlusGrad,KkMinusGrad,Rkk };
KDA_SCORE_FN Product DescribeScore(ScoreProduct kind,const Layout &l,uint64_t record,
                                   const ScoreInputs &x) {
    const Matrix q{record+l.Offset(Score::Query),C,1};
    const Matrix k{record+l.Offset(Score::Key),C,1};
    // F3 has finished consuming dA/middle before F6 reuses these two slots.
    const Matrix kp{record+l.Offset(Score::Da),C,1};
    const Matrix km{record+l.Offset(Score::Middle),C,1};
    const Matrix rkk{record+l.Offset(Score::Rkk),C,1};
    switch(kind) {
    case ScoreProduct::QPlusGrad:
        return {x.ds,x.kMinus,q,x.queryRows,x.columns,x.keyRows,Epilogue::Assign};
    case ScoreProduct::QkMinusGrad:
        return {x.ds.T(),x.qPlus,k,x.keyRows,x.columns,x.queryRows,Epilogue::Assign};
    case ScoreProduct::KPlusGrad:
        return {x.dlBeta,x.kMinus,kp,x.queryRows,x.columns,x.keyRows,Epilogue::Assign};
    case ScoreProduct::KkMinusGrad:
        return {x.dlBeta.T(),x.kPlus,km,x.keyRows,x.columns,x.queryRows,Epilogue::Assign};
    case ScoreProduct::Rkk:
        return {x.kPlus,x.kMinus.T(),rkk,x.queryRows,x.keyRows,x.columns,Epilogue::Assign};
    }
    return {};
}
} // namespace KDA::Arch22::Pipeline64
#undef KDA_SCORE_FN
