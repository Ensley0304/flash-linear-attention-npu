// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstring>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "../op_kernel/tiling.h"
namespace optiling {
static ge::graphStatus TileGateNormalize(gert::TilingContext *ctx){
    auto attrs=ctx->GetAttrs();if(!attrs)return ge::GRAPH_FAILED;
    auto dims=attrs->GetListInt(0);auto exp2=attrs->GetBool(1),safe=attrs->GetBool(2);auto lower=attrs->GetFloat(3);
    if(!dims || dims->GetSize()!=5 || !exp2 || !safe || !lower)return ge::GRAPH_FAILED;
    KDA::Arch22::GateNormalizeTiling t{};
    uint32_t fields[5];
    for(size_t i=0;i<5;++i){auto d=dims->GetData()[i];if(d<=0 || uint64_t(d)>UINT32_MAX)return ge::GRAPH_FAILED;fields[i]=uint32_t(d);}
    std::memcpy(&t,fields,sizeof(fields));t.useExp2=*exp2;t.safeGate=*safe;t.lowerBound=*lower;
    if(t.keys>256 || !std::isfinite(t.lowerBound))return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint64_t ub=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if(ub<45088 || ctx->GetRawTilingData()->GetCapacity()<sizeof(t))return ge::GRAPH_FAILED;
    std::memcpy(ctx->GetRawTilingData()->GetData(),&t,sizeof(t));ctx->GetRawTilingData()->SetDataSize(sizeof(t));
    ctx->SetBlockDim(std::min(uint64_t(t.chunks)*t.heads,uint64_t(std::max(uint32_t(platform.GetCoreNumAiv()),1U))));
    ctx->SetTilingKey(0);ctx->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();return ge::GRAPH_SUCCESS;
}
struct GateNormalizeCompileInfo {};
static ge::graphStatus ParseGateNormalize(gert::TilingParseContext*){return ge::GRAPH_SUCCESS;}
IMPL_OP_OPTILING(KdaBwdGateNormalize).Tiling(TileGateNormalize).TilingParse<GateNormalizeCompileInfo>(ParseGateNormalize);
}
