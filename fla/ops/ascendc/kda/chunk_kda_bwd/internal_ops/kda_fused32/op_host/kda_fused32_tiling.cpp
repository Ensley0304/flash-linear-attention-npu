// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "../../../op_kernel/arch22/fused32/tiling.h"

namespace optiling {
using namespace KDA::Arch22::Fused32;
static ge::graphStatus TilingKdaFused32(gert::TilingContext *ctx) {
    auto fail=[&](const char *param,const char *reason){
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("KdaFused32",param,reason);return ge::GRAPH_FAILED;
    };
    auto attrs=ctx->GetAttrs();if(!attrs)return fail("attrs","missing attributes");
    auto dims=attrs->GetListInt(0);
    auto scale=attrs->GetFloat(1),eps=attrs->GetFloat(2),lower=attrs->GetFloat(3);
    auto backward=attrs->GetBool(4);
    if(!dims || dims->GetSize()!=17 || !scale || !eps || !lower || !backward)
        return fail("config","expected 17 integer fields and three scalar attributes");
    Tiling t{};auto d=dims->GetData();
    uint32_t fields[15];
    for(size_t i=0;i<15;++i){
        if(d[i]<0 || uint64_t(d[i])>std::numeric_limits<uint32_t>::max())return fail("config","field exceeds uint32 range");
        fields[i]=uint32_t(d[i]);
    }
    std::memcpy(&t.config,fields,sizeof(fields));
    auto &c=t.config;c.scale=*scale;c.epsilon=*eps;c.lowerBound=*lower;
    if(d[15]<0 || d[15]>1 || d[16]<0 || d[16]>1)return fail("layout flags","must be boolean");
    c.gateTimeMajor=uint32_t(d[15]);c.betaTimeMajor=uint32_t(d[16]);
    if(!c.batch || !c.heads || !c.tokens || !c.sequences || !c.chunks ||
       !((c.keys==c.values && (c.keys==60 || c.keys==64 || c.keys==100 || c.keys==128 || c.keys==256)) ||
         (c.keys==64 && c.values==32)))return fail("dimensions","outside C32 general domain");
    if(c.packed>1 || c.timeMajor>1 || c.stateVFirst>1 || c.normalize>1 || c.betaMode>2 || c.gateMode>2 || c.useExp2>1 || c.recompute>1)
        return fail("flags","invalid internal flags");
    if(!std::isfinite(c.scale) || !std::isfinite(c.epsilon) || c.epsilon<=0 ||
       (c.gateMode==2 && (!std::isfinite(c.lowerBound) || c.lowerBound < -5 || c.lowerBound>=0)))
        return fail("scalars","invalid scale, epsilon or safe-gate bound");
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint64_t ub=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if(ub<116224)return fail("UB","C32 owner requires at least 116224 bytes");
    t.blocks=std::min(c.heads,std::max(uint32_t(*backward?platform.GetCoreNumAic():platform.GetCoreNumAiv()),1U));
    t.backward=*backward;t.workspace=MakeWorkspace(c,*backward);
    auto outputMask=attrs->GetInt(5);
    if(!outputMask || *outputMask<0 || *outputMask>0xffffff)return fail("output_mask","invalid output presence mask");
    t.outputMask=uint32_t(*outputMask);
    const uint64_t system=platform.GetLibApiWorkSpaceSize();
    if(t.workspace.bytes>(std::numeric_limits<uint64_t>::max()-system)/t.blocks)
        return fail("workspace","total workspace overflow");
    if(ctx->GetRawTilingData()->GetCapacity()<sizeof(t))return fail("tiling","insufficient tiling capacity");
    std::memcpy(ctx->GetRawTilingData()->GetData(),&t,sizeof(t));
    ctx->GetRawTilingData()->SetDataSize(sizeof(t));
    ctx->SetBlockDim(t.blocks);ctx->SetTilingKey(*backward?1:0);
    ctx->GetWorkspaceSizes(1)[0]=system+t.workspace.bytes*t.blocks;
    return ge::GRAPH_SUCCESS;
}
struct KdaFused32CompileInfo {};
static ge::graphStatus ParseKdaFused32(gert::TilingParseContext*){return ge::GRAPH_SUCCESS;}
IMPL_OP_OPTILING(KdaFused32).Tiling(TilingKdaFused32).TilingParse<KdaFused32CompileInfo>(ParseKdaFused32);
}
