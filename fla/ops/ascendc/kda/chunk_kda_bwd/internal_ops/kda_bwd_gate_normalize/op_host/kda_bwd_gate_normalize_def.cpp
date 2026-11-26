// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "register/op_def_registry.h"
namespace ops {
class KdaBwdGateNormalize : public OpDef {
public:
    explicit KdaBwdGateNormalize(const char *name):OpDef(name){
        this->Input("g").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        for(auto name:{"a_log","bias"})this->Input(name).ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("metadata").ParamType(REQUIRED).DataType({ge::DT_INT64,ge::DT_INT64,ge::DT_INT64}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("g_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("dimensions").AttrType(REQUIRED).ListInt();
        this->Attr("use_exp2").AttrType(REQUIRED).Bool();
        this->Attr("safe_gate").AttrType(REQUIRED).Bool();
        this->Attr("lower_bound").AttrType(REQUIRED).Float();
        OpAICoreConfig c;
        c.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
         .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
         .ExtendCfgInfo("prebuildPattern.value","Opaque").ExtendCfgInfo("coreType.value","AiCore")
         .ExtendCfgInfo("aclnnSupport.value","support_aclnn");
        this->AICore().AddConfig("ascend910b",c).AddConfig("ascend910_93",c);
    }
};
OP_ADD(KdaBwdGateNormalize);
}
