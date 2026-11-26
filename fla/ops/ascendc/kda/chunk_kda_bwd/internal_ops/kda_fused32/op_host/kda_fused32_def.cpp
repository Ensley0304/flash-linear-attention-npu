// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "register/op_def_registry.h"
namespace ops {
class KdaFused32 : public OpDef {
public:
    explicit KdaFused32(const char *name):OpDef(name){
        const std::initializer_list<ge::DataType> data={ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16};
        const std::initializer_list<ge::DataType> gate={ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_BF16,ge::DT_BF16};
        const std::initializer_list<ge::DataType> beta={ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_BF16,ge::DT_FLOAT,ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32={ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> i64={ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64};
        const std::initializer_list<ge::Format> nd={ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND};
        this->Input("q").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("g").ParamType(REQUIRED).DataType(gate).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(beta).Format(nd).UnknownShapeFormat(nd);
        this->Input("a_log").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("bias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("h0").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dht").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("g_cache").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("aqk").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("akk").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("w").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("qg").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("kg").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("v_new").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("h").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("qhat").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("khat").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("qr").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("kr").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta_eff").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("sequence_metadata").ParamType(REQUIRED).DataType(i64).Format(nd).UnknownShapeFormat(nd);
        this->Output("o").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Output("ht").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dq").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Output("dk").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg").ParamType(OPTIONAL).DataType(gate).Format(nd).UnknownShapeFormat(nd);
        this->Output("db").ParamType(OPTIONAL).DataType(beta).Format(nd).UnknownShapeFormat(nd);
        this->Output("dh0").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("d_a").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("d_bias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("g_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("aqk_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("akk_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("w_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("qg_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("kg_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("v_new_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("h_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("qhat_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("khat_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("qr_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("kr_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("beta_eff_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("u_out").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Attr("config").AttrType(REQUIRED).ListInt();
        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("epsilon").AttrType(REQUIRED).Float();
        this->Attr("lower_bound").AttrType(REQUIRED).Float();
        this->Attr("backward").AttrType(REQUIRED).Bool();
        this->Attr("output_mask").AttrType(REQUIRED).Int();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value","Opaque").ExtendCfgInfo("coreType.value","AiCore")
            .ExtendCfgInfo("aclnnSupport.value","support_aclnn");
        this->AICore().AddConfig("ascend910b",config).AddConfig("ascend910_93",config);
    }
};
OP_ADD(KdaFused32);
}
