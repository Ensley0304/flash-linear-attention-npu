#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkKdaBwdWy : public OpDef {
public:
    explicit ChunkKdaBwdWy(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> betaTypes = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("q").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v_new").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("gk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("Akk").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("h").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("dh").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("dv_scan").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("dq_base").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dk_base").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("db_base").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg_base").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dAkk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("stage").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_wy")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(ChunkKdaBwdWy);
} // namespace ops
