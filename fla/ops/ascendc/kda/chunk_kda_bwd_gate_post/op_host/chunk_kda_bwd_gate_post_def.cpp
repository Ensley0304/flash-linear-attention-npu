#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaBwdGatePost : public OpDef {
public:
    explicit ChunkKdaBwdGatePost(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND};
        this->Input("dg_hv").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_gate_post")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(ChunkKdaBwdGatePost);
} // namespace ops
