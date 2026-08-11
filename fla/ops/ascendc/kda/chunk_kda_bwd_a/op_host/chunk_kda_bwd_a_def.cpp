#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaBwdA : public OpDef {
public:
    explicit ChunkKdaBwdA(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> int64Types = {
            ge::DT_INT64, ge::DT_INT64
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("Aqk").ParamType(REQUIRED).DataType(dataTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("v_new").ParamType(REQUIRED).DataType(dataTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("h").ParamType(REQUIRED).DataType(dataTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("d_o").ParamType(REQUIRED).DataType(dataTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64Types).Format(formats).UnknownShapeFormat(formats);

        this->Output("dv0").ParamType(REQUIRED).DataType(dataTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("dq_raw").ParamType(REQUIRED).DataType(fp32Types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("dAqk").ParamType(REQUIRED).DataType(fp32Types)
            .Format(formats).UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_a")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdA);
} // namespace ops
