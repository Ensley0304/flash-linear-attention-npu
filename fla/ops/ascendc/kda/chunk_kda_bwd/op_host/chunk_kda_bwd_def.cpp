#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkKdaBwd : public OpDef {
public:
    explicit ChunkKdaBwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> betaTypes = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> gateTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("gk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("Aqk").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("Akk").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("w").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("qg").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("kg").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v_new").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("h").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("raw_g").ParamType(OPTIONAL).DataType(gateTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("a_log").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("initial_state").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dht").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);

        this->Output("dq").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("db").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dh0").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dA").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dbias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(true);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0f);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(false);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);
        this->Attr("recompute_policy").AttrType(OPTIONAL).String("NONE");
    }
};

OP_ADD(ChunkKdaBwd);
} // namespace ops
