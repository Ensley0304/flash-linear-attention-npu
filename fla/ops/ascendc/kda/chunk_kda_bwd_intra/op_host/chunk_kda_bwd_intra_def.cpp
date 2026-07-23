/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaBwdIntra : public OpDef {
public:
    explicit ChunkKdaBwdIntra(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> qkTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32Types = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> indexTypes = {ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(qkTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("k").ParamType(REQUIRED).DataType(qkTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("g").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("beta").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("dAqk").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("dAkk").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("dq").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("dk").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("db").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("dg").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats);
        // Internal tensors carry the split BF16/safe left-Cube pipeline:
        // stage 1 packs A/B, stage 2 computes C, and stage 3 consumes C.
        // Keeping these as separate launches avoids AIC/AIV cross-core flags.
        this->Input("stage_a").ParamType(OPTIONAL).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_b").ParamType(OPTIONAL).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_c").ParamType(OPTIONAL).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);

        this->Output("dq_out").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Output("dk_out").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Output("db_out").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);
        this->Output("dg_out").ParamType(REQUIRED).DataType(fp32Types).Format(formats).UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(false);
        this->Attr("total_chunks").AttrType(REQUIRED).Int(1);
        this->Attr("stage").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdIntra);
} // namespace ops
