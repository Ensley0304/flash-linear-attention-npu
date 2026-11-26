// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "kda_bwd_gate_normalize.h"
#include <algorithm>
#include <vector>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(KdaBwdGateNormalize);
const aclTensor *KdaBwdGateNormalize(const aclTensor *raw,const aclTensor *a,const aclTensor *bias,
    const aclIntArray *cu,bool useExp2,bool safeGate,double lower,const aclTensor *out,aclOpExecutor *executor){
    if(!raw || !out || !executor || !IsContiguous(raw))return nullptr;
    const auto &s=raw->GetViewShape();auto rank=s.GetDimNum();if(rank!=3 && rank!=4)return nullptr;
    const int64_t B=rank==4?s.GetDim(0):1,H=s.GetDim(rank-3),T=s.GetDim(rank-2),K=s.GetDim(rank-1);
    std::vector<int64_t> records;
    auto append=[&](int64_t b,int64_t begin,int64_t end){for(auto p=begin;p<end;p+=64)records.insert(records.end(),{b,p,std::min(int64_t(64),end-p)});};
    if(cu){for(size_t n=0;n+1<cu->Size();++n)append(0,(*cu)[n],(*cu)[n+1]);}
    else for(int64_t b=0;b<B;++b)append(b,0,T);
    int64_t dims[]={B,H,T,K,int64_t(records.size()/3)};
    auto metadata=executor->ConvertToTensor(records.data(),records.size(),DataType::DT_INT64);
    auto dimensions=executor->AllocIntArray(dims,5);if(!metadata || !dimensions)return nullptr;
    auto m=const_cast<aclTensor*>(metadata);m->SetStorageShape(metadata->GetViewShape());m->SetOriginalShape(metadata->GetViewShape());
    m->SetStorageFormat(Format::FORMAT_ND);m->SetViewFormat(Format::FORMAT_ND);m->SetOriginalFormat(Format::FORMAT_ND);
    L0_DFX(KdaBwdGateNormalize,raw,a,bias,cu,out);
    auto status=ADD_TO_LAUNCHER_LIST_AICORE(KdaBwdGateNormalize,OP_INPUT(raw,a,bias,metadata),OP_OUTPUT(out),
        OP_ATTR(dimensions,useExp2,safeGate,static_cast<float>(lower)));
    return status==ACLNN_SUCCESS?out:nullptr;
}
}
