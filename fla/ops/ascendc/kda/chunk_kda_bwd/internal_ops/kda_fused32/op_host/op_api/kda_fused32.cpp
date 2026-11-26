// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "kda_fused32.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_common/log/log.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(KdaFused32);
namespace {
op::Shape Shape(std::initializer_list<int64_t> dims){op::Shape s;for(auto d:dims)s.AppendDim(d);return s;}
bool Match(const aclTensor *t,const op::Shape &s,DataType d){return t && IsContiguous(t) && t->GetViewShape()==s && t->GetDataType()==d;}
void Meta(const aclTensor *t){
    if(!t)return;
    auto m=const_cast<aclTensor*>(t);m->SetStorageShape(t->GetViewShape());m->SetOriginalShape(t->GetViewShape());
    m->SetStorageFormat(Format::FORMAT_ND);m->SetViewFormat(Format::FORMAT_ND);m->SetOriginalFormat(Format::FORMAT_ND);
}
}
aclnnStatus KdaFused32(KdaFused32Request &r,aclOpExecutor *executor){
    auto fail=[](const char *name,const char *why){OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("KdaFused32",name,why);return ACLNN_ERR_PARAM_INVALID;};
    auto &i=r.inputs;auto &o=r.outputs;auto &c=r.config;
    if(GetCurrentPlatformInfo().GetCurNpuArch()!=NpuArch::DAV_2201)return fail("soc","C32 FP32 owner requires A2/A3");
    if(!executor || !r.layout || !i[2] || !i[3] || !i[4])return fail("inputs","v/g/beta/layout/executor are required");
    const bool packed=!std::strcmp(r.layout,"NTD") || !std::strcmp(r.layout,"TND");
    const bool tm=!std::strcmp(r.layout,"TND") || !std::strcmp(r.layout,"BSND");
    if(!packed && std::strcmp(r.layout,"BNSD") && std::strcmp(r.layout,"BSND"))return fail("layout","expected BSND/BNSD/TND/NTD");
    const size_t rank=packed?3:4,axis=packed?0:1;
    auto vshape=i[2]->GetViewShape();
    auto keySource=r.backward && c.normalize?i[18]:i[0];
    if(!keySource || vshape.GetDimNum()!=rank || keySource->GetViewShape().GetDimNum()!=rank)return fail("rank","input rank/layout mismatch");
    int64_t B=packed?1:vshape.GetDim(0),H=vshape.GetDim(axis+(tm?1:0)),T=vshape.GetDim(axis+(tm?0:1));
    int64_t K=keySource->GetViewShape().GetDim(rank-1),V=vshape.GetDim(rank-1);
    if(B<=0 || H<=0 || T<=0 || B>UINT32_MAX || H>UINT32_MAX || T>UINT32_MAX ||
       !((K==V && (K==60 || K==64 || K==100 || K==128 || K==256)) || (K==64 && V==32)))return fail("dimensions","unsupported dimensions or head counts");
    if(uint64_t(B)>uint64_t(INT64_MAX)/H/T/256/16)return fail("dimensions","tensor indexing overflow");
    c.batch=B;c.heads=H;c.tokens=T;c.keys=K;c.values=V;c.packed=packed;c.timeMajor=tm;
    if(!r.backward){c.gateTimeMajor=tm;c.betaTimeMajor=tm;}
    if(c.normalize>1 || c.betaMode>2 || c.gateMode>2 || c.useExp2>1 || c.recompute>1 || c.stateVFirst>1 ||
       !std::isfinite(c.scale) || !std::isfinite(c.epsilon) || c.epsilon<=0 ||
       (c.gateMode==2 && (!std::isfinite(c.lowerBound) || c.lowerBound < -5 || c.lowerBound>=0)))return fail("flags/scalars","invalid flag, scale, epsilon or gate bound");
    auto data=i[2]->GetDataType(),gate=i[3]->GetDataType(),beta=i[4]->GetDataType();
    if((data!=DataType::DT_FLOAT16 && data!=DataType::DT_BF16) ||
       (gate!=data && gate!=DataType::DT_FLOAT) || (beta!=data && beta!=DataType::DT_FLOAT))return fail("dtype","requires FP16/BF16 qkv and matching or FP32 gate/beta");
    auto key=packed?Shape({H,T,K}):Shape({B,H,T,K}),value=packed?Shape({H,T,V}):Shape({B,H,T,V});
    auto keyT=packed?Shape({T,H,K}):Shape({B,T,H,K}),valueT=packed?Shape({T,H,V}):Shape({B,T,H,V});
    auto scalar=packed?Shape({H,T}):Shape({B,H,T}),scalarT=packed?Shape({T,H}):Shape({B,T,H});
    auto score=packed?Shape({H,T,32}):Shape({B,H,T,32});
    auto fp=DataType::DT_FLOAT;
    if(!Match(i[2],tm?valueT:value,data) || !Match(i[4],c.betaTimeMajor?scalarT:scalar,beta))return fail("v/beta","shape, layout or contiguity mismatch");
    if((!r.backward || c.gateMode!=0) && !Match(i[3],c.gateTimeMajor?keyT:key,gate))return fail("g","raw gate layout/dtype mismatch");
    if(!r.backward || !c.normalize){
        if(!Match(i[0],tm?keyT:key,data) || !Match(i[1],tm?keyT:key,data))return fail("q/k","q/k must match v dtype, layout and heads");
    }
    if(i[5] && !Match(i[5],Shape({H}),fp))return fail("a_log","requires FP32 [H]");
    if(i[6] && (!IsContiguous(i[6]) || i[6]->GetDataType()!=fp || i[6]->GetViewShape().GetShapeSize()!=H*K))return fail("bias","requires contiguous FP32 H*K");
    std::vector<int64_t> seq,canonical;
    int64_t Q=0;
    auto add=[&](int64_t id,int64_t start,int64_t end){
        seq.insert(seq.end(),{start,end-start,Q});
        for(int64_t n=0;n<(end-start+31)/32;++n){canonical.push_back(id);canonical.push_back(n);++Q;}
    };
    if(r.cu){
        if(B!=1 || r.cu->Size()<2 || (*r.cu)[0]!=0 || (*r.cu)[r.cu->Size()-1]!=T)return fail("cu_seqlens","expected B=1 and boundaries 0..T");
        for(size_t s=0;s+1<r.cu->Size();++s){
            if((*r.cu)[s]<0 || (*r.cu)[s+1]<(*r.cu)[s] || (*r.cu)[s+1]>T)return fail("cu_seqlens","boundaries must be nondecreasing");
            add(s,(*r.cu)[s],(*r.cu)[s+1]);
        }
    }else{
        if(packed || r.chunkIndices)return fail("metadata","packed input requires cu_seqlens; chunk_indices requires cu_seqlens");
        for(int64_t b=0;b<B;++b)add(b,b*T,(b+1)*T);
    }
    if(r.chunkIndices){
        if(r.chunkIndices->Size()!=canonical.size())return fail("chunk_indices","wrong count");
        for(size_t n=0;n<canonical.size();++n)if((*r.chunkIndices)[n]!=canonical[n])return fail("chunk_indices","requires canonical sequence-major pairs");
    }
    if(Q>UINT32_MAX || seq.size()/3>UINT32_MAX)return fail("metadata","too many chunks/sequences");
    c.chunks=Q;c.sequences=seq.size()/3;
    auto state=c.stateVFirst?Shape({c.sequences,H,V,K}):Shape({c.sequences,H,K,V});
    auto tape=packed?(c.stateVFirst?Shape({Q,H,V,K}):Shape({Q,H,K,V})):
        (c.stateVFirst?Shape({B,(T+31)/32,H,V,K}):Shape({B,(T+31)/32,H,K,V}));
    // Dense B=1 plus explicit packed boundaries retains the existing rank-5 state cache.
    if(!packed && r.cu)tape=c.stateVFirst?Shape({1,Q,H,V,K}):Shape({1,Q,H,K,V});
    for(auto n:{7,8})if(i[n] && !Match(i[n],state,fp))return fail("state input","requires FP32 [S,H,K,V] or [S,H,V,K]");
    for(auto n:{1,7})if(o[n] && !Match(o[n],state,fp))return fail("state output","state shape/dtype mismatch");
    if(o[7] && !i[7])return fail("dh0","requires initial_state");
    for(auto dst:{o[1],o[7]})for(auto src:{i[7],i[8]}){
        if(dst && src && dst->GetStorageAddr() && dst->GetStorageAddr()==src->GetStorageAddr())
            return fail("state alias","state output must use storage distinct from h0/dht");
    }
    if(r.backward){
        if(!Match(i[9],valueT,data) || !Match(i[11],score,fp) || !Match(i[12],score,fp))return fail("backward inputs","dO must be token-major; scores must be C32 FP32");
        if(c.normalize && (!Match(i[18],key,fp) || !Match(i[19],key,fp) || !Match(i[20],scalar,fp) || !Match(i[21],scalar,fp)))return fail("normalization","requires saved FP32 qhat/khat/rstd");
        if(c.betaMode && !Match(i[22],scalar,fp))return fail("beta_eff","sigmoid backward requires saved effective beta");
        if((!c.recompute || c.gateMode==0) && !Match(i[10],key,fp))return fail("G","saved cumulative FP32 G required");
        if(!c.recompute){
            for(auto n:{13,14,15})if(!Match(i[n],key,fp))return fail("saved key cache","requires FP32 W/qg/kg");
            if(!Match(i[16],value,fp) || !Match(i[17],tape,fp))return fail("saved state cache","requires FP32 Vnew/h");
        }
        if(!Match(o[2],tm?keyT:key,data) || !Match(o[3],tm?keyT:key,data) || !Match(o[4],tm?valueT:value,data) ||
           !Match(o[5],c.gateTimeMajor?keyT:key,gate) || !Match(o[6],c.betaTimeMajor?scalarT:scalar,beta))return fail("gradients","input-gradient shape/dtype mismatch");
        if(o[8] && (!i[5] || !Match(o[8],Shape({H}),fp)))return fail("dA","requires A_log and FP32 [H]");
        if(o[9] && (!i[6] || !IsContiguous(o[9]) || o[9]->GetDataType()!=fp || o[9]->GetViewShape().GetShapeSize()!=H*K))return fail("dbias","requires bias and FP32 H*K");
    }else{
        if(!Match(o[0],valueT,data))return fail("o","requires token-major original dtype");
        for(auto n:{10,13,14,15,18,19})if(o[n] && !Match(o[n],key,fp))return fail("key cache","requires head-major FP32");
        for(auto n:{11,12})if(o[n] && !Match(o[n],score,fp))return fail("score cache","requires C32 FP32");
        for(auto n:{16,23})if(o[n] && !Match(o[n],value,fp))return fail("value cache","requires head-major FP32");
        for(auto n:{20,21,22})if(o[n] && !Match(o[n],scalar,fp))return fail("scalar cache","requires head-major FP32");
        if(o[17] && !Match(o[17],tape,fp))return fail("h","FP32 snapshot shape/layout mismatch");
    }
    i[23]=executor->ConvertToTensor(seq.data(),seq.size(),DataType::DT_INT64);
    if(!i[23])return ACLNN_ERR_INNER_NULLPTR;
    std::vector<int64_t> fields={c.batch,c.heads,c.tokens,c.keys,c.values,c.sequences,c.chunks,c.packed,c.timeMajor,c.stateVFirst,
        c.normalize,c.betaMode,c.gateMode,c.useExp2,c.recompute,c.gateTimeMajor,c.betaTimeMajor};
    auto config=executor->AllocIntArray(fields.data(),fields.size());if(!config)return ACLNN_ERR_INNER_NULLPTR;
    for(auto x:i)Meta(x);
    for(auto x:o)Meta(x);
    // The executor compacts null outputs. Preserve all device argument slots
    // using zero-sized descriptors; the mask prevents any dummy-buffer access.
    int64_t outputMask=0;
    for(size_t n=0;n<o.size();++n){
        if(o[n]){outputMask|=int64_t(1)<<n;continue;}
        const auto type=(n==0 || n==2 || n==3 || n==4)?data:(n==5?gate:(n==6?beta:fp));
        o[n]=executor->AllocTensor(Shape({0}),type,Format::FORMAT_ND);
        if(!o[n])return ACLNN_ERR_INNER_NULLPTR;
    }
    L0_DFX(KdaFused32,i[0],i[1],i[2],i[3],i[4],config,r.backward);
    return ADD_TO_LAUNCHER_LIST_AICORE(KdaFused32,
        OP_INPUT(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18],i[19],i[20],i[21],i[22],i[23]),
        OP_OUTPUT(o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],o[15],o[16],o[17],o[18],o[19],o[20],o[21],o[22],o[23]),
        OP_ATTR(config,c.scale,c.epsilon,c.lowerBound,r.backward,outputMask));
}
}
