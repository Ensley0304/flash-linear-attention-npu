// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "contract.h"
#include "prepare.h"
#include "state.h"
#include "ff.h"
#include "bp.h"
#include "dhu.h"
#include "bf.h"
#include "cube_service.h"

namespace KDA::Arch22::Fused32 {
using Address = GM_ADDR;
struct Buffers {
    Address q,k,v,g,beta,aLog,bias,h0,dht,dO;
    Address G,aqk,akk,w,qg,kg,vn,h,qhat,khat,qr,kr,be;
    Address o,ht,dq,dk,dv,dg,db,dh0,dA,dbias;
    Address sequenceMetadata,workspace;
    Address uOut=nullptr;
};

// One AIV owns each complete head. All stages borrow the same queues/arena.
// No stage creates a pipe, launches another kernel, or waits for another owner.
template<class DataT,class GateT,class BetaT,bool MIXED=false>
class Owner {
    using F=AscendC::LocalTensor<float>;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> in_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> out_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> meta_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_,parameter_;
    F arena_,biasSum_,scalar_;
    Config c_; Workspace ws_; Buffers b_;
    Address base_;
    uint32_t head_,ownerIndex_,ownerCount_;
    float aSum_;
    __aicore__ inline Address P(uint64_t offset) { return base_+offset; }
    __aicore__ inline void Fence() {
        auto id=static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(id);
    }
    __aicore__ inline void ScalarReady() {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    template<class T>
    __aicore__ inline void Read(Address p,uint64_t offset,F row,uint32_t count) {
        Fence();
        AscendC::GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
        gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto x=in_.template AllocTensor<T>();
        AscendC::DataCopyExtParams cp{1,uint32_t(count*sizeof(T)),0,0,0};
        AscendC::DataCopyPadExtParams<T> pad{true,0,0,0};
        AscendC::DataCopyPad(x,gm[offset],cp,pad);in_.EnQue(x);x=in_.template DeQue<T>();
        if constexpr(sizeof(T)==4) AscendC::Adds(row,x,0.0F,count);
        else AscendC::Cast(row,x,AscendC::RoundMode::CAST_NONE,count);
        AscendC::PipeBarrier<PIPE_V>();in_.FreeTensor(x);
    }
    template<class T>
    __aicore__ inline void Write(Address p,uint64_t offset,F row,uint32_t count) {
        AscendC::GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
        gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto x=out_.template AllocTensor<T>();
        if constexpr(sizeof(T)==4) AscendC::Adds(x,row,0.0F,count);
        else AscendC::Cast(x,row,AscendC::RoundMode::CAST_RINT,count);
        out_.EnQue(x);x=out_.template DeQue<T>();
        AscendC::DataCopyExtParams cp{1,uint32_t(count*sizeof(T)),0,0,0};
        AscendC::DataCopyPad(gm[offset],x,cp);out_.FreeTensor(x);
    }
    template<class T=float>
    __aicore__ inline void Copy(Address dst,Address src,uint64_t from,uint32_t rows,uint32_t width,uint64_t stride) {
        for(uint32_t i=0;i<rows;++i){Read<T>(src,from+i*stride,arena_,width);Write<float>(dst,uint64_t(i)*width,arena_,width);}
    }
    __aicore__ inline void Export(Address dst,uint64_t to,Address src,uint32_t rows,uint32_t width) {
        if(!dst)return;
        for(uint32_t i=0;i<rows;++i){Read<float>(src,uint64_t(i)*width,arena_,width);Write<float>(dst,to+uint64_t(i)*width,arena_,width);}
    }
    __aicore__ inline void State(Address dst,Address src,uint64_t offset) {
        const uint32_t size=c_.keys*c_.values;
        for(uint32_t i=0;i<size;i+=256){
            const uint32_t n=size-i<256?size-i:256;
            if(src)Read<float>(src,offset+i,arena_,n);
            else {AscendC::Duplicate(arena_,0.0F,n);AscendC::PipeBarrier<PIPE_V>();}
            Write<float>(dst,i,arena_,n);
        }
    }
    __aicore__ inline uint64_t Cache(uint64_t token,uint32_t width) {
        const uint64_t batch=c_.packed?0:token/c_.tokens,t=c_.packed?token:token%c_.tokens;
        return ((batch*c_.heads+head_)*c_.tokens+t)*width;
    }
    __aicore__ inline uint64_t Input(uint64_t token,uint32_t width) {
        return c_.timeMajor?(token*c_.heads+head_)*width:Cache(token,width);
    }
    __aicore__ inline uint64_t GateInput(uint64_t token) {
        return c_.gateTimeMajor?(token*c_.heads+head_)*c_.keys:Cache(token,c_.keys);
    }
    __aicore__ inline uint64_t BetaInput(uint64_t token) {
        return c_.betaTimeMajor?token*c_.heads+head_:Cache(token,1);
    }
    __aicore__ inline Sequence GetSequence(uint32_t s) {
        AscendC::GlobalTensor<int64_t> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(b_.sequenceMetadata));
        auto x=meta_.template AllocTensor<int64_t>();
        AscendC::DataCopyExtParams cp{1,24,0,0,0};
        AscendC::DataCopyPadExtParams<int64_t> pad{true,0,0,0};
        AscendC::DataCopyPad(x,gm[s*3],cp,pad);meta_.EnQue(x);x=meta_.template DeQue<int64_t>();
        auto id=static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(id);
        Sequence seq{x.GetValue(0),x.GetValue(1),x.GetValue(2)};meta_.FreeTensor(x);return seq;
    }
    __aicore__ inline GeneralPrepareBuffers Preparation() {
        return {b_.q,b_.k,b_.v,b_.g,b_.beta,b_.aLog?b_.aLog+head_*4:nullptr,
            b_.bias?b_.bias+uint64_t(head_)*c_.keys*4:nullptr,P(ws_.g),P(ws_.aqk),P(ws_.akk),
            P(ws_.w),P(ws_.u),P(ws_.qg),P(ws_.kg),P(ws_.qgs),P(ws_.qhat),P(ws_.khat),P(ws_.qr),P(ws_.kr),P(ws_.be)};
    }
    __aicore__ inline void Prepare(uint64_t token,uint32_t m,bool backward) {
        const uint64_t ks=c_.timeMajor?uint64_t(c_.heads)*c_.keys:c_.keys;
        const uint64_t vs=c_.timeMajor?uint64_t(c_.heads)*c_.values:c_.values;
        auto buffers=Preparation();
        GeneralPrepareChunk d{m,c_.keys,c_.values,Input(token,c_.keys),Input(token,c_.values),
            BetaInput(token),ks,vs,c_.betaTimeMajor?c_.heads:1,0,0,0,0,c_.scale,c_.epsilon,c_.lowerBound,
            bool(c_.normalize),bool(c_.useExp2),c_.gateMode,c_.betaMode};
        d.gateOffset=GateInput(token);d.gateStride=c_.gateTimeMajor?uint64_t(c_.heads)*c_.keys:c_.keys;
        if(backward){
            if(c_.normalize){
                Copy(P(ws_.qhat),b_.qhat,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.khat),b_.khat,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.qr),b_.qr,Cache(token,1),m,1,1);Copy(P(ws_.kr),b_.kr,Cache(token,1),m,1,1);
            }else{
                Copy<DataT>(P(ws_.qhat),b_.q,Input(token,c_.keys),m,c_.keys,ks);
                Copy<DataT>(P(ws_.khat),b_.k,Input(token,c_.keys),m,c_.keys,ks);
            }
            if(c_.betaMode)Copy(P(ws_.be),b_.be,Cache(token,1),m,1,1);
            else Copy<BetaT>(P(ws_.be),b_.beta,BetaInput(token),m,1,c_.betaTimeMajor?c_.heads:1);
            Copy(P(ws_.akk),b_.akk,Cache(token,CHUNK),m,CHUNK,CHUNK);
            Copy(P(ws_.aqk),b_.aqk,Cache(token,CHUNK),m,CHUNK,CHUNK);
            if(!c_.recompute){
                Copy(P(ws_.g),b_.G,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.w),b_.w,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.qg),b_.qg,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.kg),b_.kg,Cache(token,c_.keys),m,c_.keys,c_.keys);
                Copy(P(ws_.vn),b_.vn,Cache(token,c_.values),m,c_.values,c_.values);
                return;
            }
            if(c_.gateMode==0){
                // Pre-activated backward receives cumulative G in its saved unit.
                buffers.gate=b_.G;d.gateMode=3;d.gateOffset=Cache(token,c_.keys);d.gateStride=c_.keys;
                GeneralPrepare<DataT,float,BetaT> stage;stage.Init(pipe_,arena_,in_,out_);stage.Recompute(buffers,d);return;
            }
        }
        GeneralPrepare<DataT,GateT,BetaT> stage;stage.Init(pipe_,arena_,in_,out_);
        if(backward)stage.Recompute(buffers,d);else stage.Process(buffers,d);
    }
    __aicore__ inline void ForwardState(uint32_t m,Address previous,Address next) {
        GeneralFwdH<float> stage;stage.Init(pipe_,arena_,in_,out_);
        // Save()/hTape already own the entrance snapshot; do not rewrite it.
        GeneralFwdHBuffers b{P(ws_.w),P(ws_.u),P(ws_.kg),P(ws_.g),previous,nullptr,P(ws_.vn),next};
        for(uint32_t v=0;v<c_.values;v+=32){
            GeneralFwdHSequence d{m,c_.keys,c_.values,v,c_.values-v<32?c_.values-v:32,0,0,0,0,
                uint64_t(c_.keys)*c_.values,bool(c_.useExp2),bool(c_.stateVFirst)};
            if(next)stage.template Process<true>(b,d);
            else stage.template Process<false>(b,d);
        }
    }
    __aicore__ inline void Save(uint64_t token,uint32_t m,uint64_t chunk,Address previous) {
        Export(b_.G,Cache(token,c_.keys),P(ws_.g),m,c_.keys);
        Export(b_.w,Cache(token,c_.keys),P(ws_.w),m,c_.keys);
        Export(b_.qg,Cache(token,c_.keys),P(ws_.qg),m,c_.keys);
        Export(b_.kg,Cache(token,c_.keys),P(ws_.kg),m,c_.keys);
        Export(b_.vn,Cache(token,c_.values),P(ws_.vn),m,c_.values);
        Export(b_.uOut,Cache(token,c_.values),P(ws_.u),m,c_.values);
        Export(b_.aqk,Cache(token,CHUNK),P(ws_.aqk),m,CHUNK);
        Export(b_.akk,Cache(token,CHUNK),P(ws_.akk),m,CHUNK);
        Export(b_.qhat,Cache(token,c_.keys),P(ws_.qhat),m,c_.keys);
        Export(b_.khat,Cache(token,c_.keys),P(ws_.khat),m,c_.keys);
        Export(b_.qr,Cache(token,1),P(ws_.qr),m,1);Export(b_.kr,Cache(token,1),P(ws_.kr),m,1);
        Export(b_.be,Cache(token,1),P(ws_.be),m,1);
        if(b_.h)State(b_.h+(chunk*c_.heads+head_)*c_.keys*c_.values*4,previous,0);
    }
    __aicore__ inline void ForwardOutput(uint64_t token,uint32_t m,Address previous) {
        GeneralFwdFinalize<DataT> stage;stage.Init(pipe_,arena_,in_,out_);
        GeneralFinalizeBuffers b{P(ws_.qgs),P(ws_.aqk),P(ws_.vn),previous,b_.o};
        for(uint32_t v=0;v<c_.values;v+=32){
            GeneralFinalizeChunk d{m,c_.keys,c_.values,v,c_.values-v<32?c_.values-v:32,0,0,0,0,
                (token*c_.heads+head_)*c_.values,uint64_t(c_.heads)*c_.values,bool(c_.stateVFirst)};stage.Process(b,d);
        }
    }
    __aicore__ inline void CubeCommand(uint32_t m,uint32_t n,uint32_t k,uint32_t mode) {
        AscendC::Duplicate(arena_,0.0F,8);AscendC::PipeBarrier<PIPE_V>();ScalarReady();
        arena_.SetValue(0,float(m));arena_.SetValue(1,float(n));
        arena_.SetValue(2,float(k));arena_.SetValue(3,float(mode));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Write<float>(P(ws_.cube),0,arena_,8);
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(3);
        AscendC::CrossCoreWaitFlag<0x2,PIPE_MTE2>(2);
    }
    template<AscendC::HardEvent E> __aicore__ inline void TransferFence() {
        const auto id=static_cast<event_t>(pipe_.FetchEventID(E));
        AscendC::SetFlag<E>(id);AscendC::WaitFlag<E>(id);
    }
    template<class T> __aicore__ inline void PackMatrix(Address dst,Address src,uint64_t offset,
        uint32_t rows,uint32_t cols,uint32_t paddedRows,uint32_t paddedCols,uint64_t stride) {
        const uint64_t sourceGap=(stride-cols)*sizeof(T);
        if(sourceGap>0xffffffffULL) {
            // Preserve the large-stride contract beyond DataCopyPad's field range.
            for(uint32_t i=0;i<paddedRows;++i){
                AscendC::Duplicate(arena_,0.0F,paddedCols);AscendC::PipeBarrier<PIPE_V>();
                if(i<rows)Read<T>(src,offset+uint64_t(i)*stride,arena_,cols);
                Write<float>(dst,uint64_t(i)*paddedCols,arena_,paddedCols);
            }
            return;
        }
        Fence();
        const uint32_t total=paddedRows*paddedCols;
        // FP32 packing uses 64 KiB. Low-precision input plus FP32 output uses
        // 32+64 KiB, disjoint inside the existing 110 KiB stage arena.
        const uint32_t clearFloats=total*sizeof(T)/4;
        for(uint32_t i=0;i<clearFloats;i+=8192)
            AscendC::Duplicate(arena_[i],0.0F,clearFloats-i<8192?clearFloats-i:8192);
        TransferFence<AscendC::HardEvent::V_MTE2>();
        auto input=arena_.template ReinterpretCast<T>();
        auto output=sizeof(T)==4?arena_:arena_[8192];
        AscendC::GlobalTensor<T> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        const uint32_t alignedBytes=(cols*sizeof(T)+31)/32*32;
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rows),uint32_t(cols*sizeof(T)),
            static_cast<uint32_t>(sourceGap),uint32_t((paddedCols*sizeof(T)-alignedBytes)/32),0};
        AscendC::DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(alignedBytes/sizeof(T)-cols),T(0)};
        AscendC::DataCopyPad(input,source[offset],cp,pad);
        TransferFence<AscendC::HardEvent::MTE2_V>();
        if constexpr(sizeof(T)!=4) {
            for(uint32_t i=0;i<total;i+=8192)
                AscendC::Cast(output[i],input[i],AscendC::RoundMode::CAST_NONE,total-i<8192?total-i:8192);
            TransferFence<AscendC::HardEvent::V_MTE3>();
        } else TransferFence<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::GlobalTensor<float> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
        target.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::DataCopyExtParams write{1,total*4,0,0,0};
        AscendC::DataCopyPad(target,output,write);
        TransferFence<AscendC::HardEvent::MTE3_V>();
    }

    template<class AT,class BT> __aicore__ inline void Gemm(Address a,Address b,Address dst,
        uint64_t ao,uint64_t bo,uint32_t m,uint32_t n,uint32_t k,
        uint64_t as,uint64_t bs,uint64_t ds,bool ta=false,bool tb=false) {
        Address pa=P(ws_.cube)+32,pb=pa+CUBE_INPUT_BYTES,pc=pb+CUBE_INPUT_BYTES;
        for(uint32_t mi=0;mi<m;mi+=32)for(uint32_t ni=0;ni<n;ni+=128){
            const uint32_t mr=m-mi<32?m-mi:32,nr=n-ni<128?n-ni:128;
            const uint32_t mp=(mr+15)/16*16,np=(nr+15)/16*16;
            for(uint32_t ki=0;ki<k;ki+=128){
                const uint32_t kr=k-ki<128?k-ki:128,kp=(kr+15)/16*16;
                if(ta)PackMatrix<AT>(pa,a,ao+uint64_t(ki)*as+mi,kr,mr,kp,mp,as);
                else PackMatrix<AT>(pa,a,ao+uint64_t(mi)*as+ki,mr,kr,mp,kp,as);
                if(tb)PackMatrix<BT>(pb,b,bo+uint64_t(ni)*bs+ki,nr,kr,np,kp,bs);
                else PackMatrix<BT>(pb,b,bo+uint64_t(ki)*bs+ni,kr,nr,kp,np,bs);
                CubeCommand(mp,np,kp,uint32_t(ta)+2U*uint32_t(tb));
                for(uint32_t i=0;i<mr;++i){
                    Read<float>(pc,uint64_t(i)*np,arena_,nr);
                    if(ki){
                        Read<float>(dst,uint64_t(mi+i)*ds+ni,arena_[256],nr);
                        AscendC::Add(arena_,arena_,arena_[256],nr);AscendC::PipeBarrier<PIPE_V>();
                    }
                    Write<float>(dst,uint64_t(mi+i)*ds+ni,arena_,nr);
                }
            }
        }
    }
    __aicore__ inline void CubeBwdPrepare(uint64_t token,uint32_t m,Address h) {
        const uint64_t doOffset=(token*c_.heads+head_)*c_.values;
        const uint64_t doStride=uint64_t(c_.heads)*c_.values;
        Gemm<DataT,float>(b_.dO,P(ws_.vn),P(ws_.ds),doOffset,0,m,m,c_.values,doStride,c_.values,32,false,true);
        for(uint32_t i=0;i<m;++i){
            Read<float>(P(ws_.ds),i*32,arena_,m);
            AscendC::Muls(arena_,arena_,c_.scale,i+1);AscendC::PipeBarrier<PIPE_V>();
            // dAqk is inclusive lower triangular; padded columns are always zero.
            ScalarReady();for(uint32_t j=i+1;j<32;++j)arena_.SetValue(j,0.0F);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Write<float>(P(ws_.ds),i*32,arena_,32);
        }
        Gemm<float,DataT>(P(ws_.aqk),b_.dO,P(ws_.dvl),0,doOffset,m,c_.values,m,32,doStride,c_.values,true,false);
        Gemm<DataT,float>(b_.dO,h,P(ws_.dqr),doOffset,0,m,c_.keys,c_.values,doStride,
            c_.stateVFirst?c_.keys:c_.values,c_.keys,false,!c_.stateVFirst);
    }
    __aicore__ inline void CubeDhu(uint64_t token,uint32_t m,Address next,Address prev) {
        const uint32_t K=c_.keys,V=c_.values;
        const uint64_t doOffset=(token*c_.heads+head_)*V,doStride=uint64_t(c_.heads)*V;
        Gemm<float,float>(P(ws_.kg),next,P(ws_.dvn),0,0,m,V,K,K,c_.stateVFirst?K:V,V,false,bool(c_.stateVFirst));
        for(uint32_t i=0;i<m;++i){
            Read<float>(P(ws_.dvn),uint64_t(i)*V,arena_,V);
            Read<float>(P(ws_.dvl),uint64_t(i)*V,arena_[256],V);
            AscendC::Add(arena_,arena_,arena_[256],V);AscendC::PipeBarrier<PIPE_V>();
            Write<float>(P(ws_.dvn),uint64_t(i)*V,arena_,V);
        }
        if(c_.stateVFirst)
            Gemm<DataT,float>(b_.dO,P(ws_.qg),prev,doOffset,0,V,K,m,doStride,K,K,true,false);
        else
            Gemm<float,DataT>(P(ws_.qg),b_.dO,prev,0,doOffset,K,V,m,K,doStride,V,true,false);
        Read<float>(P(ws_.g),uint64_t(m-1)*K,arena_[512],K);
        if(c_.useExp2){AscendC::Muls(arena_[512],arena_[512],0.6931471805599453F,K);AscendC::PipeBarrier<PIPE_V>();}
        AscendC::Exp(arena_[512],arena_[512],K);AscendC::PipeBarrier<PIPE_V>();
        const uint32_t rows=c_.stateVFirst?V:K,width=c_.stateVFirst?K:V;
        for(uint32_t i=0;i<rows;++i){
            Read<float>(next,uint64_t(i)*width,arena_,width);
            if(c_.stateVFirst)AscendC::Mul(arena_,arena_,arena_[512],width);
            else {ScalarReady();const float decay=arena_[512].GetValue(i);AscendC::Muls(arena_,arena_,decay,width);}
            AscendC::PipeBarrier<PIPE_V>();
            Read<float>(prev,uint64_t(i)*width,arena_[256],width);
            AscendC::Muls(arena_[256],arena_[256],c_.scale,width);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(arena_,arena_,arena_[256],width);AscendC::PipeBarrier<PIPE_V>();
            Write<float>(prev,uint64_t(i)*width,arena_,width);
        }
        if(c_.stateVFirst)
            Gemm<float,float>(P(ws_.dvn),P(ws_.w),P(ws_.dhuTemp),0,0,V,K,m,V,K,K,true,false);
        else
            Gemm<float,float>(P(ws_.w),P(ws_.dvn),P(ws_.dhuTemp),0,0,K,V,m,K,V,V,true,false);
        for(uint32_t i=0;i<rows;++i){
            Read<float>(prev,uint64_t(i)*width,arena_,width);
            Read<float>(P(ws_.dhuTemp),uint64_t(i)*width,arena_[256],width);
            AscendC::Sub(arena_,arena_,arena_[256],width);AscendC::PipeBarrier<PIPE_V>();
            Write<float>(prev,uint64_t(i)*width,arena_,width);
        }
    }
    __aicore__ inline void Backward(uint64_t token,uint32_t m,Address h,Address next,Address prev) {
        if constexpr(MIXED) { CubeBwdPrepare(token,m,h); } else {
        GeneralBwdPrepare<DataT> bp;bp.Init(pipe_,arena_,in_,out_);
        GeneralBwdPrepareBuffers pb{P(ws_.aqk),P(ws_.vn),b_.dO,h,P(ws_.ds),P(ws_.dvl),P(ws_.dqr)};
        GeneralBwdPrepareChunk pd{m,c_.keys,c_.values,0,0,(token*c_.heads+head_)*c_.values,
            uint64_t(c_.heads)*c_.values,0,0,c_.scale,bool(c_.stateVFirst)};bp.Process(pb,pd);
        }
        if constexpr(MIXED) { CubeDhu(token,m,next,prev); } else {
        GeneralDhu<DataT> scan;scan.Init(pipe_,arena_,in_,out_);
        GeneralDhuBuffers db{P(ws_.qg),P(ws_.kg),P(ws_.w),P(ws_.g),b_.dO,P(ws_.dvl),next,next,P(ws_.dvn),prev};
        for(uint32_t v=0;v<c_.values;v+=32){
            GeneralDhuSequence d{m,c_.keys,c_.values,v,c_.values-v<32?c_.values-v:32,0,0,
                (token*c_.heads+head_)*c_.values,uint64_t(c_.heads)*c_.values,0,0,
                uint64_t(c_.keys)*c_.values,c_.scale,bool(c_.useExp2),bool(c_.stateVFirst)};scan.Process(db,d);
        }
        }
        GeneralBwdFinalize<DataT,GateT,BetaT> fin;fin.Init(pipe_,arena_,in_,out_);
        GeneralBwdFinalizeBuffers fb{P(ws_.qhat),P(ws_.khat),b_.v,P(ws_.g),P(ws_.be),P(ws_.akk),P(ws_.vn),h,next,
            P(ws_.dvn),P(ws_.ds),P(ws_.dqr),P(ws_.qr),P(ws_.kr),b_.g,
            b_.aLog?b_.aLog+head_*4:nullptr,b_.bias?b_.bias+uint64_t(head_)*c_.keys*4:nullptr,
            P(ws_.sw),P(ws_.sq),P(ws_.sk),P(ws_.sg),P(ws_.sb),b_.dq,b_.dk,b_.dv,b_.dg,b_.db,
            P(ws_.sw)+c_.keys*4,P(ws_.sw)};
        GeneralBwdFinalizeChunk fd{m,c_.keys,c_.values,0,0,0,0,0,Input(token,c_.keys),Input(token,c_.values),
            BetaInput(token),c_.timeMajor?uint64_t(c_.heads)*c_.keys:c_.keys,
            c_.timeMajor?uint64_t(c_.heads)*c_.values:c_.values,c_.betaTimeMajor?c_.heads:1,c_.scale,c_.lowerBound,
            bool(c_.normalize),bool(c_.useExp2),bool(c_.stateVFirst),c_.gateMode,c_.betaMode};
        fd.gateOffset=GateInput(token);fd.gateStride=c_.gateTimeMajor?uint64_t(c_.heads)*c_.keys:c_.keys;
        fin.Process(fb,fd);
        // dW's last consumer has completed. Reuse its first row for chunk parameters.
        Read<float>(P(ws_.sw),0,arena_,c_.keys);AscendC::Add(biasSum_,biasSum_,arena_,c_.keys);AscendC::PipeBarrier<PIPE_V>();
        Read<float>(P(ws_.sw),c_.keys,scalar_,1);ScalarReady();aSum_+=scalar_.GetValue(0);
    }
public:
    __aicore__ inline void Init(const Buffers &b,const Config &c,const Workspace &w,uint32_t blocks=0) {
        b_=b;c_=c;ws_=w;ownerIndex_=AscendC::GetBlockIdx();ownerCount_=blocks?blocks:AscendC::GetBlockNum();
        if constexpr(MIXED)ownerIndex_/=AscendC::GetSubBlockNum();
        base_=b.workspace+uint64_t(ownerIndex_)*w.bytes;
        pipe_.InitBuffer(in_,1,1024);pipe_.InitBuffer(out_,1,1024);pipe_.InitBuffer(meta_,1,32);
        pipe_.InitBuffer(parameter_,1056);pipe_.InitBuffer(storage_,ARENA_BYTES);
        arena_=storage_.Get<float>();biasSum_=parameter_.Get<float>();scalar_=biasSum_[256];
    }
    template<bool BACKWARD>
    __aicore__ inline void Process() {
        const uint64_t stateSize=uint64_t(c_.keys)*c_.values;
        for(head_=ownerIndex_;head_<c_.heads;head_+=ownerCount_){
            AscendC::Duplicate(biasSum_,0.0F,c_.keys);AscendC::PipeBarrier<PIPE_V>();aSum_=0.0F;
            for(uint32_t s=0;s<c_.sequences;++s){
                const auto seq=GetSequence(s);const uint64_t stateOffset=(uint64_t(s)*c_.heads+head_)*stateSize;
                Address current=P(ws_.state[0]),other=P(ws_.state[1]);
                if(!BACKWARD || c_.recompute){
                    State(current,b_.h0,stateOffset);
                    for(uint64_t offset=0,n=0;offset<uint64_t(seq.length);offset+=CHUNK,++n){
                        const uint32_t m=seq.length-offset<CHUNK?seq.length-offset:CHUNK;
                        Prepare(seq.start+offset,m,BACKWARD);
                        if constexpr(BACKWARD)State(P(ws_.tape)+(seq.firstChunk+n)*stateSize*4,current,0);
                        ForwardState(m,current,other);
                        if constexpr(!BACKWARD){ForwardOutput(seq.start+offset,m,current);Save(seq.start+offset,m,seq.firstChunk+n,current);}
                        Fence();auto tmp=current;current=other;other=tmp;
                    }
                    if constexpr(!BACKWARD){if(b_.ht)State(b_.ht+stateOffset*4,current,0);}
                }
                if constexpr(BACKWARD){
                    State(current,b_.dht,stateOffset);
                    for(uint64_t n=(seq.length+CHUNK-1)/CHUNK;n>0;--n){
                        const uint64_t offset=(n-1)*CHUNK,chunk=seq.firstChunk+n-1;
                        const uint32_t m=seq.length-offset<CHUNK?seq.length-offset:CHUNK;
                        Prepare(seq.start+offset,m,true);
                        Address h=c_.recompute?P(ws_.tape)+chunk*stateSize*4:b_.h+(chunk*c_.heads+head_)*stateSize*4;
                        if(c_.recompute)ForwardState(m,h,nullptr);
                        Backward(seq.start+offset,m,h,current,other);
                        Fence();auto tmp=current;current=other;other=tmp;
                    }
                    if(b_.dh0)State(b_.dh0+stateOffset*4,current,0);
                }
            }
            if constexpr(BACKWARD){
                if(b_.dbias)Write<float>(b_.dbias,uint64_t(head_)*c_.keys,biasSum_,c_.keys);
                if(b_.dA){AscendC::Duplicate(scalar_,aSum_,8);AscendC::PipeBarrier<PIPE_V>();Write<float>(b_.dA,head_,scalar_,1);}
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if constexpr(MIXED)CubeCommand(0,0,0,0);
    }
};
} // namespace KDA::Arch22::Fused32
