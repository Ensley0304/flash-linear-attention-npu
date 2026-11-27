// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#ifndef FWDH_RIGHT_RESIDENT
#define FWDH_RIGHT_RESIDENT 0
#endif

namespace GDN::Arch22::FwdH256 {
// P32 prototype: 64x64 packets, full reduction retained in one L0C bank.
// Each bank owns independent L1, L0AB and ready/free tokens. No partial GM sums.
class Cube {
    AscendC::TPipe &pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> l1_;
    AscendC::TBuf<AscendC::TPosition::A2> a_;
    AscendC::TBuf<AscendC::TPosition::B2> b_;
    AscendC::TBuf<AscendC::TPosition::CO1> c_;
    event_t l1Free_[2], l1Ready_[2], abFree_[2], abReady_[2], cFree_[2], cReady_[2];
    uint32_t packet_=0, output_=0;
#if FWDH_RIGHT_RESIDENT
    AscendC::TBuf<AscendC::TPosition::B1> right_;
    event_t rightFree_,rightReady_;
    bool residentRight_=false;
#endif
    __aicore__ inline void Load(GM_ADDR ap, GM_ADDR bp, uint32_t as, uint32_t bs,
                               uint32_t m, uint32_t validM, uint32_t validK,
                               uint32_t begin, bool ta, uint32_t slot) {
        using namespace AscendC;
        WaitFlag<HardEvent::MTE1_MTE2>(l1Free_[slot]);
        auto al=l1_.Get<float>()[slot*8192]; auto bl=al[4096];
        if(validM!=m || validK!=64) {
            InitConstValueParams<float> zero(1,1024,0,0.f);
            InitConstValue(al,zero); PipeBarrier<PIPE_MTE2>();
        }
        GlobalTensor<float> ag,bg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ap));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bp));
        Nd2NzParams cp{};cp.ndNum=1;cp.dstNzNStride=1;
        cp.nValue=ta?validK:validM;cp.dValue=ta?validM:validK;
        cp.srcDValue=as;cp.dstNzC0Stride=ta?64:m;
        DataCopy(al,ag[ta?uint64_t(begin)*as:begin],cp);
        cp.nValue=validK;cp.dValue=64;cp.srcDValue=bs;cp.dstNzC0Stride=64;
#if FWDH_RIGHT_RESIDENT
        if(!residentRight_)
#endif
        DataCopy(bl,bg[uint64_t(begin)*bs],cp);
        SetFlag<HardEvent::MTE2_MTE1>(l1Ready_[slot]);
    }
public:
    __aicore__ explicit Cube(AscendC::TPipe &pipe):pipe_(pipe) {}
    __aicore__ inline void Init() {
        using namespace AscendC;
        pipe_.InitBuffer(l1_,65536);pipe_.InitBuffer(a_,32768);
        pipe_.InitBuffer(b_,32768);pipe_.InitBuffer(c_,32768);
        SetHF32Mode(false);
#if FWDH_RIGHT_RESIDENT
        pipe_.InitBuffer(right_,16384);
        rightFree_=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE1_MTE2>());
        rightReady_=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_MTE1>());
        SetFlag<HardEvent::MTE1_MTE2>(rightFree_);
#endif
        for(uint32_t i=0;i<2;++i) {
            l1Free_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE1_MTE2>());
            l1Ready_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_MTE1>());
            abFree_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::M_MTE1>());
            abReady_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE1_M>());
            cFree_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::FIX_M>());
            cReady_[i]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::M_FIX>());
            SetFlag<HardEvent::MTE1_MTE2>(l1Free_[i]);
            SetFlag<HardEvent::M_MTE1>(abFree_[i]);
            SetFlag<HardEvent::FIX_M>(cFree_[i]);
        }
    }
#if FWDH_RIGHT_RESIDENT
    // One S2 window: preserve right until all four output blocks consumed it.
    __aicore__ inline void BeginRight(GM_ADDR src,uint32_t rows) {
        using namespace AscendC;
        WaitFlag<HardEvent::MTE1_MTE2>(rightFree_);
        auto dst=right_.Get<float>();
        if(rows<64) {
            InitConstValueParams<float> zero(1,512,0,0.f);
            InitConstValue(dst,zero);PipeBarrier<PIPE_MTE2>();
        }
        GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
        Nd2NzParams cp{};cp.ndNum=1;cp.nValue=rows;cp.dValue=64;
        cp.srcDValue=64;cp.dstNzC0Stride=64;cp.dstNzNStride=1;
        DataCopy(dst,gm,cp);
        SetFlag<HardEvent::MTE2_MTE1>(rightReady_);
        WaitFlag<HardEvent::MTE2_MTE1>(rightReady_);
        residentRight_=true;
    }
    __aicore__ inline void EndRight() {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(rightFree_);
        residentRight_=false;
    }
#endif
    // A: [validM,k] or physical [k,validM]; B: [k,64].
    // dst must provide ceil(validM/16)*16 rows; valid M is copied out by AIV.
    __aicore__ inline void Multiply(GM_ADDR ap,GM_ADDR bp,GM_ADDR dst,
                                   uint32_t as,uint32_t bs,uint32_t ds,
                                   uint32_t validM,uint32_t k,bool ta=false) {
        using namespace AscendC;
        const uint32_t m=(validM+15)/16*16, cs=output_++%2;
        auto cl=c_.Get<float>()[cs*4096];
        WaitFlag<HardEvent::FIX_M>(cFree_[cs]);
        Load(ap,bp,as,bs,m,validM,k<64?k:64,0,ta,packet_%2);
        for(uint32_t begin=0;begin<k;begin+=64,++packet_) {
            uint32_t slot=packet_%2;
            if(begin+64<k) Load(ap,bp,as,bs,m,validM,k-begin-64<64?k-begin-64:64,
                                begin+64,ta,(packet_+1)%2);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_[slot]);
            WaitFlag<HardEvent::M_MTE1>(abFree_[slot]);
            auto al=l1_.Get<float>()[slot*8192];auto bl=al[4096];
            auto a2=a_.Get<float>()[slot*4096];auto b2=b_.Get<float>()[slot*4096];
            LoadData3DParamsV2<float> tr{};
            tr.l1H=1;tr.l1W=64;tr.strideW=tr.strideH=tr.filterW=tr.filterH=1;
            tr.dilationFilterW=tr.dilationFilterH=1;tr.enTranspose=true;tr.mExtension=64;
            if(ta) {tr.channelSize=m;tr.kExtension=m;LoadData(a2,al,tr);}
            else {
                LoadData2DParams ld{};ld.repeatTimes=8;ld.srcStride=m/16;
                for(uint32_t r=0;r<m/16;++r) {
                    LoadData(a2[r*1024],al[r*128],ld);PipeBarrier<PIPE_MTE1>();
                }
            }
            tr.channelSize=64;tr.kExtension=64;
#if FWDH_RIGHT_RESIDENT
            if(residentRight_)LoadData(b2,right_.Get<float>(),tr);else
#endif
            LoadData(b2,bl,tr);
            SetFlag<HardEvent::MTE1_MTE2>(l1Free_[slot]);
            SetFlag<HardEvent::MTE1_M>(abReady_[slot]);
            WaitFlag<HardEvent::MTE1_M>(abReady_[slot]);
            MmadParams mm{};mm.m=m;mm.n=64;mm.k=64;
            mm.cmatrixInitVal=begin==0;mm.kDirectionAlign=ta;
            Mmad(cl,a2,b2,mm);
            SetFlag<HardEvent::M_MTE1>(abFree_[slot]);
        }
        SetFlag<HardEvent::M_FIX>(cReady_[cs]);WaitFlag<HardEvent::M_FIX>(cReady_[cs]);
        GlobalTensor<float> out;out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
        auto fp=FixpipeParamsV220(64,m,m,ds,false);
        Fixpipe<float,float,CFG_ROW_MAJOR>(out,cl,fp);
        SetFlag<HardEvent::FIX_M>(cFree_[cs]);
    }
    __aicore__ inline void End() {
        using namespace AscendC;
#if FWDH_RIGHT_RESIDENT
        WaitFlag<HardEvent::MTE1_MTE2>(rightFree_);
        pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(rightFree_);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE1>(rightReady_);
#endif
        for(uint32_t i=0;i<2;++i) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_[i]);
            WaitFlag<HardEvent::M_MTE1>(abFree_[i]);WaitFlag<HardEvent::FIX_M>(cFree_[i]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(l1Free_[i]);
            pipe_.ReleaseEventID<HardEvent::MTE2_MTE1>(l1Ready_[i]);
            pipe_.ReleaseEventID<HardEvent::M_MTE1>(abFree_[i]);
            pipe_.ReleaseEventID<HardEvent::MTE1_M>(abReady_[i]);
            pipe_.ReleaseEventID<HardEvent::FIX_M>(cFree_[i]);
            pipe_.ReleaseEventID<HardEvent::M_FIX>(cReady_[i]);
        }
    }
};
} // namespace GDN::Arch22::FwdH256
