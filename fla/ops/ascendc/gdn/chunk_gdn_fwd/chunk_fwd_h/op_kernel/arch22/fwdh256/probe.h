// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "cube.h"
#ifndef FWDH_D_BLOCK_PIPELINE
#define FWDH_D_BLOCK_PIPELINE 0
#endif
namespace GDN::Arch22::FwdH256 {
constexpr uint16_t H_READY=0,P_READY=1,P_FREE=2,RIGHT_READY=3,RIGHT_FREE=4,D_READY=5,D_FREE=6,DONE=7,ACK=8;
constexpr uint32_t SLOT_FLOATS=4096+4096+16384;
// Distinct one-credit flags: a whole-chunk D_FREE drains all four before reuse.
__aicore__ inline void PublishD(uint32_t block) {
    using namespace AscendC;
    if(block==0)CrossCoreSetFlag<0x2,PIPE_FIX>(D_READY);
    else if(block==1)CrossCoreSetFlag<0x2,PIPE_FIX>(9);
    else if(block==2)CrossCoreSetFlag<0x2,PIPE_FIX>(10);
    else CrossCoreSetFlag<0x2,PIPE_FIX>(11);
}
__aicore__ inline void WaitD(uint32_t block) {
    using namespace AscendC;
    if(block==0)CrossCoreWaitFlag<0x2,PIPE_MTE2>(D_READY);
    else if(block==1)CrossCoreWaitFlag<0x2,PIPE_MTE2>(9);
    else if(block==2)CrossCoreWaitFlag<0x2,PIPE_MTE2>(10);
    else CrossCoreWaitFlag<0x2,PIPE_MTE2>(11);
}
struct Args {
    GM_ADDR w;GM_ADDR u;GM_ADDR kg;GM_ADDR g;GM_ADDR initial;
    GM_ADDR h;GM_ADDR v;GM_ADDR final;GM_ADDR scratch;
    uint32_t batch,heads,tokens,group,exp2,blocks;
};
__aicore__ inline GM_ADDR Offset(GM_ADDR p,uint64_t floats) {return p+floats*4;}
__aicore__ inline uint32_t Min(uint32_t a,uint32_t b) {return a<b?a:b;}
struct Work {
    uint32_t batch,head,vcol,active,chunks;
    __aicore__ Work(const Args &a,uint32_t task) {
        uint32_t groups=(a.heads+a.group-1)/a.group;
        vcol=task%4*64;head=(task/4)%groups*a.group;batch=task/(4*groups);
        active=Min(a.group,a.heads-head);chunks=(a.tokens+63)/64;
    }
    __aicore__ inline uint64_t Input(const Args &a,uint32_t lane,uint32_t token) const {
        return (uint64_t(batch)*a.heads+head+lane)*a.tokens*256+uint64_t(token)*256;
    }
    __aicore__ inline uint64_t State(const Args &a,uint32_t lane) const {
        return (uint64_t(batch)*a.heads+head+lane)*65536+vcol;
    }
    __aicore__ inline uint64_t Snapshot(const Args &a,uint32_t lane,uint32_t chunk) const {
        return ((uint64_t(batch)*chunks+chunk)*a.heads+head+lane)*65536+vcol;
    }
};
class Vector {
    AscendC::TPipe &pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_,gateBuf_,broadcastBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN,2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT,2> out_;
    using F=AscendC::LocalTensor<float>;
    F state_,gate_,broadcast_;
    __aicore__ inline void Read(F dst,GM_ADDR src,uint32_t rows,uint32_t stride) {
        AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rows),256,(stride-64)*4,0,0};
        AscendC::DataCopyPadExtParams<float> pad{false,0,0,0};AscendC::DataCopyPad(dst,gm,cp,pad);
    }
    __aicore__ inline void Write(GM_ADDR dst,F src,uint32_t rows,uint32_t stride) {
        AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rows),256,0,(stride-64)*4,0};
        AscendC::DataCopyPad(gm,src,cp);
    }
    __aicore__ inline void StoreState(GM_ADDR dst) {
        for(uint32_t row=0;row<256;row+=16) {
            auto out=out_.AllocTensor<float>();AscendC::Adds(out,state_[row*64],0.f,1024);
            out_.EnQue(out);out=out_.DeQue<float>();Write(Offset(dst,row*256),out,16,256);out_.FreeTensor(out);
        }
    }
    __aicore__ inline void LoadS1(const Args &a,const Work &w,uint32_t lane,uint32_t start,
                                  uint32_t row,uint32_t rows,GM_ADDR p,bool hasP) {
        auto in=in_.AllocTensor<float>();Read(in,Offset(a.u,w.Input(a,lane,start+row)+w.vcol),rows,256);
        if(hasP)Read(in[1024],Offset(p,row*64),rows,64);
        in_.EnQue(in);
    }
    __aicore__ inline void S1(const Args &a,const Work &w,uint32_t lane,uint32_t start,
                             uint32_t rows,GM_ADDR p,GM_ADDR right,bool hasP,bool update) {
        using namespace AscendC;
        LoadS1(a,w,lane,start,0,Min(16,rows),p,hasP);
        for(uint32_t row=0;row<rows;row+=16) {
            if(row+16<rows)LoadS1(a,w,lane,start,row+16,Min(16,rows-row-16),p,hasP);
            auto in=in_.DeQue<float>();auto out=out_.AllocTensor<float>();uint32_t n=Min(16,rows-row);
            if(hasP)Sub(out,in,in[1024],n*64);else Adds(out,in,0.f,n*64);
            in_.FreeTensor(in);out_.EnQue(out);out=out_.DeQue<float>();
            Write(Offset(a.v,w.Input(a,lane,start+row)+w.vcol),out,n,256);
            if(update)Write(Offset(right,row*64),out,n,64);
            out_.FreeTensor(out);
        }
    }
    __aicore__ inline void LoadD(GM_ADDR d,uint32_t row) {
        auto in=in_.AllocTensor<float>();Read(in,Offset(d,row*64),16,64);in_.EnQue(in);
    }
    __aicore__ inline void S3(const Args &a,const Work &w,uint32_t lane,uint32_t chunk,GM_ADDR d,
                             uint32_t first=0,uint32_t end=256) {
        using namespace AscendC;
        LoadD(d,first);
        for(uint32_t row=first;row<end;row+=16) {
            if(row+16<end)LoadD(d,row+16);
            auto in=in_.DeQue<float>();
            BinaryRepeatParams rp{1,1,0,8,8,1};
            Mul(state_[row*64],state_[row*64],broadcast_[row*8],64,16,rp);PipeBarrier<PIPE_V>();
            Add(state_[row*64],state_[row*64],in,1024);in_.FreeTensor(in);PipeBarrier<PIPE_V>();
            auto out=out_.AllocTensor<float>();Adds(out,state_[row*64],0.f,1024);
            out_.EnQue(out);out=out_.DeQue<float>();
            if(chunk+1<w.chunks)Write(Offset(a.h,w.Snapshot(a,lane,chunk+1)+row*256),out,16,256);
            else if(a.final)Write(Offset(a.final,w.State(a,lane)+row*256),out,16,256);
            out_.FreeTensor(out);
        }
    }
public:
    __aicore__ explicit Vector(AscendC::TPipe &pipe):pipe_(pipe) {}
    __aicore__ inline void Init() {
        pipe_.InitBuffer(stateBuf_,65536);pipe_.InitBuffer(gateBuf_,1024);pipe_.InitBuffer(broadcastBuf_,8192);
        pipe_.InitBuffer(in_,2,8192);pipe_.InitBuffer(out_,2,4096);
        state_=stateBuf_.Get<float>();gate_=gateBuf_.Get<float>();broadcast_=broadcastBuf_.Get<float>();
    }
    __aicore__ inline void Run(const Args &a,const Work &w,uint32_t core,uint32_t lane) {
        using namespace AscendC;
        bool active=lane<w.active;
        GM_ADDR p=Offset(a.scratch,(uint64_t(core)*a.group+lane)*SLOT_FLOATS);
        GM_ADDR right=Offset(p,4096);GM_ADDR d=Offset(right,4096);
        if(active) {
            if(a.initial) {
                for(uint32_t row=0;row<256;row+=16) {
                    auto in=in_.AllocTensor<float>();Read(in,Offset(a.initial,w.State(a,lane)+row*256),16,256);
                    in_.EnQue(in);in=in_.DeQue<float>();Adds(state_[row*64],in,0.f,1024);in_.FreeTensor(in);
                }
            } else Duplicate(state_,0.f,16384);
            PipeBarrier<PIPE_V>();StoreState(Offset(a.h,w.Snapshot(a,lane,0)));
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(H_READY);
        for(uint32_t c=0;c<w.chunks;++c) {
            uint32_t start=c*64, rows=Min(64,a.tokens-start);
            bool hasP=a.initial||c>0, update=c+1<w.chunks||a.final;
            // Gate is independent of S0; prepare while Cube consumes current H.
            if(active&&update) {
                auto in=in_.AllocTensor<float>();
                GlobalTensor<float> gg;gg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.g));
                DataCopy(in,gg[w.Input(a,lane,start+rows-1)],256);
                in_.EnQue(in);in=in_.DeQue<float>();
                Muls(gate_,in,a.exp2?0.6931471805599453F:1.f,256);in_.FreeTensor(in);
                PipeBarrier<PIPE_V>();Exp(gate_,gate_,256);PipeBarrier<PIPE_V>();
                Brcb(broadcast_,gate_,32,{1,8});PipeBarrier<PIPE_V>();
            }
            if(hasP)CrossCoreWaitFlag<0x2,PIPE_MTE2>(P_READY);
            if(active)S1(a,w,lane,start,rows,p,right,hasP,update);
            if(hasP)CrossCoreSetFlag<0x2,PIPE_MTE2>(P_FREE);
            if(update) {
                CrossCoreSetFlag<0x2,PIPE_MTE3>(RIGHT_READY);
                if(FWDH_D_BLOCK_PIPELINE && a.group==1) {
                    for(uint32_t block=0;block<4;++block) {
                        WaitD(block);
                        if(active)S3(a,w,lane,c,d,block*64,(block+1)*64);
                    }
                    CrossCoreWaitFlag<0x2,PIPE_MTE2>(RIGHT_FREE);
                } else {
                    CrossCoreWaitFlag<0x2,PIPE_MTE2>(RIGHT_FREE);
                    CrossCoreWaitFlag<0x2,PIPE_MTE2>(D_READY);
                    if(active)S3(a,w,lane,c,d);
                }
                CrossCoreSetFlag<0x2,PIPE_MTE2>(D_FREE);
                if(c+1<w.chunks)CrossCoreSetFlag<0x2,PIPE_MTE3>(H_READY);
            }
        }
        CrossCoreSetFlag<0x2,PIPE_MTE3>(DONE);CrossCoreWaitFlag<0x2,PIPE_MTE2>(ACK);
    }
};
__aicore__ inline void CubeWork(Cube &op,const Args &a,const Work &w,uint32_t core) {
    using namespace AscendC;
    bool hadP=false,hadD=false;
    for(uint32_t c=0;c<w.chunks;++c) {
        uint32_t start=c*64,rows=Min(64,a.tokens-start);
        bool hasP=a.initial||c>0,update=c+1<w.chunks||a.final;
        CrossCoreWaitFlag<0x2,PIPE_MTE2>(H_READY);
        if(hasP) {
            if(hadP)CrossCoreWaitFlag<0x2,PIPE_FIX>(P_FREE);
            for(uint32_t lane=0;lane<w.active;++lane) {
                auto p=Offset(a.scratch,(uint64_t(core)*a.group+lane)*SLOT_FLOATS);
                op.Multiply(Offset(a.w,w.Input(a,lane,start)),Offset(a.h,w.Snapshot(a,lane,c)),
                            p,256,256,64,rows,256);
            }
            CrossCoreSetFlag<0x2,PIPE_FIX>(P_READY);hadP=true;
        }
        if(update) {
            CrossCoreWaitFlag<0x2,PIPE_MTE2>(RIGHT_READY);
            if(hadD)CrossCoreWaitFlag<0x2,PIPE_FIX>(D_FREE);
            for(uint32_t lane=0;lane<w.active;++lane) {
                auto p=Offset(a.scratch,(uint64_t(core)*a.group+lane)*SLOT_FLOATS);
#if FWDH_RIGHT_RESIDENT
                op.BeginRight(Offset(p,4096),rows);
#endif
                for(uint32_t row=0;row<256;row+=64) {
                    op.Multiply(Offset(a.kg,w.Input(a,lane,start)+row),Offset(p,4096),
                                Offset(p,8192+row*64),256,64,64,64,rows,true);
                    if(FWDH_D_BLOCK_PIPELINE && a.group==1)PublishD(row/64);
                }
#if FWDH_RIGHT_RESIDENT
                op.EndRight();
#endif
            }
            CrossCoreSetFlag<0x2,PIPE_MTE1>(RIGHT_FREE);
            if(!(FWDH_D_BLOCK_PIPELINE && a.group==1))CrossCoreSetFlag<0x2,PIPE_FIX>(D_READY);
            hadD=true;
        }
    }
    if(hadP)CrossCoreWaitFlag<0x2,PIPE_FIX>(P_FREE);
    if(hadD)CrossCoreWaitFlag<0x2,PIPE_FIX>(D_FREE);
    CrossCoreWaitFlag<0x2,PIPE_MTE2>(DONE);CrossCoreSetFlag<0x2,PIPE_MTE2>(ACK);
}
} // namespace GDN::Arch22::FwdH256
