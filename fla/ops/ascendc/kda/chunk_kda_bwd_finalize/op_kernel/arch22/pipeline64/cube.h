// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#include "contract.h"
#include "products.h"

namespace KDA::Arch22::Pipeline64 {
constexpr uint32_t BANKS=2;
__aicore__ inline uint16_t InputReady(uint64_t p) { return p%2; }
__aicore__ inline uint16_t InputFree(uint64_t p) { return 2+p%2; }
__aicore__ inline uint16_t ResultReady(uint64_t p) { return 4+p%2; }
__aicore__ inline uint16_t ResultFree(uint64_t p) { return 6+p%2; }
constexpr uint16_t RETIRE_READY=8, RETIRE_ACK=9;
// The aligned reduction can be copied directly from its published FP32 view.
// Retain the packing path for odd reduction tails and one-token matrices.
__aicore__ inline bool DirectOperand(const Product &p) {
    return p.m>=16 && p.n>=16 && p.k%8==0 &&
        (p.a.rowStride==1 || p.a.colStride==1) && (p.b.rowStride==1 || p.b.colStride==1);
}
// Native FP32 matrix transport for the C64 Finalize matrix stages. Operand packets
// are ND A[m,kr], B[kr,n], padded to m/n=16 and kr=8. A and B have independent
// 32 KiB regions. No dynamic GM command or scalar Tensor reads are used.
class Cube {
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> l1_;
    AscendC::TBuf<AscendC::TPosition::A1> resident_;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_;
    AscendC::GlobalTensor<float> inputs_, results_;
    uint64_t packet_ = 0, output_ = 0, packets_ = 0, outputs_ = 0;
    uint64_t load_=0;
    uint64_t transported_=0;
    bool dynamic_=false;
    struct ResidentOperand {uint64_t address=0;uint32_t n=0,d=0,stride=0,nzStride=0;bool valid=false;};
    ResidentOperand residentTags_[8]{};
    uint32_t residentVictim_=0;
    uint32_t protectedSlot_=8;
    __aicore__ inline AscendC::LocalTensor<float> LoadResident(uint64_t address,AscendC::Nd2NzParams cp) {
        using namespace AscendC;
        for(uint32_t i=0;i<8;++i) {
            const auto &t=residentTags_[i];
            if(t.valid && t.address==address && t.n==cp.nValue && t.d==cp.dValue &&
               t.stride==cp.srcDValue && t.nzStride==cp.dstNzC0Stride) {
                if(protectedSlot_==8)protectedSlot_=i;
                return resident_.Get<float>()[i*8192];
            }
        }
        uint32_t slot=residentVictim_++%8;
        if(slot==protectedSlot_)slot=residentVictim_++%8;
        if(protectedSlot_==8)protectedSlot_=slot;
        auto dst=resident_.Get<float>()[slot*8192];
        GlobalTensor<float> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
        src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);DataCopy(dst,src,cp);
        residentTags_[slot]={address,cp.nValue,cp.dValue,cp.srcDValue,cp.dstNzC0Stride,true};
        return dst;
    }

    template<AscendC::HardEvent E>
    __aicore__ inline void Set(uint32_t slot) { AscendC::SetFlag<E>(static_cast<event_t>(slot)); }
    template<AscendC::HardEvent E>
    __aicore__ inline void Wait(uint32_t slot) { AscendC::WaitFlag<E>(static_cast<event_t>(slot)); }
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR results,
                               uint64_t packets, uint64_t outputs,bool dynamic=false) {
        pipe_.InitBuffer(l1_, BANKS * INPUT_BANK_BYTES);
        if(dynamic)pipe_.InitBuffer(resident_,256*1024);
        pipe_.InitBuffer(l0a_, 64 * 1024);
        pipe_.InitBuffer(l0b_, 64 * 1024);
        pipe_.InitBuffer(l0c_, 128 * 1024);
        inputs_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputs));
        results_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(results));
        inputs_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        results_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        packets_ = packets;
        outputs_ = outputs;
        dynamic_=dynamic;
        AscendC::SetHF32Mode(false);
    }
    // Public inputs are immutable throughout the record; scratch producers
    // invalidate their range at the stage/band publication boundary.
    __aicore__ inline void InvalidateScratch(uint64_t begin,uint64_t bytes) {
        for(uint32_t i=0;i<8;++i)if(residentTags_[i].address>=begin && residentTags_[i].address<begin+bytes)
            residentTags_[i].valid=false;
    }

    __aicore__ inline void Multiply(uint32_t m, uint32_t n, uint32_t k,
                                   bool transposeA = false, bool transposeB = false,
                                   const Product *direct=nullptr,uint32_t row=0,uint32_t col=0,bool writeBack=false,
                                   bool accumulate=false,bool publish=true) {
        using namespace AscendC;
        const uint32_t cs = output_ % BANKS;
        auto c = l0c_.Get<float>()[cs * 64 * 1024 / 4];

        for (uint32_t begin = 0; begin < k; begin += 128) {
            const uint32_t kr = k - begin < 128 ? k - begin : 128;
            const uint32_t slot = load_ % BANKS;

            if(!direct)CrossCoreWaitFlag<0x2, PIPE_MTE2>(InputReady(packet_));
            auto a1 = l1_.Get<float>()[slot * INPUT_BANK_BYTES / 4];
            auto b1 = a1[32 * 1024 / 4];
            protectedSlot_=8;
            Nd2NzParams cp{};
            cp.ndNum = 1; cp.nValue = transposeA ? kr : m;
            cp.dValue = transposeA ? m : kr;
            cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue; cp.dstNzNStride = 1;
            if(direct) {
                GlobalTensor<float> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(direct->a.address));
                source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                const uint32_t validM=direct->m-row<m?direct->m-row:m;
                cp.nValue=transposeA?kr:validM;cp.dValue=transposeA?validM:kr;
                cp.srcDValue=transposeA?direct->a.colStride:direct->a.rowStride;
                a1=LoadResident(direct->a.address+4*(uint64_t(row)*direct->a.rowStride+uint64_t(begin)*direct->a.colStride),cp);
            } else DataCopy(a1, inputs_[(packet_%BANKS) * INPUT_BANK_BYTES / 4], cp);
            cp.nValue = transposeB ? n : kr; cp.dValue = transposeB ? kr : n;
            cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue;
            if(direct) {
                GlobalTensor<float> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(direct->b.address));
                source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                const uint32_t validN=direct->n-col<n?direct->n-col:n;
                cp.nValue=transposeB?validN:kr;cp.dValue=transposeB?kr:validN;
                cp.srcDValue=transposeB?direct->b.colStride:direct->b.rowStride;
                b1=LoadResident(direct->b.address+4*(uint64_t(begin)*direct->b.rowStride+uint64_t(col)*direct->b.colStride),cp);
            } else DataCopy(b1, inputs_[(packet_%BANKS) * INPUT_BANK_BYTES / 4 + 32 * 1024 / 4], cp);
            if (!direct && (dynamic_ || packet_ + BANKS < packets_))
                CrossCoreSetFlag<0x2, PIPE_MTE2>(InputFree(packet_));
            Set<HardEvent::MTE2_MTE1>(slot);

            Wait<HardEvent::MTE2_MTE1>(slot);
            auto a2 = l0a_.Get<float>()[slot * 32 * 1024 / 4];
            auto b2 = l0b_.Get<float>()[slot * 32 * 1024 / 4];
            LoadData2DParams ld{};
            LoadData3DParamsV2<float> tr{};
            tr.l1H = 1; tr.l1W = kr;
            tr.strideW = tr.strideH = tr.filterW = tr.filterH = 1;
            tr.dilationFilterW = tr.dilationFilterH = 1;
            tr.enTranspose = true; tr.mExtension = kr;
            if (transposeA) {
                tr.channelSize = m; tr.kExtension = m;
                LoadData(a2, a1, tr);
            } else {
                ld.repeatTimes = kr / 8; ld.srcStride = m / 16;
                for (uint32_t row = 0; row < m / 16; ++row) {
                    LoadData(a2[row * kr * 16], a1[row * 128], ld);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
            if (transposeB) {
                ld.repeatTimes = kr * n / 128; ld.srcStride = 1;
                LoadData(b2, b1, ld);
            } else {
                tr.channelSize = n; tr.kExtension = n;
                LoadData(b2, b1, tr);
            }
            Set<HardEvent::MTE1_MTE2>(slot);
            Wait<HardEvent::MTE1_MTE2>(slot);
            Set<HardEvent::MTE1_M>(slot);
            Wait<HardEvent::MTE1_M>(slot);
            // With the stage queue, Fixpipe owns this C slot until its next
            // reuse. The other C slot can accumulate while it drains.
            if(dynamic_ && begin==0 && output_>=BANKS && !accumulate)Wait<HardEvent::FIX_M>(cs);
            MmadParams mm{};
            mm.m = m; mm.n = n; mm.k = kr;
            mm.cmatrixInitVal = begin == 0 && !accumulate;
            mm.kDirectionAlign = transposeA;
            Mmad(c, a2, b2, mm);
            Set<HardEvent::M_MTE1>(slot);
            Wait<HardEvent::M_MTE1>(slot);
            ++load_;
            if(!direct)++packet_;
        }
        if(!publish)return; // next compatible product continues this L0C slot
        Set<HardEvent::M_FIX>(cs);
        if (!writeBack && transported_ >= BANKS)
            CrossCoreWaitFlag<0x2, PIPE_FIX>(ResultFree(transported_));
        Wait<HardEvent::M_FIX>(cs);
        auto fp = FixpipeParamsV220(n, m, m, writeBack?direct->out.rowStride:n, false);
        if(writeBack) {
            GlobalTensor<float> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(direct->out.address));
            dst.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            Fixpipe<float,float,CFG_ROW_MAJOR>(dst[uint64_t(row)*direct->out.rowStride+col],c,fp);
        } else {
            Fixpipe<float, float, CFG_ROW_MAJOR>(results_[(transported_%BANKS) * RESULT_BANK_BYTES / 4], c, fp);
            CrossCoreSetFlag<0x2, PIPE_FIX>(ResultReady(transported_));
            ++transported_;
        }
        Set<HardEvent::FIX_M>(cs);
        if(!dynamic_)Wait<HardEvent::FIX_M>(cs);
        ++output_;
    }

    __aicore__ inline void RetireRecord() {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_READY);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
    }
    // Only two scalar control words, never a tensor element loop. Invalidate
    // after the AIV publication handshake because exchange is reused per band.
    __aicore__ inline bool BandSafe(uint64_t exchange) {
        using namespace AscendC;
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<float> flags;flags.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(exchange));
        DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE>(flags);
        return flags.GetValue(0)!=0.f && flags.GetValue(8)!=0.f;
    }
    __aicore__ inline void End() {
        if(dynamic_)for(uint64_t i=output_>2?output_-2:0;i<output_;++i) {
            Wait<AscendC::HardEvent::FIX_M>(i%2);
        }
        if(dynamic_)for(uint64_t i=transported_>2?transported_-2:0;i<transported_;++i) {
            AscendC::CrossCoreWaitFlag<0x2,PIPE_FIX>(ResultFree(i));
        }
    }
};
} // namespace KDA::Arch22::Pipeline64
