// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#include "saved_contract.h"

namespace KDA::Arch22::Fused32::Saved {
// Native FP32 matrix transport for the saved backward schedule. Operand packets
// are ND A[m,kr], B[kr,n], padded to m/n=16 and kr=8. A and B have independent
// 32 KiB regions. No dynamic GM command or scalar Tensor reads are used.
class Cube {
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> l1_;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_;
    AscendC::GlobalTensor<float> inputs_, results_;
    uint64_t packet_ = 0, output_ = 0, packets_ = 0, outputs_ = 0;

    template<AscendC::HardEvent E>
    __aicore__ inline void Set(uint32_t slot) { AscendC::SetFlag<E>(static_cast<event_t>(slot)); }
    template<AscendC::HardEvent E>
    __aicore__ inline void Wait(uint32_t slot) { AscendC::WaitFlag<E>(static_cast<event_t>(slot)); }
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR results,
                               uint64_t packets, uint64_t outputs) {
        pipe_.InitBuffer(l1_, BANKS * INPUT_BANK_BYTES);
        pipe_.InitBuffer(l0a_, 64 * 1024);
        pipe_.InitBuffer(l0b_, 64 * 1024);
        pipe_.InitBuffer(l0c_, 128 * 1024);
        inputs_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputs));
        results_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(results));
        inputs_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        results_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        packets_ = packets;
        outputs_ = outputs;
        AscendC::SetHF32Mode(false);
    }

    __aicore__ inline void Multiply(uint32_t m, uint32_t n, uint32_t k,
                                   bool transposeA = false, bool transposeB = false) {
        using namespace AscendC;
        const uint32_t cs = output_ % BANKS;
        auto c = l0c_.Get<float>()[cs * 64 * 1024 / 4];

        for (uint32_t begin = 0; begin < k; begin += 128) {
            const uint32_t kr = k - begin < 128 ? k - begin : 128;
            const uint32_t slot = packet_ % BANKS;

            CrossCoreWaitFlag<0x2, PIPE_MTE2>(InputReady(packet_));
            auto a1 = l1_.Get<float>()[slot * INPUT_BANK_BYTES / 4];
            auto b1 = a1[32 * 1024 / 4];
            Nd2NzParams cp{};
            cp.ndNum = 1; cp.nValue = transposeA ? kr : m;
            cp.dValue = transposeA ? m : kr;
            cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue; cp.dstNzNStride = 1;
            DataCopy(a1, inputs_[slot * INPUT_BANK_BYTES / 4], cp);
            cp.nValue = transposeB ? n : kr; cp.dValue = transposeB ? kr : n;
            cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue;
            DataCopy(b1, inputs_[slot * INPUT_BANK_BYTES / 4 + 32 * 1024 / 4], cp);
            if (packet_ + BANKS < packets_)
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
            MmadParams mm{};
            mm.m = m; mm.n = n; mm.k = kr;
            mm.cmatrixInitVal = begin == 0;
            mm.kDirectionAlign = transposeA;
            Mmad(c, a2, b2, mm);
            Set<HardEvent::M_MTE1>(slot);
            Wait<HardEvent::M_MTE1>(slot);
            ++packet_;
        }
        Set<HardEvent::M_FIX>(cs);
        if (output_ >= BANKS)
            CrossCoreWaitFlag<0x2, PIPE_FIX>(ResultFree(output_));
        Wait<HardEvent::M_FIX>(cs);
        auto fp = FixpipeParamsV220(n, m, m, n, false);
        Fixpipe<float, float, CFG_ROW_MAJOR>(results_[cs * RESULT_BANK_BYTES / 4], c, fp);
        CrossCoreSetFlag<0x2, PIPE_FIX>(ResultReady(output_));
        Set<HardEvent::FIX_M>(cs);
        Wait<HardEvent::FIX_M>(cs);
        ++output_;
    }

    __aicore__ inline void RetireRecord() {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_READY);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
    }
    __aicore__ inline void End() {}
};
} // namespace KDA::Arch22::Fused32::Saved
