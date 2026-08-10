#ifndef CHUNK_KDA_BWD_A_VECTOR_H
#define CHUNK_KDA_BWD_A_VECTOR_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_a_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "catlass/arch/cross_core_sync.hpp"
#include "arch35/chunk_kda_bwd_a_regbase.h"
#endif

namespace KDA {

template <typename T, uint32_t V_DIM>
class ChunkKdaBwdAVector {
public:
    __aicore__ ChunkKdaBwdAVector(
        GM_ADDR q0, GM_ADDR dAqk, GM_ADDR cuSeqlens,
        GM_ADDR chunkIndices, GM_ADDR workspace)
        : q0_(q0), dAqk_(dAqk), cuSeqlens_(cuSeqlens),
          chunkIndices_(chunkIndices), workspace_(workspace)
    {
    }

    __aicore__ inline void Init(
        const ChunkKdaBwdATilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        q0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(q0_));
        dAqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk_));
        pipe_->InitBuffer(workBuf_, 64 * V_DIM * sizeof(float));
        mte2ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        vToMte3_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToMte2_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        mte3ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        if (subBlockNum == 0) {
            return;
        }
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t logicalCore = AscendC::GetBlockIdx() / subBlockNum;
        const uint64_t ownerCount =
            static_cast<uint64_t>(tiling_.chunkNum) * tiling_.headNum;

        uint32_t generation = 0;
        for (uint64_t owner = logicalCore; owner < ownerCount;
             owner += tiling_.usedCoreNum, ++generation) {
            const uint32_t slot = generation & 1U;
            const uint32_t readyFlag =
                slot == 0 ? KDA_BWD_A_READY_FLAG0 : KDA_BWD_A_READY_FLAG1;
            const uint32_t freeFlag =
                slot == 0 ? KDA_BWD_A_FREE_FLAG0 : KDA_BWD_A_FREE_FLAG1;
            WaitSlotReady(slot, readyFlag);
            const uint32_t taskIdx =
                static_cast<uint32_t>(owner / tiling_.headNum);
            const uint32_t head =
                static_cast<uint32_t>(owner % tiling_.headNum);
            GM_ADDR slotBase =
                KdaBwdAPostSlot(workspace_, logicalCore, slot, tiling_);
            ProcessQ0(taskIdx, head, subBlockIdx, slotBase);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Both AIV lanes consume one shared workspace generation.  AIC may
            // reuse the ping-pong slot only after both lane epilogues and MTE3
            // stores have retired.  On A5 the 0x2 producer mask requires both
            // Vector lanes to publish before the AIC wait can retire.
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
#endif
            PublishSlotFree(slot, freeFlag);
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

private:
    __aicore__ inline void WaitSlotReady(
        uint32_t slot, uint32_t fallbackFlag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot == 0) {
            Catlass::Arch::CrossCoreWaitFlag(readyFlag0_);
        } else {
            Catlass::Arch::CrossCoreWaitFlag(readyFlag1_);
        }
#else
        AscendC::CrossCoreWaitFlag(fallbackFlag);
#endif
    }

    __aicore__ inline void PublishSlotFree(
        uint32_t slot, uint32_t fallbackFlag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot == 0) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(freeFlag0_);
        } else {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(freeFlag1_);
        }
#else
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(fallbackFlag);
#endif
    }

    __aicore__ inline void ProcessQ0(
        uint32_t taskIdx, uint32_t head, uint32_t lane, GM_ADDR slotBase)
    {
        constexpr uint32_t rowsPerLane = KDA_BWD_A_K / 2;
        const uint32_t rowBegin = lane * rowsPerLane;
        const uint32_t elements = rowsPerLane * V_DIM;
        AscendC::GlobalTensor<float> raw;
        const uint64_t ownerOffset =
            (static_cast<uint64_t>(taskIdx) * tiling_.headNum + head) *
                KDA_BWD_A_K * V_DIM;
        const uint64_t outOffset =
            ownerOffset + static_cast<uint64_t>(rowBegin) * V_DIM;
        raw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            slotBase + tiling_.q0RawOffset));
        AscendC::LocalTensor<float> local = workBuf_.Get<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowsPerLane),
            static_cast<uint32_t>(V_DIM * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(
            local, raw[rowBegin * V_DIM], copyParams,
            {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaBwdARegbaseScale(
            (__ubuf__ float *)local.GetPhyAddr(), tiling_.scale,
            static_cast<uint16_t>(elements));
#else
        AscendC::Muls(local, local, tiling_.scale, elements);
#endif
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::DataCopyPad(q0Gm_[outOffset], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        // ProcessDAqk immediately reuses this UB as a Vector destination.
        // MTE3_MTE2 only protects the following MTE2 transfer; it does not
        // prevent Vector from overwriting the final Q0 tile still read by
        // MTE3.  Drain MTE3 to Vector before changing the UB lifetime.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void ProcessDAqk(
        const ChunkKdaBwdATask &task, uint32_t head, uint32_t lane,
        uint32_t validC, GM_ADDR slotBase)
    {
        constexpr uint32_t rowsPerLane = KDA_BWD_A_C / 2;
        const uint32_t rowBegin = lane * rowsPerLane;
        AscendC::GlobalTensor<float> raw;
        const uint64_t outOffset =
            KdaBwdAHeadTokenOffset(
                tiling_, task, head, KDA_BWD_A_C) +
            static_cast<uint64_t>(rowBegin) * KDA_BWD_A_C;
        raw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            slotBase + tiling_.dAqkRawOffset));
        AscendC::LocalTensor<float> local = workBuf_.Get<float>();
        constexpr uint32_t elements = rowsPerLane * KDA_BWD_A_C;
        AscendC::Duplicate(local, 0.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t validRows = rowBegin >= validC ? 0 :
            ((rowBegin + rowsPerLane <= validC) ? rowsPerLane :
                                                  validC - rowBegin);
        if (validRows != 0) {
            const uint32_t sourceRowGapBytes =
                (KDA_BWD_A_C - validC) * sizeof(float);
            const uint32_t alignedValidC = (validC + 7U) & ~7U;
            const uint32_t destinationRowGapBlocks =
                (KDA_BWD_A_C - alignedValidC) * sizeof(float) / 32U;
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(validC * sizeof(float)),
                sourceRowGapBytes, destinationRowGapBlocks, 0};
            AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(
                local, raw[rowBegin * KDA_BWD_A_C], copyParams, noPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        // Kernel C's intra pack applies both scale and the strict lower mask
        // when this private intermediate is consumed.  Keep the identity
        // Vector op here to establish the proven MTE2->V->MTE3 dependency.
        AscendC::Muls(local, local, 1.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (validRows != 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::DataCopyExtParams outputCopy{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(KDA_BWD_A_C * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(dAqkGm_[outOffset], local, outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        }
        return;
#endif
        if (validRows != 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::DataCopyExtParams outputCopy{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(KDA_BWD_A_C * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(
                dAqkGm_[outOffset], local, outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
            // The slot-free credit lets AIC publish the next generation, and
            // the next AIV iteration immediately reuses `local` for Q0.  An
            // MTE3_MTE2 dependency only orders a following MTE2; it does not
            // keep Vector from overwriting the UB source while the final
            // dAqk store is still in flight.  Drain MTE3 against Vector before
            // returning the generation credit.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        }
    }

    GM_ADDR q0_ = nullptr;
    GM_ADDR dAqk_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    ChunkKdaBwdATilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<float> q0Gm_;
    AscendC::GlobalTensor<float> dAqkGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TEventID mte2ToV_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToMte2_;
    AscendC::TEventID mte3ToV_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    Catlass::Arch::CrossCoreFlag readyFlag0_{KDA_BWD_A_READY_FLAG0};
    Catlass::Arch::CrossCoreFlag readyFlag1_{KDA_BWD_A_READY_FLAG1};
    Catlass::Arch::CrossCoreFlag freeFlag0_{KDA_BWD_A_FREE_FLAG0};
    Catlass::Arch::CrossCoreFlag freeFlag1_{KDA_BWD_A_FREE_FLAG1};
#endif
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_VECTOR_H
