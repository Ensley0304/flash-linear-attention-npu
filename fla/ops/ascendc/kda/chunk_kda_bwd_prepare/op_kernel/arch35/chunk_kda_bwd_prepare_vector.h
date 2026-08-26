/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_PREPARE_ARCH35_VECTOR_H

#include "chunk_kda_bwd_prepare_common.h"
#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC::MicroAPI;

__simd_vf__ inline void BuildTriScaleMaskRegbase(__ubuf__ float *mask, float scale)
{
    constexpr uint32_t ROW = KDA_PREPARE_CHUNK;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<float> scaled;
    Duplicate(zero, 0.0f, full);
    Duplicate(scaled, scale, full);
    for (uint16_t row = 0; row < ROW; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * ROW;
        StoreAlign(mask + offset, zero, full);
        uint32_t lowerCount = static_cast<uint32_t>(row) + 1U;
        MaskReg lower = UpdateMask<float>(lowerCount);
        StoreAlign(mask + offset, scaled, lower);
    }
}

__simd_vf__ inline void ApplyTriScaleMaskRegbase(
    __ubuf__ float *raw, __ubuf__ float *mask, uint16_t validRows)
{
    constexpr uint32_t ROW = KDA_PREPARE_CHUNK;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = static_cast<uint32_t>(validRows);
    MaskReg valid = UpdateMask<float>(validCount);
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, full);
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * ROW;
        RegTensor<float> rawInput;
        RegTensor<float> rawSafe;
        RegTensor<float> maskValue;
        RegTensor<float> result;
        LoadAlign(rawInput, raw + offset);
        Select(rawSafe, rawInput, zero, valid);
        LoadAlign(maskValue, mask + offset);
        Mul(result, rawSafe, maskValue, full);
        StoreAlign(raw + offset, result, full);
    }
}

class ChunkKdaBwdPrepareVector {
public:
    __aicore__ inline void Init(
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dAqk,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        pipe_->InitBuffer(rawPing_, KDA_PREPARE_RAW_BYTES);
        pipe_->InitBuffer(rawPong_, KDA_PREPARE_RAW_BYTES);
        pipe_->InitBuffer(maskBuf_, KDA_PREPARE_RAW_BYTES);
        raw_[0] = rawPing_.Get<float>();
        raw_[1] = rawPong_.Get<float>();
        mask_ = maskBuf_.Get<float>();
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
        BuildTriScaleMaskRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(mask_.GetPhyAddr()), tiling_->scale);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_V>(
                KDA_PREPARE_FREE_FLAG_BASE + slot);
        }

        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        uint64_t generation = 0;
        for (int64_t task = blockIdx; task < tiling_->chunkTaskNum; task += blockNum) {
            ChunkInfo chunk;
            ResolveChunk(task, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            for (int64_t head = 0; head < tiling_->NV; ++head, ++generation) {
                const uint32_t slot = static_cast<uint32_t>(generation & 1U);
                // Gate MTE2 itself.  Waiting on PIPE_V would still allow the
                // independent GM-to-UB transfer to overtake the FFTS ready
                // signal and consume the pre-FixPipe contents of dAqk.
                AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE2>(
                    KDA_PREPARE_READY_FLAG_BASE + slot);
                const int64_t out = TokenOffset(*tiling_, chunk, head, tiling_->chunkSize);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
                AscendC::DataCopy(raw_[slot], dAqk_[out],
                                  static_cast<uint32_t>(chunk.validRows * tiling_->chunkSize));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
                ApplyTriScaleMaskRegbase(
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(raw_[slot].GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(mask_.GetPhyAddr()),
                    static_cast<uint16_t>(chunk.validRows));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
                AscendC::DataCopy(dAqk_[out], raw_[slot],
                                  static_cast<uint32_t>(chunk.validRows * tiling_->chunkSize));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
                AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
                    KDA_PREPARE_FREE_FLAG_BASE + slot);
            }
        }
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

private:
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdPrepareTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<float> dAqk_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::LocalTensor<float> raw_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::LocalTensor<float> mask_;
    AscendC::TEventID mte2ToV_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::TEventID vToMte3_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::TEventID mte3ToMte2_[KDA_PREPARE_RAW_SLOT_COUNT];
};

} // namespace KDA

#endif
