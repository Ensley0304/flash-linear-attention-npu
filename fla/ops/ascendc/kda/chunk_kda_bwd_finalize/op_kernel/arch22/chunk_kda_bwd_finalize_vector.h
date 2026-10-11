/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_finalize_common.h"

namespace KDA::Arch22 {

class ChunkKdaBwdFinalizeVector {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR gk, GM_ADDR h, GM_ADDR dh, GM_ADDR workspace,
                                AscendC::LocalTensor<uint8_t> ub, AscendC::TPipe *pipe)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        workspace_ = workspace;
        ub_ = ub;
        pipe_ = pipe;
        inputReady_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        inputFree_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        outputReady_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        outputFree_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage1(const Stage01TilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr)
    {
        constexpr uint32_t TILE_ELEMENTS = VECTOR_ROWS * HEAD_DIM;
        constexpr float LN2 = 0.6931471805599453f;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t tokenOffset = ((batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart) * HEAD_DIM;
        const int64_t stateOffset = (chunkIndex * tiling.heads + head) * HEAD_DIM * HEAD_DIM;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        auto kSaved = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto expSaved = ub_[UB_EXP2_GK].ReinterpretCast<float>();
        auto rH = ub_[UB_RH].ReinterpretCast<float>();
        auto gLast = ub_[UB_GK_LAST].ReinterpretCast<float>();

        // W0 packs the two BF16 state tiles; W1--W4 are FP32 scratch.
        auto hTile = ub_[UB_WORK].ReinterpretCast<bfloat16_t>();
        auto dhTile = ub_[UB_WORK + TILE_ELEMENTS * sizeof(bfloat16_t)].ReinterpretCast<bfloat16_t>();
        auto hFloat = ub_[UB_WORK + UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto dhFloat = ub_[UB_WORK + 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto product = ub_[UB_WORK + 3 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto partial = ub_[UB_WORK + 4 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();

        // rH[k] = sum_v h[k,v] * dh[k,v]. The state always has 128 valid rows.
        for (uint32_t row = 0; row < HEAD_DIM; row += VECTOR_ROWS) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
            AscendC::DataCopy(hTile, h_[stateOffset + row * HEAD_DIM], TILE_ELEMENTS);
            AscendC::DataCopy(dhTile, dh_[stateOffset + row * HEAD_DIM], TILE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
            AscendC::Cast(hFloat, hTile, AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
            AscendC::Cast(dhFloat, dhTile, AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, hFloat, dhFloat, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            // Each repeat reduces 64 FP32 values; two halves form one 128-wide row.
            AscendC::WholeReduceSum(partial, product, 64, VECTOR_ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(partial[64], product[64], 64, VECTOR_ROWS, 1, 1, 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(rH[row], partial, partial[64], VECTOR_ROWS);
        }

        // Retain k, exp2_gk and gk_last in their final resident addresses.
        AscendC::Duplicate(kSaved.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), CHUNK_SIZE * HEAD_DIM);
        AscendC::Duplicate(expSaved, 0.0f, CHUNK_SIZE * HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
        AscendC::DataCopy(kSaved, k_[tokenOffset], rows * HEAD_DIM);
        AscendC::DataCopy(gLast, gk_[tokenOffset + (rows - 1) * HEAD_DIM], HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);

        // W0/W1/W2/W6 now hold raw gk, k FP32, kE FP32 and kE BF16.
        auto gateTile = ub_[UB_WORK].ReinterpretCast<float>();
        auto kFloat = hFloat;
        auto kEFloat = dhFloat;
        auto kEBf16 = ub_[UB_WORK + 6 * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
        AscendC::GlobalTensor<bfloat16_t> kEOut;
        const uint64_t slotOffset = static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES;
        kEOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ + slotOffset + WS_KE));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            const uint64_t offset = static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(capture + offset));
        }
        for (uint32_t row = 0; row < CHUNK_SIZE; row += VECTOR_ROWS) {
            const uint32_t activeRows = row < rows ? (rows - row < VECTOR_ROWS ? rows - row : VECTOR_ROWS) : 0;
            const uint32_t count = activeRows * HEAD_DIM;
            // Invalid rows never enter Exp and always write zero, including reused GM slots.
            AscendC::Duplicate(kEBf16.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), TILE_ELEMENTS);
            if (count != 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
                AscendC::DataCopy(gateTile, gk_[tokenOffset + row * HEAD_DIM], count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_);
                AscendC::Cast(kFloat, kSaved[row * HEAD_DIM], AscendC::RoundMode::CAST_NONE, count);
                AscendC::Muls(expSaved[row * HEAD_DIM], gateTile, LN2, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(expSaved[row * HEAD_DIM], expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(kEFloat, kFloat, expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kEBf16, kEFloat, AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
            AscendC::DataCopy(kEOut[row * HEAD_DIM], kEBf16, TILE_ELEMENTS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_KE + row * HEAD_DIM * sizeof(bfloat16_t)],
                                  kEBf16.ReinterpretCast<uint8_t>(), TILE_ELEMENTS * sizeof(bfloat16_t));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_);
        }
        if constexpr (CAPTURE) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
            AscendC::DataCopy(snapshot[CAPTURE_EXP2_GK], expSaved.ReinterpretCast<uint8_t>(),
                              CHUNK_SIZE * HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_RH], rH.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_GK_LAST], gLast.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_);
        }
        // The caller publishes KE_READY on PIPE_MTE3; UB residents survive Stage 1.
    }

    __aicore__ inline void End()
    {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputFree_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(outputFree_);
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::GlobalTensor<bfloat16_t> k_, h_, dh_;
    AscendC::GlobalTensor<float> gk_;
    GM_ADDR workspace_ = nullptr;
    AscendC::TEventID inputReady_, inputFree_, outputReady_, outputFree_;
};

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
