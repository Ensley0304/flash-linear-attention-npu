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
        for (uint32_t slot = 0; slot < 2; ++slot) {
            inputReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            inputFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            outputReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            outputFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    template <bool CAPTURE = false, bool SAVE_RH = false>
    __aicore__ inline void Stage1(const Stage01TilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr,
                                  uint32_t bank = 0)
    {
        constexpr uint32_t TILE_ROWS = 32;
        constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * HEAD_DIM;
        static_assert(2 * TILE_ELEMENTS * sizeof(bfloat16_t) == 2 * UB_WORK_SLOT_BYTES);
        static_assert(UB_WORK + 11 * UB_WORK_SLOT_BYTES <= UB_BYTES);
        constexpr float LN2 = 0.6931471805599453f;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t tokenOffset = ((batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart) * HEAD_DIM;
        const int64_t stateOffset = (chunkIndex * tiling.heads + head) * HEAD_DIM * HEAD_DIM;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        auto kSaved = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto expSaved = ub_[UB_EXP2_GK].ReinterpretCast<float>();
        auto rH = ub_[UB_RH].ReinterpretCast<float>();
        auto gLast = ub_[UB_GK_LAST].ReinterpretCast<float>();

        // W0--W3 hold two BF16 input pairs; W4--W10 hold shared FP32 scratch.
        auto hFloat = ub_[UB_WORK + 4 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto dhFloat = ub_[UB_WORK + 6 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto product = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto partial = ub_[UB_WORK + 10 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::DataCopy(input, h_[stateOffset + slot * TILE_ELEMENTS], TILE_ELEMENTS);
            AscendC::DataCopy(input[TILE_ELEMENTS], dh_[stateOffset + slot * TILE_ELEMENTS], TILE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        // Release each input after Cast, so the next DMA overlaps Mul/reduction.
        for (uint32_t tile = 0; tile < HEAD_DIM / TILE_ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::Cast(hFloat, input, AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
            AscendC::Cast(dhFloat, input[TILE_ELEMENTS], AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < HEAD_DIM / TILE_ROWS) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                AscendC::DataCopy(input, h_[stateOffset + (tile + 2) * TILE_ELEMENTS], TILE_ELEMENTS);
                AscendC::DataCopy(input[TILE_ELEMENTS], dh_[stateOffset + (tile + 2) * TILE_ELEMENTS], TILE_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, hFloat, dhFloat, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(partial, product, 64, TILE_ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(partial[64], product[64], 64, TILE_ROWS, 1, 1, 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(rH[tile * TILE_ROWS], partial, partial[64], TILE_ROWS);
        }

        // Retain k, exp2_gk and gk_last in their final resident addresses.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::DataCopy(kSaved, k_[tokenOffset], rows * HEAD_DIM);
        AscendC::DataCopy(gLast, gk_[tokenOffset + (rows - 1) * HEAD_DIM], HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        // Valid rows are fully overwritten; clear only the padding after DMA.
        if (rows < CHUNK_SIZE) {
            AscendC::Duplicate(kSaved[rows * HEAD_DIM].ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                               (CHUNK_SIZE - rows) * HEAD_DIM);
            AscendC::Duplicate(expSaved[rows * HEAD_DIM], 0.0f, (CHUNK_SIZE - rows) * HEAD_DIM);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        // W0--W3 now hold gate ping/pong; W8/W9 hold BF16 kE outputs.
        auto kFloat = hFloat;
        auto kEFloat = dhFloat;
        AscendC::GlobalTensor<bfloat16_t> kEOut;
        const uint64_t slotOffset =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        kEOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ + slotOffset + WS_KE));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            const uint64_t offset = static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(capture + offset));
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t row = slot * TILE_ROWS;
            const uint32_t activeRows = row < rows ? (rows - row < TILE_ROWS ? rows - row : TILE_ROWS) : 0;
            auto gate = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (activeRows != 0) {
                AscendC::DataCopy(gate, gk_[tokenOffset + row * HEAD_DIM], activeRows * HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / TILE_ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t row = tile * TILE_ROWS;
            const uint32_t activeRows = row < rows ? (rows - row < TILE_ROWS ? rows - row : TILE_ROWS) : 0;
            const uint32_t count = activeRows * HEAD_DIM;
            auto gate = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto kEBf16 = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES + slot * TILE_ELEMENTS * sizeof(bfloat16_t)]
                              .ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (count != 0) {
                AscendC::Cast(kFloat, kSaved[row * HEAD_DIM], AscendC::RoundMode::CAST_NONE, count);
                AscendC::Muls(expSaved[row * HEAD_DIM], gate, LN2, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < CHUNK_SIZE / TILE_ROWS) {
                const uint32_t nextRow = (tile + 2) * TILE_ROWS;
                const uint32_t nextRows =
                    nextRow < rows ? (rows - nextRow < TILE_ROWS ? rows - nextRow : TILE_ROWS) : 0;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (nextRows != 0) {
                    AscendC::DataCopy(gate, gk_[tokenOffset + nextRow * HEAD_DIM], nextRows * HEAD_DIM);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
            // Only the same output slot waits for its preceding MTE3 consumer.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::Duplicate(kEBf16.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), TILE_ELEMENTS);
            if (count != 0) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(expSaved[row * HEAD_DIM], expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(kEFloat, kFloat, expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kEBf16, kEFloat, AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(kEOut[row * HEAD_DIM], kEBf16, TILE_ELEMENTS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_KE + row * HEAD_DIM * sizeof(bfloat16_t)],
                                  kEBf16.ReinterpretCast<uint8_t>(), TILE_ELEMENTS * sizeof(bfloat16_t));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Acknowledge both consumed free flags before MTE3 primes them again.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        if constexpr (CAPTURE) {
            AscendC::DataCopy(snapshot[CAPTURE_EXP2_GK], expSaved.ReinterpretCast<uint8_t>(),
                              CHUNK_SIZE * HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_RH], rH.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_GK_LAST], gLast.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
        }
        if constexpr (SAVE_RH) {
            AscendC::GlobalTensor<float> saved;
            saved.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + slotOffset + WS_RH));
            AscendC::DataCopy(saved, rH, HEAD_DIM);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // The caller publishes KE_READY on PIPE_MTE3; UB residents survive Stage 1.
    }


    __aicore__ inline void InitStage3(GM_ADDR v, GM_ADDR beta, GM_ADDR dv)
    {
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        // Stage 7's band buffer is free until then. Build the triangle once per launch.
        auto lower = ub_[64 * 1024].ReinterpretCast<float>();
        AscendC::Duplicate(lower, 0.0f, CHUNK_SIZE * CHUNK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < CHUNK_SIZE; ++row) {
            AscendC::Duplicate(lower[row * CHUNK_SIZE], 1.0f, row);
        }
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage3(const Stage01TilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr,
                                  uint32_t bank = 0)
    {
        constexpr uint32_t ROWS = 32;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> dvB, zV, zW;
        AscendC::GlobalTensor<bfloat16_t> zbOut;
        dvB.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DV_B));
        zV.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_ZV));
        zW.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_ZW));
        zbOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ + base + WS_ZB));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        auto beta = ub_[UB_BETA].ReinterpretCast<float>();
        auto db = ub_[UB_DB_V].ReinterpretCast<float>();
        auto lower = ub_[64 * 1024].ReinterpretCast<float>();
        auto value = ub_[UB_WORK + 6 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto product = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto scaled = ub_[UB_WORK + 10 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto broadcast = ub_[UB_WORK + 12 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto partial = broadcast[512];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        if (rows < CHUNK_SIZE) {
            AscendC::Duplicate(beta, 0.0f, CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        }
        AscendC::DataCopyPad(beta, beta_[token], {1, rows * 4U, 0, 0, 0}, {true, 0, 0, 0.0f});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(broadcast, beta, 8, {1, 8});

        // Two independent zV/zW input pairs and two BF16 output slots.
        for (uint32_t tile = 0; tile < 2; ++tile) {
            auto input = ub_[UB_WORK + tile * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[tile]);
            AscendC::DataCopy(input, zV[tile * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::DataCopy(input[ROWS * CHUNK_SIZE], zW[tile * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[tile]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t offset = tile * ROWS * CHUNK_SIZE;
            auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto output = ub_[slot * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(value, input, input[ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(value, value, beta, 64, ROWS, {1, 1, 1, 8, 8, 0});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(value, value, lower[offset], ROWS * CHUNK_SIZE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(zbOut[offset], output, ROWS * CHUNK_SIZE);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_ZB + offset * 2], output.ReinterpretCast<uint8_t>(),
                                  ROWS * CHUNK_SIZE * 2);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }

        // The new pairs overlap both old slots. Drain both before remapping;
        // the following loads consume these permissions directly, without re-priming.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
        }
        // W0--W5 now hold two (FP32 dv_b, BF16 v) pairs. Keep beta/db_v resident.
        for (uint32_t tile = 0; tile < 2; ++tile) {
            auto input = ub_[UB_WORK + tile * 3 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            const uint32_t first = tile * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            AscendC::DataCopy(input, dvB[first * HEAD_DIM], ELEMENTS);
            if (active != 0) {
                AscendC::DataCopy(input[ELEMENTS].ReinterpretCast<bfloat16_t>(), v_[(token + first) * HEAD_DIM],
                                  active * HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[tile]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = tile * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * 3 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto vInput = input[ELEMENTS].ReinterpretCast<bfloat16_t>();
            auto output = ub_[slot * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(vInput[active * HEAD_DIM].ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(value, vInput, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            AscendC::Mul(scaled, input, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(scaled[64], input[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, value, input, ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(partial, product, 64, ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(partial[64], product[64], 64, ROWS, 1, 1, 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(db[first], partial, partial[64], ROWS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::Cast(output, scaled, AscendC::RoundMode::CAST_RINT, ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            if (active != 0) {
                AscendC::DataCopy(dv_[(token + first) * HEAD_DIM], output, active * HEAD_DIM);
            }
            if constexpr (CAPTURE) {
                if (tile + 1 == CHUNK_SIZE / ROWS) {
                    AscendC::DataCopy(snapshot[CAPTURE_DB_V], db.ReinterpretCast<uint8_t>(),
                                      CHUNK_SIZE * sizeof(float));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // The next Stage 1/3 loads slot 0 then slot 1, waiting each old reader.
        // Only shared Vector scratch needs a barrier here; output slots remain independent.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitStage5(GM_ADDR q, GM_ADDR dqRaw, float scale)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        scale_ = scale;
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage5(const Stage01TilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank)
    {
        constexpr uint32_t ROWS = 16;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        constexpr float LN2 = 0.6931471805599453f;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> state, gb, dq, dk, dg, dbOut, savedRh;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DK_STATE_RAW));
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DKGB_RAW));
        dq.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DQ_BASE));
        dk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DK_BASE));
        dg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DG_BASE));
        dbOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DB_BASE));
        savedRh.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_RH));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        auto kFloat = ub_[48 * 1024].ReinterpretCast<float>();
        auto qFloat = ub_[56 * 1024].ReinterpretCast<float>();
        auto exponent = ub_[168 * 1024 + 256].ReinterpretCast<float>();
        auto product = ub_[176 * 1024 + 512].ReinterpretCast<float>();
        auto temp = ub_[56 * 1024].ReinterpretCast<float>();
        auto last = ub_[UB_GK_LAST].ReinterpretCast<float>();
        auto rh = ub_[UB_RH].ReinterpretCast<float>();
        auto gate = ub_[83 * 1024].ReinterpretCast<float>();
        auto broadcast = ub_[86 * 1024].ReinterpretCast<float>();
        auto db = ub_[UB_DB_V].ReinterpretCast<float>();
        // Both old Stage 3 input pairs overlap this layout. Its output DMA also
        // reads the region reused for FP32 outputs and the resident db_v snapshot.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::DataCopy(last, gk_[(token + rows - 1) * HEAD_DIM], HEAD_DIM);
        AscendC::DataCopy(rh, savedRh, HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::Muls(gate, last, LN2, HEAD_DIM);
        AscendC::Brcb(broadcast, ub_[UB_BETA].ReinterpretCast<float>(), 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gate, gate, HEAD_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gate, gate, rh, HEAD_DIM);
        // Acknowledge the two old output permissions before re-priming.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            const uint32_t first = slot * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            LoadStage5Tile(state, gb, token, first, active, slot);
        }
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = tile * ROWS;
            const uint32_t offset = first * HEAD_DIM;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * 40 * 1024].ReinterpretCast<float>();
            auto kIn = input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>();
            auto qIn = kIn[ELEMENTS];
            auto dqOut = ub_[slot * 24 * 1024].ReinterpretCast<float>();
            auto dkOut = dqOut[ELEMENTS];
            auto dgOut = dqOut[2 * ELEMENTS];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(input[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(input[3 * ELEMENTS + active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(kIn[active * HEAD_DIM].ReinterpretCast<uint16_t>(), uint16_t(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(qIn[active * HEAD_DIM].ReinterpretCast<uint16_t>(), uint16_t(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(kFloat, kIn, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            // Form the exponent difference before Exp; never divide by exp(gk).
            AscendC::Sub(temp, last, input, 64, ROWS, {1, 1, 1, 16, 0, 16});
            AscendC::Sub(temp[64], last[64], input[64], 64, ROWS, {1, 1, 1, 16, 0, 16});
            AscendC::Muls(exponent, input, LN2, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(temp, temp, LN2, ELEMENTS);
            AscendC::Exp(exponent, exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(temp, temp, ELEMENTS);
            AscendC::Mul(dqOut, input[3 * ELEMENTS], exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dkOut, input[ELEMENTS], temp, ELEMENTS);
            AscendC::Muls(dqOut, dqOut, scale_, ELEMENTS);
            AscendC::Mul(exponent, exponent, input[2 * ELEMENTS], ELEMENTS);
            // Ratio is dead after dkOut; reuse its scratch for q before releasing input.
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qFloat, qIn, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < 4) {
                const uint32_t next = (tile + 2) * ROWS;
                const uint32_t count = next < rows ? (rows - next < ROWS ? rows - next : ROWS) : 0;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                LoadStage5Tile(state, gb, token, next, count, slot);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, kFloat, dkOut, ELEMENTS);
            AscendC::Mul(dgOut, qFloat, dqOut, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(dgOut, dgOut, product, ELEMENTS);
            // Tree reduction across tokens, retaining all 128 channels.
            for (uint32_t count = ELEMENTS / 2; count >= HEAD_DIM; count /= 2) {
                AscendC::Add(product, product, product[count], count);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(gate, gate, product, HEAD_DIM);
            AscendC::Mul(product, kFloat, exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(temp, product, 64, ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(temp[64], product[64], 64, ROWS, 1, 1, 16);
            AscendC::Mul(product, product, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(product[64], product[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(exponent, exponent, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(exponent[64], exponent[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(temp, temp, temp[64], ROWS);
            AscendC::Sub(dkOut, dkOut, exponent, ELEMENTS);
            AscendC::Sub(dgOut, dgOut, product, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(db[first], db[first], temp, ROWS);
            if (first < rows && rows <= first + ROWS) {
                AscendC::Add(dgOut[(rows - first - 1) * HEAD_DIM], dgOut[(rows - first - 1) * HEAD_DIM], gate,
                             HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(dq[offset], dqOut, ELEMENTS);
            AscendC::DataCopy(dk[offset], dkOut, ELEMENTS);
            AscendC::DataCopy(dg[offset], dgOut, ELEMENTS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_DQ_BASE + offset * 4], dqOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
                AscendC::DataCopy(snapshot[CAPTURE_DK_BASE + offset * 4], dkOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
                AscendC::DataCopy(snapshot[CAPTURE_DG_BASE + offset * 4], dgOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
            }
            if (tile == 3) {
                AscendC::DataCopy(dbOut, db, CHUNK_SIZE);
                if constexpr (CAPTURE) {
                    AscendC::DataCopy(snapshot[CAPTURE_DB_BASE], db.ReinterpretCast<uint8_t>(), CHUNK_SIZE * 4);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Stage 1 remaps the scratch/output regions and writes rH/gk_last again.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }


    __aicore__ inline void InitStage7(GM_ADDR dAqk)
    {
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        // CreateVecIndex starts with scalar stores on A2. Initialize once in a
        // persistent region, outside every Stage 1/3/5 scratch lifetime.
        auto indices = ub_[84 * 1024 + 768].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(indices, 0, 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(indices, indices, static_cast<int32_t>((CHUNK_SIZE + 1) * 4), 16);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool CAPTURE = false, bool CONSUME_BANDS = false>
    __aicore__ inline void Stage7(const Stage01TilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank)
    {
        constexpr uint32_t ROWS = 16;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        constexpr float LN2 = 0.6931471805599453f;
        const uint32_t rows = chunk.validRows;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> dAkk;
        AscendC::GlobalTensor<uint8_t> out, snapshot;
        dAkk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DAKK_RAW));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(workspace_ + base));
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        // 0--64 KiB: q/k/original gk; 64--80: persistent strict-lower mask.
        // 88--104: two input pairs; 104--120: value/reconstruction;
        // 120--136: two hi/lo outputs; 136--160: arithmetic scratch;
        // 160--176: band exponent; 176--178: beta broadcast; gather offsets stay at 84.75 KiB.
        auto q = ub_.ReinterpretCast<bfloat16_t>();
        auto k = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto g = ub_[UB_EXP2_GK].ReinterpretCast<float>();
        auto lower = ub_[64 * 1024].ReinterpretCast<float>();
        auto diag = ub_[80 * 1024 + 512].ReinterpretCast<float>();
        auto center = ub_[82 * 1024 + 768].ReinterpretCast<float>();
        auto value = ub_[104 * 1024].ReinterpretCast<float>();
        auto rebuilt = ub_[112 * 1024].ReinterpretCast<float>();
        auto operand = ub_[136 * 1024].ReinterpretCast<float>();
        auto exponent = ub_[144 * 1024].ReinterpretCast<float>();
        auto bandExp = ub_[160 * 1024].ReinterpretCast<float>();
        auto broadcast = ub_[176 * 1024].ReinterpretCast<float>();
        auto indices = ub_[84 * 1024 + 768].ReinterpretCast<int32_t>();
        // Drain old layouts before remapping. Input and output permissions remain independent.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // MTE3_V drains on V only; bridge to MTE2 before overwriting Stage 5 outputs.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::DataCopy(q, q_[token * HEAD_DIM], rows * HEAD_DIM);
        AscendC::DataCopy(k, k_[token * HEAD_DIM], rows * HEAD_DIM);
        AscendC::DataCopy(g, gk_[token * HEAD_DIM], rows * HEAD_DIM);
        AscendC::Brcb(broadcast, ub_[UB_BETA].ReinterpretCast<float>(), 8, {1, 8});
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[UB_WORK + slot * 8 * 1024].ReinterpretCast<float>();
            const uint32_t active = rows > slot * ROWS ? (rows - slot * ROWS < ROWS ? rows - slot * ROWS : ROWS) : 0;
            if (active != 0) {
                AscendC::DataCopy(input, dAqk_[(token + slot * ROWS) * CHUNK_SIZE], active * CHUNK_SIZE);
            }
            AscendC::DataCopy(input[ROWS * CHUNK_SIZE], dAkk[slot * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Chunk-wide strict lower matrices are generated once. Output planes are hi then lo.
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint32_t first = tile * ROWS;
            const uint32_t slot = tile & 1;
            const uint32_t active = rows > first ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * 8 * 1024].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(input[active * CHUNK_SIZE], 0.0f, (ROWS - active) * CHUNK_SIZE);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(diag[first], input, indices.ReinterpretCast<uint32_t>(), first * 4, ROWS);
            for (uint32_t matrix = 0; matrix < 2; ++matrix) {
                auto output = ub_[120 * 1024 + slot * 8 * 1024].ReinterpretCast<bfloat16_t>();
                AscendC::Mul(value, input[matrix * ROWS * CHUNK_SIZE], lower[first * CHUNK_SIZE], ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(value, value, matrix == 0 ? 65536.0f : -65536.0f, ROWS * CHUNK_SIZE);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(rebuilt, output, AscendC::RoundMode::CAST_NONE, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(rebuilt, value, rebuilt, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(output[ROWS * CHUNK_SIZE], rebuilt, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                const uint32_t dst = matrix == 0 ? WS_DAQK_LOWER : WS_DAKK_LOWER;
                const uint32_t saved = matrix == 0 ? CAPTURE_DAQK_LOWER : CAPTURE_DAKK_LOWER;
                for (uint32_t part = 0; part < 2; ++part) {
                    AscendC::DataCopy(out[dst + part * 8192 + first * CHUNK_SIZE * 2],
                                      output[part * ROWS * CHUNK_SIZE].ReinterpretCast<uint8_t>(),
                                      ROWS * CHUNK_SIZE * 2);
                    if constexpr (CAPTURE) {
                        AscendC::DataCopy(snapshot[saved + part * 8192 + first * CHUNK_SIZE * 2],
                                          output[part * ROWS * CHUNK_SIZE].ReinterpretCast<uint8_t>(),
                                          ROWS * CHUNK_SIZE * 2);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < 4) {
                const uint32_t next = first + 2 * ROWS;
                const uint32_t count = rows > next ? (rows - next < ROWS ? rows - next : ROWS) : 0;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (count != 0) {
                    AscendC::DataCopy(input, dAqk_[(token + next) * CHUNK_SIZE], count * CHUNK_SIZE);
                }
                AscendC::DataCopy(input[ROWS * CHUNK_SIZE], dAkk[next * CHUNK_SIZE], ROWS * CHUNK_SIZE);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
        }
        // The partial harness captures both bands; its reusable workspace retains band s=0.
        // Stage 8/10 must consume each band before the next iteration overwrites these slots.
        for (int32_t start = ((rows - 1) / 32) * 32; start >= 0; start -= 32) {
            const uint32_t end = (static_cast<uint32_t>(start + 32) < rows ? start + 32 : rows);
            const uint32_t saved = CAPTURE_BANDS + (start / 32) * CAPTURE_BAND_BYTES;
            AscendC::Add(center, g[start * HEAD_DIM], g[(end - 1) * HEAD_DIM], HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(center, center, 0.5f, HEAD_DIM);
            AscendC::Duplicate(bandExp, 0.0f, 32 * HEAD_DIM);
            // qPos and bkPos use the same positive exponent in each tile.
            for (uint32_t phase = 0; phase < (CONSUME_BANDS ? 2 : 1); ++phase) {
                for (uint32_t tile = 0; tile < 4; ++tile) {
                    const uint32_t first = tile * ROWS;
                    for (uint32_t matrix = CONSUME_BANDS && phase == 1 ? 1 : 0;
                         matrix < (CONSUME_BANDS && phase == 0 ? 1 : 3); ++matrix) {
                        const uint32_t dst = matrix == 0 ? WS_KNEG : (matrix == 1 ? WS_QPOS : WS_BKPOS);
                        const uint32_t slot = (3 * tile + matrix) & 1;
                        const uint32_t limit = matrix == 0 ? end : rows;
                        const uint32_t active = first < limit && (matrix == 0 || first >= start) ?
                                                    (limit - first < ROWS ? limit - first : ROWS) :
                                                    0;
                        auto output = ub_[120 * 1024 + slot * 8 * 1024].ReinterpretCast<bfloat16_t>();
                        AscendC::Duplicate(value, 0.0f, ELEMENTS);
                        if (active != 0) {
                            // Never evaluate Exp on masked rows or reconstruct gk from Stage 1's exponent.
                            AscendC::PipeBarrier<PIPE_V>();
                            if (matrix == 0) {
                                AscendC::Sub(exponent, center, g[first * HEAD_DIM], 64, active, {1, 1, 1, 16, 0, 16});
                                AscendC::Sub(exponent[64], center[64], g[first * HEAD_DIM + 64], 64, active,
                                             {1, 1, 1, 16, 0, 16});
                            } else if (matrix == 1) {
                                AscendC::Sub(exponent, g[first * HEAD_DIM], center, 64, active, {1, 1, 1, 16, 16, 0});
                                AscendC::Sub(exponent[64], g[first * HEAD_DIM + 64], center[64], 64, active,
                                             {1, 1, 1, 16, 16, 0});
                            }
                            AscendC::Cast(operand, matrix == 1 ? q[first * HEAD_DIM] : k[first * HEAD_DIM],
                                          AscendC::RoundMode::CAST_NONE, active * HEAD_DIM);
                            AscendC::PipeBarrier<PIPE_V>();
                            if (matrix != 2) {
                                AscendC::Muls(exponent, exponent, LN2, active * HEAD_DIM);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Exp(exponent, exponent, active * HEAD_DIM);
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            AscendC::Mul(value, operand, exponent, active * HEAD_DIM);
                            if (matrix == 1 && first < end) {
                                AscendC::Adds(bandExp[(first - start) * HEAD_DIM], exponent, 0.0f, active * HEAD_DIM);
                            }
                            if (matrix == 2) {
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Mul(value, value, broadcast[first * 8], 64, active, {1, 1, 0, 16, 16, 1});
                                AscendC::Mul(value[64], value[64], broadcast[first * 8], 64, active,
                                             {1, 1, 0, 16, 16, 1});
                            }
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, ELEMENTS);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(rebuilt, output, AscendC::RoundMode::CAST_NONE, ELEMENTS);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Sub(rebuilt, value, rebuilt, ELEMENTS);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(output[ELEMENTS], rebuilt, AscendC::RoundMode::CAST_RINT, ELEMENTS);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                        for (uint32_t part = 0; part < 2; ++part) {
                            AscendC::DataCopy(out[dst + part * 16384 + first * HEAD_DIM * 2],
                                              output[part * ELEMENTS].ReinterpretCast<uint8_t>(), ELEMENTS * 2);
                            if constexpr (CAPTURE) {
                                AscendC::DataCopy(
                                    snapshot[saved + matrix * 32768 + part * 16384 + first * HEAD_DIM * 2],
                                    output[part * ELEMENTS].ReinterpretCast<uint8_t>(), ELEMENTS * 2);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                    }
                }
                if constexpr (CONSUME_BANDS) {
                    if (phase == 0) {
                        // kNeg is complete; Cube need not wait for qPos/bkPos.
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                    }
                }
            }
            // These Vector operands stay in UB for Stage 9/11; only the test build exports them.
            if constexpr (CAPTURE) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
                AscendC::DataCopy(snapshot[CAPTURE_DAQK_DIAG], diag.ReinterpretCast<uint8_t>(), CHUNK_SIZE * 4);
                AscendC::DataCopy(snapshot[saved + 96 * 1024], bandExp.ReinterpretCast<uint8_t>(), 32 * HEAD_DIM * 4);
                AscendC::DataCopy(snapshot[saved + 112 * 1024], center.ReinterpretCast<uint8_t>(), HEAD_DIM * 4);
            }
            // Consume both output frees before the band/resident regions can change.
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            if constexpr (CAPTURE) {
                // The preceding frees were issued before the small capture stores.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[0]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            if constexpr (CONSUME_BANDS) {
                // The next band may overwrite GM only after Cube's MTE2 read.
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
            }
        }
        // Stage 1's next MTE2 writes overlap Stage 7 arithmetic scratch. Its old
        // input-free flags only covered the strict-lower reads, not the band loop.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
        }
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

private:
    __aicore__ inline void LoadStage5Tile(AscendC::GlobalTensor<float> state, AscendC::GlobalTensor<float> gb,
                                          int64_t token, uint32_t first, uint32_t rows, uint32_t slot)
    {
        constexpr uint32_t ELEMENTS = 16 * HEAD_DIM;
        auto input = ub_[UB_WORK + slot * 40 * 1024].ReinterpretCast<float>();
        AscendC::DataCopy(input[ELEMENTS], state[first * HEAD_DIM], ELEMENTS);
        AscendC::DataCopy(input[2 * ELEMENTS], gb[first * HEAD_DIM], ELEMENTS);
        if (rows != 0) {
            AscendC::DataCopy(input, gk_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
            AscendC::DataCopy(input[3 * ELEMENTS], dqRaw_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
            AscendC::DataCopy(input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>(), k_[(token + first) * HEAD_DIM],
                              rows * HEAD_DIM);
            AscendC::DataCopy(input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>()[ELEMENTS],
                              q_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::GlobalTensor<bfloat16_t> k_, h_, dh_, v_, dv_;
    AscendC::GlobalTensor<float> gk_, beta_, dqRaw_, dAqk_;
    AscendC::GlobalTensor<bfloat16_t> q_;
    float scale_ = 1.0f;
    GM_ADDR workspace_ = nullptr;
    AscendC::TEventID inputReady_[2], inputFree_[2], outputReady_[2], outputFree_[2];
};

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
