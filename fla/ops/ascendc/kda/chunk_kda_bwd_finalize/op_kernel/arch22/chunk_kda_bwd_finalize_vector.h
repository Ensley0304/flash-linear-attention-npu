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

    template <bool CAPTURE = false>
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
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::GlobalTensor<bfloat16_t> k_, h_, dh_, v_, dv_;
    AscendC::GlobalTensor<float> gk_, beta_;
    GM_ADDR workspace_ = nullptr;
    AscendC::TEventID inputReady_[2], inputFree_[2], outputReady_[2], outputFree_[2];
};

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
