/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_PRE_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_PRE_VECTOR_H

#include "../arch35/chunk_kda_bwd_finalize_common.h"
#include "kernel_operator.h"

namespace KDA {

constexpr uint32_t KDA_FINALIZE_A2_UB_BYTES = 192 * 1024;
constexpr float KDA_FINALIZE_A2_LN2 = 0.6931471805599453f;
constexpr bool KDA_FINALIZE_A2_PERF_RUN_STATE_BASE = true;

template <typename BetaT>
class ChunkKdaBwdFinalizeA2PreVector {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh, GM_ADDR dqRaw,
        GM_ADDR dqBase, GM_ADDR dkBase, GM_ADDR dv, GM_ADDR dbBase,
        GM_ADDR dgBase, GM_ADDR dAkk,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        dqBase_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqBase));
        dkBase_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkBase));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        dbBase_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbBase));
        dgBase_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgBase));
        dAkk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk));
        workspaceAddr_ = workspace;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        subBlockNum_ = AscendC::GetSubBlockNum();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        pipe_->InitBuffer(ubBuf_, KDA_FINALIZE_A2_UB_BYTES);
        ub_ = ubBuf_.Get<uint8_t>();
        mte2ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        vToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        vToMte3_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        const int64_t logicalCore = AscendC::GetBlockIdx() / subBlockNum_;
        const int64_t coreNum = tiling_->a2.usedCoreNum;
        const int64_t headWindowCount =
            (tiling_->NV + KDA_FINALIZE_HEADS_PER_WINDOW - 1) /
            KDA_FINALIZE_HEADS_PER_WINDOW;
        uint64_t groupGeneration = 0;
        for (int64_t workTask = logicalCore; workTask < tiling_->workTaskNum;
             workTask += coreNum, ++groupGeneration) {
            const int64_t chunkTask = workTask / headWindowCount;
            const int64_t headWindow =
                workTask - chunkTask * headWindowCount;
            const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
            const int64_t headEnd = FinalizeMin(
                headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
            FinalizeChunkInfo chunk;
            ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }

            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                if ((owner & 1U) != subBlockIdx_) {
                    continue;
                }
                RunStage0(chunk, head, owner, logicalCore, groupGeneration);
            }

            WaitAicStage(KDA_FINALIZE_A2_AIC_STAGE0_READY);
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                if ((owner & 1U) == subBlockIdx_) {
                    RunDwResidual(owner, logicalCore, groupGeneration);
                }
            }
            SignalAicStage(KDA_FINALIZE_A2_VEC_STAGE0_READY);

            WaitAicStage(KDA_FINALIZE_A2_AIC_STAGE1_READY);
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                if ((owner & 1U) == subBlockIdx_) {
                    RunStage2(chunk, head, owner, logicalCore, groupGeneration);
                }
            }
            SignalAicStage(KDA_FINALIZE_A2_VEC_ZB_READY);

            WaitAicStage(KDA_FINALIZE_A2_AIC_STAGE3_READY);
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                if ((owner & 1U) == subBlockIdx_) {
                    if constexpr (KDA_FINALIZE_A2_PERF_RUN_STATE_BASE) {
                        RunStateAndBase(
                            chunk, head, owner, logicalCore, groupGeneration);
                    }
                    RunTzaResidual(owner, logicalCore, groupGeneration);
                }
            }
            SignalAicStage(KDA_FINALIZE_A2_VEC_TZA_READY);

            WaitAicStage(KDA_FINALIZE_A2_AIC_STAGE4_READY);
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                if ((owner & 1U) == subBlockIdx_) {
                    RunDakk(chunk, head, owner, logicalCore, groupGeneration);
                }
            }
            SignalAicStage(KDA_FINALIZE_A2_VEC_DONE);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void WaitAicStage(uint32_t flag)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag);
    }

    __aicore__ inline void SignalAicStage(uint32_t flag)
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> Ub(uint32_t byteOffset)
    {
        return ub_[byteOffset].ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::GlobalTensor<T> Ws(
        uint64_t slotBase, uint32_t byteOffset)
    {
        AscendC::GlobalTensor<T> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(
            workspaceAddr_ + slotBase + byteOffset));
        return tensor;
    }

    __aicore__ inline void WaitLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StartStores()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
    }

    __aicore__ inline void FinishVectorBeforeLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
    }

    __aicore__ inline void DrainStores()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void SplitBf16(
        AscendC::LocalTensor<float> value,
        AscendC::LocalTensor<bfloat16_t> high,
        AscendC::LocalTensor<bfloat16_t> low,
        AscendC::LocalTensor<float> reconstructed, uint32_t count)
    {
        AscendC::Cast(high, value, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(reconstructed, high, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(value, value, reconstructed, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(low, value, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBeta(
        AscendC::LocalTensor<float> dst, uint64_t offset,
        AscendC::LocalTensor<BetaT> typed, uint32_t rows)
    {
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            AscendC::DataCopy(dst, beta_[offset], rows);
        } else {
            AscendC::DataCopy(typed, beta_[offset], rows);
            WaitLoads();
            AscendC::Cast(dst, typed, AscendC::RoundMode::CAST_NONE, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void BroadcastRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> scalar,
        uint32_t rows)
    {
        AscendC::Brcb(dst, scalar, static_cast<uint8_t>((rows + 7U) / 8U), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulRowsByScalar(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        AscendC::LocalTensor<float> broadcast, uint32_t rows, uint32_t cols)
    {
        const uint8_t stride = static_cast<uint8_t>(cols * sizeof(float) / 32);
        for (uint32_t col = 0; col < cols; col += 64U) {
            const uint32_t mask = cols - col < 64U ? cols - col : 64U;
            AscendC::Mul(dst[col], src[col], broadcast, mask, rows,
                         {1, 1, 0, stride, stride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduce128(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint8_t kRowRepStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::Add(
            src[kHalfRow], src, src[kHalfRow], kHalfRow,
            static_cast<uint8_t>(rows),
            {1, 1, 1, kRowRepStride, kRowRepStride, kRowRepStride});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            dst, src[kHalfRow], kHalfRow,
            static_cast<uint8_t>(rows), 1, 1, kRowRepStride);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceRows128InPlace(
        AscendC::LocalTensor<float> values, uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint32_t kRowBlocks =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        for (uint32_t stride = 1; stride < rows; stride <<= 1U) {
            if (stride <= 4U) {
                const uint32_t repeats =
                    (rows + stride - 1U) / (stride << 1U);
                const uint32_t repStride = (stride << 1U) * kRowBlocks;
                AscendC::BinaryRepeatParams params;
                params.dstBlkStride = 1;
                params.src0BlkStride = 1;
                params.src1BlkStride = 1;
                params.dstRepStride = static_cast<uint8_t>(repStride);
                params.src0RepStride = static_cast<uint8_t>(repStride);
                params.src1RepStride = static_cast<uint8_t>(repStride);
                AscendC::Add(
                    values, values,
                    values[stride * KDA_FINALIZE_DIM],
                    static_cast<uint64_t>(kHalfRow),
                    static_cast<uint8_t>(repeats), params);
                AscendC::Add(
                    values[kHalfRow], values[kHalfRow],
                    values[stride * KDA_FINALIZE_DIM + kHalfRow],
                    static_cast<uint64_t>(kHalfRow),
                    static_cast<uint8_t>(repeats), params);
            } else {
                for (uint32_t row = 0; row + stride < rows;
                     row += stride << 1U) {
                    AscendC::Add(
                        values[row * KDA_FINALIZE_DIM],
                        values[row * KDA_FINALIZE_DIM],
                        values[(row + stride) * KDA_FINALIZE_DIM],
                        KDA_FINALIZE_DIM);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ZeroRowFromColumn(
        AscendC::LocalTensor<float> row, uint32_t column)
    {
        uint64_t mask[1] = {
            column == 0U ? ~0ULL : (~0ULL << column)
        };
        AscendC::Duplicate(row, 0.0f, mask, 1, 1, 8);
    }

    __aicore__ inline void RunStage0(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint32_t elems = static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto kTyped = Ub<bfloat16_t>(0);
        auto expG = Ub<float>(16 * 1024);
        auto value = Ub<float>(48 * 1024);
        auto high = kTyped;
        AscendC::DataCopy(kTyped, k_[token], elems);
        AscendC::DataCopy(expG, gk_[token], elems);
        WaitLoads();
        AscendC::Muls(expG, expG, KDA_FINALIZE_A2_LN2, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expG, expG, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(value, kTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(value, value, expG, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(high, value, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK), expG, elems);
        AscendC::DataCopy(Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_KE), high, elems);
        DrainStores();
    }

    __aicore__ inline void RunDwResidual(
        uint32_t owner, int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto value = Ub<float>(0);
        auto high = Ub<bfloat16_t>(64 * 1024);
        AscendC::DataCopy(
            value, Ws<float>(ws, KDA_FINALIZE_WS_A2_RELAY_DW),
            KDA_FINALIZE_VECTOR_ELEMS);
        WaitLoads();
        AscendC::Cast(
            high, value, AscendC::RoundMode::CAST_RINT,
            KDA_FINALIZE_VECTOR_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_A2_RELAY_DW), high,
            KDA_FINALIZE_VECTOR_ELEMS);
        DrainStores();
    }

    __aicore__ inline void RunStage2(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto zV = Ub<float>(0);
        auto zW = Ub<float>(16 * 1024);
        auto result = Ub<float>(32 * 1024);
        auto betaFp = Ub<float>(48 * 1024);
        auto betaTyped = Ub<BetaT>(49 * 1024);
        auto reconstructed = Ub<float>(64 * 1024);
        auto high = Ub<bfloat16_t>(80 * 1024);
        auto low = Ub<bfloat16_t>(88 * 1024);
        AscendC::DataCopy(
            zV, Ws<float>(ws, KDA_FINALIZE_WS_A2_RELAY_ZV),
            KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::DataCopy(
            zW, Ws<float>(ws, KDA_FINALIZE_WS_A2_RELAY_ZW),
            KDA_FINALIZE_MATRIX_ELEMS);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        LoadBeta(betaFp, betaOffset, betaTyped, static_cast<uint32_t>(chunk.validRows));
        WaitLoads();
        AscendC::Sub(result, zV, zW, KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        // dL[i,j] scales causal columns by beta[j], not rows by beta[i].
        AscendC::Mul(
            result, result, betaFp, KDA_FINALIZE_CHUNK,
            static_cast<uint8_t>(chunk.validRows), {1, 1, 1, 8, 8, 0});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < KDA_FINALIZE_CHUNK; ++row) {
            const uint32_t keep = row < static_cast<uint32_t>(chunk.validRows) ? row : 0U;
            ZeroRowFromColumn(result[row * KDA_FINALIZE_CHUNK], keep);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SplitBf16(
            result, high, low, reconstructed, KDA_FINALIZE_MATRIX_ELEMS);
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_A2_RELAY_DW), high,
            KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::DataCopy(
            Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_A2_RELAY_DW +
                KDA_FINALIZE_MATRIX_BF16_BYTES), low,
            KDA_FINALIZE_MATRIX_ELEMS);
        DrainStores();
    }

    __aicore__ inline void ComputeRh(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        auto hTyped = Ub<bfloat16_t>(0);
        auto dhTyped = Ub<bfloat16_t>(32 * 1024);
        auto product = Ub<float>(64 * 1024);
        auto dhFp = Ub<float>(128 * 1024);
        auto rhResident = Ub<float>(144 * 1024);
        const int64_t hState = FinalizeHOffset(*tiling_, chunk, head);
        const int64_t dhState = FinalizeDhOffset(*tiling_, chunk, head);
        AscendC::DataCopy(hTyped, h_[hState], KDA_FINALIZE_STATE_ELEMS);
        AscendC::DataCopy(dhTyped, dh_[dhState], KDA_FINALIZE_STATE_ELEMS);
        WaitLoads();
        AscendC::Cast(product, hTyped, AscendC::RoundMode::CAST_NONE,
                      KDA_FINALIZE_STATE_ELEMS);
        AscendC::Cast(dhFp, dhTyped, AscendC::RoundMode::CAST_NONE,
                      KDA_FINALIZE_STATE_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, product, dhFp, KDA_FINALIZE_STATE_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128(rhResident, product, KDA_FINALIZE_DIM);
        FinishVectorBeforeLoads();
    }

    __aicore__ inline void ComputeDkStateAndGate(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto dkState = Ub<float>(0);
        auto gateRaw = Ub<float>(32 * 1024);
        auto kTyped = Ub<bfloat16_t>(64 * 1024);
        auto product = Ub<float>(80 * 1024);
        auto gate = Ub<float>(112 * 1024);
        auto rh = Ub<float>(113 * 1024);
        auto expLast = Ub<float>(114 * 1024);
        AscendC::DataCopy(dkState, Ws<float>(ws, KDA_FINALIZE_WS_DK_STATE_RAW), elems);
        AscendC::DataCopy(gateRaw, gk_[token], elems);
        AscendC::DataCopy(kTyped, k_[token], elems);
        AscendC::DataCopy(rh, Ws<float>(ws, KDA_FINALIZE_WS_GK_LAST), KDA_FINALIZE_DIM);
        AscendC::DataCopy(
            expLast,
            Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK)[(rows - 1U) * KDA_FINALIZE_DIM],
            KDA_FINALIZE_DIM);
        WaitLoads();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Sub(
                gateRaw[row * KDA_FINALIZE_DIM],
                gateRaw[(rows - 1U) * KDA_FINALIZE_DIM],
                gateRaw[row * KDA_FINALIZE_DIM], KDA_FINALIZE_DIM);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gateRaw, gateRaw, KDA_FINALIZE_A2_LN2, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateRaw, gateRaw, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dkState, dkState, gateRaw, elems);
        AscendC::Cast(product, kTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, product, dkState, elems);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows128InPlace(product, rows);
        AscendC::Adds(gate, product, 0.0f, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(rh, rh, expLast, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(gate, gate, rh, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(Ws<float>(ws, KDA_FINALIZE_WS_DK_STATE_RAW), dkState, elems);
        AscendC::DataCopy(Ws<float>(ws, KDA_FINALIZE_WS_GK_LAST), gate, KDA_FINALIZE_DIM);
        DrainStores();
    }

    __aicore__ inline void ComputeDvDb(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto dvb = Ub<float>(0);
        auto vTyped = Ub<bfloat16_t>(32 * 1024);
        auto product = Ub<float>(48 * 1024);
        auto betaFp = Ub<float>(80 * 1024);
        auto betaTyped = Ub<BetaT>(81 * 1024);
        auto betaBroadcast = Ub<float>(82 * 1024);
        auto db = Ub<float>(84 * 1024);
        auto dvOut = Ub<bfloat16_t>(88 * 1024);
        AscendC::DataCopy(dvb, Ws<float>(ws, KDA_FINALIZE_WS_DVB), elems);
        AscendC::DataCopy(vTyped, v_[token], elems);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        LoadBeta(betaFp, betaOffset, betaTyped, rows);
        WaitLoads();
        AscendC::Cast(product, vTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, product, dvb, elems);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128(db, product, rows);
        BroadcastRows(betaBroadcast, betaFp, rows);
        MulRowsByScalar(dvb, dvb, betaBroadcast, rows, KDA_FINALIZE_DIM);
        AscendC::Cast(dvOut, dvb, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dv_[token], dvOut, elems);
        AscendC::DataCopy(Ws<float>(ws, KDA_FINALIZE_WS_DB_V), db, rows);
        DrainStores();
    }

    __aicore__ inline void ComputeDqBase(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t elems = static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto dq = Ub<float>(0);
        auto expG = Ub<float>(32 * 1024);
        AscendC::DataCopy(dq, dqRaw_[token], elems);
        AscendC::DataCopy(expG, Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK), elems);
        WaitLoads();
        AscendC::Mul(dq, dq, expG, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(dq, dq, tiling_->scale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dqBase_[token], dq, elems);
        DrainStores();
    }

    __aicore__ inline void ComputeDkDbBase(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto dkState = Ub<float>(0);
        auto dkgExp = Ub<float>(32 * 1024);
        auto betaTerm = Ub<float>(64 * 1024);
        auto kTyped = Ub<bfloat16_t>(96 * 1024);
        auto kFp = Ub<float>(112 * 1024);
        auto betaFp = Ub<float>(144 * 1024);
        auto betaTyped = Ub<BetaT>(145 * 1024);
        auto betaBroadcast = Ub<float>(146 * 1024);
        auto dbDelta = Ub<float>(148 * 1024);
        // RowReduce128 uses 64 scalar outputs plus 64 eight-float partials;
        // keep dbBase beyond that 2304-byte scratch range.
        auto dbBase = Ub<float>(152 * 1024);
        AscendC::DataCopy(dkState, Ws<float>(ws, KDA_FINALIZE_WS_DK_STATE_RAW), elems);
        AscendC::DataCopy(dkgExp, Ws<float>(ws, KDA_FINALIZE_WS_DKGB_RAW), elems);
        AscendC::DataCopy(betaTerm, Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK), elems);
        AscendC::DataCopy(kTyped, k_[token], elems);
        AscendC::DataCopy(dbBase, Ws<float>(ws, KDA_FINALIZE_WS_DB_V), rows);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        LoadBeta(betaFp, betaOffset, betaTyped, rows);
        WaitLoads();
        AscendC::Mul(dkgExp, dkgExp, betaTerm, elems);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(betaBroadcast, betaFp, rows);
        MulRowsByScalar(betaTerm, dkgExp, betaBroadcast, rows, KDA_FINALIZE_DIM);
        AscendC::Sub(dkState, dkState, betaTerm, elems);
        AscendC::Cast(kFp, kTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(kFp, kFp, dkgExp, elems);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128(dbDelta, kFp, rows);
        AscendC::Sub(dbBase, dbBase, dbDelta, rows);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dkBase_[token], dkState, elems);
        AscendC::DataCopy(dbBase_[token / KDA_FINALIZE_DIM], dbBase, rows);
        AscendC::DataCopy(Ws<float>(ws, KDA_FINALIZE_WS_DKGB_RAW), dkgExp, elems);
        DrainStores();
    }

    __aicore__ inline void ComputeDgBase(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto dg = Ub<float>(0);
        auto base = Ub<float>(32 * 1024);
        auto typed = Ub<bfloat16_t>(64 * 1024);
        auto operand = Ub<float>(80 * 1024);
        auto betaFp = Ub<float>(112 * 1024);
        auto betaTyped = Ub<BetaT>(113 * 1024);
        auto betaBroadcast = Ub<float>(114 * 1024);
        auto gate = Ub<float>(116 * 1024);

        AscendC::DataCopy(base, dqBase_[token], elems);
        AscendC::DataCopy(typed, q_[token], elems);
        WaitLoads();
        AscendC::Cast(operand, typed, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dg, operand, base, elems);
        AscendC::PipeBarrier<PIPE_V>();

        FinishVectorBeforeLoads();
        AscendC::DataCopy(base, Ws<float>(ws, KDA_FINALIZE_WS_DK_STATE_RAW), elems);
        AscendC::DataCopy(typed, k_[token], elems);
        WaitLoads();
        AscendC::Cast(operand, typed, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(base, base, operand, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dg, dg, base, elems);
        AscendC::PipeBarrier<PIPE_V>();

        FinishVectorBeforeLoads();
        AscendC::DataCopy(base, Ws<float>(ws, KDA_FINALIZE_WS_DKGB_RAW), elems);
        AscendC::DataCopy(gate, Ws<float>(ws, KDA_FINALIZE_WS_GK_LAST), KDA_FINALIZE_DIM);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        LoadBeta(betaFp, betaOffset, betaTyped, rows);
        WaitLoads();
        BroadcastRows(betaBroadcast, betaFp, rows);
        MulRowsByScalar(base, base, betaBroadcast, rows, KDA_FINALIZE_DIM);
        AscendC::Mul(base, base, operand, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dg, dg, base, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            dg[(rows - 1U) * KDA_FINALIZE_DIM],
            dg[(rows - 1U) * KDA_FINALIZE_DIM], gate,
            KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dgBase_[token], dg, elems);
        DrainStores();
    }

    __aicore__ inline void ComputeDkStateAndBaseFused(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t ws)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM);

        // Build gated dkState and its last-row correction first. The 32-KiB
        // dkState plane then stays resident through dG; the other state
        // temporaries are reused by the base-gradient phase. Peak UB is
        // 192 KiB: four FP32 planes, one BF16 plane, one FP32 operand plane,
        // and 16 KiB of reduction/scalar scratch.
        auto dkState = Ub<float>(0);
        auto gateRaw = Ub<float>(32 * 1024);
        auto stateKTyped = Ub<bfloat16_t>(64 * 1024);
        auto product = Ub<float>(80 * 1024);
        auto rh = Ub<float>(144 * 1024);
        auto gate = Ub<float>(184 * 1024);
        auto expLast = Ub<float>(185 * 1024);

        AscendC::DataCopy(
            dkState, Ws<float>(ws, KDA_FINALIZE_WS_DK_STATE_RAW), elems);
        AscendC::DataCopy(gateRaw, gk_[token], elems);
        AscendC::DataCopy(stateKTyped, k_[token], elems);
        AscendC::DataCopy(
            expLast,
            Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK)[
                (rows - 1U) * KDA_FINALIZE_DIM],
            KDA_FINALIZE_DIM);
        WaitLoads();
        constexpr uint32_t kHalfCols = KDA_FINALIZE_DIM / 2;
        constexpr uint8_t kGateRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        auto gateLast = gateRaw[(rows - 1U) * KDA_FINALIZE_DIM];
        AscendC::Sub(
            gateRaw, gateLast, gateRaw, kHalfCols,
            static_cast<uint8_t>(rows),
            {1, 1, 1, kGateRowStride, 0, kGateRowStride});
        AscendC::Sub(
            gateRaw[kHalfCols], gateLast[kHalfCols], gateRaw[kHalfCols],
            kHalfCols, static_cast<uint8_t>(rows),
            {1, 1, 1, kGateRowStride, 0, kGateRowStride});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gateRaw, gateRaw, KDA_FINALIZE_A2_LN2, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateRaw, gateRaw, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dkState, dkState, gateRaw, elems);
        AscendC::Cast(
            product, stateKTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();

        // gateRaw is dead after scaling dkState. Start the next 32-KiB
        // workspace loads while the row tree and last-row correction run.
        auto prefetchedExp = Ub<float>(112 * 1024);
        FinishVectorBeforeLoads();
        AscendC::DataCopy(
            gateRaw, Ws<float>(ws, KDA_FINALIZE_WS_DKGB_RAW), elems);
        AscendC::DataCopy(
            prefetchedExp, Ws<float>(ws, KDA_FINALIZE_WS_EXP2_GK), elems);
        AscendC::Mul(product, product, dkState, elems);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows128InPlace(product, rows);
        AscendC::Adds(gate, product, 0.0f, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(rh, rh, expLast, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(gate, gate, rh, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();

        WaitLoads();
        FinishVectorBeforeLoads();
        auto dkgExp = Ub<float>(32 * 1024);
        auto typed = stateKTyped;
        auto dqOrDg = Ub<float>(80 * 1024);
        auto expOrDk = prefetchedExp;
        auto operand = Ub<float>(144 * 1024);
        auto dbDelta = Ub<float>(176 * 1024);
        auto dbBase = Ub<float>(179 * 1024);
        auto betaFp = Ub<float>(181 * 1024);
        auto betaTyped = Ub<BetaT>(182 * 1024);

        AscendC::DataCopy(dqOrDg, dqRaw_[token], elems);
        AscendC::DataCopy(
            dbBase, Ws<float>(ws, KDA_FINALIZE_WS_DB_V), rows);
        const int64_t betaOffset = FinalizeTokenOffset(
            *tiling_, chunk, head, 1);
        LoadBeta(betaFp, betaOffset, betaTyped, rows);
        WaitLoads();

        AscendC::Mul(dqOrDg, dqOrDg, expOrDk, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(dqOrDg, dqOrDg, tiling_->scale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dkgExp, dkgExp, expOrDk, elems);
        AscendC::PipeBarrier<PIPE_V>();

        // operand temporarily holds the row-broadcast beta. expOrDk changes
        // from exp(g) to beta*dKgb, then becomes dk-base after dG consumes it.
        BroadcastRows(operand, betaFp, rows);
        MulRowsByScalar(
            expOrDk, dkgExp, operand, rows, KDA_FINALIZE_DIM);
        StartStores();
        AscendC::DataCopy(dqBase_[token], dqOrDg, elems);
        DrainStores();

        AscendC::Cast(operand, typed, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(operand, operand, dkgExp, elems);
        AscendC::PipeBarrier<PIPE_V>();

        // dkgExp is dead. Reuse its UB storage to fetch Q while the row
        // reduction updates dbBase; K remains resident from the first phase.
        auto qTyped = Ub<bfloat16_t>(32 * 1024);
        auto kTyped = typed;
        FinishVectorBeforeLoads();
        AscendC::DataCopy(qTyped, q_[token], elems);
        RowReduce128(dbDelta, operand, rows);
        AscendC::Sub(dbBase, dbBase, dbDelta, rows);
        AscendC::PipeBarrier<PIPE_V>();
        WaitLoads();
        AscendC::Cast(operand, qTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dqOrDg, operand, dqOrDg, elems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(operand, kTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(operand, operand, dkState, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dqOrDg, dqOrDg, operand, elems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(operand, kTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(operand, operand, expOrDk, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dqOrDg, dqOrDg, operand, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            dqOrDg[(rows - 1U) * KDA_FINALIZE_DIM],
            dqOrDg[(rows - 1U) * KDA_FINALIZE_DIM], gate,
            KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(expOrDk, dkState, expOrDk, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dkBase_[token], expOrDk, elems);
        AscendC::DataCopy(dbBase_[token / KDA_FINALIZE_DIM], dbBase, rows);
        AscendC::DataCopy(dgBase_[token], dqOrDg, elems);
        DrainStores();
    }

    __aicore__ inline void RunStateAndBase(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        ComputeRh(chunk, head, ws);
        ComputeDvDb(chunk, head, ws);
        ComputeDkStateAndBaseFused(chunk, head, ws);
    }

    __aicore__ inline void RunTzaResidual(
        uint32_t owner, int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto value = Ub<float>(0);
        auto reconstructed = Ub<float>(16 * 1024);
        auto high = Ub<bfloat16_t>(32 * 1024);
        auto low = Ub<bfloat16_t>(40 * 1024);
        AscendC::DataCopy(
            value, Ws<float>(ws, KDA_FINALIZE_WS_A2_RELAY_ZV),
            KDA_FINALIZE_MATRIX_ELEMS);
        WaitLoads();
        SplitBf16(value, high, low, reconstructed, KDA_FINALIZE_MATRIX_ELEMS);
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_A2_RELAY_DW), high,
            KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::DataCopy(
            Ws<bfloat16_t>(ws, KDA_FINALIZE_WS_A2_RELAY_DW +
                KDA_FINALIZE_MATRIX_BF16_BYTES), low,
            KDA_FINALIZE_MATRIX_ELEMS);
        DrainStores();
    }

    __aicore__ inline void RunDakk(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto raw = Ub<float>(0);
        AscendC::DataCopy(
            raw, Ws<float>(ws, KDA_FINALIZE_WS_A2_RELAY_ZV),
            KDA_FINALIZE_MATRIX_ELEMS);
        WaitLoads();
        AscendC::Muls(
            raw, raw, -1.0f, KDA_FINALIZE_MATRIX_ELEMS);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < KDA_FINALIZE_CHUNK; ++row) {
            ZeroRowFromColumn(raw[row * KDA_FINALIZE_CHUNK], row);
        }
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t matrixToken =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        StartStores();
        AscendC::DataCopy(dAkk_[matrixToken], raw, KDA_FINALIZE_MATRIX_ELEMS);
        DrainStores();
    }

    AscendC::GlobalTensor<bfloat16_t> q_, k_, v_, h_, dh_;
    AscendC::GlobalTensor<float> gk_, dqRaw_, dqBase_, dkBase_, dbBase_, dgBase_, dAkk_;
    AscendC::GlobalTensor<BetaT> beta_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    GM_ADDR workspaceAddr_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_;
    AscendC::TEventID vToMte2_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToMte2_;
    uint32_t subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
    uint32_t subBlockIdx_ = 0;
};

static_assert(KDA_FINALIZE_A2_UB_BYTES <= 192 * 1024,
              "A2 finalize Vector exceeds the 192 KiB UB budget.");

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_PRE_VECTOR_H
