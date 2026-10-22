/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_VECTOR_H

#include "chunk_kda_bwd_finalize_resident_common.h"

namespace KDA {

constexpr float KDA_FINALIZE_A2_RESIDENT_LN2 = 0.6931471805599453f;

#ifndef KDA_FINALIZE_A2_DIAG_SKIP_BASE_GRADIENTS
#define KDA_FINALIZE_A2_DIAG_SKIP_BASE_GRADIENTS 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_ZB
#define KDA_FINALIZE_A2_DIAG_SKIP_ZB 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_S6_DQ
#define KDA_FINALIZE_A2_DIAG_SKIP_S6_DQ 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_S6_DK
#define KDA_FINALIZE_A2_DIAG_SKIP_S6_DK 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_S6_RIGHT
#define KDA_FINALIZE_A2_DIAG_SKIP_S6_RIGHT 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_S6_FINAL
#define KDA_FINALIZE_A2_DIAG_SKIP_S6_FINAL 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_GATE_CARRY
#define KDA_FINALIZE_A2_DIAG_SKIP_GATE_CARRY 0
#endif
#ifndef KDA_FINALIZE_A2_DIAG_GATE_SCAN_ONLY
#define KDA_FINALIZE_A2_DIAG_GATE_SCAN_ONLY 0
#endif

// Bring-up implementation for the first resident stage. It is deliberately
// not selected by the production dispatcher until the dependent Cube stages
// are complete and the intermediate result has passed differential tests.
template <typename BetaT = bfloat16_t, typename RawGateT = float>
class ChunkKdaBwdFinalizeA2ResidentVectorStage0 {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh, GM_ADDR dAqk, GM_ADDR dqRaw,
        GM_ADDR dqOut, GM_ADDR dkOut, GM_ADDR dvOut,
        GM_ADDR dBetaOut, GM_ADDR dGOut,
        GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR workspace, const ChunkKdaBwdFinalizeTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        dqOut_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(dqOut));
        dkOut_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(dkOut));
        dvOut_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(dvOut));
        dBetaOut_.SetGlobalBuffer(
            reinterpret_cast<__gm__ BetaT *>(dBetaOut));
        dGOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dGOut));
        rawGFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawG));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        if (tiling->a2.hasDtBias != 0) {
            dtBias_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(dtBias));
        }
        // Streamed inputs and write-only outputs should not evict relay/state data.
        dAqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dqRaw_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        rawGFloat_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dqOut_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dkOut_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dvOut_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dBetaOut_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        workspaceAddr_ = workspace;
        tiling_ = tiling;
        pipe_ = pipe;
        subBlockNum_ = AscendC::GetSubBlockNum();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        pipe_->InitBuffer(ubBuf_, KDA_FINALIZE_A2_UB_END);
        ub_ = ubBuf_.Get<uint8_t>();
        mte2ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        vToMte3_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToMte2_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        mte3ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        vToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        scalarVToS_ = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        const int64_t logicalCore = AscendC::GetBlockIdx() / subBlockNum_;
        const bool fusedGate = ShouldUseFusedGate();
        if (fusedGate) {
            InitFusedGateState();
        }
        uint64_t workOrdinal = 0;
        bool kePrepared = false;
        bool baseEarlyPrepared = false;
        int64_t preparedChunkTask = -1;
        int64_t preparedHead = -1;
        for (int64_t head = 0; head < tiling_->NV; ++head) {
            const FinalizeA2ResidentChunkRange range =
                ResolveFinalizeA2ResidentChunkRange(
                    logicalCore, head, *tiling_);
            float gateScale = 0.0f;
            if (fusedGate) {
                BeginFusedGateHead(head);
                gateScale = ReadGateScalar(
                    Ub<float>(KDA_FINALIZE_A2_UB_GATE_EXPA)[head]);
            }
            for (int64_t local = 0; local < range.count; ++local) {
                const int64_t chunkTask = range.begin + local;
                FinalizeChunkInfo chunk;
                ResolveFinalizeChunk(chunkTask, cuSeqlens_,
                                     chunkIndices_, *tiling_, chunk);
                if (!chunk.valid) {
                    continue;
                }
                const uint64_t slotBase =
                    FinalizeA2ResidentSlotBase(
                        logicalCore, static_cast<uint32_t>(
                            workOrdinal % KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH));
                const bool usePreparedKe =
                    kePrepared && preparedChunkTask == chunkTask &&
                    preparedHead == head;
                if (usePreparedKe) {
                    kePrepared = false;
                } else {
                    BuildKe(chunk, head, slotBase);
                    PublishKe();
                }
                const bool usePreparedBase =
                    baseEarlyPrepared && preparedChunkTask == chunkTask &&
                    preparedHead == head;
                if (usePreparedBase) {
                    baseEarlyPrepared = false;
                } else {
                    BuildBaseEarly(chunk, head);
                }
                WaitCubeStage(KDA_FINALIZE_A2_FLAG_BASE0_READY);
                DrainVectorToLoads();
                BuildBasePrologue(chunk, head, slotBase);
                WaitCubeStage(KDA_FINALIZE_A2_FLAG_BASE1_READY);
                if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_ZB) {
                    BuildZb(chunk, head, slotBase);
                }
                PublishZbReady();
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE4) {
                    BuildIntraPre(chunk, head, slotBase);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        KDA_FINALIZE_A2_FLAG_INTRA_INPUT_READY);
                }
                if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_BASE_GRADIENTS) {
                    BuildBaseGradients(chunk, head, slotBase);
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3) {
                    WaitCubeStage(KDA_FINALIZE_A2_FLAG_DAKK_READY);
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE4) {
                    BuildIntraPost(chunk, head, slotBase);
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        KDA_FINALIZE_A2_FLAG_INTRA_DK_INPUT_READY);
                    BuildIntraRemaining(chunk, head, slotBase);
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5) {
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_INPUT_READY);
                    WaitCubeStage(KDA_FINALIZE_A2_FLAG_INTRA_DQ_READY);
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE6) {
                        if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_S6_DQ) {
                            PrepareIntraDq(chunk, head, slotBase);
                        }
                    }
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_SPLIT_INTRA) {
                        WaitCubeStage(KDA_FINALIZE_A2_FLAG_INTRA_DK_READY);
                    }
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE6) {
                        if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_S6_DK) {
                            PrepareIntraDk(chunk, head, slotBase);
                        }
                    }
                    if constexpr (
                            KDA_FINALIZE_A2_RESIDENT_GATE_SIGMOID_PREFETCH) {
                        if (fusedGate) {
                            PrecomputeGateSigmoid(
                                chunk, head, gateScale);
                        }
                    }
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_SPLIT_INTRA) {
                        WaitCubeStage(
                            KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_READY);
                    }
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE6) {
                    if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_S6_RIGHT) {
                        PrepareIntraRight(
                            chunk, slotBase, fusedGate);
                    }
                    if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_S6_FINAL) {
                        FinalizeIntraLeft(chunk, head, slotBase);
                    }
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE6) {
                    if (fusedGate) {
                        AscendC::PipeBarrier<PIPE_V>();
                        DrainStoresToVector();
                        DrainStoresToLoads();
                        PrepareFusedGateScan(logicalCore);
                        bool nextWorkPrepared = false;
                        FinalizeChunkInfo nextChunk{};
                        int64_t nextHead = 0;
                        int64_t nextChunkTask = -1;
                        if (FindNextWork(
                                logicalCore, head, local, nextChunk,
                                nextHead, nextChunkTask)) {
                            const uint64_t nextSlotBase =
                                FinalizeA2ResidentSlotBase(
                                    logicalCore, static_cast<uint32_t>(
                                        (workOrdinal + 1U) %
                                        KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH));
                            DrainVectorToLoads();
                            BuildKe(nextChunk, nextHead, nextSlotBase);
                            PublishKe();
                            kePrepared = true;
                            preparedChunkTask = nextChunkTask;
                            preparedHead = nextHead;
                            nextWorkPrepared = true;
                        }
                        ApplyFusedGate(
                            chunk, head, logicalCore, slotBase, gateScale);
                        if (nextWorkPrepared) {
                            DrainVectorToLoads();
                            BuildBaseEarly(nextChunk, nextHead);
                            baseEarlyPrepared = true;
                        }
                        DrainStoresToVector();
                        DrainStoresToLoads();
                        PublishWorkDone();
                    } else {
                        StoreUngatedDg(chunk, head);
                        DrainStoresToVector();
                        DrainStoresToLoads();
                        PublishWorkDone();
                    }
                }
                ++workOrdinal;
            }
            if (fusedGate) {
                EndFusedGateHead(head);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        DrainStoresToVector();
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(scalarVToS_);
    }

private:
    __aicore__ inline bool FindNextWork(
        int64_t logicalCore, int64_t currentHead, int64_t currentLocal,
        FinalizeChunkInfo &nextChunk, int64_t &nextHead,
        int64_t &nextChunkTask)
    {
        for (int64_t head = currentHead; head < tiling_->NV; ++head) {
            const FinalizeA2ResidentChunkRange range =
                ResolveFinalizeA2ResidentChunkRange(
                    logicalCore, head, *tiling_);
            const int64_t firstLocal =
                head == currentHead ? currentLocal + 1 : 0;
            for (int64_t local = firstLocal; local < range.count; ++local) {
                nextChunkTask = range.begin + local;
                ResolveFinalizeChunk(
                    nextChunkTask, cuSeqlens_, chunkIndices_,
                    *tiling_, nextChunk);
                if (nextChunk.valid) {
                    nextHead = head;
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline void PublishWorkDone()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
            KDA_FINALIZE_A2_FLAG_WORK_DONE);
    }

    __aicore__ inline void PublishZbReady()
    {
        DrainStoresToLoads();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
            KDA_FINALIZE_A2_FLAG_ZB_READY);
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

    __aicore__ inline void WaitCubeStage(uint32_t flag)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag);
    }

    __aicore__ inline void DrainVectorToLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
    }

    __aicore__ inline void StartStores()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
    }

    __aicore__ inline void DrainStoresToLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void DrainStoresToVector()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void SyncAivWritersToLoads()
    {
        DrainStoresToLoads();
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
    }

    __aicore__ inline void SyncAivReadersToStores()
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
    }

    __aicore__ inline void SplitBf16(
        AscendC::LocalTensor<float> value,
        AscendC::LocalTensor<bfloat16_t> high,
        AscendC::LocalTensor<bfloat16_t> low,
        AscendC::LocalTensor<float> reconstructed, uint32_t count)
    {
        AscendC::Cast(high, value, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            reconstructed, high, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(value, value, reconstructed, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(low, value, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> scalar,
        uint32_t rows)
    {
        AscendC::Brcb(
            dst, scalar, static_cast<uint8_t>((rows + 7U) / 8U), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulRowsByScalar(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        AscendC::LocalTensor<float> broadcast, uint32_t rows, uint32_t cols)
    {
        const uint8_t stride =
            static_cast<uint8_t>(cols * sizeof(float) / 32U);
        for (uint32_t col = 0; col < cols; col += 64U) {
            const uint32_t mask =
                cols - col < 64U ? cols - col : 64U;
            AscendC::Mul(dst[col], src[col], broadcast, mask,
                         static_cast<uint8_t>(rows),
                         {1, 1, 0, stride, stride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SubRowsByVector(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> rowVector,
        AscendC::LocalTensor<float> src, uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint16_t kRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = kRowStride;
        params.src0RepStride = 0;
        params.src1RepStride = kRowStride;
        AscendC::Sub(
            dst, rowVector, src, static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::Sub(
            dst[kHalfRow], rowVector[kHalfRow], src[kHalfRow],
            static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildCenteredExponentials(
        AscendC::LocalTensor<float> negative,
        AscendC::LocalTensor<float> positive,
        AscendC::LocalTensor<float> centerExp,
        AscendC::LocalTensor<float> expG, uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint16_t kRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::BinaryRepeatParams vectorOverRows;
        vectorOverRows.dstBlkStride = 1;
        vectorOverRows.src0BlkStride = 1;
        vectorOverRows.src1BlkStride = 1;
        vectorOverRows.dstRepStride = kRowStride;
        vectorOverRows.src0RepStride = 0;
        vectorOverRows.src1RepStride = kRowStride;
        AscendC::BinaryRepeatParams rowsOverVector = vectorOverRows;
        rowsOverVector.src0RepStride = kRowStride;
        rowsOverVector.src1RepStride = 0;
        AscendC::Div(
            negative, centerExp, expG, static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), vectorOverRows);
        AscendC::Div(
            negative[kHalfRow], centerExp[kHalfRow], expG[kHalfRow],
            static_cast<uint64_t>(kHalfRow), static_cast<uint8_t>(rows),
            vectorOverRows);
        AscendC::Div(
            positive, expG, centerExp, static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), rowsOverVector);
        AscendC::Div(
            positive[kHalfRow], expG[kHalfRow], centerExp[kHalfRow],
            static_cast<uint64_t>(kHalfRow), static_cast<uint8_t>(rows),
            rowsOverVector);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduce128(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint16_t kRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = kRowStride;
        params.src0RepStride = kRowStride;
        params.src1RepStride = kRowStride;
        AscendC::Add(
            src, src, src[kHalfRow], static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            dst, src, kHalfRow, static_cast<uint8_t>(rows),
            1, 1, kRowStride);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduce128InPlace(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint16_t kRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = kRowStride;
        params.src0RepStride = kRowStride;
        params.src1RepStride = kRowStride;
        AscendC::Add(
            src, src, src[kHalfRow], static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            dst, src, kHalfRow, static_cast<uint8_t>(rows),
            1, 1, kRowStride);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool ShouldUseFusedGate() const
    {
        if constexpr (!AscendC::IsSameType<RawGateT, float>::value) {
            return false;
        }
        return FinalizeA2CanUseFusedGate(*tiling_);
    }

    template <typename T>
    __aicore__ inline AscendC::GlobalTensor<T> PartialWs(uint64_t offset)
    {
        AscendC::GlobalTensor<T> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(
            workspaceAddr_ + offset));
        return tensor;
    }

    __aicore__ inline uint64_t GatePaOffset(int64_t head) const
    {
        const uint64_t physicalAiv = AscendC::GetBlockIdx();
        return tiling_->gatePartialOffset +
            (physicalAiv * static_cast<uint64_t>(tiling_->NV) +
             static_cast<uint64_t>(head)) *
                KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES;
    }

    __aicore__ inline uint64_t GatePbOffset(int64_t head) const
    {
        const uint64_t physicalAiv = AscendC::GetBlockIdx();
        return tiling_->dtBiasPartialOffset +
            (physicalAiv * static_cast<uint64_t>(tiling_->NV) +
             static_cast<uint64_t>(head)) *
                KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES;
    }

    __aicore__ inline void InitFusedGateState()
    {
        auto expA = Ub<float>(KDA_FINALIZE_A2_UB_GATE_EXPA);
        auto one = Ub<float>(KDA_FINALIZE_A2_UB_GATE_ONE);
        AscendC::DataCopyPad(
            expA, aLog_[0],
            {1, static_cast<uint32_t>(tiling_->NV * sizeof(float)),
             0, 0, 0},
            {false, 0, 0, 0});
        WaitLoads();
        AscendC::Duplicate(one, 1.0f, 64);
        AscendC::Exp(expA, expA, static_cast<uint32_t>(tiling_->NV));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BeginFusedGateHead(int64_t head)
    {
        auto paAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PA_ACC);
        auto pbAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PB_ACC);
        auto bias = Ub<float>(KDA_FINALIZE_A2_UB_GATE_BIAS);
        AscendC::Duplicate(paAcc, 0.0f, 8);
        if (tiling_->a2.hasDtBias != 0) {
            AscendC::Duplicate(pbAcc, 0.0f, KDA_FINALIZE_DIM);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (tiling_->a2.hasDtBias != 0) {
            AscendC::DataCopyPad(
                bias,
                dtBias_[static_cast<uint64_t>(head) * KDA_FINALIZE_DIM],
                {1, KDA_FINALIZE_DIM * sizeof(float), 0, 0, 0},
                {false, 0, 0, 0});
            WaitLoads();
        }
    }

    __aicore__ inline void EndFusedGateHead(int64_t head)
    {
        auto paAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PA_ACC);
        auto pbAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PB_ACC);
        StartStores();
        AscendC::DataCopyPad(
            PartialWs<float>(GatePaOffset(head)), paAcc,
            {1, KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES, 0, 0, 0});
        if (tiling_->a2.hasDtBias != 0) {
            AscendC::DataCopyPad(
                PartialWs<float>(GatePbOffset(head)), pbAcc,
                {1, KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES, 0, 0, 0});
        }
        DrainStoresToVector();
        DrainStoresToLoads();
    }

    template <uint32_t STEP>
    __aicore__ inline void GateScan32(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src)
    {
        constexpr uint32_t activeRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS - STEP;
        AscendC::Add(
            dst, src, src[STEP * KDA_FINALIZE_DIM],
            activeRows * KDA_FINALIZE_DIM);
        AscendC::Adds(
            dst[activeRows * KDA_FINALIZE_DIM],
            src[activeRows * KDA_FINALIZE_DIM], 0.0f,
            STEP * KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddRowsByVector(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src,
        AscendC::LocalTensor<float> rowVector, uint32_t rows)
    {
        constexpr uint32_t kHalfRow = KDA_FINALIZE_DIM / 2;
        constexpr uint16_t kRowStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = kRowStride;
        params.src0RepStride = kRowStride;
        params.src1RepStride = 0;
        AscendC::Add(
            dst, src, rowVector, static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::Add(
            dst[kHalfRow], src[kHalfRow], rowVector[kHalfRow],
            static_cast<uint64_t>(kHalfRow),
            static_cast<uint8_t>(rows), params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReadGateScalar(
        AscendC::LocalTensor<float> value)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        return ((__ubuf__ float *)value.GetPhyAddr())[0];
    }

    __aicore__ inline void GateSigmoid(
        AscendC::LocalTensor<float> value,
        AscendC::LocalTensor<float> one, uint32_t count)
    {
        AscendC::Exp(value, value, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(value, value, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 0;
        params.src1RepStride = 8;
        AscendC::Div(
            value, one, value, static_cast<uint64_t>(64),
            static_cast<uint8_t>(count / 64U), params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceGateColumns(
        AscendC::LocalTensor<float> partial,
        AscendC::LocalTensor<float> values)
    {
        constexpr uint32_t kHalfRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS / 2U;
        AscendC::Add(
            partial, values, values[kHalfRows * KDA_FINALIZE_DIM],
            kHalfRows * KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            partial, partial, partial[8U * KDA_FINALIZE_DIM],
            8U * KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            partial, partial, partial[4U * KDA_FINALIZE_DIM],
            4U * KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            partial, partial, partial[2U * KDA_FINALIZE_DIM],
            2U * KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            partial, partial, partial[KDA_FINALIZE_DIM],
            KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateGateColumns(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> partial,
        AscendC::LocalTensor<float> values)
    {
        ReduceGateColumns(partial, values);
        AscendC::Add(acc, acc, partial, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreUngatedDg(
        const FinalizeChunkInfo &chunk, int64_t head)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM) +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        StartStores();
        AscendC::DataCopyPad(
            dGOut_[token], Ub<float>(KDA_FINALIZE_A2_UB_WORK3),
            {1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0});
    }

    __aicore__ inline void PrecomputeGateSigmoid(
        const FinalizeChunkInfo &chunk, int64_t head, float gateScale)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM) +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        auto raw = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto sigmoid = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        auto one = Ub<float>(KDA_FINALIZE_A2_UB_GATE_ONE);
        auto bias = Ub<float>(KDA_FINALIZE_A2_UB_GATE_BIAS);

        DrainStoresToLoads();
        AscendC::DataCopyPad(
            raw, rawGFloat_[token],
            {1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        WaitLoads();
        if (tiling_->a2.hasDtBias != 0) {
            AddRowsByVector(raw, raw, bias, rows);
        }
        AscendC::Muls(sigmoid, raw, -gateScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        GateSigmoid(sigmoid, one, elems);
        DrainVectorToLoads();
    }

    __aicore__ inline void PrecomputeGateDerivative(
        const FinalizeChunkInfo &chunk)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        auto derivative = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        auto oneMinusSigmoid = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto one = Ub<float>(KDA_FINALIZE_A2_UB_GATE_ONE);
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 0;
        params.src1RepStride = 8;
        AscendC::Sub(
            oneMinusSigmoid, one, derivative, static_cast<uint64_t>(64),
            static_cast<uint8_t>(elems / 64U), params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(
            derivative, derivative, oneMinusSigmoid, elems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareFusedGateScan(int64_t logicalCore)
    {
        auto upstream0 = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto upstream1 = Ub<float>(KDA_FINALIZE_A2_UB_WORK5);
        GateScan32<1>(upstream1, upstream0);
        GateScan32<2>(upstream0, upstream1);
        GateScan32<4>(upstream1, upstream0);
        GateScan32<8>(upstream0, upstream1);
        GateScan32<16>(upstream1, upstream0);
        if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_GATE_CARRY) {
            if (subBlockIdx_ == 1U) {
                StartStores();
                AscendC::DataCopyPad(
                    PartialWs<float>(
                        FinalizeA2GateCarryOffset(logicalCore, *tiling_)),
                    upstream1,
                    {1, KDA_FINALIZE_DIM * sizeof(float), 0, 0, 0});
            }
        }
    }

    __aicore__ inline void ApplyFusedGate(
        const FinalizeChunkInfo &chunk, int64_t head, int64_t logicalCore,
        uint64_t slotBase, float gateScale)
    {
        AscendC::PipeBarrier<PIPE_V>();
        DrainStoresToVector();
        constexpr uint32_t rows = KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowBegin = subBlockIdx_ * rows;
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM) +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;

        auto raw = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto sigmoidDerivative = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        // The 220x UB maps 16 KiB-spaced buffers to the same bank group.
        // These offsets use storage whose previous owner is dead after the
        // scan, separating the hot Gate operands without growing the TBuf.
        auto product = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto partial = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto pbAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PB_ACC);
        auto bias = Ub<float>(KDA_FINALIZE_A2_UB_GATE_BIAS);
        auto paAcc = Ub<float>(KDA_FINALIZE_A2_UB_GATE_PA_ACC);
        auto upstream1 = Ub<float>(KDA_FINALIZE_A2_UB_WORK5);

        if constexpr (KDA_FINALIZE_A2_DIAG_GATE_SCAN_ONLY) {
            StartStores();
            AscendC::DataCopyPad(
                dGOut_[token], upstream1,
                {1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0});
            return;
        }

        if constexpr (!KDA_FINALIZE_A2_DIAG_SKIP_GATE_CARRY) {
            DrainVectorToLoads();
            DrainStoresToLoads();
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            if (subBlockIdx_ == 0U) {
                AscendC::DataCopyPad(
                    partial,
                    PartialWs<float>(
                        FinalizeA2GateCarryOffset(logicalCore, *tiling_)),
                    {1, KDA_FINALIZE_DIM * sizeof(float), 0, 0, 0},
                    {false, 0, 0, 0});
                WaitLoads();
                AddRowsByVector(upstream1, upstream1, partial, rows);
            }
        }
        AscendC::Mul(
            upstream1, upstream1, sigmoidDerivative, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            upstream1, upstream1,
            tiling_->a2.lowerBound * gateScale, elems);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(product, upstream1, raw, elems);
        AscendC::PipeBarrier<PIPE_V>();
        if (tiling_->a2.hasDtBias != 0) {
            AccumulateGateColumns(pbAcc, partial, upstream1);
        }
        AscendC::BlockReduceSum(
            partial, product, 64U, 64U, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(
            partial, partial, 8U, 64U, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            partial[64], partial, 64U, 1U, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(paAcc, paAcc, partial[64], 1U);
        AscendC::PipeBarrier<PIPE_V>();

        StartStores();
        AscendC::DataCopyPad(
            dGOut_[token], upstream1,
            {1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0});
    }

    __aicore__ inline void BuildBaseEarly(
        const FinalizeChunkInfo &chunk, int64_t head)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t tokenBase = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t token = tokenBase +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;

        auto expG = Ub<float>(KDA_FINALIZE_A2_UB_EXP);
        auto rawGk = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);
        auto dqBase = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        auto gateRaw = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto gate = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto centerExp = Ub<float>(KDA_FINALIZE_A2_UB_CENTER_EXP);
        AscendC::DataCopy(dqBase, dqRaw_[token], elems);
        AscendC::DataCopy(
            gate,
            gk_[tokenBase +
                (static_cast<int64_t>(chunk.validRows) - 1) *
                    KDA_FINALIZE_DIM],
            KDA_FINALIZE_DIM);
        if (subBlockIdx_ != 0U) {
            AscendC::DataCopy(centerExp, gk_[tokenBase], KDA_FINALIZE_DIM);
        }
        WaitLoads();
        if (subBlockIdx_ == 0U) {
            AscendC::Add(
                centerExp, rawGk, gate, KDA_FINALIZE_DIM);
        } else {
            AscendC::Add(
                centerExp, centerExp, gate, KDA_FINALIZE_DIM);
        }
        AscendC::Mul(dqBase, dqBase, expG, elems);
        SubRowsByVector(gateRaw, gate, rawGk, rows);
        AscendC::Muls(dqBase, dqBase, tiling_->scale, elems);
        AscendC::Muls(
            gateRaw, gateRaw, KDA_FINALIZE_A2_RESIDENT_LN2, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(centerExp, centerExp, 0.5f, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            centerExp, centerExp, KDA_FINALIZE_A2_RESIDENT_LN2,
            KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateRaw, gateRaw, elems);
        AscendC::Exp(centerExp, centerExp, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildBasePrologue(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        constexpr uint32_t kStateRowsPerAiv =
            KDA_FINALIZE_A2_RESIDENT_STATE_ROWS;
        constexpr uint32_t kStateElems =
            kStateRowsPerAiv * KDA_FINALIZE_DIM;
        const uint32_t stateRowBegin =
            subBlockIdx_ * kStateRowsPerAiv;
        const int64_t hState =
            FinalizeHOffset(*tiling_, chunk, head) +
            static_cast<int64_t>(stateRowBegin) * KDA_FINALIZE_DIM;
        const int64_t dhState =
            FinalizeDhOffset(*tiling_, chunk, head) +
            static_cast<int64_t>(stateRowBegin) * KDA_FINALIZE_DIM;
        const int64_t tokenBase = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM);
        auto hTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK0);
        auto dhTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK1);
        auto product = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto dhFp = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto rhHalf = Ub<float>(KDA_FINALIZE_A2_UB_IO);
        AscendC::DataCopy(hTyped, h_[hState], kStateElems);
        AscendC::DataCopy(dhTyped, dh_[dhState], kStateElems);
        WaitLoads();
        AscendC::Cast(
            product, hTyped, AscendC::RoundMode::CAST_NONE, kStateElems);
        AscendC::Cast(
            dhFp, dhTyped, AscendC::RoundMode::CAST_NONE, kStateElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, product, dhFp, kStateElems);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128InPlace(rhHalf, product, kStateRowsPerAiv);
        AscendC::Duplicate(
            rhHalf[KDA_FINALIZE_A2_RESIDENT_STATE_ROWS], 0.0f,
            KDA_FINALIZE_A2_RESIDENT_STATE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_FINAL_RH +
                subBlockIdx_ * 512),
            rhHalf, KDA_FINALIZE_DIM);
        DrainStoresToLoads();
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        const int64_t token = tokenBase +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        const int64_t scalarToken =
            FinalizeTokenOffset(*tiling_, chunk, head, 1) + rowBegin;

        auto dvb = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto vTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK3);
        auto productDv = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto betaBroadcast = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto dvTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK5);
        auto betaTyped = Ub<BetaT>(KDA_FINALIZE_A2_UB_IO);
        auto betaFp = Ub<float>(KDA_FINALIZE_A2_UB_IO + 256);
        auto dbBase = Ub<float>(KDA_FINALIZE_A2_UB_IO + 512);
        DrainVectorToLoads();
        AscendC::DataCopy(
            dvb,
            Ws<float>(slotBase,
                KDA_FINALIZE_A2_WS_DVB + rowByteOffset),
            elems);
        AscendC::DataCopy(vTyped, v_[token], elems);
        AscendC::DataCopy(betaTyped, beta_[scalarToken], rows);
        WaitLoads();
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            AscendC::Adds(betaFp, betaTyped, 0.0f, rows);
        } else {
            AscendC::Cast(
                betaFp, betaTyped, AscendC::RoundMode::CAST_NONE, rows);
        }
        AscendC::Cast(
            productDv, vTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(productDv, productDv, dvb, elems);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128(dbBase, productDv, rows);
        BroadcastRows(betaBroadcast, betaFp, rows);
        MulRowsByScalar(
            dvb, dvb, betaBroadcast, rows, KDA_FINALIZE_DIM);
        AscendC::Cast(
            dvTyped, dvb, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dvOut_[token], dvTyped, elems);
        DrainStoresToLoads();
    }

    __aicore__ inline void PreloadBaseGradientRelay(
        const FinalizeChunkInfo &chunk, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        auto dkState = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto dkgExp = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);
        AscendC::DataCopy(
            dkState,
            Ws<float>(slotBase,
                KDA_FINALIZE_A2_WS_DK_STATE + rowByteOffset),
            elems);
        AscendC::DataCopy(
            dkgExp,
            Ws<float>(slotBase,
                KDA_FINALIZE_A2_WS_DKGB + rowByteOffset),
            elems);
        WaitLoads();
    }

    __aicore__ inline void BuildBaseGradients(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        const uint32_t validRows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t rows = rowBegin < validRows
            ? FinalizeMin(
                  KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
                  validRows - rowBegin)
            : 0U;
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        const int64_t tokenBase = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t token = tokenBase +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        const int64_t scalarToken =
            FinalizeTokenOffset(*tiling_, chunk, head, 1) + rowBegin;
        auto expG = Ub<float>(KDA_FINALIZE_A2_UB_EXP);
        auto qTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_Q);
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto dqBase = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        auto dkState = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto dgBase = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);
        auto gateRaw = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto dkBase = Ub<float>(KDA_FINALIZE_A2_UB_WORK5);
        auto dkgExp = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto betaTerm = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto scratch = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto kDk = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto betaTyped = Ub<BetaT>(KDA_FINALIZE_A2_UB_IO);
        auto betaFp = Ub<float>(KDA_FINALIZE_A2_UB_IO + 256);
        auto dbBase = Ub<float>(KDA_FINALIZE_A2_UB_IO + 512);
        auto dbDelta = Ub<float>(KDA_FINALIZE_A2_UB_IO + 768);
        auto gatePeer = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto betaBroadcast = Ub<float>(KDA_FINALIZE_A2_UB_IO + 3072);

        if (rows != 0U) {
            AscendC::DataCopy(
                dkState,
                Ws<float>(slotBase,
                    KDA_FINALIZE_A2_WS_DK_STATE + rowByteOffset),
                elems);
            AscendC::DataCopy(
                dkgExp,
                Ws<float>(slotBase,
                    KDA_FINALIZE_A2_WS_DKGB + rowByteOffset),
                elems);
            AscendC::Cast(
                scratch, qTyped, AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dgBase, scratch, dqBase, elems);
            AscendC::PipeBarrier<PIPE_V>();
            WaitLoads();
            AscendC::Mul(dkState, dkState, gateRaw, elems);
            AscendC::Mul(dkgExp, dkgExp, expG, elems);
            AscendC::PipeBarrier<PIPE_V>();
            BroadcastRows(betaBroadcast, betaFp, rows);
            MulRowsByScalar(
                betaTerm, dkgExp, betaBroadcast,
                rows, KDA_FINALIZE_DIM);
            AscendC::Mul(scratch, kFp, dkgExp, elems);
            AscendC::Mul(kDk, kFp, dkState, elems);
            AscendC::Sub(dkBase, dkState, betaTerm, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dkState, kFp, betaTerm, elems);
            AscendC::Sub(dgBase, dgBase, kDk, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(dgBase, dgBase, dkState, elems);
            AscendC::PipeBarrier<PIPE_V>();
            if (rows == KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS) {
                ReduceGateColumns(gatePeer, kDk);
            } else {
                AscendC::Duplicate(
                    gatePeer, 0.0f, KDA_FINALIZE_DIM / 2U);
                AscendC::Duplicate(
                    gatePeer[KDA_FINALIZE_DIM / 2U], 0.0f,
                    KDA_FINALIZE_DIM / 2U);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t localRow = 0; localRow < rows; ++localRow) {
                    AscendC::Add(
                        gatePeer, gatePeer,
                        kDk[localRow * KDA_FINALIZE_DIM],
                        KDA_FINALIZE_DIM / 2U);
                    AscendC::Add(
                        gatePeer[KDA_FINALIZE_DIM / 2U],
                        gatePeer[KDA_FINALIZE_DIM / 2U],
                        kDk[localRow * KDA_FINALIZE_DIM +
                                KDA_FINALIZE_DIM / 2U],
                        KDA_FINALIZE_DIM / 2U);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        } else {
            AscendC::Duplicate(
                gatePeer, 0.0f, KDA_FINALIZE_DIM / 2U);
            AscendC::Duplicate(
                gatePeer[KDA_FINALIZE_DIM / 2U], 0.0f,
                KDA_FINALIZE_DIM / 2U);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (rows != 0U) {
            RowReduce128(dbDelta, scratch, rows);
            AscendC::Sub(dbBase, dbBase, dbDelta, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        StartStores();
        AscendC::DataCopy(
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_FINAL_GATE +
                subBlockIdx_ * KDA_FINALIZE_DIM * sizeof(float)),
            gatePeer,
            KDA_FINALIZE_DIM);
        DrainStoresToLoads();
    }

    __aicore__ inline void BuildKe(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM) +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        const uint32_t keByteOffset = KDA_FINALIZE_A2_WS_KE +
            rowBegin * KDA_FINALIZE_DIM * sizeof(bfloat16_t);

        auto expG = Ub<float>(KDA_FINALIZE_A2_UB_EXP);
        auto qTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_Q);
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto rawGk = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);
        auto value = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto keTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK1);
        AscendC::DataCopy(qTyped, q_[token], elems);
        AscendC::DataCopy(keTyped, k_[token], elems);
        AscendC::DataCopy(expG, gk_[token], elems);
        WaitLoads();
        AscendC::Adds(rawGk, expG, 0.0f, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(expG, expG, KDA_FINALIZE_A2_RESIDENT_LN2, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expG, expG, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kFp, keTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(value, kFp, expG, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(keTyped, value, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(Ws<bfloat16_t>(slotBase, keByteOffset),
                          keTyped, elems);
    }

    __aicore__ inline void PublishKe()
    {
        DrainStoresToLoads();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
            KDA_FINALIZE_A2_FLAG_KE_READY);
    }

    __aicore__ inline void ZeroUpperTriangle(
        AscendC::LocalTensor<float> value,
        AscendC::LocalTensor<uint8_t> rowMasks,
        AscendC::LocalTensor<float> zero,
        uint32_t rowBegin, uint32_t rows,
        uint32_t diagonalOffset = 0U)
    {
        __ubuf__ uint64_t *maskPtr =
            reinterpret_cast<__ubuf__ uint64_t *>(rowMasks.GetPhyAddr());
        for (uint32_t localRow = 0; localRow < rows; ++localRow) {
            const uint32_t column =
                rowBegin + localRow + diagonalOffset;
            maskPtr[localRow] = column >= KDA_FINALIZE_CHUNK
                ? 0ULL
                : (column == 0U ? ~0ULL : (~0ULL << column));
        }
        AscendC::Duplicate(zero, 0.0f, KDA_FINALIZE_CHUNK);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::BinaryRepeatParams params = {1, 0, 1, 8, 0, 8};
        AscendC::Select(
            value, rowMasks, zero, value,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            KDA_FINALIZE_CHUNK, static_cast<uint8_t>(rows), params);
    }

    __aicore__ inline void BuildZb(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_CHUNK;
        const int64_t betaOffset =
            FinalizeTokenOffset(*tiling_, chunk, head, 1);
        const uint32_t matrixFp32ByteOffset =
            rowBegin * KDA_FINALIZE_CHUNK * sizeof(float);
        const uint32_t matrixBf16ByteOffset =
            rowBegin * KDA_FINALIZE_CHUNK * sizeof(bfloat16_t);

        auto zbTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK0);
        auto zv = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto zw = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto result = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto betaFp = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto betaTyped = Ub<BetaT>(KDA_FINALIZE_A2_UB_WORK4 + 1024);
        auto rowMasks = Ub<uint8_t>(KDA_FINALIZE_A2_UB_WORK4 + 2048);
        auto zero = Ub<float>(KDA_FINALIZE_A2_UB_WORK4 + 3072);

        AscendC::DataCopy(
            zv,
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_ZV +
                matrixFp32ByteOffset), elems);
        AscendC::DataCopy(
            zw,
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_ZW +
                matrixFp32ByteOffset), elems);
        AscendC::DataCopy(
            betaTyped, beta_[betaOffset], KDA_FINALIZE_CHUNK);
        WaitLoads();
        AscendC::Cast(
            betaFp, betaTyped, AscendC::RoundMode::CAST_NONE,
            KDA_FINALIZE_CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(result, zw, zv, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(
            result, result, betaFp, KDA_FINALIZE_CHUNK,
            static_cast<uint8_t>(rows), {1, 1, 1, 8, 8, 0});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        ZeroUpperTriangle(result, rowMasks, zero, rowBegin, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            zbTyped, result, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(slotBase, KDA_FINALIZE_A2_WS_ZB +
                matrixBf16ByteOffset), zbTyped, elems);
    }

    __aicore__ inline void PackMatrix(
        AscendC::GlobalTensor<float> src, uint64_t srcOffset,
        uint64_t slotBase,
        uint32_t dstByteOffset, uint32_t rowBegin, uint32_t rows,
        float scale, uint32_t diagonalOffset)
    {
        constexpr uint32_t sliceRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t sliceElems =
            sliceRows * KDA_FINALIZE_CHUNK;
        constexpr uint32_t planeBytes =
            KDA_FINALIZE_MATRIX_BF16_BYTES;
        const uint32_t elems = rows * KDA_FINALIZE_CHUNK;
        const uint32_t rowElems = rowBegin * KDA_FINALIZE_CHUNK;
        const uint32_t rowBytes = rowElems * sizeof(bfloat16_t);
        auto value = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto reconstructed = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto high = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK2);
        auto low = Ub<bfloat16_t>(
            KDA_FINALIZE_A2_UB_WORK2 + sliceElems * sizeof(bfloat16_t));
        auto rowMasks = Ub<uint8_t>(
            KDA_FINALIZE_A2_UB_WORK0 + sliceElems * sizeof(float));
        auto zero = Ub<float>(
            KDA_FINALIZE_A2_UB_WORK0 + sliceElems * sizeof(float) + 512);
        if (rows != sliceRows) {
            AscendC::Duplicate(
                high, static_cast<bfloat16_t>(0), sliceElems);
            if constexpr (!KDA_FINALIZE_A2_RESIDENT_INTRA_HIGH_ONLY) {
                AscendC::Duplicate(
                    low, static_cast<bfloat16_t>(0), sliceElems);
            }
        }
        if (elems != 0U) {
            AscendC::DataCopy(value, src[srcOffset + rowElems], elems);
            WaitLoads();
            if (scale != 1.0f) {
                AscendC::Muls(value, value, scale, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
            ZeroUpperTriangle(
                value, rowMasks, zero, rowBegin, rows, diagonalOffset);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (KDA_FINALIZE_A2_RESIDENT_INTRA_HIGH_ONLY) {
                AscendC::Cast(
                    high, value, AscendC::RoundMode::CAST_RINT, elems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                SplitBf16(value, high, low, reconstructed, elems);
            }
        }
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(slotBase, dstByteOffset + rowBytes),
            high, sliceElems);
        if constexpr (!KDA_FINALIZE_A2_RESIDENT_INTRA_HIGH_ONLY) {
            AscendC::DataCopy(
                Ws<bfloat16_t>(
                    slotBase, dstByteOffset + planeBytes + rowBytes),
                low, sliceElems);
        }
        DrainStoresToLoads();
    }

    template <bool DRAIN_TO_VECTOR = true>
    __aicore__ inline void StoreVectorSplit(
        AscendC::LocalTensor<float> value,
        AscendC::LocalTensor<float> reconstructed,
        AscendC::LocalTensor<bfloat16_t> high,
        AscendC::LocalTensor<bfloat16_t> low,
        uint64_t slotBase, uint32_t dstByteOffset,
        uint32_t rowBegin, uint32_t rows)
    {
        constexpr uint32_t sliceRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t sliceElems =
            sliceRows * KDA_FINALIZE_DIM;
        constexpr uint32_t planeBytes =
            KDA_FINALIZE_VECTOR_BF16_BYTES;
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowBytes =
            rowBegin * KDA_FINALIZE_DIM * sizeof(bfloat16_t);
        if constexpr (KDA_FINALIZE_A2_RESIDENT_INTRA_HIGH_ONLY) {
            AscendC::Cast(
                high, value, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SplitBf16(value, high, low, reconstructed, elems);
        }
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(slotBase, dstByteOffset + rowBytes),
            high, sliceElems);
        if constexpr (!KDA_FINALIZE_A2_RESIDENT_INTRA_HIGH_ONLY) {
            AscendC::DataCopy(
                Ws<bfloat16_t>(
                    slotBase, dstByteOffset + planeBytes + rowBytes),
                low, sliceElems);
        }
        if constexpr (DRAIN_TO_VECTOR) {
            DrainStoresToVector();
        }
    }

    __aicore__ inline void MaskDakkBf16(
        uint64_t slotBase, uint32_t rowBegin, uint32_t rows)
    {
        constexpr uint32_t sliceRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t sliceElems =
            sliceRows * KDA_FINALIZE_CHUNK;
        const uint32_t elems = rows * KDA_FINALIZE_CHUNK;
        const uint32_t rowBytes =
            rowBegin * KDA_FINALIZE_CHUNK * sizeof(bfloat16_t);
        auto value = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK3);
        if (rows != sliceRows) {
            AscendC::Duplicate(
                value, static_cast<bfloat16_t>(0), sliceElems);
        }
        if (elems != 0U) {
            AscendC::DataCopy(
                value,
                Ws<bfloat16_t>(
                    slotBase,
                    KDA_FINALIZE_A2_WS_INTRA_DAKK + rowBytes),
                elems);
            WaitLoads();
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t column = rowBegin + row;
                if (column < KDA_FINALIZE_CHUNK) {
                    uint64_t mask[2] = {
                        column == 0U ? ~0ULL : (~0ULL << column), 0ULL};
                    AscendC::Duplicate(
                        value[row * KDA_FINALIZE_CHUNK],
                        static_cast<bfloat16_t>(0),
                        mask, 1, 1, 4);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(
                slotBase, KDA_FINALIZE_A2_WS_INTRA_DAKK + rowBytes),
            value, sliceElems);
        DrainStoresToLoads();
    }

    __aicore__ inline void BuildIntraVectorPrefix(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase,
        uint32_t rowBegin, uint32_t rows)
    {
        constexpr uint32_t sliceRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t sliceElems =
            sliceRows * KDA_FINALIZE_DIM;
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto negative = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto value = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto high = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK4);
        auto low = Ub<bfloat16_t>(
            KDA_FINALIZE_A2_UB_WORK4 +
            sliceElems * sizeof(bfloat16_t));

        if (rows != sliceRows) {
            AscendC::Duplicate(
                high, static_cast<bfloat16_t>(0), sliceElems);
        }
        if (elems == 0U) {
            StartStores();
            AscendC::DataCopy(
                Ws<bfloat16_t>(slotBase,
                    subBlockIdx_ == 0U
                        ? KDA_FINALIZE_A2_WS_INTRA_K_NEG0
                        : KDA_FINALIZE_A2_WS_INTRA_K_NEG1),
                high, sliceElems);
            DrainStoresToLoads();
            return;
        }
        AscendC::Mul(value, kFp, negative, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            high, value, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(
            Ws<bfloat16_t>(slotBase,
                subBlockIdx_ == 0U
                    ? KDA_FINALIZE_A2_WS_INTRA_K_NEG0
                    : KDA_FINALIZE_A2_WS_INTRA_K_NEG1),
            high, sliceElems);
        DrainStoresToVector();
        DrainStoresToLoads();
    }

    __aicore__ inline void BuildIntraVectorSuffix(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase,
        uint32_t rowBegin, uint32_t rows)
    {
        constexpr uint32_t sliceRows =
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        constexpr uint32_t sliceElems =
            sliceRows * KDA_FINALIZE_DIM;
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        if (elems == 0U) {
            return;
        }
        auto qTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_Q);
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto expG = Ub<float>(KDA_FINALIZE_A2_UB_EXP);
        auto center = Ub<float>(KDA_FINALIZE_A2_UB_CENTER_EXP);
        auto posExponent = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto value = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto reconstructed = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto high = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK4);
        auto low = Ub<bfloat16_t>(
            KDA_FINALIZE_A2_UB_WORK4 +
            sliceElems * sizeof(bfloat16_t));
        auto betaFp = Ub<float>(KDA_FINALIZE_A2_UB_IO + 256);
        auto betaBroadcast = Ub<float>(KDA_FINALIZE_A2_UB_IO + 1024);

        AscendC::BinaryRepeatParams rowsOverVector;
        rowsOverVector.dstBlkStride = 1;
        rowsOverVector.src0BlkStride = 1;
        rowsOverVector.src1BlkStride = 1;
        rowsOverVector.dstRepStride =
            KDA_FINALIZE_DIM * sizeof(float) / 32;
        rowsOverVector.src0RepStride = rowsOverVector.dstRepStride;
        rowsOverVector.src1RepStride = 0;
        AscendC::Div(
            posExponent, expG, center,
            static_cast<uint64_t>(KDA_FINALIZE_DIM / 2U),
            static_cast<uint8_t>(rows), rowsOverVector);
        AscendC::Div(
            posExponent[KDA_FINALIZE_DIM / 2U],
            expG[KDA_FINALIZE_DIM / 2U], center[KDA_FINALIZE_DIM / 2U],
            static_cast<uint64_t>(KDA_FINALIZE_DIM / 2U),
            static_cast<uint8_t>(rows), rowsOverVector);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            value, qTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(value, value, posExponent, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StoreVectorSplit(
            value, reconstructed, high, low, slotBase,
            KDA_FINALIZE_A2_WS_INTRA_Q_POS, rowBegin, rows);

        AscendC::Mul(value, kFp, posExponent, elems);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(betaBroadcast, betaFp, rows);
        MulRowsByScalar(
            value, value, betaBroadcast, rows, KDA_FINALIZE_DIM);
        StoreVectorSplit<false>(
            value, reconstructed, high, low, slotBase,
            KDA_FINALIZE_A2_WS_INTRA_BK_POS, rowBegin, rows);
        DrainStoresToLoads();
        DrainVectorToLoads();
    }

    __aicore__ inline void BuildIntraPre(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        const uint32_t rows = rowBegin <
                static_cast<uint32_t>(chunk.validRows)
            ? FinalizeMin(
                  KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
                  static_cast<uint32_t>(chunk.validRows) - rowBegin)
            : 0U;
        const int64_t matrixToken =
            FinalizeTokenOffset(
                *tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        PackMatrix(
            dAqk_, static_cast<uint64_t>(matrixToken), slotBase,
            KDA_FINALIZE_A2_WS_INTRA_DAQK,
            rowBegin, rows, 1.0f, 1U);
        BuildIntraVectorPrefix(chunk, head, slotBase, rowBegin, rows);
    }

    __aicore__ inline void BuildIntraRemaining(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        const uint32_t rows = rowBegin <
                static_cast<uint32_t>(chunk.validRows)
            ? FinalizeMin(
                  KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
                  static_cast<uint32_t>(chunk.validRows) - rowBegin)
            : 0U;
        BuildIntraVectorSuffix(chunk, head, slotBase, rowBegin, rows);
    }

    __aicore__ inline void BuildIntraPost(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        const uint32_t rows = rowBegin <
                static_cast<uint32_t>(chunk.validRows)
            ? FinalizeMin(
                  KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
                  static_cast<uint32_t>(chunk.validRows) - rowBegin)
            : 0U;
        MaskDakkBf16(slotBase, rowBegin, rows);
    }

    __aicore__ inline void PrepareIntraRight(
        const FinalizeChunkInfo &chunk, uint64_t slotBase,
        bool precomputeGateDerivative)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        auto rawDkUpper = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto negScale = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        DrainStoresToLoads();
        AscendC::DataCopy(
            rawDkUpper,
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_INTRA_RIGHT +
                rowByteOffset),
            elems);
        if (precomputeGateDerivative) {
            PrecomputeGateDerivative(chunk);
        }
        WaitLoads();
        AscendC::Mul(rawDkUpper, rawDkUpper, negScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareIntraDq(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        const int64_t token = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM) +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;

        auto qTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_Q);
        auto rawDq = Ub<float>(KDA_FINALIZE_A2_UB_WORK0);
        auto posScale = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto scratch = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto outTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_WORK0);
        auto dqBase = Ub<float>(KDA_FINALIZE_A2_UB_DQ_BASE);
        auto dgBase = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);

        AscendC::DataCopy(
            rawDq,
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_INTRA_DQ_EARLY +
                rowByteOffset),
            elems);
        AscendC::Cast(
            scratch, qTyped, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        WaitLoads();
        AscendC::Mul(rawDq, rawDq, posScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(scratch, scratch, rawDq, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dgBase, dgBase, scratch, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch, dqBase, rawDq, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            outTyped, scratch, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dqOut_[token], outTyped, elems);
    }

    __aicore__ inline void PrepareIntraDk(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const uint32_t rowByteOffset =
            rowBegin * KDA_FINALIZE_DIM * sizeof(float);
        const int64_t scalarToken =
            FinalizeTokenOffset(*tiling_, chunk, head, 1) + rowBegin;
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto rawDkLower = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto posScale = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto scratch = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto betaTyped = Ub<BetaT>(KDA_FINALIZE_A2_UB_IO);
        auto betaFp = Ub<float>(KDA_FINALIZE_A2_UB_IO + 256);
        auto dbBase = Ub<float>(KDA_FINALIZE_A2_UB_IO + 512);
        auto dbDelta = Ub<float>(KDA_FINALIZE_A2_UB_IO + 768);
        auto gateCorrection0 =
            Ub<float>(KDA_FINALIZE_A2_UB_GATE_CORRECTION);
        auto gateCorrection1 =
            Ub<float>(KDA_FINALIZE_A2_UB_GATE_CORRECTION + 512);
        auto finalRh =
            Ub<float>(KDA_FINALIZE_A2_UB_FINAL_RH);
        auto expG = Ub<float>(KDA_FINALIZE_A2_UB_EXP);
        const uint32_t lastOwner =
            (static_cast<uint32_t>(chunk.validRows) - 1U) /
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        const uint32_t lastLocalRow =
            static_cast<uint32_t>(chunk.validRows) - 1U -
            lastOwner * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        auto finalExpG = expG[lastLocalRow * KDA_FINALIZE_DIM];

        AscendC::DataCopy(
            rawDkLower,
            Ws<float>(slotBase, KDA_FINALIZE_A2_WS_INTRA_DK +
                rowByteOffset),
            elems);
        if (subBlockIdx_ == lastOwner) {
            AscendC::DataCopy(
                gateCorrection0,
                Ws<float>(slotBase, KDA_FINALIZE_A2_WS_FINAL_GATE),
                2U * KDA_FINALIZE_DIM);
            AscendC::DataCopy(
                finalRh,
                Ws<float>(slotBase, KDA_FINALIZE_A2_WS_FINAL_RH),
                2U * KDA_FINALIZE_DIM);
        }
        WaitLoads();
        AscendC::Mul(rawDkLower, rawDkLower, posScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(scratch, kFp, rawDkLower, elems);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduce128(dbDelta, scratch, rows);
        BroadcastRows(scratch, betaFp, rows);
        MulRowsByScalar(
            rawDkLower, rawDkLower, scratch, rows, KDA_FINALIZE_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        if (subBlockIdx_ == lastOwner) {
            AscendC::Mul(
                finalRh, finalRh, finalExpG, KDA_FINALIZE_DIM / 2U);
            AscendC::Mul(
                finalRh[KDA_FINALIZE_DIM / 2U],
                finalRh[KDA_FINALIZE_DIM / 2U],
                finalExpG[KDA_FINALIZE_DIM / 2U],
                KDA_FINALIZE_DIM / 2U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                gateCorrection0, gateCorrection0,
                gateCorrection1, KDA_FINALIZE_DIM / 2U);
            AscendC::Add(
                gateCorrection0[KDA_FINALIZE_DIM / 2U],
                gateCorrection0[KDA_FINALIZE_DIM / 2U],
                gateCorrection1[KDA_FINALIZE_DIM / 2U],
                KDA_FINALIZE_DIM / 2U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                gateCorrection0, gateCorrection0,
                finalRh, KDA_FINALIZE_DIM / 2U);
            AscendC::Add(
                gateCorrection0[KDA_FINALIZE_DIM / 2U],
                gateCorrection0[KDA_FINALIZE_DIM / 2U],
                finalRh[KDA_FINALIZE_DIM / 2U],
                KDA_FINALIZE_DIM / 2U);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(betaFp, dbBase, dbDelta, rows);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<BetaT, float>::value) {
            AscendC::Adds(betaTyped, betaFp, 0.0f, rows);
        } else {
            AscendC::Cast(
                betaTyped, betaFp, AscendC::RoundMode::CAST_RINT, rows);
        }
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dBetaOut_[scalarToken], betaTyped, rows);
    }

    __aicore__ inline void FinalizeIntraLeft(
        const FinalizeChunkInfo &chunk, int64_t head, uint64_t slotBase)
    {
        const uint32_t rowBegin =
            subBlockIdx_ * KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;
        if (rowBegin >= static_cast<uint32_t>(chunk.validRows)) {
            return;
        }
        const uint32_t rows = FinalizeMin(
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS,
            static_cast<uint32_t>(chunk.validRows) - rowBegin);
        const uint32_t elems = rows * KDA_FINALIZE_DIM;
        const int64_t tokenBase = FinalizeTokenOffset(
            *tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t token = tokenBase +
            static_cast<int64_t>(rowBegin) * KDA_FINALIZE_DIM;
        auto kFp = Ub<float>(KDA_FINALIZE_A2_UB_K);
        auto rawDkLower = Ub<float>(KDA_FINALIZE_A2_UB_WORK1);
        auto rawDkUpper = Ub<float>(KDA_FINALIZE_A2_UB_WORK2);
        auto posScale = Ub<float>(KDA_FINALIZE_A2_UB_WORK3);
        auto scratch = Ub<float>(KDA_FINALIZE_A2_UB_WORK4);
        auto negScale = Ub<float>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto outTyped = Ub<bfloat16_t>(KDA_FINALIZE_A2_UB_DK_BASE);
        auto dkBase = Ub<float>(KDA_FINALIZE_A2_UB_WORK5);
        auto dgBase = Ub<float>(KDA_FINALIZE_A2_UB_DG_BASE);
        auto gateCorrection0 =
            Ub<float>(KDA_FINALIZE_A2_UB_GATE_CORRECTION);
        auto gateCorrection1 =
            Ub<float>(KDA_FINALIZE_A2_UB_GATE_CORRECTION + 512);

        const uint32_t lastOwner =
            (static_cast<uint32_t>(chunk.validRows) - 1U) /
            KDA_FINALIZE_A2_RESIDENT_TOKEN_ROWS;

        AscendC::Add(negScale, rawDkLower, rawDkUpper, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(posScale, dkBase, negScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            outTyped, posScale, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        StartStores();
        AscendC::DataCopy(dkOut_[token], outTyped, elems);

        AscendC::Sub(scratch, rawDkLower, rawDkUpper, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(scratch, kFp, scratch, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(posScale, dgBase, scratch, elems);
        AscendC::PipeBarrier<PIPE_V>();
        if (subBlockIdx_ == lastOwner) {
            const uint32_t lastLocalRow =
                static_cast<uint32_t>(chunk.validRows) - 1U - rowBegin;
            AscendC::Add(
                posScale[lastLocalRow * KDA_FINALIZE_DIM],
                posScale[lastLocalRow * KDA_FINALIZE_DIM],
                gateCorrection0, KDA_FINALIZE_DIM / 2U);
            AscendC::Add(
                posScale[lastLocalRow * KDA_FINALIZE_DIM +
                         KDA_FINALIZE_DIM / 2U],
                posScale[lastLocalRow * KDA_FINALIZE_DIM +
                         KDA_FINALIZE_DIM / 2U],
                gateCorrection0[KDA_FINALIZE_DIM / 2U],
                KDA_FINALIZE_DIM / 2U);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<bfloat16_t> q_, k_, v_, h_, dh_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<float> dAqk_, dqRaw_;
    AscendC::GlobalTensor<BetaT> beta_;
    AscendC::GlobalTensor<bfloat16_t> dqOut_, dkOut_, dvOut_;
    AscendC::GlobalTensor<BetaT> dBetaOut_;
    AscendC::GlobalTensor<float> dGOut_;
    AscendC::GlobalTensor<float> rawGFloat_, aLog_, dtBias_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspaceAddr_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToMte2_;
    AscendC::TEventID mte3ToV_;
    AscendC::TEventID vToMte2_;
    AscendC::TEventID scalarVToS_;
    uint32_t subBlockNum_ = KDA_FINALIZE_A2_RESIDENT_AIVS;
    uint32_t subBlockIdx_ = 0;
};

class ChunkKdaBwdFinalizeA2GatePartialReduce {
public:
    __aicore__ inline void Init(
        GM_ADDR workspace, GM_ADDR dAOut, GM_ADDR dBiasOut,
        const ChunkKdaBwdFinalizeTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        workspaceAddr_ = workspace;
        tiling_ = tiling;
        dAOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAOut));
        if (tiling_->a2.hasDtBias != 0) {
            dBiasOut_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(dBiasOut));
        }
        pipe_ = pipe;
        pipe_->InitBuffer(ubBuf_, 24 * 1024);
        ub_ = ubBuf_.Get<uint8_t>();
        mte2ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        vToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        vToMte3_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t head = AscendC::GetBlockIdx();
        if (head < static_cast<uint32_t>(tiling_->NV)) {
            auto paAcc = Ub<float>(0);
            auto pbAcc = Ub<float>(512);
            auto paRecords = Ub<float>(1024);
            auto pbRecords = Ub<float>(2560);
            AscendC::Duplicate(paAcc, 0.0f, 8);
            if (tiling_->a2.hasDtBias != 0) {
                AscendC::Duplicate(pbAcc, 0.0f, KDA_FINALIZE_DIM);
            }
            AscendC::PipeBarrier<PIPE_V>();

            const uint32_t physicalAivNum =
                static_cast<uint32_t>(tiling_->a2.usedCoreNum) * 2U;
            const uint32_t paStride =
                (static_cast<uint32_t>(tiling_->NV) - 1U) *
                KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES;
            AscendC::DataCopyPad(
                paRecords,
                Partial<float>(
                    tiling_->gatePartialOffset +
                    static_cast<uint64_t>(head) *
                        KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES),
                {static_cast<uint16_t>(physicalAivNum),
                 KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES,
                 paStride, 0, 0},
                {false, 0, 0, 0});
            if (tiling_->a2.hasDtBias != 0) {
                const uint32_t pbStride =
                    (static_cast<uint32_t>(tiling_->NV) - 1U) *
                    KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES;
                AscendC::DataCopyPad(
                    pbRecords,
                    Partial<float>(
                        tiling_->dtBiasPartialOffset +
                        static_cast<uint64_t>(head) *
                            KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES),
                    {static_cast<uint16_t>(physicalAivNum),
                     KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES,
                     pbStride, 0, 0},
                    {false, 0, 0, 0});
            }
            WaitLoads();
            for (uint32_t physicalAiv = 0;
                 physicalAiv < physicalAivNum; ++physicalAiv) {
                AscendC::Add(
                    paAcc, paAcc,
                    paRecords[
                        physicalAiv *
                        (KDA_FINALIZE_A2_GATE_PA_RECORD_BYTES /
                         sizeof(float))],
                    8);
                if (tiling_->a2.hasDtBias != 0) {
                    AscendC::Add(
                        pbAcc, pbAcc,
                        pbRecords[
                            physicalAiv *
                            (KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES /
                             sizeof(float))],
                        KDA_FINALIZE_DIM);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            StartStores();
            AscendC::DataCopyPad(
                dAOut_[head], paAcc,
                {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            if (tiling_->a2.hasDtBias != 0) {
                AscendC::DataCopyPad(
                    dBiasOut_[static_cast<uint64_t>(head) *
                              KDA_FINALIZE_DIM],
                    pbAcc,
                    {1, KDA_FINALIZE_A2_GATE_PB_RECORD_BYTES, 0, 0, 0});
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
    }

private:
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> Ub(uint32_t byteOffset)
    {
        return ub_[byteOffset].ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::GlobalTensor<T> Partial(uint64_t offset)
    {
        AscendC::GlobalTensor<T> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(
            workspaceAddr_ + offset));
        return tensor;
    }

    __aicore__ inline void WaitLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void DrainVectorToLoads()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2_);
    }

    __aicore__ inline void StartStores()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
    }

    GM_ADDR workspaceAddr_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<float> dAOut_, dBiasOut_;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_;
    AscendC::TEventID vToMte2_;
    AscendC::TEventID vToMte3_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_VECTOR_H
