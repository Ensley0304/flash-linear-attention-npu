/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_KDA_BWD_INTRA_MIXED_HPP
#define CHUNK_KDA_BWD_INTRA_MIXED_HPP

constexpr uint32_t KDA_MIXED_BT = 64;
constexpr uint32_t KDA_MIXED_BC = 16;
constexpr uint32_t KDA_MIXED_ROWS_PER_AIV = 8;
constexpr uint32_t KDA_MIXED_K = 128;
constexpr uint32_t KDA_MIXED_BLOCKS = KDA_MIXED_BT / KDA_MIXED_BC;
constexpr uint32_t KDA_MIXED_PAIRS = KDA_MIXED_BLOCKS * (KDA_MIXED_BLOCKS + 1) / 2;
constexpr uint32_t KDA_MIXED_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_MIXED_SELECTED_ELEMENTS =
    KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
constexpr uint32_t KDA_MIXED_ROW_BLOCK_ELEMENTS = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;

// One slot contains two packed FP32 GEMMs:
//   [32,16] x [16,128] -> [32,128]  (dq and dk-left)
//   [16,32] x [32,128] -> [16,128]  (combined dk-right, A column-major)
constexpr uint32_t KDA_MIXED_A_LEFT = 0;
constexpr uint32_t KDA_MIXED_B_LEFT = KDA_MIXED_A_LEFT + 32 * 16;
constexpr uint32_t KDA_MIXED_C_LEFT = KDA_MIXED_B_LEFT + 16 * KDA_MIXED_K;
constexpr uint32_t KDA_MIXED_A_RIGHT = KDA_MIXED_C_LEFT + 32 * KDA_MIXED_K;
constexpr uint32_t KDA_MIXED_B_RIGHT = KDA_MIXED_A_RIGHT + 16 * 32;
constexpr uint32_t KDA_MIXED_C_RIGHT = KDA_MIXED_B_RIGHT + 32 * KDA_MIXED_K;
// The logical right result has 16 rows. Reserve the full 32-row M tile so a
// conservative CATLASS tail store can never cross into the next ring slot.
constexpr uint32_t KDA_MIXED_C_RIGHT_ALLOC_ROWS = 32;
constexpr uint32_t KDA_MIXED_SLOT_ELEMENTS =
    KDA_MIXED_C_RIGHT + KDA_MIXED_C_RIGHT_ALLOC_ROWS * KDA_MIXED_K;
constexpr uint32_t KDA_MIXED_UB_BYTES =
    2 * KDA_MIXED_SELECTED_ELEMENTS * sizeof(bfloat16_t) +
    4 * KDA_MIXED_SELECTED_ELEMENTS * sizeof(float) +
    2 * KDA_MIXED_BLOCKS * KDA_MIXED_K * sizeof(float) +
    KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * sizeof(float) +
    KDA_MIXED_QUEUE_DEPTH * 2 * KDA_MIXED_ROW_BLOCK_ELEMENTS * sizeof(float) +
    3 * KDA_MIXED_SELECTED_ELEMENTS * sizeof(float) +
    3 * KDA_MIXED_ROW_BLOCK_ELEMENTS * sizeof(float) +
    4 * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC * sizeof(float) +
    2 * KDA_MIXED_ROW_BLOCK_ELEMENTS * sizeof(float) +
    256 * sizeof(float) + 64;
static_assert(KDA_MIXED_SLOT_ELEMENTS == 15360, "Host and device workspace layouts must match");
static_assert(KDA_MIXED_UB_BYTES == 175296, "Update the mixed-path UB accounting after buffer changes");
static_assert(KDA_MIXED_UB_BYTES <= 192 * 1024, "ChunkKdaBwdIntra mixed path exceeds the A2 UB budget");
static_assert(KDA_MIXED_PAIRS % KDA_MIXED_QUEUE_DEPTH == 0,
              "Keep the workspace slot and cross-core reverse-flag phases aligned across tasks");

constexpr uint8_t KDA_MIXED_DONE_FLAG0 = 2;
constexpr uint8_t KDA_MIXED_DONE_FLAG1 = 3;
constexpr uint8_t KDA_MIXED_READY_FLAG0 = 4;
constexpr uint8_t KDA_MIXED_READY_FLAG1 = 5;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaBwdMixedArchTag = Catlass::Arch::Ascend950;
#else
using KdaBwdMixedArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaBwdMixedDispatchPolicy =
    Catlass::Gemm::MmadPingpongTlaMulti<KdaBwdMixedArchTag, true, false>;
static_assert(!KdaBwdMixedDispatchPolicy::USE_HF32_MODE,
              "ChunkKdaBwdIntra mixed path must use IEEE FP32 Cube mode");
using KdaBwdMixedTileShape = tla::Shape<tla::Int<32>, tla::Int<128>, tla::Int<32>>;
using KdaBwdMixedLeftTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
    KdaBwdMixedArchTag, float, Catlass::layout::RowMajor, float, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
using KdaBwdMixedRightTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
    KdaBwdMixedArchTag, float, Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
using KdaBwdMixedLeftBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
    KdaBwdMixedDispatchPolicy, KdaBwdMixedTileShape, KdaBwdMixedTileShape,
    float, float, float, void, KdaBwdMixedLeftTileCopy>;
using KdaBwdMixedRightBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
    KdaBwdMixedDispatchPolicy, KdaBwdMixedTileShape, KdaBwdMixedTileShape,
    float, float, float, void, KdaBwdMixedRightTileCopy>;

template <typename T>
class ChunkKdaBwdIntraMixedKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk,
                                GM_ADDR dAkk, GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg,
                                GM_ADDR dqOut, GM_ADDR dkOut, GM_ADDR dbOut, GM_ADDR dgOut,
                                GM_ADDR workspace, const ChunkKdaBwdIntraTilingData &tiling,
                                TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta));
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dAkk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk));
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk));
        db_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db));
        dg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg));
        dqOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqOut));
        dkOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkOut));
        dbOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbOut));
        dgOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgOut));
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        batch_ = static_cast<uint64_t>(tiling.batch);
        heads_ = static_cast<uint64_t>(tiling.vHeadNum);
        seqlen_ = static_cast<uint64_t>(tiling.seqlen);
        chunks_ = static_cast<uint64_t>(tiling.totalChunks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        pipe_ = pipe;

        if ASCEND_IS_AIC {
            logicalCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if ASCEND_IS_AIV {
            const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            logicalCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
            InitVectorBuffers();
        }
    }

    __aicore__ inline void ProcessAic()
    {
        Catlass::Arch::Resource<KdaBwdMixedArchTag> resource;
        KdaBwdMixedLeftBlockMmad leftBlockMmad(resource);
        KdaBwdMixedRightBlockMmad rightBlockMmad(resource);
        const uint64_t taskCount = batch_ * chunks_ * heads_;
        for (uint64_t task = logicalCoreIdx_; task < taskCount; task += usedCoreNum_) {
            for (uint32_t pair = 0; pair < KDA_MIXED_PAIRS; ++pair) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
                ComputePairAic(leftBlockMmad, rightBlockMmad,
                               pair % KDA_MIXED_QUEUE_DEPTH);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);
            }
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint64_t taskCount = batch_ * chunks_ * heads_;
        for (uint64_t task = logicalCoreIdx_; task < taskCount; task += usedCoreNum_) {
            uint64_t b = 0;
            uint64_t hv = 0;
            uint64_t chunkStart = 0;
            ResolveTask(task, b, hv, chunkStart);
            LoadTaskFeatures(b, hv, chunkStart, subBlockIdx);
            ZeroAccumulators();
            for (uint32_t pair = 0; pair < KDA_MIXED_PAIRS; ++pair) {
                uint32_t earlyBlock = 0;
                uint32_t lateBlock = 0;
                ResolvePair(pair, earlyBlock, lateBlock);
                const uint32_t slot = pair % KDA_MIXED_QUEUE_DEPTH;
                PreparePairAiv(b, hv, chunkStart, subBlockIdx, earlyBlock, lateBlock, slot);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
                if (pair > 0) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
                    uint32_t previousEarly = 0;
                    uint32_t previousLate = 0;
                    ResolvePair(pair - 1, previousEarly, previousLate);
                    ConsumePairAiv(previousEarly, previousLate,
                                   (pair - 1) % KDA_MIXED_QUEUE_DEPTH);
                }
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            uint32_t finalEarly = 0;
            uint32_t finalLate = 0;
            ResolvePair(KDA_MIXED_PAIRS - 1, finalEarly, finalLate);
            ConsumePairAiv(finalEarly, finalLate,
                           (KDA_MIXED_PAIRS - 1) % KDA_MIXED_QUEUE_DEPTH);
            StoreTaskOutputs(b, hv, chunkStart, subBlockIdx);
        }
    }

private:
    __aicore__ inline void InitVectorBuffers()
    {
        constexpr uint32_t selectedElements = KDA_MIXED_SELECTED_ELEMENTS;
        constexpr uint32_t rowBlockElements = KDA_MIXED_ROW_BLOCK_ELEMENTS;
        pipe_->InitBuffer(qTypedBuf_, selectedElements * sizeof(T));
        pipe_->InitBuffer(kTypedBuf_, selectedElements * sizeof(T));
        pipe_->InitBuffer(qCacheBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(kCacheBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(kBetaCacheBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(gCacheBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(refBuf_, 2 * KDA_MIXED_BLOCKS * KDA_MIXED_K * sizeof(float));
        pipe_->InitBuffer(betaBuf_, KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * sizeof(float));
        pipe_->InitBuffer(gateSlotsBuf_, KDA_MIXED_QUEUE_DEPTH * 2 * rowBlockElements * sizeof(float));
        pipe_->InitBuffer(dqAccBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(dkLeftAccBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(dkRightAccBuf_, selectedElements * sizeof(float));
        pipe_->InitBuffer(operand0Buf_, rowBlockElements * sizeof(float));
        pipe_->InitBuffer(operand1Buf_, rowBlockElements * sizeof(float));
        pipe_->InitBuffer(operand2Buf_, rowBlockElements * sizeof(float));
        pipe_->InitBuffer(dARowQBuf_, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC * sizeof(float));
        pipe_->InitBuffer(dARowKBuf_, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC * sizeof(float));
        pipe_->InitBuffer(dAColQBuf_, KDA_MIXED_BC * KDA_MIXED_ROWS_PER_AIV * sizeof(float));
        pipe_->InitBuffer(dAColKBuf_, KDA_MIXED_BC * KDA_MIXED_ROWS_PER_AIV * sizeof(float));
        pipe_->InitBuffer(resultBuf_, 2 * rowBlockElements * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(dbBuf_, 32);
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t) const
    {
        return ((b * heads_ + h) * seqlen_ + t) * KDA_MIXED_K;
    }

    __aicore__ inline uint64_t VOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_MIXED_K;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * heads_ + hv) * seqlen_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t,
                                       uint64_t localColumn) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_MIXED_BT + localColumn;
    }

    __aicore__ inline uint64_t SlotBase(uint32_t slot) const
    {
        return (logicalCoreIdx_ * KDA_MIXED_QUEUE_DEPTH + slot) * KDA_MIXED_SLOT_ELEMENTS;
    }

    __aicore__ inline uint32_t CacheBlockOffset(uint32_t block) const
    {
        return block * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
    }

    __aicore__ inline void ResolveTask(uint64_t task, uint64_t &b, uint64_t &hv,
                                       uint64_t &chunkStart) const
    {
        hv = task % heads_;
        const uint64_t flatChunk = task / heads_;
        b = flatChunk / chunks_;
        chunkStart = (flatChunk % chunks_) * KDA_MIXED_BT;
    }

    __aicore__ inline void ResolvePair(uint32_t pair, uint32_t &earlyBlock,
                                       uint32_t &lateBlock) const
    {
        if (pair == 0) {
            earlyBlock = 0; lateBlock = 0;
        } else if (pair == 1) {
            earlyBlock = 0; lateBlock = 1;
        } else if (pair == 2) {
            earlyBlock = 1; lateBlock = 1;
        } else if (pair == 3) {
            earlyBlock = 0; lateBlock = 2;
        } else if (pair == 4) {
            earlyBlock = 1; lateBlock = 2;
        } else if (pair == 5) {
            earlyBlock = 2; lateBlock = 2;
        } else if (pair == 6) {
            earlyBlock = 0; lateBlock = 3;
        } else if (pair == 7) {
            earlyBlock = 1; lateBlock = 3;
        } else if (pair == 8) {
            earlyBlock = 2; lateBlock = 3;
        } else {
            earlyBlock = 3; lateBlock = 3;
        }
    }

    __aicore__ inline void SyncMte2ToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventId);
    }

    __aicore__ inline void SyncSToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void ScaleEightRowsByBeta(LocalTensor<float> dst,
                                                 LocalTensor<float> src,
                                                 uint32_t dataOffset,
                                                 __ubuf__ float *betaPtr,
                                                 uint32_t betaOffset)
    {
        // Read the whole 8-row beta tile on PIPE_S before publishing one S_V
        // event.  Auto-sync is disabled, so passing betaPtr[...] directly to
        // Muls could let PIPE_V consume the scalar before the UB read finishes.
        const float beta0 = betaPtr[betaOffset];
        const float beta1 = betaPtr[betaOffset + 1];
        const float beta2 = betaPtr[betaOffset + 2];
        const float beta3 = betaPtr[betaOffset + 3];
        const float beta4 = betaPtr[betaOffset + 4];
        const float beta5 = betaPtr[betaOffset + 5];
        const float beta6 = betaPtr[betaOffset + 6];
        const float beta7 = betaPtr[betaOffset + 7];
        SyncSToV();
        Muls(dst[dataOffset], src[dataOffset], beta0, KDA_MIXED_K);
        Muls(dst[dataOffset + KDA_MIXED_K], src[dataOffset + KDA_MIXED_K],
             beta1, KDA_MIXED_K);
        Muls(dst[dataOffset + 2 * KDA_MIXED_K], src[dataOffset + 2 * KDA_MIXED_K],
             beta2, KDA_MIXED_K);
        Muls(dst[dataOffset + 3 * KDA_MIXED_K], src[dataOffset + 3 * KDA_MIXED_K],
             beta3, KDA_MIXED_K);
        Muls(dst[dataOffset + 4 * KDA_MIXED_K], src[dataOffset + 4 * KDA_MIXED_K],
             beta4, KDA_MIXED_K);
        Muls(dst[dataOffset + 5 * KDA_MIXED_K], src[dataOffset + 5 * KDA_MIXED_K],
             beta5, KDA_MIXED_K);
        Muls(dst[dataOffset + 6 * KDA_MIXED_K], src[dataOffset + 6 * KDA_MIXED_K],
             beta6, KDA_MIXED_K);
        Muls(dst[dataOffset + 7 * KDA_MIXED_K], src[dataOffset + 7 * KDA_MIXED_K],
             beta7, KDA_MIXED_K);
    }

    __aicore__ inline void SyncVToS()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void LoadTaskFeatures(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                            uint32_t subBlockIdx)
    {
        LocalTensor<T> qTyped = qTypedBuf_.Get<T>();
        LocalTensor<T> kTyped = kTypedBuf_.Get<T>();
        LocalTensor<float> qCache = qCacheBuf_.Get<float>();
        LocalTensor<float> kCache = kCacheBuf_.Get<float>();
        LocalTensor<float> gCache = gCacheBuf_.Get<float>();
        LocalTensor<float> refs = refBuf_.Get<float>();
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        constexpr uint32_t blockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        for (uint32_t block = 0; block < KDA_MIXED_BLOCKS; ++block) {
            const uint64_t token = chunkStart + block * KDA_MIXED_BC +
                                   subBlockIdx * KDA_MIXED_ROWS_PER_AIV;
            const uint32_t localOffset = CacheBlockOffset(block);
            DataCopy(qTyped[localOffset], q_[QOffset(b, hv, token)], blockElements);
            DataCopy(kTyped[localOffset], k_[QOffset(b, hv, token)], blockElements);
            DataCopy(gCache[localOffset], g_[VOffset(b, hv, token)], blockElements);
            DataCopy(betaLocal[block * KDA_MIXED_ROWS_PER_AIV],
                     beta_[BetaOffset(b, hv, token)], KDA_MIXED_ROWS_PER_AIV);
            DataCopy(refs[block * KDA_MIXED_K],
                     g_[VOffset(b, hv, chunkStart + block * KDA_MIXED_BC)], KDA_MIXED_K);
            DataCopy(refs[(KDA_MIXED_BLOCKS + block) * KDA_MIXED_K],
                     g_[VOffset(b, hv, chunkStart + block * KDA_MIXED_BC + KDA_MIXED_BC / 2)],
                     KDA_MIXED_K);
        }
        SyncMte2ToV();
        Cast(qCache, qTyped, RoundMode::CAST_NONE,
             KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K);
        Cast(kCache, kTyped, RoundMode::CAST_NONE,
             KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K);
        PipeBarrier<PIPE_V>();
        SyncMte2ToS();
        LocalTensor<float> kBetaCache = kBetaCacheBuf_.Get<float>();
        __ubuf__ float *betaPtr =
            reinterpret_cast<__ubuf__ float *>(betaBuf_.Get<float>().GetPhyAddr());
        for (uint32_t block = 0; block < KDA_MIXED_BLOCKS; ++block) {
            const uint32_t rowOffset =
                block * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
            ScaleEightRowsByBeta(kBetaCache, kCache, rowOffset, betaPtr,
                                 block * KDA_MIXED_ROWS_PER_AIV);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ZeroAccumulators()
    {
        constexpr uint32_t elements = KDA_MIXED_BLOCKS * KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        Duplicate(dqAccBuf_.Get<float>(), 0.0f, elements);
        Duplicate(dkLeftAccBuf_.Get<float>(), 0.0f, elements);
        Duplicate(dkRightAccBuf_.Get<float>(), 0.0f, elements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastReference(LocalTensor<float> dst, LocalTensor<float> reference)
    {
        Copy(dst, reference, 64, KDA_MIXED_ROWS_PER_AIV, {1, 1, 16, 0});
        Copy(dst[64], reference[64], 64, KDA_MIXED_ROWS_PER_AIV, {1, 1, 16, 0});
    }

    __aicore__ inline void BuildPairGates(uint32_t earlyBlock, uint32_t lateBlock,
                                          uint32_t slot)
    {
        constexpr uint32_t rowBlockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        LocalTensor<float> gateSlots = gateSlotsBuf_.Get<float>();
        LocalTensor<float> gateLate = gateSlots[slot * 2 * rowBlockElements];
        LocalTensor<float> gateEarly = gateSlots[slot * 2 * rowBlockElements + rowBlockElements];
        // ResolvePair groups off-diagonal pairs with the same late block.  For
        // every pair after the first one in such a group, gateLate is bitwise
        // identical to the value produced in the previous ring slot.  Copy it
        // before that slot is consumed instead of repeating Broadcast/Sub/Exp.
        const bool reuseLate = earlyBlock < lateBlock && earlyBlock > 0;
        LocalTensor<float> refs = refBuf_.Get<float>();
        LocalTensor<float> reference = earlyBlock == lateBlock ?
            refs[(KDA_MIXED_BLOCKS + lateBlock) * KDA_MIXED_K] :
            refs[lateBlock * KDA_MIXED_K];
        LocalTensor<float> gCache = gCacheBuf_.Get<float>();
        if (reuseLate) {
            const uint32_t previousSlot =
                (slot + KDA_MIXED_QUEUE_DEPTH - 1) % KDA_MIXED_QUEUE_DEPTH;
            Adds(gateLate, gateSlots[previousSlot * 2 * rowBlockElements],
                 0.0f, rowBlockElements);
        } else {
            BroadcastReference(gateLate, reference);
        }
        BroadcastReference(gateEarly, reference);
        PipeBarrier<PIPE_V>();
        if (!reuseLate) {
            Sub(gateLate, gCache[CacheBlockOffset(lateBlock)], gateLate, rowBlockElements);
        }
        Sub(gateEarly, gateEarly, gCache[CacheBlockOffset(earlyBlock)], rowBlockElements);
        PipeBarrier<PIPE_V>();
        if (!reuseLate) {
            Muls(gateLate, gateLate, LN2, rowBlockElements);
        }
        Muls(gateEarly, gateEarly, LN2, rowBlockElements);
        PipeBarrier<PIPE_V>();
        if (!reuseLate) {
            Exp(gateLate, gateLate, rowBlockElements);
        }
        Exp(gateEarly, gateEarly, rowBlockElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPairOperands(uint32_t earlyBlock, uint32_t lateBlock,
                                             uint32_t slot)
    {
        constexpr uint32_t rowBlockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        LocalTensor<float> gates = gateSlotsBuf_.Get<float>();
        LocalTensor<float> gateLate = gates[slot * 2 * rowBlockElements];
        LocalTensor<float> gateEarly = gates[slot * 2 * rowBlockElements + rowBlockElements];
        LocalTensor<float> qCache = qCacheBuf_.Get<float>();
        LocalTensor<float> kCache = kCacheBuf_.Get<float>();
        LocalTensor<float> kBetaCache = kBetaCacheBuf_.Get<float>();
        LocalTensor<float> operand0 = operand0Buf_.Get<float>();
        LocalTensor<float> operand1 = operand1Buf_.Get<float>();
        LocalTensor<float> operand2 = operand2Buf_.Get<float>();
        Mul(operand0, kCache[CacheBlockOffset(earlyBlock)], gateEarly, rowBlockElements);
        Mul(operand1, qCache[CacheBlockOffset(lateBlock)], gateLate, rowBlockElements);
        Mul(operand2, kBetaCache[CacheBlockOffset(lateBlock)], gateLate, rowBlockElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PreparePairAiv(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                          uint32_t subBlockIdx, uint32_t earlyBlock,
                                          uint32_t lateBlock, uint32_t slot)
    {
        LocalTensor<float> rowQ = dARowQBuf_.Get<float>();
        LocalTensor<float> rowK = dARowKBuf_.Get<float>();
        LocalTensor<float> colQ = dAColQBuf_.Get<float>();
        LocalTensor<float> colK = dAColKBuf_.Get<float>();
        const uint32_t rowStart = subBlockIdx * KDA_MIXED_ROWS_PER_AIV;
        const uint64_t sourceToken = chunkStart + lateBlock * KDA_MIXED_BC + rowStart;
        DataCopyExtParams rowParams{KDA_MIXED_ROWS_PER_AIV,
                                    KDA_MIXED_BC * sizeof(float),
                                    (KDA_MIXED_BT - KDA_MIXED_BC) * sizeof(float), 0, 0};
        DataCopyExtParams colParams{KDA_MIXED_BC,
                                    KDA_MIXED_ROWS_PER_AIV * sizeof(float),
                                    (KDA_MIXED_BT - KDA_MIXED_ROWS_PER_AIV) * sizeof(float), 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(rowQ, dAqk_[AOffset(b, hv, sourceToken, earlyBlock * KDA_MIXED_BC)],
                    rowParams, noPad);
        DataCopyPad(rowK, dAkk_[AOffset(b, hv, sourceToken, earlyBlock * KDA_MIXED_BC)],
                    rowParams, noPad);
        DataCopyPad(colQ, dAqk_[AOffset(b, hv, chunkStart + lateBlock * KDA_MIXED_BC,
                                       earlyBlock * KDA_MIXED_BC + rowStart)],
                    colParams, noPad);
        DataCopyPad(colK, dAkk_[AOffset(b, hv, chunkStart + lateBlock * KDA_MIXED_BC,
                                       earlyBlock * KDA_MIXED_BC + rowStart)],
                    colParams, noPad);
        TEventID dAReady = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(dAReady);

        BuildPairGates(earlyBlock, lateBlock, slot);
        BuildPairOperands(earlyBlock, lateBlock, slot);

        WaitFlag<HardEvent::MTE2_S>(dAReady);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(dAReady);
        const bool diagonal = earlyBlock == lateBlock;
        if (diagonal) {
            __ubuf__ float *rowQPtr = reinterpret_cast<__ubuf__ float *>(rowQ.GetPhyAddr());
            __ubuf__ float *rowKPtr = reinterpret_cast<__ubuf__ float *>(rowK.GetPhyAddr());
            __ubuf__ float *colQPtr = reinterpret_cast<__ubuf__ float *>(colQ.GetPhyAddr());
            __ubuf__ float *colKPtr = reinterpret_cast<__ubuf__ float *>(colK.GetPhyAddr());
            for (uint32_t row = 0; row < KDA_MIXED_ROWS_PER_AIV; ++row) {
                const uint32_t logicalRow = rowStart + row;
                for (uint32_t col = 0; col < KDA_MIXED_BC; ++col) {
                    if (col > logicalRow) {
                        rowQPtr[row * KDA_MIXED_BC + col] = 0.0f;
                        rowKPtr[row * KDA_MIXED_BC + col] = 0.0f;
                    }
                }
                for (uint32_t source = 0; source < KDA_MIXED_BC; ++source) {
                    if (source < logicalRow) {
                        colQPtr[source * KDA_MIXED_ROWS_PER_AIV + row] = 0.0f;
                        colKPtr[source * KDA_MIXED_ROWS_PER_AIV + row] = 0.0f;
                    }
                }
            }
            SyncSToV();
        } else {
            SyncMte2ToV();
        }
        Adds(rowQ, rowQ, 0.0f, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC);
        Adds(rowK, rowK, 0.0f, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC);
        Adds(colQ, colQ, 0.0f, KDA_MIXED_BC * KDA_MIXED_ROWS_PER_AIV);
        Adds(colK, colK, 0.0f, KDA_MIXED_BC * KDA_MIXED_ROWS_PER_AIV);
        PipeBarrier<PIPE_V>();

        constexpr uint32_t rowBlockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        const uint64_t slotBase = SlotBase(slot);
        DataCopyExtParams colToARightParams{
            KDA_MIXED_BC,
            KDA_MIXED_ROWS_PER_AIV * sizeof(float),
            0,
            (KDA_MIXED_BC - KDA_MIXED_ROWS_PER_AIV) * sizeof(float),
            0};
        SyncVToMte3();
        DataCopy(workspace_[slotBase + KDA_MIXED_A_LEFT + rowStart * KDA_MIXED_BC],
                 rowQ, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC);
        DataCopy(workspace_[slotBase + KDA_MIXED_A_LEFT + (KDA_MIXED_BC + rowStart) * KDA_MIXED_BC],
                 rowK, KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_BC);
        DataCopyPad(workspace_[slotBase + KDA_MIXED_A_RIGHT + rowStart],
                    colQ, colToARightParams);
        DataCopyPad(workspace_[slotBase + KDA_MIXED_A_RIGHT +
                               KDA_MIXED_BC * KDA_MIXED_BC + rowStart],
                    colK, colToARightParams);
        DataCopy(workspace_[slotBase + KDA_MIXED_B_LEFT + rowStart * KDA_MIXED_K],
                 operand0Buf_.Get<float>(), rowBlockElements);
        DataCopy(workspace_[slotBase + KDA_MIXED_B_RIGHT + rowStart * KDA_MIXED_K],
                 operand1Buf_.Get<float>(), rowBlockElements);
        DataCopy(workspace_[slotBase + KDA_MIXED_B_RIGHT +
                            (KDA_MIXED_BC + rowStart) * KDA_MIXED_K],
                 operand2Buf_.Get<float>(), rowBlockElements);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ComputePairAic(KdaBwdMixedLeftBlockMmad &leftBlockMmad,
                                           KdaBwdMixedRightBlockMmad &rightBlockMmad,
                                           uint32_t slot)
    {
        const uint64_t slotBase = SlotBase(slot);
        auto layoutALeft = tla::MakeLayout<float, Catlass::layout::RowMajor>(32, 16);
        auto layoutBLeft = tla::MakeLayout<float, Catlass::layout::RowMajor>(16, KDA_MIXED_K);
        auto layoutCLeft = tla::MakeLayout<float, Catlass::layout::RowMajor>(32, KDA_MIXED_K);
        auto tensorALeft = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_A_LEFT], layoutALeft,
                                           Catlass::Arch::PositionGM{});
        auto tensorBLeft = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_B_LEFT], layoutBLeft,
                                           Catlass::Arch::PositionGM{});
        auto tensorCLeft = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_C_LEFT], layoutCLeft,
                                           Catlass::Arch::PositionGM{});
        Catlass::GemmCoord leftShape{32, KDA_MIXED_K, 16};
        auto blockALeft = GetTile(tensorALeft, tla::MakeCoord(0, 0), tla::MakeShape(32, 16));
        auto blockBLeft = GetTile(tensorBLeft, tla::MakeCoord(0, 0), tla::MakeShape(16, KDA_MIXED_K));
        auto blockCLeft = GetTile(tensorCLeft, tla::MakeCoord(0, 0), tla::MakeShape(32, KDA_MIXED_K));
        leftBlockMmad.preSetFlags();
        leftBlockMmad(blockALeft, blockBLeft, blockCLeft, leftShape);
        leftBlockMmad.finalWaitFlags();

        auto layoutARight = tla::MakeLayout<float, Catlass::layout::ColumnMajor>(16, 32);
        auto layoutBRight = tla::MakeLayout<float, Catlass::layout::RowMajor>(32, KDA_MIXED_K);
        auto layoutCRight = tla::MakeLayout<float, Catlass::layout::RowMajor>(16, KDA_MIXED_K);
        auto tensorARight = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_A_RIGHT], layoutARight,
                                            Catlass::Arch::PositionGM{});
        auto tensorBRight = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_B_RIGHT], layoutBRight,
                                            Catlass::Arch::PositionGM{});
        auto tensorCRight = tla::MakeTensor(workspace_[slotBase + KDA_MIXED_C_RIGHT], layoutCRight,
                                            Catlass::Arch::PositionGM{});
        Catlass::GemmCoord rightShape{16, KDA_MIXED_K, 32};
        auto blockARight = GetTile(tensorARight, tla::MakeCoord(0, 0), tla::MakeShape(16, 32));
        auto blockBRight = GetTile(tensorBRight, tla::MakeCoord(0, 0), tla::MakeShape(32, KDA_MIXED_K));
        auto blockCRight = GetTile(tensorCRight, tla::MakeCoord(0, 0), tla::MakeShape(16, KDA_MIXED_K));
        rightBlockMmad.preSetFlags();
        rightBlockMmad(blockARight, blockBRight, blockCRight, rightShape);
        rightBlockMmad.finalWaitFlags();
    }

    __aicore__ inline void ConsumePairAiv(uint32_t earlyBlock, uint32_t lateBlock,
                                          uint32_t slot)
    {
        constexpr uint32_t rowBlockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        const uint32_t rowStart = static_cast<uint32_t>(GetSubBlockIdx()) * KDA_MIXED_ROWS_PER_AIV;
        const uint64_t slotBase = SlotBase(slot);
        LocalTensor<float> result = resultBuf_.Get<float>();
        LocalTensor<float> rightResult = operand0Buf_.Get<float>();
        // Defer this dependency until the next MTE2 consumer so the following
        // pair can prepare its Vector operands after the previous accumulation.
        SyncVToMte2();
        DataCopy(result,
                 workspace_[slotBase + KDA_MIXED_C_LEFT + rowStart * KDA_MIXED_K],
                 rowBlockElements);
        DataCopy(result[rowBlockElements],
                 workspace_[slotBase + KDA_MIXED_C_LEFT +
                            (KDA_MIXED_BC + rowStart) * KDA_MIXED_K],
                 rowBlockElements);
        DataCopy(rightResult,
                 workspace_[slotBase + KDA_MIXED_C_RIGHT + rowStart * KDA_MIXED_K],
                 rowBlockElements);
        SyncMte2ToV();
        LocalTensor<float> gates = gateSlotsBuf_.Get<float>();
        LocalTensor<float> gateLate = gates[slot * 2 * rowBlockElements];
        LocalTensor<float> gateEarly = gates[slot * 2 * rowBlockElements + rowBlockElements];
        MulAddDst(dqAccBuf_.Get<float>()[CacheBlockOffset(lateBlock)], result,
                  gateLate, rowBlockElements);
        MulAddDst(dkLeftAccBuf_.Get<float>()[CacheBlockOffset(lateBlock)],
                  result[rowBlockElements], gateLate, rowBlockElements);
        MulAddDst(dkRightAccBuf_.Get<float>()[CacheBlockOffset(earlyBlock)],
                  rightResult, gateEarly, rowBlockElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreTaskOutputs(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                            uint32_t subBlockIdx)
    {
        constexpr uint32_t rowBlockElements = KDA_MIXED_ROWS_PER_AIV * KDA_MIXED_K;
        LocalTensor<float> outQ = resultBuf_.Get<float>();
        LocalTensor<float> outK = resultBuf_.Get<float>()[rowBlockElements];
        LocalTensor<float> outG = operand0Buf_.Get<float>();
        LocalTensor<float> scaledLeft = operand1Buf_.Get<float>();
        LocalTensor<float> product = operand2Buf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceBuf_.Get<float>();
        LocalTensor<float> dbLocal = dbBuf_.Get<float>();
        // Vector destinations must be 32-byte aligned. Reuse the now-dead dA
        // buffers as padded [8, 8] db storage plus an aligned compact vector.
        LocalTensor<float> dbAcc = dARowQBuf_.Get<float>();
        LocalTensor<float> dbCompact = dARowKBuf_.Get<float>();
        __ubuf__ float *betaPtr = reinterpret_cast<__ubuf__ float *>(betaBuf_.Get<float>().GetPhyAddr());
        const uint32_t rowStart = subBlockIdx * KDA_MIXED_ROWS_PER_AIV;
        // The final pair leaves resultBuf_/operand0Buf_ live on Vector.
        SyncVToMte2();
        for (uint32_t block = 0; block < KDA_MIXED_BLOCKS; ++block) {
            const uint64_t token = chunkStart + block * KDA_MIXED_BC + rowStart;
            const uint32_t cacheOffset = CacheBlockOffset(block);
            DataCopy(outQ, dq_[VOffset(b, hv, token)], rowBlockElements);
            DataCopy(outK, dk_[VOffset(b, hv, token)], rowBlockElements);
            DataCopy(outG, dg_[VOffset(b, hv, token)], rowBlockElements);
            DataCopy(dbLocal, db_[BetaOffset(b, hv, token)], KDA_MIXED_ROWS_PER_AIV);
            SyncMte2ToV();

            Duplicate(dbAcc, 0.0f, KDA_MIXED_ROWS_PER_AIV * 8);
            Mul(product, dkLeftAccBuf_.Get<float>()[cacheOffset],
                kCacheBuf_.Get<float>()[cacheOffset], rowBlockElements);
            Add(outQ, outQ, dqAccBuf_.Get<float>()[cacheOffset], rowBlockElements);
            Adds(scaledLeft, dkLeftAccBuf_.Get<float>()[cacheOffset], 0.0f, rowBlockElements);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < KDA_MIXED_ROWS_PER_AIV; ++row) {
                ReduceSum<float, true>(scalar, product[row * KDA_MIXED_K], reduceTmp,
                                       KDA_MIXED_K);
                PipeBarrier<PIPE_V>();
                Add(dbAcc[row * 8], dbAcc[row * 8], scalar, 1);
                PipeBarrier<PIPE_V>();
            }
            ScaleEightRowsByBeta(scaledLeft, scaledLeft, 0, betaPtr,
                                 block * KDA_MIXED_ROWS_PER_AIV);
            PipeBarrier<PIPE_V>();
            SyncVToS();
            __ubuf__ float *dbAccPtr = reinterpret_cast<__ubuf__ float *>(dbAcc.GetPhyAddr());
            __ubuf__ float *dbCompactPtr = reinterpret_cast<__ubuf__ float *>(dbCompact.GetPhyAddr());
            for (uint32_t row = 0; row < KDA_MIXED_ROWS_PER_AIV; ++row) {
                dbCompactPtr[row] = dbAccPtr[row * 8];
            }
            SyncSToV();
            Add(dbLocal, dbLocal, dbCompact, KDA_MIXED_ROWS_PER_AIV);
            PipeBarrier<PIPE_V>();
            Add(outK, outK, scaledLeft, rowBlockElements);
            PipeBarrier<PIPE_V>();
            Add(outK, outK, dkRightAccBuf_.Get<float>()[cacheOffset], rowBlockElements);
            MulAddDst(outG, qCacheBuf_.Get<float>()[cacheOffset],
                      dqAccBuf_.Get<float>()[cacheOffset], rowBlockElements);
            Sub(product, scaledLeft, dkRightAccBuf_.Get<float>()[cacheOffset], rowBlockElements);
            PipeBarrier<PIPE_V>();
            MulAddDst(outG, product, kCacheBuf_.Get<float>()[cacheOffset], rowBlockElements);
            PipeBarrier<PIPE_V>();

            SyncVToMte3();
            DataCopy(dqOut_[VOffset(b, hv, token)], outQ, rowBlockElements);
            DataCopy(dkOut_[VOffset(b, hv, token)], outK, rowBlockElements);
            DataCopy(dgOut_[VOffset(b, hv, token)], outG, rowBlockElements);
            DataCopy(dbOut_[BetaOffset(b, hv, token)], dbLocal, KDA_MIXED_ROWS_PER_AIV);
            SyncMte3ToMte2();
            SyncMte3ToV();
        }
    }

    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<float> g_;
    GlobalTensor<float> beta_;
    GlobalTensor<float> dAqk_;
    GlobalTensor<float> dAkk_;
    GlobalTensor<float> dq_;
    GlobalTensor<float> dk_;
    GlobalTensor<float> db_;
    GlobalTensor<float> dg_;
    GlobalTensor<float> dqOut_;
    GlobalTensor<float> dkOut_;
    GlobalTensor<float> dbOut_;
    GlobalTensor<float> dgOut_;
    GlobalTensor<float> workspace_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> qTypedBuf_, kTypedBuf_, qCacheBuf_, kCacheBuf_, kBetaCacheBuf_, gCacheBuf_;
    TBuf<TPosition::VECCALC> refBuf_, betaBuf_, gateSlotsBuf_;
    TBuf<TPosition::VECCALC> dqAccBuf_, dkLeftAccBuf_, dkRightAccBuf_;
    TBuf<TPosition::VECCALC> operand0Buf_, operand1Buf_, operand2Buf_;
    TBuf<TPosition::VECCALC> dARowQBuf_, dARowKBuf_, dAColQBuf_, dAColKBuf_;
    TBuf<TPosition::VECCALC> resultBuf_, reduceBuf_, scalarBuf_, dbBuf_;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_MIXED_QUEUE_DEPTH> readyFlag_{
        KDA_MIXED_READY_FLAG0, KDA_MIXED_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_MIXED_QUEUE_DEPTH> doneFlag_{
        KDA_MIXED_DONE_FLAG0, KDA_MIXED_DONE_FLAG1};
    uint64_t batch_ = 0;
    uint64_t heads_ = 0;
    uint64_t seqlen_ = 0;
    uint64_t chunks_ = 0;
    uint64_t usedCoreNum_ = 1;
    uint64_t logicalCoreIdx_ = 0;
};

#endif // CHUNK_KDA_BWD_INTRA_MIXED_HPP
