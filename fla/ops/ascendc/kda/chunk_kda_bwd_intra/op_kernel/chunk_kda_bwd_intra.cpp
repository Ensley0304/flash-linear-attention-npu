/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t BC = 16;
constexpr uint32_t BK = 32;
constexpr uint32_t MAX_BT = 128;
constexpr uint32_t MAX_K = 256;
constexpr float LN2 = 0.69314718055994530942f;

template <typename T, bool SAFE_GATE>
class ChunkKdaBwdIntraKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
                                GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR chunkIndices,
                                const ChunkKdaBwdIntraTilingData &tiling, TPipe *pipe)
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

        b_ = static_cast<uint64_t>(tiling.batch);
        h_ = static_cast<uint64_t>(tiling.qHeadNum);
        hv_ = static_cast<uint64_t>(tiling.vHeadNum);
        t_ = static_cast<uint64_t>(tiling.seqlen);
        kDim_ = static_cast<uint64_t>(tiling.headDim);
        bt_ = static_cast<uint64_t>(tiling.chunkSize);
        nt_ = static_cast<uint64_t>(tiling.totalChunks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        isVarLen_ = tiling.isVarLen != 0;
        if (isVarLen_) {
            chunkMeta_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
        pipe_ = pipe;

        pipe_->InitBuffer(dARowQBuf_, BC * MAX_BT * sizeof(float));
        pipe_->InitBuffer(dARowKBuf_, BC * MAX_BT * sizeof(float));
        pipe_->InitBuffer(dAColQBuf_, MAX_BT * BC * sizeof(float));
        pipe_->InitBuffer(dAColKBuf_, MAX_BT * BC * sizeof(float));
        pipe_->InitBuffer(betaBuf_, MAX_BT * sizeof(float));
        pipe_->InitBuffer(qTypedBuf_, BK * sizeof(T));
        pipe_->InitBuffer(kTypedBuf_, BK * sizeof(T));
        pipe_->InitBuffer(qSrcBuf_, BK * sizeof(float));
        pipe_->InitBuffer(kSrcBuf_, BK * sizeof(float));
        pipe_->InitBuffer(gSrcBuf_, BK * sizeof(float));
        pipe_->InitBuffer(qSelfBuf_, BK * sizeof(float));
        pipe_->InitBuffer(kSelfBuf_, BK * sizeof(float));
        pipe_->InitBuffer(gSelfBuf_, BK * sizeof(float));
        pipe_->InitBuffer(gLeftRefBuf_, MAX_K * sizeof(float));
        pipe_->InitBuffer(gRightRefBuf_, MAX_K * sizeof(float));
        if constexpr (SAFE_GATE) {
            pipe_->InitBuffer(gDiagRefBuf_, MAX_K * sizeof(float));
        }
        pipe_->InitBuffer(gateBuf_, BK * sizeof(float));
        pipe_->InitBuffer(tmp0Buf_, BK * sizeof(float));
        pipe_->InitBuffer(tmp1Buf_, BK * sizeof(float));
        pipe_->InitBuffer(dqAccBuf_, BK * sizeof(float));
        pipe_->InitBuffer(dkLeftBuf_, BK * sizeof(float));
        pipe_->InitBuffer(dkRightBuf_, BK * sizeof(float));
        pipe_->InitBuffer(out0Buf_, BK * sizeof(float));
        pipe_->InitBuffer(out1Buf_, BK * sizeof(float));
        pipe_->InitBuffer(out2Buf_, BK * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(chunkMetaBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint64_t nc = bt_ / BC;
        const uint64_t flatChunks = isVarLen_ ? nt_ : b_ * nt_;
        const uint64_t taskCount = flatChunks * hv_ * nc;
        for (uint64_t task = static_cast<uint64_t>(GetBlockIdx()); task < taskCount; task += usedCoreNum_) {
            ProcessTask(task, nc);
        }
    }

private:
    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        return ((b * h_ + h) * t_ + t) * kDim_ + d;
    }

    __aicore__ inline uint64_t VOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        return ((b * hv_ + hv) * t_ + t) * kDim_ + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * hv_ + hv) * t_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t col) const
    {
        return ((b * hv_ + hv) * t_ + t) * bt_ + col;
    }

    __aicore__ inline bool ResolveTask(uint64_t task, uint64_t nc, uint64_t &b, uint64_t &h, uint64_t &hv,
                                       uint64_t &start, uint64_t &end, uint64_t &rowBlock)
    {
        rowBlock = task % nc;
        uint64_t headTask = task / nc;
        hv = headTask % hv_;
        uint64_t flatChunk = headTask / hv_;
        if (!isVarLen_) {
            b = flatChunk / nt_;
            uint64_t localChunk = flatChunk % nt_;
            start = localChunk * bt_;
            end = start + bt_;
            if (end > t_) {
                end = t_;
            }
        } else {
            LocalTensor<int64_t> metadata = chunkMetaBuf_.Get<int64_t>();
            DataCopy(metadata, chunkMeta_[flatChunk * 4], 4);
            SyncMte2ToS();
            __ubuf__ int64_t *metadataPtr = reinterpret_cast<__ubuf__ int64_t *>(metadata.GetPhyAddr());
            start = static_cast<uint64_t>(metadataPtr[1]);
            end = static_cast<uint64_t>(metadataPtr[2]);
            b = 0;
        }
        h = hv / (hv_ / h_);
        return start < end && rowBlock * BC < end - start;
    }

    __aicore__ inline void CopyFp32In(LocalTensor<float> dst, GlobalTensor<float> &src, uint64_t offset,
                                      uint64_t count)
    {
        const uint64_t bytes = count * sizeof(float);
        if (bytes >= 32 && (bytes % 32) == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(dst, src[offset], params, pad);
        }
    }

    __aicore__ inline void CopyFp32Out(GlobalTensor<float> &dst, uint64_t offset, LocalTensor<float> src,
                                       uint64_t count)
    {
        const uint64_t bytes = count * sizeof(float);
        if (bytes >= 32 && (bytes % 32) == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(bytes), 0, 0, 0};
            DataCopyPad(dst[offset], src, params);
        }
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void SyncSToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToS()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void LoadQkg(uint64_t qOffset, uint64_t vOffset, LocalTensor<float> qDst,
                                   LocalTensor<float> kDst, LocalTensor<float> gDst, uint32_t count)
    {
        LocalTensor<T> qTyped = qTypedBuf_.Get<T>();
        LocalTensor<T> kTyped = kTypedBuf_.Get<T>();
        SyncVToMte2();
        DataCopy(qTyped, q_[qOffset], count);
        DataCopy(kTyped, k_[qOffset], count);
        DataCopy(gDst, g_[vOffset], count);
        SyncMte2ToV();
        Cast(qDst, qTyped, RoundMode::CAST_NONE, count);
        Cast(kDst, kTyped, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadKg(uint64_t qOffset, uint64_t vOffset, LocalTensor<float> kDst,
                                  LocalTensor<float> gDst, uint32_t count)
    {
        LocalTensor<T> kTyped = kTypedBuf_.Get<T>();
        SyncVToMte2();
        DataCopy(kTyped, k_[qOffset], count);
        DataCopy(gDst, g_[vOffset], count);
        SyncMte2ToV();
        Cast(kDst, kTyped, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadG(uint64_t vOffset, LocalTensor<float> gDst, uint32_t count)
    {
        SyncVToMte2();
        DataCopy(gDst, g_[vOffset], count);
        SyncMte2ToV();
        Adds(gDst, gDst, 0.0f, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildGate(LocalTensor<float> gate, LocalTensor<float> lhs, LocalTensor<float> rhs,
                                     uint32_t count)
    {
        Sub(gate, lhs, rhs, count);
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, LN2, count);
        PipeBarrier<PIPE_V>();
        Exp(gate, gate, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddScaled(LocalTensor<float> acc, LocalTensor<float> source,
                                     LocalTensor<float> gate, float scale, uint32_t count)
    {
        LocalTensor<float> tmp0 = tmp0Buf_.Get<float>();
        Mul(tmp0, source, gate, count);
        PipeBarrier<PIPE_V>();
        Muls(tmp0, tmp0, scale, count);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp0, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddScaledPair(LocalTensor<float> acc0, float scale0, LocalTensor<float> acc1,
                                         float scale1, LocalTensor<float> source, LocalTensor<float> gate,
                                         uint32_t count)
    {
        LocalTensor<float> common = tmp0Buf_.Get<float>();
        LocalTensor<float> scaled = tmp1Buf_.Get<float>();
        Mul(common, source, gate, count);
        PipeBarrier<PIPE_V>();
        Muls(scaled, common, scale0, count);
        PipeBarrier<PIPE_V>();
        Add(acc0, acc0, scaled, count);
        PipeBarrier<PIPE_V>();
        Muls(scaled, common, scale1, count);
        PipeBarrier<PIPE_V>();
        Add(acc1, acc1, scaled, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadDABlock(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                       uint64_t curT, uint64_t rowBegin, uint64_t rowCount)
    {
        LocalTensor<float> rowQ = dARowQBuf_.Get<float>();
        LocalTensor<float> rowK = dARowKBuf_.Get<float>();
        LocalTensor<float> colQ = dAColQBuf_.Get<float>();
        LocalTensor<float> colK = dAColKBuf_.Get<float>();
        DataCopyExtParams rowParams{static_cast<uint16_t>(rowCount),
                                    static_cast<uint32_t>(bt_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(rowQ, dAqk_[AOffset(b, hv, chunkStart + rowBegin, 0)], rowParams, noPad);
        DataCopyPad(rowK, dAkk_[AOffset(b, hv, chunkStart + rowBegin, 0)], rowParams, noPad);

        // Load the 16 columns owned by this task as a packed [curT, BC]
        // slab. BT and BC are both 32-byte aligned for FP32, so every block
        // has an exact layout and ProcessRow can address one column with a
        // fixed BC stride. The full BC columns are valid storage even for a
        // short tail task; only rowCount columns are consumed.
        DataCopyExtParams colParams{static_cast<uint16_t>(curT), BC * sizeof(float),
                                    static_cast<uint32_t>((bt_ - BC) * sizeof(float)), 0, 0};
        DataCopyPad(colQ, dAqk_[AOffset(b, hv, chunkStart, rowBegin)], colParams, noPad);
        DataCopyPad(colK, dAkk_[AOffset(b, hv, chunkStart, rowBegin)], colParams, noPad);
        SyncMte2ToS();
    }

    __aicore__ inline float ReduceDb(LocalTensor<float> dkLeft, LocalTensor<float> kSelf, uint32_t count)
    {
        LocalTensor<float> product = tmp0Buf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceBuf_.Get<float>();
        Mul(product, dkLeft, kSelf, count);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, true>(scalar, product, reduceTmp, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        SyncVToS();
        return reinterpret_cast<__ubuf__ float *>(scalar.GetPhyAddr())[0];
    }

    __aicore__ inline void ProcessRow(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkStart,
                                      uint64_t curT, uint64_t localRow, __ubuf__ float *betaPtr)
    {
        const uint64_t globalRow = chunkStart + localRow;
        const uint64_t rowInBlock = localRow % BC;
        __ubuf__ float *rowQPtr = reinterpret_cast<__ubuf__ float *>(dARowQBuf_.Get<float>().GetPhyAddr()) +
                                 rowInBlock * bt_;
        __ubuf__ float *rowKPtr = reinterpret_cast<__ubuf__ float *>(dARowKBuf_.Get<float>().GetPhyAddr()) +
                                 rowInBlock * bt_;
        __ubuf__ float *colQPtr = reinterpret_cast<__ubuf__ float *>(dAColQBuf_.Get<float>().GetPhyAddr()) +
                                 rowInBlock;
        __ubuf__ float *colKPtr = reinterpret_cast<__ubuf__ float *>(dAColKBuf_.Get<float>().GetPhyAddr()) +
                                 rowInBlock;
        const float betaValue = betaPtr[localRow];
        float dbSum = 0.0f;

        for (uint64_t d = 0; d < kDim_; d += BK) {
            const uint32_t curK = static_cast<uint32_t>((kDim_ - d < BK) ? kDim_ - d : BK);
            LocalTensor<float> qSelf = qSelfBuf_.Get<float>();
            LocalTensor<float> kSelf = kSelfBuf_.Get<float>();
            LocalTensor<float> gSelf = gSelfBuf_.Get<float>();
            LoadQkg(QOffset(b, h, globalRow, d), VOffset(b, hv, globalRow, d), qSelf, kSelf, gSelf, curK);

            LocalTensor<float> dqAcc = dqAccBuf_.Get<float>();
            LocalTensor<float> dkLeft = dkLeftBuf_.Get<float>();
            LocalTensor<float> dkRight = dkRightBuf_.Get<float>();
            Duplicate(dqAcc, 0.0f, curK);
            Duplicate(dkLeft, 0.0f, curK);
            Duplicate(dkRight, 0.0f, curK);
            PipeBarrier<PIPE_V>();

            if constexpr (SAFE_GATE) {
                // Match the Triton safe-gate path: factor each causal exponent around
                // the current 16-token block's first/middle/last gate. This avoids a
                // single long-range Exp without changing the mathematical result.
                const uint64_t blockBegin = (localRow / BC) * BC;
                uint64_t blockEnd = blockBegin + BC;
                if (blockEnd > curT) {
                    blockEnd = curT;
                }
                LocalTensor<float> gLeftRef = gLeftRefBuf_.Get<float>()[d];
                LocalTensor<float> gDiagRef = gDiagRefBuf_.Get<float>()[d];
                LocalTensor<float> gRightRef = gRightRefBuf_.Get<float>()[d];

                LocalTensor<float> dqDiag = out0Buf_.Get<float>();
                LocalTensor<float> dkLeftDiag = out1Buf_.Get<float>();
                LocalTensor<float> dkRightFuture = out2Buf_.Get<float>();
                Duplicate(dqDiag, 0.0f, curK);
                Duplicate(dkLeftDiag, 0.0f, curK);
                Duplicate(dkRightFuture, 0.0f, curK);
                PipeBarrier<PIPE_V>();

                for (uint64_t j = 0; j < curT; ++j) {
                    const uint64_t globalJ = chunkStart + j;
                    LocalTensor<float> qSrc = qSrcBuf_.Get<float>();
                    LocalTensor<float> kSrc = kSrcBuf_.Get<float>();
                    LocalTensor<float> gSrc = gSrcBuf_.Get<float>();
                    LocalTensor<float> gate = gateBuf_.Get<float>();
                    if (j < localRow) {
                        LoadKg(QOffset(b, h, globalJ, d), VOffset(b, hv, globalJ, d), kSrc, gSrc, curK);
                    } else {
                        LoadQkg(QOffset(b, h, globalJ, d), VOffset(b, hv, globalJ, d), qSrc, kSrc, gSrc, curK);
                    }
                    float rowQScale = 0.0f;
                    float rowKScale = 0.0f;
                    float colQScale = 0.0f;
                    float colKScale = 0.0f;
                    if (j <= localRow) {
                        rowQScale = rowQPtr[j];
                        rowKScale = rowKPtr[j];
                    }
                    if (j >= localRow) {
                        colQScale = colQPtr[j * BC];
                        colKScale = colKPtr[j * BC] * betaPtr[j];
                    }
                    // dA and beta coefficients are read by PIPE_S from UB.
                    // Auto-sync is disabled for this project, so PIPE_V must
                    // explicitly wait before consuming those scalar values.
                    // Reading after issuing LoadQkg lets PIPE_S overlap the
                    // coefficient access with the source-token MTE2/Cast path.
                    SyncSToV();
                    if (j <= localRow) {
                        if (j < blockBegin) {
                            BuildGate(gate, gLeftRef, gSrc, curK);
                            AddScaledPair(dqAcc, rowQScale, dkLeft, rowKScale, kSrc, gate, curK);
                        } else {
                            BuildGate(gate, gDiagRef, gSrc, curK);
                            AddScaledPair(dqDiag, rowQScale, dkLeftDiag, rowKScale, kSrc, gate, curK);
                        }
                    }
                    if (j >= localRow) {
                        if (j < blockEnd) {
                            BuildGate(gate, gSrc, gDiagRef, curK);
                            AddScaled(dkRight, qSrc, gate, colQScale, curK);
                            AddScaled(dkRight, kSrc, gate, colKScale, curK);
                        } else {
                            BuildGate(gate, gSrc, gRightRef, curK);
                            AddScaled(dkRightFuture, qSrc, gate, colQScale, curK);
                            AddScaled(dkRightFuture, kSrc, gate, colKScale, curK);
                        }
                    }
                }

                LocalTensor<float> gate = gateBuf_.Get<float>();
                if (blockBegin > 0) {
                    BuildGate(gate, gSelf, gLeftRef, curK);
                    Mul(dqAcc, dqAcc, gate, curK);
                    Mul(dkLeft, dkLeft, gate, curK);
                    PipeBarrier<PIPE_V>();
                }
                BuildGate(gate, gSelf, gDiagRef, curK);
                Mul(dqDiag, dqDiag, gate, curK);
                Mul(dkLeftDiag, dkLeftDiag, gate, curK);
                PipeBarrier<PIPE_V>();
                Add(dqAcc, dqAcc, dqDiag, curK);
                Add(dkLeft, dkLeft, dkLeftDiag, curK);
                PipeBarrier<PIPE_V>();
                BuildGate(gate, gDiagRef, gSelf, curK);
                Mul(dkRight, dkRight, gate, curK);
                PipeBarrier<PIPE_V>();
                if (blockEnd < curT) {
                    BuildGate(gate, gRightRef, gSelf, curK);
                    Mul(dkRightFuture, dkRightFuture, gate, curK);
                    PipeBarrier<PIPE_V>();
                    Add(dkRight, dkRight, dkRightFuture, curK);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                // Match the upstream non-safe calculation order. Inter-block
                // contributions are still factored around the current block's
                // first/last gate; only the current 16x16 diagonal block uses
                // direct pairwise exp2 differences.
                const uint64_t blockBegin = (localRow / BC) * BC;
                uint64_t blockEnd = blockBegin + BC;
                if (blockEnd > curT) {
                    blockEnd = curT;
                }
                LocalTensor<float> gLeftRef = gLeftRefBuf_.Get<float>()[d];
                LocalTensor<float> gRightRef = gRightRefBuf_.Get<float>()[d];
                LocalTensor<float> dqPrevious = out0Buf_.Get<float>();
                LocalTensor<float> dkLeftPrevious = out1Buf_.Get<float>();
                LocalTensor<float> dkRightFuture = out2Buf_.Get<float>();
                Duplicate(dqPrevious, 0.0f, curK);
                Duplicate(dkLeftPrevious, 0.0f, curK);
                Duplicate(dkRightFuture, 0.0f, curK);
                PipeBarrier<PIPE_V>();

                for (uint64_t j = 0; j < curT; ++j) {
                    const uint64_t globalJ = chunkStart + j;
                    LocalTensor<float> qSrc = qSrcBuf_.Get<float>();
                    LocalTensor<float> kSrc = kSrcBuf_.Get<float>();
                    LocalTensor<float> gSrc = gSrcBuf_.Get<float>();
                    LocalTensor<float> gate = gateBuf_.Get<float>();
                    if (j < localRow) {
                        LoadKg(QOffset(b, h, globalJ, d), VOffset(b, hv, globalJ, d), kSrc, gSrc, curK);
                    } else {
                        LoadQkg(QOffset(b, h, globalJ, d), VOffset(b, hv, globalJ, d), qSrc, kSrc, gSrc, curK);
                    }
                    float rowQScale = 0.0f;
                    float rowKScale = 0.0f;
                    float colQScale = 0.0f;
                    float colKScale = 0.0f;
                    if (j <= localRow) {
                        rowQScale = rowQPtr[j];
                        rowKScale = rowKPtr[j];
                    }
                    if (j >= localRow) {
                        colQScale = colQPtr[j * BC];
                        colKScale = colKPtr[j * BC] * betaPtr[j];
                    }
                    SyncSToV();
                    if (j <= localRow) {
                        if (j < blockBegin) {
                            BuildGate(gate, gLeftRef, gSrc, curK);
                            AddScaledPair(dqPrevious, rowQScale, dkLeftPrevious, rowKScale, kSrc, gate, curK);
                        } else if (j == localRow) {
                            Duplicate(gate, 1.0f, curK);
                            PipeBarrier<PIPE_V>();
                            AddScaledPair(dqAcc, rowQScale, dkLeft, rowKScale, kSrc, gate, curK);
                        } else {
                            BuildGate(gate, gSelf, gSrc, curK);
                            AddScaledPair(dqAcc, rowQScale, dkLeft, rowKScale, kSrc, gate, curK);
                        }
                    }
                    if (j >= localRow) {
                        if (j < blockEnd) {
                            if (j != localRow) {
                                BuildGate(gate, gSrc, gSelf, curK);
                            }
                            AddScaled(dkRight, qSrc, gate, colQScale, curK);
                            AddScaled(dkRight, kSrc, gate, colKScale, curK);
                        } else {
                            BuildGate(gate, gSrc, gRightRef, curK);
                            AddScaled(dkRightFuture, qSrc, gate, colQScale, curK);
                            AddScaled(dkRightFuture, kSrc, gate, colKScale, curK);
                        }
                    }
                }

                LocalTensor<float> gate = gateBuf_.Get<float>();
                if (blockBegin > 0) {
                    BuildGate(gate, gSelf, gLeftRef, curK);
                    Mul(dqPrevious, dqPrevious, gate, curK);
                    Mul(dkLeftPrevious, dkLeftPrevious, gate, curK);
                    PipeBarrier<PIPE_V>();
                    Add(dqAcc, dqAcc, dqPrevious, curK);
                    Add(dkLeft, dkLeft, dkLeftPrevious, curK);
                    PipeBarrier<PIPE_V>();
                }
                if (blockEnd < curT) {
                    BuildGate(gate, gRightRef, gSelf, curK);
                    Mul(dkRightFuture, dkRightFuture, gate, curK);
                    PipeBarrier<PIPE_V>();
                    Add(dkRight, dkRight, dkRightFuture, curK);
                    PipeBarrier<PIPE_V>();
                }
            }

            dbSum += ReduceDb(dkLeft, kSelf, curK);
            // ReduceDb returns a PIPE_S read from scalarBuf_.  Synchronize it
            // before the next vector operation (and before scalarBuf_ can be
            // reused by a later feature tile).
            SyncSToV();
            Muls(dkLeft, dkLeft, betaValue, curK);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> outQ = out0Buf_.Get<float>();
            LocalTensor<float> outK = out1Buf_.Get<float>();
            LocalTensor<float> outG = out2Buf_.Get<float>();
            SyncVToMte2();
            CopyFp32In(outQ, dq_, VOffset(b, hv, globalRow, d), curK);
            CopyFp32In(outK, dk_, VOffset(b, hv, globalRow, d), curK);
            CopyFp32In(outG, dg_, VOffset(b, hv, globalRow, d), curK);
            SyncMte2ToV();
            Add(outQ, outQ, dqAcc, curK);
            Add(outK, outK, dkLeft, curK);
            Add(outK, outK, dkRight, curK);
            LocalTensor<float> tmp0 = tmp0Buf_.Get<float>();
            LocalTensor<float> tmp1 = tmp1Buf_.Get<float>();
            Mul(tmp0, qSelf, dqAcc, curK);
            Sub(tmp1, dkLeft, dkRight, curK);
            PipeBarrier<PIPE_V>();
            Mul(tmp1, tmp1, kSelf, curK);
            PipeBarrier<PIPE_V>();
            Add(outG, outG, tmp0, curK);
            Add(outG, outG, tmp1, curK);
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            CopyFp32Out(dqOut_, VOffset(b, hv, globalRow, d), outQ, curK);
            CopyFp32Out(dkOut_, VOffset(b, hv, globalRow, d), outK, curK);
            CopyFp32Out(dgOut_, VOffset(b, hv, globalRow, d), outG, curK);
            SyncMte3ToMte2();
            SyncMte3ToV();
        }

        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        SyncVToMte2();
        CopyFp32In(scalar, db_, BetaOffset(b, hv, globalRow), 1);
        SyncMte2ToV();
        Adds(scalar, scalar, dbSum, 1);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        CopyFp32Out(dbOut_, BetaOffset(b, hv, globalRow), scalar, 1);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessTask(uint64_t task, uint64_t nc)
    {
        uint64_t b = 0, h = 0, hv = 0, start = 0, end = 0, rowBlock = 0;
        if (!ResolveTask(task, nc, b, h, hv, start, end, rowBlock)) {
            return;
        }
        const uint64_t curT = end - start;
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        SyncVToMte2();
        CopyFp32In(betaLocal, beta_, BetaOffset(b, hv, start), curT);
        SyncMte2ToS();
        __ubuf__ float *betaPtr = reinterpret_cast<__ubuf__ float *>(betaLocal.GetPhyAddr());
        uint64_t rowBegin = rowBlock * BC;
        uint64_t rowEnd = rowBegin + BC;
        if (rowEnd > curT) {
            rowEnd = curT;
        }
        LoadG(VOffset(b, hv, start + rowBegin, 0), gLeftRefBuf_.Get<float>(),
              static_cast<uint32_t>(kDim_));
        LoadG(VOffset(b, hv, start + rowEnd - 1, 0), gRightRefBuf_.Get<float>(),
              static_cast<uint32_t>(kDim_));
        if constexpr (SAFE_GATE) {
            const uint64_t rowMid = rowBegin +
                ((BC / 2 < rowEnd - rowBegin - 1) ? BC / 2 : rowEnd - rowBegin - 1);
            LoadG(VOffset(b, hv, start + rowMid, 0), gDiagRefBuf_.Get<float>(),
                  static_cast<uint32_t>(kDim_));
        }
        LoadDABlock(b, hv, start, curT, rowBegin, rowEnd - rowBegin);
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            ProcessRow(b, h, hv, start, curT, row, betaPtr);
        }
    }

private:
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
    GlobalTensor<int64_t> chunkMeta_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> dARowQBuf_, dARowKBuf_, dAColQBuf_, dAColKBuf_, betaBuf_;
    TBuf<TPosition::VECCALC> qTypedBuf_, kTypedBuf_, qSrcBuf_, kSrcBuf_, gSrcBuf_;
    TBuf<TPosition::VECCALC> qSelfBuf_, kSelfBuf_, gSelfBuf_, gLeftRefBuf_, gDiagRefBuf_, gRightRefBuf_;
    TBuf<TPosition::VECCALC> gateBuf_, tmp0Buf_, tmp1Buf_;
    TBuf<TPosition::VECCALC> dqAccBuf_, dkLeftBuf_, dkRightBuf_, out0Buf_, out1Buf_, out2Buf_;
    TBuf<TPosition::VECCALC> reduceBuf_, scalarBuf_, chunkMetaBuf_;
    uint64_t b_ = 0, h_ = 0, hv_ = 0, t_ = 0, kDim_ = 0, bt_ = 0, nt_ = 0;
    uint64_t usedCoreNum_ = 1;
    bool isVarLen_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_bwd_intra(
    GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR db, GM_ADDR dg, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dqOut, GM_ADDR dkOut,
    GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cuSeqlens;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        ChunkKdaBwdIntraKernel<half, false> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ChunkKdaBwdIntraKernel<half, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ChunkKdaBwdIntraKernel<bfloat16_t, false> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ChunkKdaBwdIntraKernel<bfloat16_t, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    }
}
