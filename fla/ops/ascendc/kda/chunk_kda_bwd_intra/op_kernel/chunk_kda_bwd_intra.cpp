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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;

namespace {
constexpr uint32_t BC = 16;
constexpr uint32_t BK = 32;
constexpr uint32_t MAX_BT = 128;
constexpr uint32_t MAX_K = 256;
constexpr float LN2 = 0.69314718055994530942f;

template <typename T, bool SAFE_GATE, bool BLOCKWISE = false>
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
        if constexpr (BLOCKWISE) {
            // The optimized safe-gate path keeps one complete [curT, BK]
            // source feature tile in UB.  q/k stay typed only for the GM
            // transfer and are cast once into the FP32 resident cache.
            pipe_->InitBuffer(dARowQTransBuf_, MAX_BT * BC * sizeof(float));
            pipe_->InitBuffer(dARowKTransBuf_, MAX_BT * BC * sizeof(float));
            pipe_->InitBuffer(qTypedBuf_, MAX_BT * BK * sizeof(T));
            pipe_->InitBuffer(kTypedBuf_, MAX_BT * BK * sizeof(T));
            pipe_->InitBuffer(qSrcBuf_, MAX_BT * BK * sizeof(float));
            pipe_->InitBuffer(kSrcBuf_, MAX_BT * BK * sizeof(float));
            pipe_->InitBuffer(gSrcBuf_, MAX_BT * BK * sizeof(float));
            pipe_->InitBuffer(gateBuf_, BK * sizeof(float));
            pipe_->InitBuffer(tmp0Buf_, BK * sizeof(float));
            pipe_->InitBuffer(tmp1Buf_, BC * 8 * sizeof(float));
            pipe_->InitBuffer(dqAccBuf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(dkLeftBuf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(dkRightBuf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(out0Buf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(out1Buf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(out2Buf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(blockTmpBuf_, BC * BK * sizeof(float));
            pipe_->InitBuffer(dbCompactBuf_, BC * sizeof(float));
        } else {
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
        }
        pipe_->InitBuffer(reduceBuf_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(dbAccBuf_, BLOCKWISE ? BC * 8 * sizeof(float) : 32);
        pipe_->InitBuffer(chunkMetaBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint64_t nc = bt_ / BC;
        const uint64_t flatChunks = isVarLen_ ? nt_ : b_ * nt_;
        const uint64_t taskCount = flatChunks * hv_ * nc;
        for (uint64_t task = static_cast<uint64_t>(GetBlockIdx()); task < taskCount; task += usedCoreNum_) {
            if constexpr (BLOCKWISE) {
                ProcessTaskBlockwise(task, nc);
            } else {
                ProcessTask(task, nc);
            }
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
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void SyncSToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void SyncVToS()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventId);
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

    __aicore__ inline void AccumulateDb(LocalTensor<float> dbAcc, LocalTensor<float> dkLeft,
                                        LocalTensor<float> kSelf, uint32_t count)
    {
        LocalTensor<float> product = tmp0Buf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceBuf_.Get<float>();
        Mul(product, dkLeft, kSelf, count);
        PipeBarrier<PIPE_V>();
        ReduceSum<float, true>(scalar, product, reduceTmp, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(dbAcc, dbAcc, scalar, 1);
        PipeBarrier<PIPE_V>();
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
        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        Duplicate(dbAcc, 0.0f, 1);
        PipeBarrier<PIPE_V>();

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

            AccumulateDb(dbAcc, dkLeft, kSelf, curK);
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
            // The following Add reads the value written by the previous Add.
            // Auto-sync is disabled, so make the in-place RAW dependency
            // explicit; otherwise the dkLeft contribution can be lost.
            PipeBarrier<PIPE_V>();
            Add(outK, outK, dkRight, curK);
            LocalTensor<float> tmp0 = tmp0Buf_.Get<float>();
            LocalTensor<float> tmp1 = tmp1Buf_.Get<float>();
            Mul(tmp0, qSelf, dqAcc, curK);
            Sub(tmp1, dkLeft, dkRight, curK);
            PipeBarrier<PIPE_V>();
            Mul(tmp1, tmp1, kSelf, curK);
            PipeBarrier<PIPE_V>();
            Add(outG, outG, tmp0, curK);
            // Preserve qSelf * dqAcc before accumulating the second dg term.
            PipeBarrier<PIPE_V>();
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
        Add(scalar, scalar, dbAcc, 1);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        CopyFp32Out(dbOut_, BetaOffset(b, hv, globalRow), scalar, 1);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void LoadDABlockBlockwise(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                                uint64_t curT, uint64_t rowBegin, uint64_t rowCount,
                                                __ubuf__ float *betaPtr)
    {
        LocalTensor<float> rowQ = dARowQBuf_.Get<float>();
        LocalTensor<float> rowK = dARowKBuf_.Get<float>();
        LocalTensor<float> colQ = dAColQBuf_.Get<float>();
        LocalTensor<float> colK = dAColKBuf_.Get<float>();
        SyncVToMte2();
        DataCopyExtParams rowParams{static_cast<uint16_t>(rowCount),
                                    static_cast<uint32_t>(bt_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(rowQ, dAqk_[AOffset(b, hv, chunkStart + rowBegin, 0)], rowParams, noPad);
        DataCopyPad(rowK, dAkk_[AOffset(b, hv, chunkStart + rowBegin, 0)], rowParams, noPad);
        DataCopyExtParams colParams{static_cast<uint16_t>(curT), BC * sizeof(float),
                                    static_cast<uint32_t>((bt_ - BC) * sizeof(float)), 0, 0};
        DataCopyPad(colQ, dAqk_[AOffset(b, hv, chunkStart, rowBegin)], colParams, noPad);
        DataCopyPad(colK, dAkk_[AOffset(b, hv, chunkStart, rowBegin)], colParams, noPad);
        SyncMte2ToS();

        // Prepare all four coefficient streams as [source, row].  The row
        // streams arrive as [row, source], while the column streams are
        // already packed.  Causal invalid entries and tail rows are zeroed
        // here once, so the hot source loop contains no scalar reads or
        // Scalar-to-Vector event per row.
        __ubuf__ float *rowQPtr = reinterpret_cast<__ubuf__ float *>(rowQ.GetPhyAddr());
        __ubuf__ float *rowKPtr = reinterpret_cast<__ubuf__ float *>(rowK.GetPhyAddr());
        __ubuf__ float *rowQTransPtr = reinterpret_cast<__ubuf__ float *>(
            dARowQTransBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float *rowKTransPtr = reinterpret_cast<__ubuf__ float *>(
            dARowKTransBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float *colQPtr = reinterpret_cast<__ubuf__ float *>(colQ.GetPhyAddr());
        __ubuf__ float *colKPtr = reinterpret_cast<__ubuf__ float *>(colK.GetPhyAddr());
        for (uint64_t source = 0; source < curT; ++source) {
            const float betaValue = betaPtr[source];
            for (uint64_t row = 0; row < BC; ++row) {
                const uint64_t packedOffset = source * BC + row;
                if (row < rowCount) {
                    const uint64_t localRow = rowBegin + row;
                    const uint64_t rawOffset = row * bt_ + source;
                    if (source <= localRow) {
                        rowQTransPtr[packedOffset] = rowQPtr[rawOffset];
                        rowKTransPtr[packedOffset] = rowKPtr[rawOffset];
                    } else {
                        rowQTransPtr[packedOffset] = 0.0f;
                        rowKTransPtr[packedOffset] = 0.0f;
                    }
                    if (source >= localRow) {
                        colKPtr[packedOffset] = colKPtr[packedOffset] * betaValue;
                    } else {
                        colQPtr[packedOffset] = 0.0f;
                        colKPtr[packedOffset] = 0.0f;
                    }
                } else {
                    rowQTransPtr[packedOffset] = 0.0f;
                    rowKTransPtr[packedOffset] = 0.0f;
                    colQPtr[packedOffset] = 0.0f;
                    colKPtr[packedOffset] = 0.0f;
                }
            }
        }
        SyncSToV();
    }

    __aicore__ inline void LoadSourceFeatureTile(uint64_t b, uint64_t h, uint64_t hv,
                                                 uint64_t chunkStart, uint64_t curT,
                                                 uint64_t d, uint32_t curK)
    {
        LocalTensor<T> qTyped = qTypedBuf_.Get<T>();
        LocalTensor<T> kTyped = kTypedBuf_.Get<T>();
        LocalTensor<float> qCache = qSrcBuf_.Get<float>();
        LocalTensor<float> kCache = kSrcBuf_.Get<float>();
        LocalTensor<float> gCache = gSrcBuf_.Get<float>();
        // DataCopyPad padding can only complete the current 32-byte data
        // block.  It must not be used to create the fixed BK row pitch (for
        // example, K=16 would request 16 FP32 padding elements).  Keep the
        // source rows compact and express the UB gap with dstStride instead.
        const uint32_t qkDstStride =
            static_cast<uint32_t>((BK - curK) * sizeof(T) / 32);
        const uint32_t gDstStride =
            static_cast<uint32_t>((BK - curK) * sizeof(float) / 32);
        DataCopyExtParams qkParams{static_cast<uint16_t>(curT),
                                   static_cast<uint32_t>(curK * sizeof(T)),
                                   static_cast<uint32_t>((kDim_ - curK) * sizeof(T)),
                                   qkDstStride, 0};
        DataCopyExtParams gParams{static_cast<uint16_t>(curT),
                                  static_cast<uint32_t>(curK * sizeof(float)),
                                  static_cast<uint32_t>((kDim_ - curK) * sizeof(float)),
                                  gDstStride, 0};
        DataCopyPadExtParams<T> typedPad{false, 0, 0, 0};
        DataCopyPadExtParams<float> floatPad{false, 0, 0, 0.0f};
        SyncVToMte2();
        DataCopyPad(qTyped, q_[QOffset(b, h, chunkStart, d)], qkParams, typedPad);
        DataCopyPad(kTyped, k_[QOffset(b, h, chunkStart, d)], qkParams, typedPad);
        DataCopyPad(gCache, g_[VOffset(b, hv, chunkStart, d)], gParams, floatPad);
        SyncMte2ToV();
        const uint32_t cacheCount = static_cast<uint32_t>(curT * BK);
        Cast(qCache, qTyped, RoundMode::CAST_NONE, cacheCount);
        Cast(kCache, kTyped, RoundMode::CAST_NONE, cacheCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void OuterAccumulate(LocalTensor<float> acc, LocalTensor<float> common,
                                            LocalTensor<float> coefficients, uint32_t curK)
    {
        LocalTensor<float> coefficientBrcb = tmp1Buf_.Get<float>();
        Brcb(coefficientBrcb, coefficients, static_cast<uint8_t>(BC / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        // Each source contributes one rank-1 update to all BC output rows.
        // Keep the broadcast coefficient layout, but fuse product formation
        // and FP32 accumulation so the hot source loop issues one vector
        // instruction instead of a Mul followed by an Add.
        BinaryRepeatParams maddParams{1, 1, 0,
                                     static_cast<uint8_t>(BK * sizeof(float) / 32), 0, 1};
        MulAddDst(acc, common, coefficientBrcb, curK, static_cast<uint8_t>(BC), maddParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateLeftSource(LocalTensor<float> dqAcc, LocalTensor<float> dkAcc,
                                                LocalTensor<float> kSource, LocalTensor<float> gate,
                                                LocalTensor<float> rowQCoefficients,
                                                LocalTensor<float> rowKCoefficients, uint32_t curK)
    {
        LocalTensor<float> common = tmp0Buf_.Get<float>();
        Mul(common, kSource, gate, curK);
        PipeBarrier<PIPE_V>();
        OuterAccumulate(dqAcc, common, rowQCoefficients, curK);
        OuterAccumulate(dkAcc, common, rowKCoefficients, curK);
    }

    __aicore__ inline void AccumulateRightSource(LocalTensor<float> dkAcc, LocalTensor<float> qSource,
                                                 LocalTensor<float> kSource, LocalTensor<float> gate,
                                                 LocalTensor<float> colQCoefficients,
                                                 LocalTensor<float> colKCoefficients, uint32_t curK)
    {
        LocalTensor<float> common = tmp0Buf_.Get<float>();
        Mul(common, qSource, gate, curK);
        PipeBarrier<PIPE_V>();
        OuterAccumulate(dkAcc, common, colQCoefficients, curK);
        Mul(common, kSource, gate, curK);
        PipeBarrier<PIPE_V>();
        OuterAccumulate(dkAcc, common, colKCoefficients, curK);
    }

    __aicore__ inline void LoadOutputFeatureBlock(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                                  uint64_t rowBegin, uint64_t rowCount,
                                                  uint64_t d, uint32_t curK)
    {
        LocalTensor<float> outQ = out0Buf_.Get<float>();
        LocalTensor<float> outK = out1Buf_.Get<float>();
        LocalTensor<float> outG = out2Buf_.Get<float>();
        const uint32_t dstStride =
            static_cast<uint32_t>((BK - curK) * sizeof(float) / 32);
        DataCopyExtParams params{static_cast<uint16_t>(rowCount),
                                 static_cast<uint32_t>(curK * sizeof(float)),
                                 static_cast<uint32_t>((kDim_ - curK) * sizeof(float)),
                                 dstStride, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        SyncVToMte2();
        DataCopyPad(outQ, dq_[VOffset(b, hv, chunkStart + rowBegin, d)], params, pad);
        DataCopyPad(outK, dk_[VOffset(b, hv, chunkStart + rowBegin, d)], params, pad);
        DataCopyPad(outG, dg_[VOffset(b, hv, chunkStart + rowBegin, d)], params, pad);
        SyncMte2ToV();
    }

    __aicore__ inline void StoreOutputFeatureBlock(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                                   uint64_t rowBegin, uint64_t rowCount,
                                                   uint64_t d, uint32_t curK)
    {
        LocalTensor<float> outQ = out0Buf_.Get<float>();
        LocalTensor<float> outK = out1Buf_.Get<float>();
        LocalTensor<float> outG = out2Buf_.Get<float>();
        DataCopyExtParams params{static_cast<uint16_t>(rowCount),
                                 static_cast<uint32_t>(curK * sizeof(float)),
                                 static_cast<uint32_t>((BK - curK) * sizeof(float) / 32),
                                 static_cast<uint32_t>((kDim_ - curK) * sizeof(float)), 0};
        SyncVToMte3();
        DataCopyPad(dqOut_[VOffset(b, hv, chunkStart + rowBegin, d)], outQ, params);
        DataCopyPad(dkOut_[VOffset(b, hv, chunkStart + rowBegin, d)], outK, params);
        DataCopyPad(dgOut_[VOffset(b, hv, chunkStart + rowBegin, d)], outG, params);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessSafeFeatureBlock(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                                   uint64_t curT, uint64_t rowBegin, uint64_t rowCount,
                                                   uint64_t d, uint32_t curK)
    {
        LocalTensor<float> qCache = qSrcBuf_.Get<float>();
        LocalTensor<float> kCache = kSrcBuf_.Get<float>();
        LocalTensor<float> gCache = gSrcBuf_.Get<float>();
        LocalTensor<float> rowQ = dARowQTransBuf_.Get<float>();
        LocalTensor<float> rowK = dARowKTransBuf_.Get<float>();
        LocalTensor<float> colQ = dAColQBuf_.Get<float>();
        LocalTensor<float> colK = dAColKBuf_.Get<float>();
        LocalTensor<float> dqAcc = dqAccBuf_.Get<float>();
        LocalTensor<float> dkLeft = dkLeftBuf_.Get<float>();
        LocalTensor<float> dkRight = dkRightBuf_.Get<float>();
        LocalTensor<float> dqDiag = out0Buf_.Get<float>();
        LocalTensor<float> dkLeftDiag = out1Buf_.Get<float>();
        LocalTensor<float> dkRightFuture = out2Buf_.Get<float>();
        LocalTensor<float> gate = gateBuf_.Get<float>();
        const uint32_t blockElements = BC * BK;
        Duplicate(dqAcc, 0.0f, blockElements);
        Duplicate(dkLeft, 0.0f, blockElements);
        Duplicate(dkRight, 0.0f, blockElements);
        Duplicate(dqDiag, 0.0f, blockElements);
        Duplicate(dkLeftDiag, 0.0f, blockElements);
        Duplicate(dkRightFuture, 0.0f, blockElements);
        PipeBarrier<PIPE_V>();

        const uint64_t rowEnd = rowBegin + rowCount;
        LocalTensor<float> gLeftRef = gCache[rowBegin * BK];
        LocalTensor<float> gRightRef = gCache[(rowEnd - 1) * BK];

        for (uint64_t source = 0; source < rowBegin; ++source) {
            LocalTensor<float> kSource = kCache[source * BK];
            BuildGate(gate, gLeftRef, gCache[source * BK], curK);
            AccumulateLeftSource(dqAcc, dkLeft, kSource, gate,
                                 rowQ[source * BC], rowK[source * BC], curK);
        }
        for (uint64_t source = rowBegin; source < rowEnd; ++source) {
            LocalTensor<float> qSource = qCache[source * BK];
            LocalTensor<float> kSource = kCache[source * BK];
            LocalTensor<float> gSource = gCache[source * BK];
            // Directional endpoint references keep the feature-side gate <= 1.
            // A midpoint reference can first form Inf from a large, but legal,
            // BF16 q/k value and then feed it to a zero dA coefficient, yielding
            // 0 * Inf = NaN even though the final pairwise result is finite.
            BuildGate(gate, gRightRef, gSource, curK);
            AccumulateLeftSource(dqDiag, dkLeftDiag, kSource, gate,
                                 rowQ[source * BC], rowK[source * BC], curK);
            BuildGate(gate, gSource, gLeftRef, curK);
            AccumulateRightSource(dkRight, qSource, kSource, gate,
                                  colQ[source * BC], colK[source * BC], curK);
        }
        for (uint64_t source = rowEnd; source < curT; ++source) {
            LocalTensor<float> qSource = qCache[source * BK];
            LocalTensor<float> kSource = kCache[source * BK];
            LocalTensor<float> gSource = gCache[source * BK];
            BuildGate(gate, gSource, gRightRef, curK);
            AccumulateRightSource(dkRightFuture, qSource, kSource, gate,
                                  colQ[source * BC], colK[source * BC], curK);
        }

        for (uint64_t row = 0; row < rowCount; ++row) {
            const uint64_t rowOffset = row * BK;
            LocalTensor<float> gSelf = gCache[(rowBegin + row) * BK];
            if (rowBegin > 0) {
                BuildGate(gate, gSelf, gLeftRef, curK);
                Mul(dqAcc[rowOffset], dqAcc[rowOffset], gate, curK);
                Mul(dkLeft[rowOffset], dkLeft[rowOffset], gate, curK);
                PipeBarrier<PIPE_V>();
            }
            BuildGate(gate, gSelf, gRightRef, curK);
            Mul(dqDiag[rowOffset], dqDiag[rowOffset], gate, curK);
            Mul(dkLeftDiag[rowOffset], dkLeftDiag[rowOffset], gate, curK);
            PipeBarrier<PIPE_V>();
            Add(dqAcc[rowOffset], dqAcc[rowOffset], dqDiag[rowOffset], curK);
            Add(dkLeft[rowOffset], dkLeft[rowOffset], dkLeftDiag[rowOffset], curK);
            PipeBarrier<PIPE_V>();
            BuildGate(gate, gLeftRef, gSelf, curK);
            Mul(dkRight[rowOffset], dkRight[rowOffset], gate, curK);
            PipeBarrier<PIPE_V>();
            if (rowEnd < curT) {
                BuildGate(gate, gRightRef, gSelf, curK);
                Mul(dkRightFuture[rowOffset], dkRightFuture[rowOffset], gate, curK);
                PipeBarrier<PIPE_V>();
                Add(dkRight[rowOffset], dkRight[rowOffset], dkRightFuture[rowOffset], curK);
                PipeBarrier<PIPE_V>();
            }
        }

        // db uses dkLeft before beta scaling, matching the legacy and Triton
        // order.  Each row reduction remains independent and FP32.
        LocalTensor<float> blockTmp = blockTmpBuf_.Get<float>();
        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        const uint32_t validBlockElements = static_cast<uint32_t>(rowCount * BK);
        Mul(blockTmp, dkLeft, kCache[rowBegin * BK], validBlockElements);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceBuf_.Get<float>();
        for (uint64_t row = 0; row < rowCount; ++row) {
            ReduceSum<float, true>(scalar, blockTmp[row * BK], reduceTmp, static_cast<int32_t>(curK));
            PipeBarrier<PIPE_V>();
            Add(dbAcc[row * 8], dbAcc[row * 8], scalar, 1);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> betaBrcb = tmp1Buf_.Get<float>();
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        Brcb(betaBrcb, betaLocal[rowBegin], static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams betaParams{1, 1, 0,
                                     static_cast<uint8_t>(BK * sizeof(float) / 32),
                                     static_cast<uint8_t>(BK * sizeof(float) / 32), 1};
        Mul(dkLeft, dkLeft, betaBrcb, curK, static_cast<uint8_t>(rowCount), betaParams);
        PipeBarrier<PIPE_V>();

        LoadOutputFeatureBlock(b, hv, chunkStart, rowBegin, rowCount, d, curK);
        LocalTensor<float> outQ = out0Buf_.Get<float>();
        LocalTensor<float> outK = out1Buf_.Get<float>();
        LocalTensor<float> outG = out2Buf_.Get<float>();
        Add(outQ, outQ, dqAcc, validBlockElements);
        Add(outK, outK, dkLeft, validBlockElements);
        PipeBarrier<PIPE_V>();
        Add(outK, outK, dkRight, validBlockElements);
        MulAddDst(outG, qCache[rowBegin * BK], dqAcc, validBlockElements);
        Sub(dkRight, dkLeft, dkRight, validBlockElements);
        PipeBarrier<PIPE_V>();
        MulAddDst(outG, dkRight, kCache[rowBegin * BK], validBlockElements);
        PipeBarrier<PIPE_V>();
        StoreOutputFeatureBlock(b, hv, chunkStart, rowBegin, rowCount, d, curK);
    }

    __aicore__ inline void StoreDbBlock(uint64_t b, uint64_t hv, uint64_t chunkStart,
                                        uint64_t rowBegin, uint64_t rowCount)
    {
        LocalTensor<float> outDb = out0Buf_.Get<float>();
        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        LocalTensor<float> dbCompact = dbCompactBuf_.Get<float>();
        SyncVToS();
        __ubuf__ float *dbAccPtr = reinterpret_cast<__ubuf__ float *>(dbAcc.GetPhyAddr());
        __ubuf__ float *dbCompactPtr = reinterpret_cast<__ubuf__ float *>(dbCompact.GetPhyAddr());
        for (uint64_t row = 0; row < rowCount; ++row) {
            dbCompactPtr[row] = dbAccPtr[row * 8];
        }
        SyncSToV();
        SyncVToMte2();
        CopyFp32In(outDb, db_, BetaOffset(b, hv, chunkStart + rowBegin), rowCount);
        SyncMte2ToV();
        Add(outDb, outDb, dbCompact, static_cast<uint32_t>(rowCount));
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        CopyFp32Out(dbOut_, BetaOffset(b, hv, chunkStart + rowBegin), outDb, rowCount);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessTaskBlockwise(uint64_t task, uint64_t nc)
    {
        static_assert(SAFE_GATE, "block-wise ChunkKdaBwdIntra is enabled only for safe_gate");
        uint64_t b = 0, h = 0, hv = 0, start = 0, end = 0, rowBlock = 0;
        if (!ResolveTask(task, nc, b, h, hv, start, end, rowBlock)) {
            return;
        }
        const uint64_t curT = end - start;
        const uint64_t rowBegin = rowBlock * BC;
        const uint64_t rowEnd = (rowBegin + BC < curT) ? rowBegin + BC : curT;
        const uint64_t rowCount = rowEnd - rowBegin;
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        SyncVToMte2();
        CopyFp32In(betaLocal, beta_, BetaOffset(b, hv, start), curT);
        SyncMte2ToS();
        __ubuf__ float *betaPtr = reinterpret_cast<__ubuf__ float *>(betaLocal.GetPhyAddr());
        LoadDABlockBlockwise(b, hv, start, curT, rowBegin, rowCount, betaPtr);

        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        Duplicate(dbAcc, 0.0f, BC * 8);
        PipeBarrier<PIPE_V>();
        for (uint64_t d = 0; d < kDim_; d += BK) {
            const uint32_t curK = static_cast<uint32_t>((kDim_ - d < BK) ? kDim_ - d : BK);
            LoadSourceFeatureTile(b, h, hv, start, curT, d, curK);
            ProcessSafeFeatureBlock(b, hv, start, curT, rowBegin, rowCount, d, curK);
        }
        StoreDbBlock(b, hv, start, rowBegin, rowCount);
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
    TBuf<TPosition::VECCALC> dARowQTransBuf_, dARowKTransBuf_;
    TBuf<TPosition::VECCALC> qTypedBuf_, kTypedBuf_, qSrcBuf_, kSrcBuf_, gSrcBuf_;
    TBuf<TPosition::VECCALC> qSelfBuf_, kSelfBuf_, gSelfBuf_, gLeftRefBuf_, gDiagRefBuf_, gRightRefBuf_;
    TBuf<TPosition::VECCALC> gateBuf_, tmp0Buf_, tmp1Buf_;
    TBuf<TPosition::VECCALC> dqAccBuf_, dkLeftBuf_, dkRightBuf_, out0Buf_, out1Buf_, out2Buf_;
    TBuf<TPosition::VECCALC> blockTmpBuf_, reduceBuf_, scalarBuf_, dbAccBuf_, dbCompactBuf_, chunkMetaBuf_;
    uint64_t b_ = 0, h_ = 0, hv_ = 0, t_ = 0, kDim_ = 0, bt_ = 0, nt_ = 0;
    uint64_t usedCoreNum_ = 1;
    bool isVarLen_ = false;
};

#include "chunk_kda_bwd_intra_mixed.hpp"
#include "chunk_kda_bwd_intra_grouped.hpp"
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_bwd_intra(
    GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR db, GM_ADDR dg, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dqOut, GM_ADDR dkOut,
    GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cuSeqlens;
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<half, false> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<half, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<bfloat16_t, false> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<bfloat16_t, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<half, true, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaBwdIntraKernel<bfloat16_t, true, true> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut, chunkIndices,
                tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(15)) {
        KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_2);
        ChunkKdaBwdIntraMixedKernel<bfloat16_t> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut,
                userWS, tilingData, &pipe);
        if ASCEND_IS_AIC {
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            op.ProcessAiv();
        }
    } else if (TILING_KEY_IS(23)) {
        KERNEL_TASK_TYPE(23, KERNEL_TYPE_MIX_AIC_1_2);
        ChunkKdaBwdIntraGroupedKernel<bfloat16_t> op;
        op.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dqOut, dkOut, dbOut, dgOut,
                userWS, tilingData, &pipe);
        if ASCEND_IS_AIC {
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            op.ProcessAiv();
        }
    }
}
