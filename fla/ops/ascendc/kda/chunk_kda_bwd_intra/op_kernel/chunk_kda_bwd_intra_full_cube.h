/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_KDA_BWD_INTRA_FULL_CUBE_H
#define CHUNK_KDA_BWD_INTRA_FULL_CUBE_H

namespace KdaFullCube {

constexpr uint32_t BLOCK = 16;
constexpr uint32_t CHUNK = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t FEATURE_TILE = 64;
constexpr uint32_t FEATURE_STRIDE = FEATURE_TILE * sizeof(float) / 32;
constexpr uint32_t TILE_ELEMENTS = BLOCK * FEATURE_TILE;
constexpr float LOG2_E_SCALE = 0.69314718055994530942f;

// Six block-diagonal GEMMs cover every source contraction in a full 64-token
// chunk.  Rows belonging to different 16-token target blocks use disjoint K
// ranges.  This preserves the safe-gate reference point of each block while
// increasing the amount of work in each direct BlockMmad call.
constexpr uint32_t A_LEFT_PREV_M = 96;
constexpr uint32_t A_LEFT_PREV_K = 96;
constexpr uint32_t A_LEFT_DIAG_M = 128;
constexpr uint32_t A_LEFT_DIAG_K = 64;
constexpr uint32_t A_RIGHT_FUTURE_M = 48;
constexpr uint32_t A_RIGHT_FUTURE_K = 96;
constexpr uint32_t A_RIGHT_DIAG_M = 64;
constexpr uint32_t A_RIGHT_DIAG_K = 64;

constexpr uint32_t A_LEFT_PREV_ELEMENTS = A_LEFT_PREV_M * A_LEFT_PREV_K;
constexpr uint32_t A_LEFT_DIAG_ELEMENTS = A_LEFT_DIAG_M * A_LEFT_DIAG_K;
constexpr uint32_t A_RIGHT_FUTURE_ELEMENTS = A_RIGHT_FUTURE_M * A_RIGHT_FUTURE_K;
constexpr uint32_t A_RIGHT_DIAG_ELEMENTS = A_RIGHT_DIAG_M * A_RIGHT_DIAG_K;

constexpr uint32_t B_LEFT_PREV_ELEMENTS = A_LEFT_PREV_K * HEAD_DIM;
constexpr uint32_t B_LEFT_DIAG_ELEMENTS = A_LEFT_DIAG_K * HEAD_DIM;
constexpr uint32_t B_RIGHT_FUTURE_ELEMENTS = A_RIGHT_FUTURE_K * HEAD_DIM;
constexpr uint32_t B_RIGHT_DIAG_ELEMENTS = A_RIGHT_DIAG_K * HEAD_DIM;

constexpr uint32_t C_LEFT_PREV_ELEMENTS = A_LEFT_PREV_M * HEAD_DIM;
constexpr uint32_t C_LEFT_DIAG_ELEMENTS = A_LEFT_DIAG_M * HEAD_DIM;
constexpr uint32_t C_RIGHT_FUTURE_ELEMENTS = A_RIGHT_FUTURE_M * HEAD_DIM;
constexpr uint32_t C_RIGHT_DIAG_ELEMENTS = A_RIGHT_DIAG_M * HEAD_DIM;

constexpr uint32_t A_LEFT_PREV_OFFSET = 0;
constexpr uint32_t A_LEFT_DIAG_OFFSET = A_LEFT_PREV_OFFSET + A_LEFT_PREV_ELEMENTS;
constexpr uint32_t A_RIGHT_FUTURE_Q_OFFSET = A_LEFT_DIAG_OFFSET + A_LEFT_DIAG_ELEMENTS;
constexpr uint32_t A_RIGHT_FUTURE_K_OFFSET =
    A_RIGHT_FUTURE_Q_OFFSET + A_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t A_RIGHT_DIAG_Q_OFFSET =
    A_RIGHT_FUTURE_K_OFFSET + A_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t A_RIGHT_DIAG_K_OFFSET =
    A_RIGHT_DIAG_Q_OFFSET + A_RIGHT_DIAG_ELEMENTS;
constexpr uint32_t A_END_OFFSET = A_RIGHT_DIAG_K_OFFSET + A_RIGHT_DIAG_ELEMENTS;

constexpr uint32_t B_LEFT_PREV_OFFSET = A_END_OFFSET;
constexpr uint32_t B_LEFT_DIAG_OFFSET = B_LEFT_PREV_OFFSET + B_LEFT_PREV_ELEMENTS;
constexpr uint32_t B_RIGHT_FUTURE_Q_OFFSET =
    B_LEFT_DIAG_OFFSET + B_LEFT_DIAG_ELEMENTS;
constexpr uint32_t B_RIGHT_FUTURE_K_OFFSET =
    B_RIGHT_FUTURE_Q_OFFSET + B_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t B_RIGHT_DIAG_Q_OFFSET =
    B_RIGHT_FUTURE_K_OFFSET + B_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t B_RIGHT_DIAG_K_OFFSET =
    B_RIGHT_DIAG_Q_OFFSET + B_RIGHT_DIAG_ELEMENTS;
constexpr uint32_t B_END_OFFSET = B_RIGHT_DIAG_K_OFFSET + B_RIGHT_DIAG_ELEMENTS;

constexpr uint32_t C_LEFT_PREV_OFFSET = B_END_OFFSET;
constexpr uint32_t C_LEFT_DIAG_OFFSET = C_LEFT_PREV_OFFSET + C_LEFT_PREV_ELEMENTS;
constexpr uint32_t C_RIGHT_FUTURE_Q_OFFSET =
    C_LEFT_DIAG_OFFSET + C_LEFT_DIAG_ELEMENTS;
constexpr uint32_t C_RIGHT_FUTURE_K_OFFSET =
    C_RIGHT_FUTURE_Q_OFFSET + C_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t C_RIGHT_DIAG_Q_OFFSET =
    C_RIGHT_FUTURE_K_OFFSET + C_RIGHT_FUTURE_ELEMENTS;
constexpr uint32_t C_RIGHT_DIAG_K_OFFSET =
    C_RIGHT_DIAG_Q_OFFSET + C_RIGHT_DIAG_ELEMENTS;
constexpr uint32_t SLOT_ELEMENTS = C_RIGHT_DIAG_K_OFFSET + C_RIGHT_DIAG_ELEMENTS;
constexpr uint64_t SLOT_BYTES = static_cast<uint64_t>(SLOT_ELEMENTS) * sizeof(float);

constexpr bool IsWorkspaceOffsetAligned(uint32_t offset)
{
    return (static_cast<uint64_t>(offset) * sizeof(float)) % 512 == 0;
}

static_assert(IsWorkspaceOffsetAligned(A_LEFT_PREV_OFFSET) &&
              IsWorkspaceOffsetAligned(A_LEFT_DIAG_OFFSET) &&
              IsWorkspaceOffsetAligned(A_RIGHT_FUTURE_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(A_RIGHT_FUTURE_K_OFFSET) &&
              IsWorkspaceOffsetAligned(A_RIGHT_DIAG_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(A_RIGHT_DIAG_K_OFFSET) &&
              IsWorkspaceOffsetAligned(B_LEFT_PREV_OFFSET) &&
              IsWorkspaceOffsetAligned(B_LEFT_DIAG_OFFSET) &&
              IsWorkspaceOffsetAligned(B_RIGHT_FUTURE_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(B_RIGHT_FUTURE_K_OFFSET) &&
              IsWorkspaceOffsetAligned(B_RIGHT_DIAG_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(B_RIGHT_DIAG_K_OFFSET) &&
              IsWorkspaceOffsetAligned(C_LEFT_PREV_OFFSET) &&
              IsWorkspaceOffsetAligned(C_LEFT_DIAG_OFFSET) &&
              IsWorkspaceOffsetAligned(C_RIGHT_FUTURE_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(C_RIGHT_FUTURE_K_OFFSET) &&
              IsWorkspaceOffsetAligned(C_RIGHT_DIAG_Q_OFFSET) &&
              IsWorkspaceOffsetAligned(C_RIGHT_DIAG_K_OFFSET),
              "KDA full-Cube matrices must remain 512B aligned");
static_assert((SLOT_BYTES % 512) == 0, "KDA full-Cube per-core slot must be 512B aligned");
static_assert(SLOT_BYTES == 614400, "KDA full-Cube workspace layout changed unexpectedly");

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using ArchTag = Catlass::Arch::Ascend950;
#else
using ArchTag = Catlass::Arch::AtlasA2;
#endif
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, false, false>;
static_assert(!DispatchPolicy::USE_HF32_MODE,
              "ChunkKdaBwdIntra full Cube must use IEEE FP32 mode");
// This CATLASS MmadPingpong implementation requires identical L1/L0 M/N
// basic blocks.  Keep the complete tile shape identical to the already-proven
// FP32 rowBlock3 Cube configuration so later tuning cannot silently violate
// that compile-time contract.
using L1TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
using L0TileShape = L1TileShape;

enum FeatureKind : uint32_t {
    FEATURE_K = 0,
    FEATURE_Q = 1,
    FEATURE_BETA_K = 2,
};

class AivKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
        GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR workspace,
        const ChunkKdaBwdIntraTilingData &tiling, TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
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
        h_ = static_cast<uint64_t>(tiling.qHeadNum);
        hv_ = static_cast<uint64_t>(tiling.vHeadNum);
        t_ = static_cast<uint64_t>(tiling.seqlen);
        nt_ = static_cast<uint64_t>(tiling.totalChunks);
        pipe_ = pipe;

        // The prep and consume phases never overlap on one AIV lane.  Reuse
        // these arenas across phases to keep the UB budget below 192 KiB.
        pipe_->InitBuffer(dAQBuf_, CHUNK * CHUNK * sizeof(float));
        pipe_->InitBuffer(dAKBuf_, CHUNK * CHUNK * sizeof(float));
        pipe_->InitBuffer(matrixBuf_, 32 * A_LEFT_PREV_K * sizeof(float));
        pipe_->InitBuffer(typedBuf_, 2 * 48 * FEATURE_TILE * sizeof(bfloat16_t));
        pipe_->InitBuffer(featureBuf_, 3 * 48 * FEATURE_TILE * sizeof(float));
        pipe_->InitBuffer(gateBuf_, 48 * FEATURE_TILE * sizeof(float));
        pipe_->InitBuffer(betaBuf_, CHUNK * sizeof(float));
        pipe_->InitBuffer(betaBrcbBuf_, CHUNK * 8 * sizeof(float));
        pipe_->InitBuffer(auxBuf_, 3 * TILE_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 256 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(dbAccBuf_, BLOCK * 8 * sizeof(float));
        pipe_->InitBuffer(dbCompactBuf_, BLOCK * sizeof(float));
    }

    __aicore__ inline void PackTask(uint64_t task, uint64_t logicalCore, uint64_t lane)
    {
        const uint64_t valueHead = task % hv_;
        const uint64_t chunk = task / hv_;
        const uint64_t chunkStart = chunk * CHUNK;
        const uint64_t slotBase = logicalCore * SLOT_ELEMENTS;

        LocalTensor<float> dAQ = dAQBuf_.Get<float>();
        LocalTensor<float> dAK = dAKBuf_.Get<float>();
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        SyncVToMte2();
        DataCopy(dAQ, dAqk_[AOffset(valueHead, chunkStart, 0)], CHUNK * CHUNK);
        DataCopy(dAK, dAkk_[AOffset(valueHead, chunkStart, 0)], CHUNK * CHUNK);
        DataCopy(betaLocal, beta_[BetaOffset(valueHead, chunkStart)], CHUNK);
        SyncMte2ToS();

        __ubuf__ float *dAQPtr = reinterpret_cast<__ubuf__ float *>(dAQ.GetPhyAddr());
        __ubuf__ float *dAKPtr = reinterpret_cast<__ubuf__ float *>(dAK.GetPhyAddr());
        const uint32_t rowBlock0 = lane == 0 ? 0 : 1;
        const uint32_t rowBlock1 = lane == 0 ? 3 : 2;
        PackAMatrices(slotBase, rowBlock0, dAQPtr, dAKPtr);
        PackAMatrices(slotBase, rowBlock1, dAQPtr, dAKPtr);

        // Both lanes have equal B-row work: lane0 handles row blocks 0+3,
        // lane1 handles 1+2 (240 source-feature rows per feature tile).
        PackBMatrices(slotBase, valueHead, chunkStart, rowBlock0);
        PackBMatrices(slotBase, valueHead, chunkStart, rowBlock1);
    }

    __aicore__ inline void ConsumeTask(uint64_t task, uint64_t logicalCore, uint64_t lane)
    {
        const uint64_t valueHead = task % hv_;
        const uint64_t chunk = task / hv_;
        const uint64_t chunkStart = chunk * CHUNK;
        const uint64_t slotBase = logicalCore * SLOT_ELEMENTS;
        const uint32_t rowBlock0 = lane == 0 ? 0 : 1;
        const uint32_t rowBlock1 = lane == 0 ? 3 : 2;
        ConsumeRowBlock(slotBase, valueHead, chunkStart, rowBlock0);
        ConsumeRowBlock(slotBase, valueHead, chunkStart, rowBlock1);
    }

private:
    __aicore__ inline uint64_t QOffset(uint64_t h, uint64_t token, uint64_t d) const
    {
        return (h * t_ + token) * HEAD_DIM + d;
    }

    __aicore__ inline uint64_t VOffset(uint64_t hv, uint64_t token, uint64_t d) const
    {
        return (hv * t_ + token) * HEAD_DIM + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t hv, uint64_t token) const
    {
        return hv * t_ + token;
    }

    __aicore__ inline uint64_t AOffset(uint64_t hv, uint64_t token, uint64_t col) const
    {
        return (hv * t_ + token) * CHUNK + col;
    }

    __aicore__ inline void SyncVToMte2()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(id);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(id);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(id);
        WaitFlag<HardEvent::MTE2_S>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(id);
    }

    __aicore__ inline void SyncSToV()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(id);
    }

    __aicore__ inline void SyncVToS()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(id);
    }

    __aicore__ inline void SyncVToMte3()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(id);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(id);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        TEventID id = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(id);
    }

    __aicore__ inline void StoreScalarMatrixGroup(uint64_t gmOffset, uint32_t elements)
    {
        LocalTensor<float> matrix = matrixBuf_.Get<float>();
        SyncSToV();
        Adds(matrix, matrix, 0.0f, elements);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        DataCopy(workspace_[gmOffset], matrix, elements);
        SyncMte3ToV();
    }

    __aicore__ inline void PrepareScalarMatrix(uint32_t elements)
    {
        LocalTensor<float> matrix = matrixBuf_.Get<float>();
        Duplicate(matrix, 0.0f, elements);
        PipeBarrier<PIPE_V>();
        SyncVToS();
    }

    __aicore__ inline void PackAMatrices(uint64_t slotBase, uint32_t rowBlock,
                                         __ubuf__ float *dAQ, __ubuf__ float *dAK)
    {
        const uint32_t rowBegin = rowBlock * BLOCK;
        __ubuf__ float *matrix =
            reinterpret_cast<__ubuf__ float *>(matrixBuf_.Get<float>().GetPhyAddr());

        if (rowBlock > 0) {
            const uint32_t sourceCount = rowBegin;
            const uint32_t group = rowBlock - 1;
            const uint32_t rowBase = group * 2 * BLOCK;
            const uint32_t colBase = rowBlock == 1 ? 0 : (rowBlock == 2 ? 16 : 48);
            const uint32_t elements = 2 * BLOCK * A_LEFT_PREV_K;
            PrepareScalarMatrix(elements);
            for (uint32_t row = 0; row < BLOCK; ++row) {
                for (uint32_t source = 0; source < sourceCount; ++source) {
                    matrix[row * A_LEFT_PREV_K + colBase + source] =
                        dAQ[(rowBegin + row) * CHUNK + source];
                    matrix[(BLOCK + row) * A_LEFT_PREV_K + colBase + source] =
                        dAK[(rowBegin + row) * CHUNK + source];
                }
            }
            StoreScalarMatrixGroup(slotBase + A_LEFT_PREV_OFFSET +
                                       rowBase * A_LEFT_PREV_K,
                                   elements);
        }

        {
            const uint32_t rowBase = rowBlock * 2 * BLOCK;
            const uint32_t colBase = rowBlock * BLOCK;
            const uint32_t elements = 2 * BLOCK * A_LEFT_DIAG_K;
            PrepareScalarMatrix(elements);
            for (uint32_t row = 0; row < BLOCK; ++row) {
                for (uint32_t source = 0; source <= row; ++source) {
                    matrix[row * A_LEFT_DIAG_K + colBase + source] =
                        dAQ[(rowBegin + row) * CHUNK + rowBegin + source];
                    matrix[(BLOCK + row) * A_LEFT_DIAG_K + colBase + source] =
                        dAK[(rowBegin + row) * CHUNK + rowBegin + source];
                }
            }
            StoreScalarMatrixGroup(slotBase + A_LEFT_DIAG_OFFSET +
                                       rowBase * A_LEFT_DIAG_K,
                                   elements);
        }

        if (rowBlock < 3) {
            const uint32_t sourceStart = rowBegin + BLOCK;
            const uint32_t sourceCount = CHUNK - sourceStart;
            const uint32_t rowBase = rowBlock * BLOCK;
            const uint32_t colBase = rowBlock == 0 ? 0 : (rowBlock == 1 ? 48 : 80);
            const uint32_t elements = BLOCK * A_RIGHT_FUTURE_K;
            for (uint32_t kind = 0; kind < 2; ++kind) {
                PrepareScalarMatrix(elements);
                __ubuf__ float *sourceDA = kind == 0 ? dAQ : dAK;
                for (uint32_t row = 0; row < BLOCK; ++row) {
                    for (uint32_t source = 0; source < sourceCount; ++source) {
                        matrix[row * A_RIGHT_FUTURE_K + colBase + source] =
                            sourceDA[(sourceStart + source) * CHUNK + rowBegin + row];
                    }
                }
                const uint32_t base = kind == 0 ? A_RIGHT_FUTURE_Q_OFFSET
                                                : A_RIGHT_FUTURE_K_OFFSET;
                StoreScalarMatrixGroup(slotBase + base +
                                           rowBase * A_RIGHT_FUTURE_K,
                                       elements);
            }
        }

        {
            const uint32_t rowBase = rowBlock * BLOCK;
            const uint32_t colBase = rowBlock * BLOCK;
            const uint32_t elements = BLOCK * A_RIGHT_DIAG_K;
            for (uint32_t kind = 0; kind < 2; ++kind) {
                PrepareScalarMatrix(elements);
                __ubuf__ float *sourceDA = kind == 0 ? dAQ : dAK;
                for (uint32_t row = 0; row < BLOCK; ++row) {
                    for (uint32_t source = row; source < BLOCK; ++source) {
                        matrix[row * A_RIGHT_DIAG_K + colBase + source] =
                            sourceDA[(rowBegin + source) * CHUNK + rowBegin + row];
                    }
                }
                const uint32_t base = kind == 0 ? A_RIGHT_DIAG_Q_OFFSET
                                                : A_RIGHT_DIAG_K_OFFSET;
                StoreScalarMatrixGroup(slotBase + base + rowBase * A_RIGHT_DIAG_K,
                                       elements);
            }
        }
    }

    template <bool FIXED_LHS, FeatureKind KIND>
    __aicore__ inline void PackBGroup(
        uint64_t slotBase, uint64_t valueHead, uint64_t chunkStart,
        uint32_t sourceStart, uint32_t sourceCount, uint32_t referenceToken,
        uint32_t matrixOffset, uint32_t groupKOffset)
    {
        const uint64_t queryHead = valueHead / (hv_ / h_);
        LocalTensor<bfloat16_t> typed = typedBuf_.Get<bfloat16_t>();
        LocalTensor<float> feature = featureBuf_.Get<float>();
        LocalTensor<float> sourceG = featureBuf_.Get<float>()[48 * FEATURE_TILE];
        LocalTensor<float> referenceG = featureBuf_.Get<float>()[2 * 48 * FEATURE_TILE];
        LocalTensor<float> gate = gateBuf_.Get<float>();
        LocalTensor<float> betaBrcb = betaBrcbBuf_.Get<float>();
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        DataCopyPadExtParams<bfloat16_t> typedPad{false, 0, 0, 0};
        DataCopyPadExtParams<float> floatPad{false, 0, 0, 0.0f};

        for (uint32_t d = 0; d < HEAD_DIM; d += FEATURE_TILE) {
            DataCopyExtParams typedParams{
                static_cast<uint16_t>(sourceCount),
                static_cast<uint32_t>(FEATURE_TILE * sizeof(bfloat16_t)),
                static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(bfloat16_t)),
                0, 0};
            DataCopyExtParams floatParams{
                static_cast<uint16_t>(sourceCount),
                static_cast<uint32_t>(FEATURE_TILE * sizeof(float)),
                static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)),
                0, 0};
            SyncVToMte2();
            if constexpr (KIND == FEATURE_Q) {
                DataCopyPad(typed, q_[QOffset(queryHead, chunkStart + sourceStart, d)],
                            typedParams, typedPad);
            } else {
                DataCopyPad(typed, k_[QOffset(queryHead, chunkStart + sourceStart, d)],
                            typedParams, typedPad);
            }
            DataCopyPad(sourceG, g_[VOffset(valueHead, chunkStart + sourceStart, d)],
                        floatParams, floatPad);
            DataCopy(referenceG,
                     g_[VOffset(valueHead, chunkStart + referenceToken, d)],
                     FEATURE_TILE);
            SyncMte2ToV();
            Cast(feature, typed, RoundMode::CAST_NONE, sourceCount * FEATURE_TILE);
            PipeBarrier<PIPE_V>();

            const uint8_t repeats = static_cast<uint8_t>(sourceCount);
            UnaryRepeatParams unary{1, 1, FEATURE_STRIDE, FEATURE_STRIDE};
            if constexpr (FIXED_LHS) {
                BinaryRepeatParams subParams{1, 1, 1, FEATURE_STRIDE, 0, FEATURE_STRIDE};
                Sub(gate, referenceG, sourceG, FEATURE_TILE, repeats, subParams);
            } else {
                BinaryRepeatParams subParams{1, 1, 1, FEATURE_STRIDE, FEATURE_STRIDE, 0};
                Sub(gate, sourceG, referenceG, FEATURE_TILE, repeats, subParams);
            }
            PipeBarrier<PIPE_V>();
            Muls(gate, gate, LOG2_E_SCALE, FEATURE_TILE, repeats, unary);
            PipeBarrier<PIPE_V>();
            Exp(gate, gate, FEATURE_TILE, repeats, unary);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams featureParams{
                1, 1, 1, FEATURE_STRIDE, FEATURE_STRIDE, FEATURE_STRIDE};
            Mul(feature, feature, gate, FEATURE_TILE, repeats, featureParams);
            PipeBarrier<PIPE_V>();
            if constexpr (KIND == FEATURE_BETA_K) {
                Brcb(betaBrcb, betaLocal[sourceStart],
                     static_cast<uint8_t>((sourceCount + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>();
                BinaryRepeatParams betaParams{
                    1, 1, 0, FEATURE_STRIDE, FEATURE_STRIDE, 1};
                Mul(feature, feature, betaBrcb, FEATURE_TILE, repeats, betaParams);
                PipeBarrier<PIPE_V>();
            }

            DataCopyExtParams outParams{
                static_cast<uint16_t>(sourceCount),
                static_cast<uint32_t>(FEATURE_TILE * sizeof(float)), 0,
                static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)), 0};
            SyncVToMte3();
            DataCopyPad(workspace_[slotBase + matrixOffset +
                                   groupKOffset * HEAD_DIM + d],
                        feature, outParams);
            SyncMte3ToV();
        }
    }

    __aicore__ inline void PackBMatrices(
        uint64_t slotBase, uint64_t valueHead, uint64_t chunkStart,
        uint32_t rowBlock)
    {
        const uint32_t rowBegin = rowBlock * BLOCK;
        const uint32_t rowEnd = rowBegin + BLOCK;
        const uint32_t rowMid = rowBegin + BLOCK / 2;
        if (rowBlock > 0) {
            const uint32_t groupK = rowBlock == 1 ? 0 : (rowBlock == 2 ? 16 : 48);
            PackBGroup<true, FEATURE_K>(
                slotBase, valueHead, chunkStart, 0, rowBegin, rowBegin,
                B_LEFT_PREV_OFFSET, groupK);
        }
        PackBGroup<true, FEATURE_K>(
            slotBase, valueHead, chunkStart, rowBegin, BLOCK, rowMid,
            B_LEFT_DIAG_OFFSET, rowBegin);

        if (rowBlock < 3) {
            const uint32_t sourceCount = CHUNK - rowEnd;
            const uint32_t groupK = rowBlock == 0 ? 0 : (rowBlock == 1 ? 48 : 80);
            PackBGroup<false, FEATURE_Q>(
                slotBase, valueHead, chunkStart, rowEnd, sourceCount, rowEnd - 1,
                B_RIGHT_FUTURE_Q_OFFSET, groupK);
            PackBGroup<false, FEATURE_BETA_K>(
                slotBase, valueHead, chunkStart, rowEnd, sourceCount, rowEnd - 1,
                B_RIGHT_FUTURE_K_OFFSET, groupK);
        }
        PackBGroup<false, FEATURE_Q>(
            slotBase, valueHead, chunkStart, rowBegin, BLOCK, rowMid,
            B_RIGHT_DIAG_Q_OFFSET, rowBegin);
        PackBGroup<false, FEATURE_BETA_K>(
            slotBase, valueHead, chunkStart, rowBegin, BLOCK, rowMid,
            B_RIGHT_DIAG_K_OFFSET, rowBegin);
    }

    __aicore__ inline void LoadCBlock(
        LocalTensor<float> dst, uint64_t gmOffset, uint32_t rowBase, uint32_t d)
    {
        DataCopyExtParams params{
            static_cast<uint16_t>(BLOCK),
            static_cast<uint32_t>(FEATURE_TILE * sizeof(float)),
            static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)),
            0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(dst, workspace_[gmOffset + rowBase * HEAD_DIM + d], params, noPad);
    }

    template <bool FIXED_LHS>
    __aicore__ inline void BuildRowGate(
        LocalTensor<float> gate, LocalTensor<float> reference,
        LocalTensor<float> rows)
    {
        UnaryRepeatParams unary{1, 1, FEATURE_STRIDE, FEATURE_STRIDE};
        if constexpr (FIXED_LHS) {
            BinaryRepeatParams params{1, 1, 1, FEATURE_STRIDE, 0, FEATURE_STRIDE};
            Sub(gate, reference, rows, FEATURE_TILE, BLOCK, params);
        } else {
            BinaryRepeatParams params{1, 1, 1, FEATURE_STRIDE, FEATURE_STRIDE, 0};
            Sub(gate, rows, reference, FEATURE_TILE, BLOCK, params);
        }
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, LOG2_E_SCALE, FEATURE_TILE, BLOCK, unary);
        PipeBarrier<PIPE_V>();
        Exp(gate, gate, FEATURE_TILE, BLOCK, unary);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulRows(
        LocalTensor<float> dst, LocalTensor<float> gate)
    {
        BinaryRepeatParams params{
            1, 1, 1, FEATURE_STRIDE, FEATURE_STRIDE, FEATURE_STRIDE};
        Mul(dst, dst, gate, FEATURE_TILE, BLOCK, params);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadSelfAndC(
        uint64_t slotBase, uint64_t valueHead, uint64_t chunkStart,
        uint32_t rowBlock, uint32_t d,
        LocalTensor<float> qSelf, LocalTensor<float> kSelf,
        LocalTensor<float> gSelf, LocalTensor<float> c0,
        LocalTensor<float> c1, LocalTensor<float> c2,
        LocalTensor<float> c3, LocalTensor<float> c4,
        LocalTensor<float> c5, LocalTensor<float> c6,
        LocalTensor<float> c7)
    {
        const uint32_t rowBegin = rowBlock * BLOCK;
        const uint64_t queryHead = valueHead / (hv_ / h_);
        LocalTensor<bfloat16_t> qTyped = typedBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> kTyped =
            typedBuf_.Get<bfloat16_t>()[BLOCK * FEATURE_TILE];
        DataCopyExtParams typedParams{
            static_cast<uint16_t>(BLOCK),
            static_cast<uint32_t>(FEATURE_TILE * sizeof(bfloat16_t)),
            static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(bfloat16_t)),
            0, 0};
        DataCopyExtParams floatParams{
            static_cast<uint16_t>(BLOCK),
            static_cast<uint32_t>(FEATURE_TILE * sizeof(float)),
            static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)),
            0, 0};
        DataCopyPadExtParams<bfloat16_t> typedPad{false, 0, 0, 0};
        DataCopyPadExtParams<float> floatPad{false, 0, 0, 0.0f};

        SyncVToMte2();
        DataCopyPad(qTyped, q_[QOffset(queryHead, chunkStart + rowBegin, d)],
                    typedParams, typedPad);
        DataCopyPad(kTyped, k_[QOffset(queryHead, chunkStart + rowBegin, d)],
                    typedParams, typedPad);
        DataCopyPad(gSelf, g_[VOffset(valueHead, chunkStart + rowBegin, d)],
                    floatParams, floatPad);
        if (rowBlock > 0) {
            const uint32_t group = rowBlock - 1;
            LoadCBlock(c0, slotBase + C_LEFT_PREV_OFFSET, group * 2 * BLOCK, d);
            LoadCBlock(c1, slotBase + C_LEFT_PREV_OFFSET,
                       group * 2 * BLOCK + BLOCK, d);
        }
        LoadCBlock(c2, slotBase + C_LEFT_DIAG_OFFSET, rowBlock * 2 * BLOCK, d);
        LoadCBlock(c3, slotBase + C_LEFT_DIAG_OFFSET,
                   rowBlock * 2 * BLOCK + BLOCK, d);
        if (rowBlock < 3) {
            LoadCBlock(c4, slotBase + C_RIGHT_FUTURE_Q_OFFSET,
                       rowBlock * BLOCK, d);
            LoadCBlock(c5, slotBase + C_RIGHT_FUTURE_K_OFFSET,
                       rowBlock * BLOCK, d);
        }
        LoadCBlock(c6, slotBase + C_RIGHT_DIAG_Q_OFFSET,
                   rowBlock * BLOCK, d);
        LoadCBlock(c7, slotBase + C_RIGHT_DIAG_K_OFFSET,
                   rowBlock * BLOCK, d);
        SyncMte2ToV();
        Cast(qSelf, qTyped, RoundMode::CAST_NONE, TILE_ELEMENTS);
        Cast(kSelf, kTyped, RoundMode::CAST_NONE, TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        if (rowBlock == 0) {
            Duplicate(c0, 0.0f, TILE_ELEMENTS);
            Duplicate(c1, 0.0f, TILE_ELEMENTS);
        }
        if (rowBlock == 3) {
            Duplicate(c4, 0.0f, TILE_ELEMENTS);
            Duplicate(c5, 0.0f, TILE_ELEMENTS);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadOutputBlock(
        uint64_t valueHead, uint64_t chunkStart, uint32_t rowBegin, uint32_t d,
        LocalTensor<float> outQ, LocalTensor<float> outK, LocalTensor<float> outG)
    {
        DataCopyExtParams params{
            static_cast<uint16_t>(BLOCK),
            static_cast<uint32_t>(FEATURE_TILE * sizeof(float)),
            static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)),
            0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        SyncVToMte2();
        DataCopyPad(outQ, dq_[VOffset(valueHead, chunkStart + rowBegin, d)], params, noPad);
        DataCopyPad(outK, dk_[VOffset(valueHead, chunkStart + rowBegin, d)], params, noPad);
        DataCopyPad(outG, dg_[VOffset(valueHead, chunkStart + rowBegin, d)], params, noPad);
        SyncMte2ToV();
    }

    __aicore__ inline void StoreOutputBlock(
        uint64_t valueHead, uint64_t chunkStart, uint32_t rowBegin, uint32_t d,
        LocalTensor<float> outQ, LocalTensor<float> outK, LocalTensor<float> outG)
    {
        DataCopyExtParams params{
            static_cast<uint16_t>(BLOCK),
            static_cast<uint32_t>(FEATURE_TILE * sizeof(float)), 0,
            static_cast<uint32_t>((HEAD_DIM - FEATURE_TILE) * sizeof(float)), 0};
        SyncVToMte3();
        DataCopyPad(dqOut_[VOffset(valueHead, chunkStart + rowBegin, d)], outQ, params);
        DataCopyPad(dkOut_[VOffset(valueHead, chunkStart + rowBegin, d)], outK, params);
        DataCopyPad(dgOut_[VOffset(valueHead, chunkStart + rowBegin, d)], outG, params);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ConsumeRowBlock(
        uint64_t slotBase, uint64_t valueHead, uint64_t chunkStart,
        uint32_t rowBlock)
    {
        const uint32_t rowBegin = rowBlock * BLOCK;
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        SyncVToMte2();
        DataCopy(betaLocal, beta_[BetaOffset(valueHead, chunkStart + rowBegin)], BLOCK);
        SyncMte2ToV();

        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        Duplicate(dbAcc, 0.0f, BLOCK * 8);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cArena0 = dAQBuf_.Get<float>();
        LocalTensor<float> cArena1 = dAKBuf_.Get<float>();
        LocalTensor<float> c0 = cArena0;
        LocalTensor<float> c1 = cArena0[TILE_ELEMENTS];
        LocalTensor<float> c2 = cArena0[2 * TILE_ELEMENTS];
        LocalTensor<float> c3 = cArena0[3 * TILE_ELEMENTS];
        LocalTensor<float> c4 = cArena1;
        LocalTensor<float> c5 = cArena1[TILE_ELEMENTS];
        LocalTensor<float> c6 = cArena1[2 * TILE_ELEMENTS];
        LocalTensor<float> c7 = cArena1[3 * TILE_ELEMENTS];
        LocalTensor<float> qSelf = featureBuf_.Get<float>();
        LocalTensor<float> kSelf = featureBuf_.Get<float>()[TILE_ELEMENTS];
        LocalTensor<float> gSelf = featureBuf_.Get<float>()[2 * TILE_ELEMENTS];
        LocalTensor<float> gate = gateBuf_.Get<float>();
        LocalTensor<float> outQ = matrixBuf_.Get<float>();
        LocalTensor<float> outK = matrixBuf_.Get<float>()[TILE_ELEMENTS];
        LocalTensor<float> outG = matrixBuf_.Get<float>()[2 * TILE_ELEMENTS];
        LocalTensor<float> blockTmp = auxBuf_.Get<float>();

        for (uint32_t d = 0; d < HEAD_DIM; d += FEATURE_TILE) {
            LoadSelfAndC(slotBase, valueHead, chunkStart, rowBlock, d,
                         qSelf, kSelf, gSelf, c0, c1, c2, c3, c4, c5, c6, c7);

            if (rowBlock > 0) {
                BuildRowGate<false>(gate, gSelf, gSelf);
                MulRows(c0, gate);
                MulRows(c1, gate);
            }
            BuildRowGate<false>(gate, gSelf[(BLOCK / 2) * FEATURE_TILE], gSelf);
            MulRows(c2, gate);
            MulRows(c3, gate);
            Add(c0, c0, c2, TILE_ELEMENTS);
            Add(c1, c1, c3, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();

            Add(c6, c6, c7, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            BuildRowGate<true>(gate, gSelf[(BLOCK / 2) * FEATURE_TILE], gSelf);
            MulRows(c6, gate);
            if (rowBlock < 3) {
                Add(c4, c4, c5, TILE_ELEMENTS);
                PipeBarrier<PIPE_V>();
                BuildRowGate<true>(gate, gSelf[(BLOCK - 1) * FEATURE_TILE], gSelf);
                MulRows(c4, gate);
                Add(c6, c6, c4, TILE_ELEMENTS);
                PipeBarrier<PIPE_V>();
            }

            Mul(blockTmp, c1, kSelf, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> scalar = scalarBuf_.Get<float>();
            LocalTensor<float> reduceTmp = reduceBuf_.Get<float>();
            for (uint32_t reduceOffset = 0; reduceOffset < FEATURE_TILE;
                 reduceOffset += 32) {
                for (uint32_t row = 0; row < BLOCK; ++row) {
                    ReduceSum<float, true>(
                        scalar, blockTmp[row * FEATURE_TILE + reduceOffset],
                        reduceTmp, 32);
                    PipeBarrier<PIPE_V>();
                    Add(dbAcc[row * 8], dbAcc[row * 8], scalar, 1);
                    PipeBarrier<PIPE_V>();
                }
            }

            LocalTensor<float> betaBrcb = betaBrcbBuf_.Get<float>();
            Brcb(betaBrcb, betaLocal, BLOCK / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams betaParams{
                1, 1, 0, FEATURE_STRIDE, FEATURE_STRIDE, 1};
            Mul(c1, c1, betaBrcb, FEATURE_TILE, BLOCK, betaParams);
            PipeBarrier<PIPE_V>();

            LoadOutputBlock(valueHead, chunkStart, rowBegin, d, outQ, outK, outG);
            Add(outQ, outQ, c0, TILE_ELEMENTS);
            Add(outK, outK, c1, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(outK, outK, c6, TILE_ELEMENTS);
            Mul(blockTmp, qSelf, c0, TILE_ELEMENTS);
            Sub(c6, c1, c6, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Mul(c6, c6, kSelf, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(outG, outG, blockTmp, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(outG, outG, c6, TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            StoreOutputBlock(valueHead, chunkStart, rowBegin, d, outQ, outK, outG);
        }

        LocalTensor<float> dbCompact = dbCompactBuf_.Get<float>();
        SyncVToS();
        __ubuf__ float *dbAccPtr =
            reinterpret_cast<__ubuf__ float *>(dbAcc.GetPhyAddr());
        __ubuf__ float *dbCompactPtr =
            reinterpret_cast<__ubuf__ float *>(dbCompact.GetPhyAddr());
        for (uint32_t row = 0; row < BLOCK; ++row) {
            dbCompactPtr[row] = dbAccPtr[row * 8];
        }
        SyncSToV();
        LocalTensor<float> outDb = auxBuf_.Get<float>();
        SyncVToMte2();
        DataCopy(outDb, db_[BetaOffset(valueHead, chunkStart + rowBegin)], BLOCK);
        SyncMte2ToV();
        Add(outDb, outDb, dbCompact, BLOCK);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        DataCopy(dbOut_[BetaOffset(valueHead, chunkStart + rowBegin)], outDb, BLOCK);
        SyncMte3ToV();
    }

private:
    GlobalTensor<bfloat16_t> q_, k_;
    GlobalTensor<float> g_, beta_, dAqk_, dAkk_, dq_, dk_, db_, dg_;
    GlobalTensor<float> dqOut_, dkOut_, dbOut_, dgOut_, workspace_;
    TBuf<TPosition::VECCALC> dAQBuf_, dAKBuf_, matrixBuf_, typedBuf_;
    TBuf<TPosition::VECCALC> featureBuf_, gateBuf_, betaBuf_, betaBrcbBuf_;
    TBuf<TPosition::VECCALC> auxBuf_, reduceBuf_, scalarBuf_, dbAccBuf_, dbCompactBuf_;
    uint64_t h_ = 0, hv_ = 0, t_ = 0, nt_ = 0;
    TPipe *pipe_ = nullptr;
};

class AicKernel {
public:
    __aicore__ inline void Init(GM_ADDR workspace)
    {
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    }

    __aicore__ inline void ProcessSlot(uint64_t logicalCore)
    {
        using Element = float;
        using Layout = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, Layout, Element, Layout, Element, Layout>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape,
            Element, Element, Element, void, TileCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        const uint64_t slotBase = logicalCore * SLOT_ELEMENTS;
        Run(blockMmad, slotBase, A_LEFT_PREV_OFFSET, B_LEFT_PREV_OFFSET,
            C_LEFT_PREV_OFFSET, A_LEFT_PREV_M, A_LEFT_PREV_K);
        Run(blockMmad, slotBase, A_LEFT_DIAG_OFFSET, B_LEFT_DIAG_OFFSET,
            C_LEFT_DIAG_OFFSET, A_LEFT_DIAG_M, A_LEFT_DIAG_K);
        Run(blockMmad, slotBase, A_RIGHT_FUTURE_Q_OFFSET,
            B_RIGHT_FUTURE_Q_OFFSET, C_RIGHT_FUTURE_Q_OFFSET,
            A_RIGHT_FUTURE_M, A_RIGHT_FUTURE_K);
        Run(blockMmad, slotBase, A_RIGHT_FUTURE_K_OFFSET,
            B_RIGHT_FUTURE_K_OFFSET, C_RIGHT_FUTURE_K_OFFSET,
            A_RIGHT_FUTURE_M, A_RIGHT_FUTURE_K);
        Run(blockMmad, slotBase, A_RIGHT_DIAG_Q_OFFSET,
            B_RIGHT_DIAG_Q_OFFSET, C_RIGHT_DIAG_Q_OFFSET,
            A_RIGHT_DIAG_M, A_RIGHT_DIAG_K);
        Run(blockMmad, slotBase, A_RIGHT_DIAG_K_OFFSET,
            B_RIGHT_DIAG_K_OFFSET, C_RIGHT_DIAG_K_OFFSET,
            A_RIGHT_DIAG_M, A_RIGHT_DIAG_K);
    }

private:
    template <typename BlockMmad>
    __aicore__ inline void Run(
        BlockMmad &blockMmad, uint64_t slotBase, uint32_t aOffset,
        uint32_t bOffset, uint32_t cOffset, uint32_t m, uint32_t k)
    {
        using Element = float;
        using Layout = Catlass::layout::RowMajor;
        auto layoutA = tla::MakeLayout<Element, Layout>(m, k);
        auto layoutB = tla::MakeLayout<Element, Layout>(k, HEAD_DIM);
        auto layoutC = tla::MakeLayout<Element, Layout>(m, HEAD_DIM);
        auto tensorA = tla::MakeTensor(workspace_[slotBase + aOffset], layoutA,
                                       Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(workspace_[slotBase + bOffset], layoutB,
                                       Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(workspace_[slotBase + cOffset], layoutC,
                                       Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, HEAD_DIM, k};
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                              tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0),
                              tla::MakeShape(shape.m(), shape.n()));
        blockMmad(blockA, blockB, blockC, shape);
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<float> workspace_;
};

class MixedKernel {
public:
    __aicore__ inline void Init(GM_ADDR workspace,
                                const ChunkKdaBwdIntraTilingData &tiling)
    {
        workspace_ = workspace;
        taskCount_ = static_cast<uint64_t>(tiling.totalChunks) *
                     static_cast<uint64_t>(tiling.vHeadNum);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
    }

    __aicore__ inline void ProcessAic()
    {
        AicKernel cube;
        cube.Init(workspace_);
        const uint64_t logicalCore = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t task = logicalCore; task < taskCount_;
             task += usedCoreNum_) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
            cube.ProcessSlot(logicalCore);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(doneFlag_);
        }
    }

    __aicore__ inline void ProcessAiv(
        GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
        GM_ADDR dbOut, GM_ADDR dgOut, const ChunkKdaBwdIntraTilingData &tiling,
        TPipe *pipe)
    {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        const uint64_t lane = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t logicalCore =
            static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        AivKernel vector;
        vector.Init(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                    dqOut, dkOut, dbOut, dgOut, workspace_, tiling, pipe);

        for (uint64_t task = logicalCore; task < taskCount_;
             task += usedCoreNum_) {
            vector.PackTask(task, logicalCore, lane);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(readyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(doneFlag_);
            vector.ConsumeTask(task, logicalCore, lane);
        }
    }

private:
    Catlass::Arch::CrossCoreFlagWithReverse<> readyFlag_{0, 1};
    Catlass::Arch::CrossCoreFlagWithReverse<> doneFlag_{2, 3};
    GM_ADDR workspace_ = nullptr;
    uint64_t taskCount_ = 0;
    uint64_t usedCoreNum_ = 1;
};

} // namespace KdaFullCube

#endif // CHUNK_KDA_BWD_INTRA_FULL_CUBE_H
