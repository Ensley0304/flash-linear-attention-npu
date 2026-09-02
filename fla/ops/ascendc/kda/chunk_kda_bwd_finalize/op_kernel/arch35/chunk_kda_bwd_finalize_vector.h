/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC::MicroAPI;

constexpr float KDA_FINALIZE_LN2 = 0.6931471805599453f;

// Stage0 VectorPre.  One VF call covers a complete head/chunk.  kE is emitted
// in both layouts from the same register result: ND for its later Vector user
// and NZ for the Stage1 Cube B operand.  This is one computation, not a second
// exp/mul pass.
__simd_vf__ inline void FinalizeStage0VF(
    __ubuf__ bfloat16_t *kENz, __ubuf__ bfloat16_t *kENd,
    __ubuf__ float *exp2Gk, __ubuf__ float *gkLast, __ubuf__ float *rH,
    __ubuf__ bfloat16_t *k, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *h, __ubuf__ bfloat16_t *dh,
    uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> g0;
    RegTensor<float> g1;
    RegTensor<float> e0;
    RegTensor<float> e1;
    RegTensor<float> k0;
    RegTensor<float> k1;
    RegTensor<bfloat16_t> kb;
    RegTensor<bfloat16_t> out;

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + rowOffset);
        Muls(e0, g0, KDA_FINALIZE_LN2, fpMask);
        Muls(e1, g1, KDA_FINALIZE_LN2, fpMask);
        Exp(e0, e0, fpMask);
        Exp(e1, e1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(exp2Gk + rowOffset, e0, e1, fpMask);

        LoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, e0, fpMask);
        Mul(k1, k1, e1, fpMask);
        CastFloat2Half<bfloat16_t>(out, k0, k1, fpMask);
        StoreAlign(kENd + rowOffset, out, bfMask);

        // NZ physical layout: [N1, M, C0].  Four 16-column fractals form one
        // logical 64x128 kE tile.  Store only the valid logical row.
        for (uint16_t n1 = 0; n1 < KDA_FINALIZE_DIM / 16; ++n1) {
            uint32_t srcCount = 16;
            MaskReg sixteen = UpdateMask<bfloat16_t>(srcCount);
            RegTensor<bfloat16_t> segment;
            LoadAlign(segment, kENd + rowOffset + n1 * 16);
            const uint32_t nzOffset =
                static_cast<uint32_t>(n1) * KDA_FINALIZE_CHUNK * 16 + row * 16;
            StoreAlign(kENz + nzOffset, segment, sixteen);
        }
    }

    // gk_last is a full K row.
    const uint32_t lastOffset = static_cast<uint32_t>(validRows - 1) * KDA_FINALIZE_DIM;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + lastOffset);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(gkLast, g0, g1, fpMask);

    // r_h[k] = sum_v h[k,v] * dh[k,v].
    for (uint16_t row = 0; row < KDA_FINALIZE_DIM; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<bfloat16_t> hb;
        RegTensor<bfloat16_t> dhb;
        RegTensor<float> h0;
        RegTensor<float> h1;
        RegTensor<float> dh0;
        RegTensor<float> dh1;
        RegTensor<float> product;
        RegTensor<float> sum;
        LoadIn<bfloat16_t, false>(hb, h + rowOffset);
        LoadIn<bfloat16_t, false>(dhb, dh + rowOffset);
        CastHalf2Float<bfloat16_t>(h0, h1, hb, bfMask);
        CastHalf2Float<bfloat16_t>(dh0, dh1, dhb, bfMask);
        Mul(h0, h0, dh0, fpMask);
        Mul(h1, h1, dh1, fpMask);
        Add(product, h0, h1, fpMask);
        ReduceSum(sum, product, fpMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + row, sum, fpMask);
    }
}

// Stage2 BuildZ.  zV/zW are direct FP32 FixPipe results in row-major UB.
// Zb is written directly in the NZ layout consumed by Stage3, avoiding an
// on-chip relocation before UB->L1.
__simd_vf__ inline void FinalizeStage2VF(
    __ubuf__ bfloat16_t *zbNz, __ubuf__ bfloat16_t *zbNd,
    __ubuf__ float *zV, __ubuf__ float *zW,
    __ubuf__ bfloat16_t *beta, uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<bfloat16_t> betaBf;
    RegTensor<float> betaFp;
    LoadIn<bfloat16_t, false>(betaBf, beta);
    Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);

    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> zv;
        RegTensor<float> zw;
        RegTensor<float> result;
        RegTensor<bfloat16_t> packed;
        LoadAlign(zv, zV + rowOffset);
        LoadAlign(zw, zW + rowOffset);
        Sub(result, zv, zw, fpMask);
        Mul(result, result, betaFp, fpMask);
        uint32_t lowerCount = row;
        MaskReg lower = UpdateMask<float>(lowerCount);
        Select(result, result, zero, lower);
        Cast<bfloat16_t, float, ctFp322HalfZero>(packed, result, fpMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(zbNd + rowOffset, packed, fpMask);
        for (uint16_t n1 = 0; n1 < KDA_FINALIZE_CHUNK / 16; ++n1) {
            uint32_t segmentCount = 16;
            MaskReg sixteen = UpdateMask<bfloat16_t>(segmentCount);
            RegTensor<bfloat16_t> segment;
            LoadAlign(segment, zbNd + rowOffset + n1 * 16);
            StoreAlign(
                zbNz + static_cast<uint32_t>(n1) * KDA_FINALIZE_CHUNK * 16 + row * 16,
                segment, sixteen);
        }
    }
}

class ChunkKdaBwdFinalizeVectorStage02 {
public:
    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR gk, GM_ADDR beta, GM_ADDR h, GM_ADDR dh,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(workspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        subBlockNum_ = AscendC::GetSubBlockNum();
        if (subBlockNum_ == 0U) {
            subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
        }
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        pipe_->InitBuffer(ubBuf_, KDA_FINALIZE_UB_BYTES);
        ub_ = ubBuf_.Get<uint8_t>();
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t logicalCore = AscendC::GetBlockIdx() / subBlockNum_;
        const int64_t coreNum = AscendC::GetBlockNum();
        uint64_t generation = 0;
        uint64_t groupGeneration = 0;
        for (int64_t workTask = logicalCore; workTask < tiling_->workTaskNum;
             workTask += coreNum, ++groupGeneration) {
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
            const int64_t headEnd = FinalizeMin(
                headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
            FinalizeChunkInfo chunk;
            ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }

            // Stage0 AIV is independent of Stage0 AIC.  Each AIV owns alternate
            // heads and alternates its two physical UB slots.
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage0(chunk, head, owner, slot, logicalCore, groupGeneration);
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage2(chunk, head, owner, slot);
            }
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBytes(uint32_t offset)
    {
        return ub_[offset];
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Bf16(uint32_t offset)
    {
        AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, 0, 512 * 1024);
        return l1[offset].ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline void RunStage0(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        auto kUb = UbBytes(32 * 1024).ReinterpretCast<bfloat16_t>();
        auto gkUb = UbBytes(48 * 1024).ReinterpretCast<float>();
        auto hUb = UbBytes(80 * 1024).ReinterpretCast<bfloat16_t>();
        auto dhUb = UbBytes(112 * 1024).ReinterpretCast<bfloat16_t>();
        auto expUb = UbBytes(144 * 1024).ReinterpretCast<float>();
        auto kENd = UbBytes(176 * 1024).ReinterpretCast<bfloat16_t>();
        auto gkLast = UbBytes(192 * 1024).ReinterpretCast<float>();
        auto rH = UbBytes(193 * 1024).ReinterpretCast<float>();
        auto kENz = UbBytes(slot * KDA_FINALIZE_VECTOR_BF16_BYTES).ReinterpretCast<bfloat16_t>();

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t state = FinalizeStateOffset(*tiling_, chunk, head);
        AscendC::DataCopy(kUb, k_[token], chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(gkUb, gk_[token], chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(hUb, h_[state], KDA_FINALIZE_STATE_ELEMS);
        AscendC::DataCopy(dhUb, dh_[state], KDA_FINALIZE_STATE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage0VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(kENz.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kENd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(expUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(hUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dhUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsExp = workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>();
        auto wsKe = workspace_[ws + KDA_FINALIZE_WS_KE].ReinterpretCast<bfloat16_t>();
        auto wsLast = workspace_[ws + KDA_FINALIZE_WS_GK_LAST].ReinterpretCast<float>();
        auto wsRh = workspace_[ws + KDA_FINALIZE_WS_RH].ReinterpretCast<float>();
        AscendC::DataCopy(wsExp, expUb, chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(wsKe, kENd, chunk.validRows * KDA_FINALIZE_DIM);
        AscendC::DataCopy(wsLast, gkLast, KDA_FINALIZE_DIM);
        AscendC::DataCopy(wsRh, rH, KDA_FINALIZE_DIM);

        // Direct UB->L1 egress for Stage1.  The fixed destination is the
        // corresponding owner in the four-head kE resident window.
        auto kEL1 = L1Bf16(96 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        AscendC::DataCopy(kEL1, kENz,
            AscendC::DataCopyParams(1, KDA_FINALIZE_VECTOR_BF16_BYTES / 32, 0, 0));
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_KE_READY_BASE + owner);

        // Only after all Stage0 uses of these physical ranges are drained may
        // AIC overwrite them with zV/zW.
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZV_FREE_BASE + slot);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZW_FREE_BASE + slot);
    }

    __aicore__ inline void RunStage2(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot)
    {
        const uint64_t flagOffset =
            static_cast<uint64_t>(subBlockIdx_) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZV_READY_BASE + flagOffset + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZW_READY_BASE + flagOffset + slot);
        auto zV = UbBytes(KDA_FINALIZE_UB_ZV + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .ReinterpretCast<float>();
        auto zW = UbBytes(KDA_FINALIZE_UB_ZW + slot * KDA_FINALIZE_MATRIX_FP32_BYTES)
                      .ReinterpretCast<float>();
        auto zB = UbBytes(KDA_FINALIZE_UB_ZB + slot * KDA_FINALIZE_MATRIX_BF16_BYTES)
                      .ReinterpretCast<bfloat16_t>();
        auto betaUb = UbBytes(KDA_FINALIZE_UB_BETA).ReinterpretCast<bfloat16_t>();
        auto zBNd = UbBytes(KDA_FINALIZE_UB_WORK).ReinterpretCast<bfloat16_t>();
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        // beta is a scalar per token.  The last chunk is not necessarily
        // 32-byte aligned, so use the byte-count interface and zero-fill the
        // rest of the fixed 64-element VF tile instead of reading past T.
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> betaPad{
            true, 0,
            static_cast<uint8_t>(KDA_FINALIZE_CHUNK - chunk.validRows),
            static_cast<bfloat16_t>(0)};
        AscendC::DataCopyPad(betaUb, beta_[betaOffset], betaCopy, betaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage2VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(zB.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(zBNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(zW.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(betaUb.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZV_FREE_BASE + slot);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZW_FREE_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        auto zBL1 = L1Bf16(160 * 1024 + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        AscendC::DataCopy(zBL1, zB,
            AscendC::DataCopyParams(1, KDA_FINALIZE_MATRIX_BF16_BYTES / 32, 0, 0));
        // Stage3 will consume this flag and return the L1 slot credit.  During
        // the Stage0--2-only milestone there is intentionally no consumer.
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZB_READY_BASE + owner);
    }

    AscendC::GlobalTensor<bfloat16_t> k_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<bfloat16_t> beta_;
    AscendC::GlobalTensor<bfloat16_t> h_;
    AscendC::GlobalTensor<bfloat16_t> dh_;
    AscendC::GlobalTensor<uint8_t> workspace_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID vToMte3_[KDA_FINALIZE_AIV_SLOTS];
    uint32_t subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
    uint32_t subBlockIdx_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
