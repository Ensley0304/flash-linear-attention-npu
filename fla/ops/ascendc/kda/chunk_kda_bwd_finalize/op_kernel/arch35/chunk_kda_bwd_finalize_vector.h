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
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rH + row, sum, fpMask);
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

// Stage3 StatePre.  One VF invocation consumes one complete head/chunk.  The
// two 64-lane FP32 register halves cover K/V=128 without a second pass.
__simd_vf__ inline void FinalizeStage3VF(
    __ubuf__ float *dkState, __ubuf__ float *dvb,
    __ubuf__ bfloat16_t *dv, __ubuf__ float *gateState, __ubuf__ float *dbV,
    __ubuf__ float *dqRaw, __ubuf__ float *exp2Gk, __ubuf__ float *gk,
    __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *v,
    __ubuf__ bfloat16_t *beta, __ubuf__ float *gkLast, __ubuf__ float *rH,
    float scale, uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> last0;
    RegTensor<float> last1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(last0, last1, gkLast);
    Duplicate(gate0, 0.0f, fpMask);
    Duplicate(gate1, 0.0f, fpMask);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> g0;
        RegTensor<float> g1;
        RegTensor<float> decay0;
        RegTensor<float> decay1;
        RegTensor<float> state0;
        RegTensor<float> state1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, gk + rowOffset);
        Sub(decay0, last0, g0, fpMask);
        Sub(decay1, last1, g1, fpMask);
        Muls(decay0, decay0, KDA_FINALIZE_LN2, fpMask);
        Muls(decay1, decay1, KDA_FINALIZE_LN2, fpMask);
        Exp(decay0, decay0, fpMask);
        Exp(decay1, decay1, fpMask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            state0, state1, dkState + rowOffset);
        Mul(state0, state0, decay0, fpMask);
        Mul(state1, state1, decay1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dkState + rowOffset, state0, state1, fpMask);

        RegTensor<bfloat16_t> kb;
        RegTensor<float> k0;
        RegTensor<float> k1;
        LoadIn<bfloat16_t, false>(kb, k + rowOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        Mul(k0, k0, state0, fpMask);
        Mul(k1, k1, state1, fpMask);
        Add(gate0, gate0, k0, fpMask);
        Add(gate1, gate1, k1, fpMask);

        RegTensor<float> dvb0;
        RegTensor<float> dvb1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dvb0, dvb1, dvb + rowOffset);
        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        LoadIn<bfloat16_t, true>(betaBf, beta + row);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
        RegTensor<float> dv0;
        RegTensor<float> dv1;
        Mul(dv0, dvb0, betaFp, fpMask);
        Mul(dv1, dvb1, betaFp, fpMask);
        RegTensor<bfloat16_t> dvBf;
        CastFloat2Half<bfloat16_t>(dvBf, dv0, dv1, fpMask);
        StoreAlign(dv + rowOffset, dvBf, bfMask);

        RegTensor<bfloat16_t> vb;
        RegTensor<float> v0;
        RegTensor<float> v1;
        RegTensor<float> product;
        RegTensor<float> sum;
        LoadIn<bfloat16_t, false>(vb, v + rowOffset);
        CastHalf2Float<bfloat16_t>(v0, v1, vb, bfMask);
        Mul(v0, v0, dvb0, fpMask);
        Mul(v1, v1, dvb1, fpMask);
        Add(product, v0, v1, fpMask);
        ReduceSum(sum, product, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, sum, fpMask);

        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dq0, dq1, dqRaw + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(exp0, exp1, exp2Gk + rowOffset);
        Mul(dq0, dq0, exp0, fpMask);
        Mul(dq1, dq1, exp1, fpMask);
        Muls(dq0, dq0, scale, fpMask);
        Muls(dq1, dq1, scale, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dvb + rowOffset, dq0, dq1, fpMask);
    }

    RegTensor<float> lastExp0;
    RegTensor<float> lastExp1;
    RegTensor<float> rh0;
    RegTensor<float> rh1;
    const uint32_t lastOffset = static_cast<uint32_t>(validRows - 1) * KDA_FINALIZE_DIM;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
        lastExp0, lastExp1, exp2Gk + lastOffset);
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(rh0, rh1, rH);
    Mul(rh0, rh0, lastExp0, fpMask);
    Mul(rh1, rh1, lastExp1, fpMask);
    Add(gate0, gate0, rh0, fpMask);
    Add(gate1, gate1, rh1, fpMask);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(gateState, gate0, gate1, fpMask);
}

class ChunkKdaBwdFinalizeVectorStage03 {
public:
    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR h, GM_ADDR dh,
        GM_ADDR dqRaw, GM_ADDR dv,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
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
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            // The first Stage0 has no preceding zB MTE3 reader.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            zBPublishCount_[slot] = 0;
        }
        stage3Mte3ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        stage3Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        stage0Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
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

            // Stage3 StatePre uses one phase-wide UB working set.  The next
            // work task starts from slot0, while the previous task finishes
            // on slot1, so a per-slot credit alone cannot protect the shared
            // range.  Drain the preceding task's final MTE3 before any Stage0
            // MTE2 reinterprets UB, then seed the same event for the first
            // StatePre head in this task.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

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
            // Stage0 uses one shared UB working set for all local heads.  Its
            // final workspace egress must finish before Stage2 reinterprets
            // the same physical UB range.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

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

            // StatePre reinterprets nearly the whole UB and therefore
            // overlaps both BuildZ ping/pong egress slots.  Drain both Zb
            // UB->L1 readers before changing the phase-wide UB semantics.
            for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            // StatePre is independent of the Stage3 Cube MMAD.  Each AIV
            // starts it after completing its own BuildZ heads.
            uint32_t stage3ActiveMask = 0;
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                stage3ActiveMask |= 1U << slot;
                RunStage3(chunk, head, owner, slot, logicalCore, groupGeneration);
            }
            // Tail head windows may leave one or both local slots unused.
            // Restore those consumed credits explicitly for the next task.
            for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
                if ((stage3ActiveMask & (1U << slot)) == 0U) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
                }
            }
        }
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
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
        // Stage0 reinterprets the same physical UB range used by the previous
        // work task's zB source.  Do not let MTE2 overwrite it until MTE3 has
        // finished the UB->L1 handoff.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // All heads on this AIV share the Stage0 UB layout.  Do not let this
        // head's MTE2/V overwrite the preceding head while its four MTE3
        // workspace stores are still reading that layout.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
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
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);

        // Direct UB->L1 egress for Stage1.  The fixed destination is the
        // corresponding owner in the four-head kE resident window.
        auto kEL1 = L1Bf16(96 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        AscendC::DataCopy(kEL1, kENz,
            AscendC::DataCopyParams(1, KDA_FINALIZE_VECTOR_BF16_BYTES / 32, 0, 0));
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_KE_READY_BASE + slot);

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
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZV_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_ZW_READY_BASE + slot);
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
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        auto zBL1 = L1Bf16(160 * 1024 + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        if (zBPublishCount_[slot] != 0U) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_ZB_FREE_BASE + slot);
        }
        AscendC::DataCopy(zBL1, zB,
            AscendC::DataCopyParams(1, KDA_FINALIZE_MATRIX_BF16_BYTES / 32, 0, 0));
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_ZB_READY_BASE + slot);
        ++zBPublishCount_[slot];
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    __aicore__ inline void RunStage3(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        // The shared StatePre UB has two future writers: MTE2 and VF.  This
        // phase-level credit prevents the next head's MTE2 from overtaking
        // the preceding head's MTE3->V ownership barrier.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        auto dkState = UbBytes(0).ReinterpretCast<float>();
        auto dvb = UbBytes(32 * 1024).ReinterpretCast<float>();
        auto dqRaw = UbBytes(64 * 1024).ReinterpretCast<float>();
        auto exp2Gk = UbBytes(96 * 1024).ReinterpretCast<float>();
        auto gk = UbBytes(128 * 1024).ReinterpretCast<float>();
        auto k = UbBytes(160 * 1024).ReinterpretCast<bfloat16_t>();
        auto v = UbBytes(176 * 1024).ReinterpretCast<bfloat16_t>();
        auto dv = UbBytes(192 * 1024).ReinterpretCast<bfloat16_t>();
        auto beta = UbBytes(208 * 1024).ReinterpretCast<bfloat16_t>();
        auto gkLast = UbBytes(209 * 1024).ReinterpretCast<float>();
        auto rH = UbBytes(210 * 1024).ReinterpretCast<float>();
        auto gateState = UbBytes(211 * 1024).ReinterpretCast<float>();
        auto dbV = UbBytes(212 * 1024).ReinterpretCast<float>();

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsDkState = workspace_[ws + KDA_FINALIZE_WS_DK_STATE_RAW].ReinterpretCast<float>();
        auto wsDvb = workspace_[ws + KDA_FINALIZE_WS_DVB].ReinterpretCast<float>();
        auto wsExp = workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>();
        auto wsLast = workspace_[ws + KDA_FINALIZE_WS_GK_LAST].ReinterpretCast<float>();
        auto wsRh = workspace_[ws + KDA_FINALIZE_WS_RH].ReinterpretCast<float>();

        const uint32_t vectorElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        AscendC::DataCopy(dkState, wsDkState, vectorElems);
        AscendC::DataCopy(dvb, wsDvb, vectorElems);
        AscendC::DataCopy(dqRaw, dqRaw_[token], vectorElems);
        AscendC::DataCopy(exp2Gk, wsExp, vectorElems);
        AscendC::DataCopy(gk, gk_[token], vectorElems);
        AscendC::DataCopy(k, k_[token], vectorElems);
        AscendC::DataCopy(v, v_[token], vectorElems);
        AscendC::DataCopy(gkLast, wsLast, KDA_FINALIZE_DIM);
        AscendC::DataCopy(rH, wsRh, KDA_FINALIZE_DIM);
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> betaPad{
            true, 0,
            static_cast<uint8_t>(KDA_FINALIZE_CHUNK - chunk.validRows),
            static_cast<bfloat16_t>(0)};
        AscendC::DataCopyPad(beta, beta_[betaOffset], betaCopy, betaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage3VF(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dv.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
            tiling_->scale, static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        auto wsGate = workspace_[ws + KDA_FINALIZE_WS_GATE_STATE].ReinterpretCast<float>();
        auto wsDbV = workspace_[ws + KDA_FINALIZE_WS_DB_V].ReinterpretCast<float>();
        AscendC::DataCopy(wsDkState, dkState, vectorElems);
        AscendC::DataCopy(wsDvb, dvb, vectorElems);
        AscendC::DataCopy(wsGate, gateState, KDA_FINALIZE_DIM);
        AscendC::DataCopyExtParams dbCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(wsDbV, dbV, dbCopy);
        AscendC::DataCopy(dv_[token], dv, vectorElems);
        // StatePre has one shared 213-KiB UB working set, not per-slot
        // ping/pong storage.  Drain every egress before the next head lets VF
        // overwrite it; this is the same MTE3->V ownership rule used by the
        // mature fwd_h implementation.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
    }

    AscendC::GlobalTensor<bfloat16_t> k_;
    AscendC::GlobalTensor<bfloat16_t> v_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<bfloat16_t> beta_;
    AscendC::GlobalTensor<bfloat16_t> h_;
    AscendC::GlobalTensor<bfloat16_t> dh_;
    AscendC::GlobalTensor<float> dqRaw_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    AscendC::GlobalTensor<uint8_t> workspace_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID vToMte3_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID mte3ToMte2_[KDA_FINALIZE_AIV_SLOTS];
    AscendC::TEventID stage3Mte3ToV_;
    AscendC::TEventID stage3Mte3ToMte2_;
    AscendC::TEventID stage0Mte3ToMte2_;
    uint32_t zBPublishCount_[KDA_FINALIZE_AIV_SLOTS];
    uint32_t subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
    uint32_t subBlockIdx_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
