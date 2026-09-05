/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_utils/vector/regbase.hpp"
#include "chunk_kda_bwd_finalize_gate.h"

namespace KDA {

using namespace AscendC::MicroAPI;

constexpr float KDA_FINALIZE_LN2 = 0.6931471805599453f;
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT};
constexpr CastTrait KDA_FINALIZE_FP32_TO_BF16_RNE_ONE = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_callee__ inline void FinalizeCastBf16(
    RegTensor<bfloat16_t> &dst, RegTensor<float> &even,
    RegTensor<float> &odd, MaskReg &mask)
{
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE_ONE>(dst, odd, mask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(dst, even, mask);
}

__simd_callee__ inline void FinalizeZeroNzTail(
    __ubuf__ bfloat16_t *dst, uint16_t validRows, uint16_t cols)
{
    uint32_t count = 16;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, static_cast<bfloat16_t>(0), mask);
    for (uint16_t col = 0; col < cols / 16; ++col) {
        for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
            StoreAlign(dst + col * KDA_FINALIZE_CHUNK * 16 + row * 16, zero, mask);
        }
    }
}

__simd_callee__ inline void FinalizeZeroNdTail(
    __ubuf__ bfloat16_t *dst, uint16_t validRows, uint16_t cols)
{
    uint32_t count = cols;
    MaskReg mask = UpdateMask<bfloat16_t>(count);
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, static_cast<bfloat16_t>(0), mask);
    for (uint16_t row = validRows; row < KDA_FINALIZE_CHUNK; ++row) {
        StoreAlign(dst + row * cols, zero, mask);
    }
}

__simd_vf__ inline void FinalizeStage5TailVF(
    __ubuf__ bfloat16_t *daq, __ubuf__ bfloat16_t *dak,
    __ubuf__ bfloat16_t *kn, __ubuf__ bfloat16_t *qp,
    __ubuf__ bfloat16_t *bp, uint16_t validRows)
{
    FinalizeZeroNdTail(daq, validRows, KDA_FINALIZE_CHUNK);
    FinalizeZeroNdTail(dak, validRows, KDA_FINALIZE_CHUNK);
    FinalizeZeroNdTail(kn, validRows, KDA_FINALIZE_DIM);
    FinalizeZeroNdTail(qp, validRows, KDA_FINALIZE_DIM);
    FinalizeZeroNdTail(bp, validRows, KDA_FINALIZE_DIM);
}

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
        FinalizeCastBf16(out, k0, k1, fpMask);
        StoreAlign(kENd + rowOffset, out, bfMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

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

    if (validRows < KDA_FINALIZE_CHUNK) {
        FinalizeZeroNzTail(kENz, validRows, KDA_FINALIZE_DIM);
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
    DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(betaBf, beta);
    Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, fpMask);

    RegTensor<float> zero;
    Duplicate(zero, 0.0f, fpMask);
    uint32_t betaCount = validRows;
    MaskReg betaMask = UpdateMask<float>(betaCount);
    Select(betaFp, betaFp, zero, betaMask);
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
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, result, fpMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(zbNd + rowOffset, packed, fpMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
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
    if (validRows < KDA_FINALIZE_CHUNK) {
        FinalizeZeroNzTail(zbNz, validRows, KDA_FINALIZE_CHUNK);
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
        FinalizeCastBf16(dvBf, dv0, dv1, fpMask);
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

// Stage4 BaseFinalize.  The three 64x128 FP32 regions are updated in place:
// dkState -> dkBase, dKgbRaw -> dgBase, and dbV -> dbBase.  dqBase and exp2Gk
// remain live in their fixed workspace regions for later stages.
__simd_vf__ inline void FinalizeStage4VF(
    __ubuf__ float *dkState, __ubuf__ float *dqBase,
    __ubuf__ float *dKgbRaw, __ubuf__ float *exp2Gk,
    __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *beta,
    __ubuf__ float *gateState, __ubuf__ float *dbV,
    uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gate0, gate1, gateState);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> state0;
        RegTensor<float> state1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        RegTensor<float> dkg0;
        RegTensor<float> dkg1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            state0, state1, dkState + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dq0, dq1, dqBase + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dkg0, dkg1, dKgbRaw + rowOffset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            exp0, exp1, exp2Gk + rowOffset);

        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        LoadIn<bfloat16_t, true>(betaBf, beta + row);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);

        RegTensor<float> dkTerm0;
        RegTensor<float> dkTerm1;
        RegTensor<float> dkgExp0;
        RegTensor<float> dkgExp1;
        Mul(dkgExp0, exp0, dkg0, fpMask);
        Mul(dkgExp1, exp1, dkg1, fpMask);
        Mul(dkTerm0, dkgExp0, betaFp, fpMask);
        Mul(dkTerm1, dkgExp1, betaFp, fpMask);
        Sub(dkTerm0, state0, dkTerm0, fpMask);
        Sub(dkTerm1, state1, dkTerm1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dkState + rowOffset, dkTerm0, dkTerm1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<bfloat16_t> kBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        RegTensor<float> k0;
        RegTensor<float> k1;
        LoadIn<bfloat16_t, false>(qBf, q + rowOffset);
        LoadIn<bfloat16_t, false>(kBf, k + rowOffset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);

        RegTensor<float> dg0;
        RegTensor<float> dg1;
        RegTensor<float> stateK0;
        RegTensor<float> stateK1;
        RegTensor<float> dkgKe0;
        RegTensor<float> dkgKe1;
        Mul(dg0, q0, dq0, fpMask);
        Mul(dg1, q1, dq1, fpMask);
        Mul(stateK0, k0, state0, fpMask);
        Mul(stateK1, k1, state1, fpMask);
        Sub(dg0, dg0, stateK0, fpMask);
        Sub(dg1, dg1, stateK1, fpMask);
        // Vector gradients must not consume Cube's rounded BF16 kE.
        // Reuse exp*dKgb from dkBase instead of recomputing or storing kE.
        Mul(dkgKe0, dkgExp0, k0, fpMask);
        Mul(dkgKe1, dkgExp1, k1, fpMask);

        RegTensor<float> reduceInput;
        RegTensor<float> reduceSum;
        RegTensor<float> dbBase;
        Add(reduceInput, dkgKe0, dkgKe1, fpMask);
        ReduceSum(reduceSum, reduceInput, fpMask);
        LoadIn<float, true>(dbBase, dbV + row);
        Sub(dbBase, dbBase, reduceSum, fpMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, dbBase, fpMask);

        Mul(dkgKe0, dkgKe0, betaFp, fpMask);
        Mul(dkgKe1, dkgKe1, betaFp, fpMask);
        Sub(dg0, dg0, dkgKe0, fpMask);
        Sub(dg1, dg1, dkgKe1, fpMask);
        if (row + 1U == validRows) {
            Add(dg0, dg0, gate0, fpMask);
            Add(dg1, dg1, gate1, fpMask);
        }
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dKgbRaw + rowOffset, dg0, dg1, fpMask);
    }
}

// Stage5 IntraPre. One VF invocation covers a complete head and leaves all
// four results in row-major ND form. The following UB->L1 MTE3 copies perform
// the ND->NZ layout conversion with strided vector-mode DataCopy, matching the
// mature DHU producer path and avoiding an extra UB read/write transpose.
__simd_vf__ inline void FinalizeStage5VF(
    __ubuf__ bfloat16_t *dAkkNd, __ubuf__ bfloat16_t *kNegNd,
    __ubuf__ bfloat16_t *qPosNd, __ubuf__ bfloat16_t *bkPosNd,
    __ubuf__ bfloat16_t *dAkkRaw, __ubuf__ bfloat16_t *q,
    __ubuf__ bfloat16_t *k, __ubuf__ float *exp2Gk,
    __ubuf__ bfloat16_t *beta, uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<float> one;
    Duplicate(zero, 0.0f, fpMask);
    Duplicate(one, 1.0f, fpMask);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t matrixOffset =
            static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<bfloat16_t> rawBf;
        RegTensor<float> raw0;
        RegTensor<bfloat16_t> dakkPacked;
        // A matrix row has 64 BF16 elements, unlike the 128-element vectors.
        // Unpack one row into FP32 lanes and pack it back without dropping
        // odd columns or reading the following row.
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(rawBf, dAkkRaw + matrixOffset);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(raw0, rawBf, fpMask);
        Muls(raw0, raw0, -1.0f, fpMask);
        uint32_t lowerCount = row;
        MaskReg lower = UpdateMask<float>(lowerCount);
        Select(raw0, raw0, zero, lower);
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(dakkPacked, raw0, fpMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dAkkNd + matrixOffset, dakkPacked, fpMask);

        const uint32_t vectorOffset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> e0;
        RegTensor<float> e1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            e0, e1, exp2Gk + vectorOffset);

        RegTensor<bfloat16_t> kBf;
        RegTensor<float> k0;
        RegTensor<float> k1;
        RegTensor<float> kNeg0;
        RegTensor<float> kNeg1;
        RegTensor<bfloat16_t> packed;
        LoadIn<bfloat16_t, false>(kBf, k + vectorOffset);
        CastHalf2Float<bfloat16_t>(k0, k1, kBf, bfMask);
        Div(kNeg0, one, e0, fpMask);
        Div(kNeg1, one, e1, fpMask);
        Mul(kNeg0, kNeg0, k0, fpMask);
        Mul(kNeg1, kNeg1, k1, fpMask);
        FinalizeCastBf16(packed, kNeg0, kNeg1, fpMask);
        StoreAlign(kNegNd + vectorOffset, packed, bfMask);

        // Reuse this row's K/exp registers without changing rounding boundaries.
        RegTensor<bfloat16_t> qBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        LoadIn<bfloat16_t, false>(qBf, q + vectorOffset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);
        Mul(q0, q0, e0, fpMask);
        Mul(q1, q1, e1, fpMask);
        FinalizeCastBf16(packed, q0, q1, fpMask);
        StoreAlign(qPosNd + vectorOffset, packed, bfMask);

        RegTensor<bfloat16_t> betaBf;
        RegTensor<float> betaFp;
        LoadIn<bfloat16_t, true>(betaBf, beta + row);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(betaFp, betaBf, bfMask);
        Mul(k0, k0, e0, fpMask);
        Mul(k1, k1, e1, fpMask);
        Mul(k0, k0, betaFp, fpMask);
        Mul(k1, k1, betaFp, fpMask);
        FinalizeCastBf16(packed, k0, k1, fpMask);
        StoreAlign(bkPosNd + vectorOffset, packed, bfMask);
    }
}

// dAqk is a public FP32 input, while Cube consumes BF16 operands. Convert it
// once at the Stage5 handoff so the complete LocalOperand owner slot has the
// representation used by Stage6 and Stage8.
__simd_vf__ inline void FinalizeStage5DaqkVF(
    __ubuf__ bfloat16_t *dAqkBf16, __ubuf__ float *dAqk,
    uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_CHUNK;
        RegTensor<float> value;
        RegTensor<bfloat16_t> packed;
        LoadAlign(value, dAqk + offset);
        // Match the public FP32 operand's PyTorch/Triton BF16 tie-to-even cast.
        Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packed, value, fpMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
            dAqkBf16 + offset, packed, fpMask);
    }
}

// Stage7 consumes the FP32 Cube result, writes the final BF16 dq, and updates
// dg_base in FP32 for the later K and gate stages.
__simd_vf__ inline void FinalizeStage7VF(
    __ubuf__ bfloat16_t *dqOut, __ubuf__ float *dgBase,
    __ubuf__ float *dqLocalRaw, __ubuf__ float *dqBase,
    __ubuf__ float *exp2Gk, __ubuf__ bfloat16_t *q,
    __ubuf__ float *qRstd, uint32_t hasQkL2Norm, uint16_t validRows)
{
    MaskReg fpMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * KDA_FINALIZE_DIM;
        RegTensor<float> raw0;
        RegTensor<float> raw1;
        RegTensor<float> exp0;
        RegTensor<float> exp1;
        RegTensor<float> dq0;
        RegTensor<float> dq1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            raw0, raw1, dqLocalRaw + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            exp0, exp1, exp2Gk + offset);
        Mul(raw0, raw0, exp0, fpMask);
        Mul(raw1, raw1, exp1, fpMask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dq0, dq1, dqBase + offset);
        Add(dq0, dq0, raw0, fpMask);
        Add(dq1, dq1, raw1, fpMask);

        RegTensor<bfloat16_t> qBf;
        RegTensor<float> q0;
        RegTensor<float> q1;
        LoadIn<bfloat16_t, false>(qBf, q + offset);
        CastHalf2Float<bfloat16_t>(q0, q1, qBf, bfMask);

        RegTensor<float> dg0;
        RegTensor<float> dg1;
        RegTensor<float> product0;
        RegTensor<float> product1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            dg0, dg1, dgBase + offset);
        Mul(product0, q0, raw0, fpMask);
        Mul(product1, q1, raw1, fpMask);
        Add(dg0, dg0, product0, fpMask);
        Add(dg1, dg1, product1, fpMask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            dgBase + offset, dg0, dg1, fpMask);

        if (hasQkL2Norm != 0U) {
            RegTensor<float> dot;
            RegTensor<float> rstd;
            Mul(product0, dq0, q0, fpMask);
            Mul(product1, dq1, q1, fpMask);
            Add(product0, product0, product1, fpMask);
            ReduceSum(dot, product0, fpMask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstd, qRstd + row);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                qRstd + row, dot, scalarMask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(dot, qRstd + row);
            Mul(product0, q0, dot, fpMask);
            Mul(product1, q1, dot, fpMask);
            Sub(dq0, dq0, product0, fpMask);
            Sub(dq1, dq1, product1, fpMask);
            Mul(dq0, dq0, rstd, fpMask);
            Mul(dq1, dq1, rstd, fpMask);
        }

        RegTensor<bfloat16_t> packed;
        FinalizeCastBf16(packed, dq0, dq1, fpMask);
        StoreAlign(dqOut + offset, packed, bfMask);
    }
}

__simd_vf__ inline void FinalizeStage9VF(
    __ubuf__ float *left, __ubuf__ float *right, __ubuf__ float *dbDelta,
    __ubuf__ float *dg, __ubuf__ float *expG,
    __ubuf__ bfloat16_t *k, __ubuf__ bfloat16_t *beta, uint16_t rows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> l0, l1, r0, r1, e0, e1, k0, k1, g0, g1, b;
        RegTensor<bfloat16_t> kb, bb;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(l0, l1, left + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(r0, r1, right + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(e0, e1, expG + offset);
        Mul(l0, l0, e0, mask);
        Mul(l1, l1, e1, mask);
        Div(r0, r0, e0, mask);
        Div(r1, r1, e1, mask);
        LoadIn<bfloat16_t, false>(kb, k + offset);
        CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(bb, beta + row);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(b, bb, mask);
        RegTensor<float> p0, p1, sum;
        Mul(p0, l0, k0, mask);
        Mul(p1, l1, k1, mask);
        Add(p0, p0, p1, mask);
        ReduceSum(sum, p0, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbDelta + row, sum, one);
        Mul(l0, l0, b, mask);
        Mul(l1, l1, b, mask);
        Sub(p0, l0, r0, mask);
        Sub(p1, l1, r1, mask);
        Mul(p0, p0, k0, mask);
        Mul(p1, p1, k1, mask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, dg + offset);
        Add(g0, g0, p0, mask);
        Add(g1, g1, p1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dg + offset, g0, g1, mask);
        Add(l0, l0, r0, mask);
        Add(l1, l1, r1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(left + offset, l0, l1, mask);
    }
}

__simd_vf__ inline void FinalizeStage10VF(
    __ubuf__ bfloat16_t *out, __ubuf__ bfloat16_t *dbOut,
    __ubuf__ float *delta, __ubuf__ float *base,
    __ubuf__ float *dbDelta, __ubuf__ float *dbBase,
    __ubuf__ bfloat16_t *k, __ubuf__ float *rstd,
    uint32_t hasNorm, uint16_t rows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bfMask = CreateMask<half, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t offset = row * KDA_FINALIZE_DIM;
        RegTensor<float> d0, d1, b0, b1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(d0, d1, delta + offset);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(b0, b1, base + offset);
        Add(d0, d0, b0, mask);
        Add(d1, d1, b1, mask);
        if (hasNorm != 0U) {
            RegTensor<bfloat16_t> kb;
            RegTensor<float> k0, k1, p0, p1, dot, rs;
            LoadIn<bfloat16_t, false>(kb, k + offset);
            CastHalf2Float<bfloat16_t>(k0, k1, kb, bfMask);
            Mul(p0, d0, k0, mask);
            Mul(p1, d1, k1, mask);
            Add(p0, p0, p1, mask);
            ReduceSum(dot, p0, mask);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rs, rstd + row);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstd + row, dot, one);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(dot, rstd + row);
            Mul(p0, k0, dot, mask);
            Mul(p1, k1, dot, mask);
            Sub(d0, d0, p0, mask);
            Sub(d1, d1, p1, mask);
            Mul(d0, d0, rs, mask);
            Mul(d1, d1, rs, mask);
        }
        RegTensor<bfloat16_t> packed;
        FinalizeCastBf16(packed, d0, d1, mask);
        StoreAlign(out + offset, packed, bfMask);
    }
    uint32_t count = rows;
    MaskReg rowMask = UpdateMask<float>(count);
    RegTensor<float> db, deltaB;
    RegTensor<bfloat16_t> packedB;
    LoadAlign(db, dbBase);
    LoadAlign(deltaB, dbDelta);
    Add(db, db, deltaB, rowMask);
    Cast<bfloat16_t, float, KDA_FINALIZE_FP32_TO_BF16_RNE>(packedB, db, rowMask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dbOut, packedB, rowMask);
}

class ChunkKdaBwdFinalizeVectorStage12 {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh,
        GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
        GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
        GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        rawG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rawG));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_A_LOG *>(aLog));
        dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        dG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dG));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        if (tiling->hasQkL2Norm != 0U) {
            qRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
        }
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dk));
        dBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dBeta));
        if (tiling->hasQkL2Norm != 0U) {
            kRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
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
        stage5VToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        stage7Mte3ToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        gateVToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
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

            // Even an idle AIV must consume this task's credit. Otherwise a
            // one-head window can run ahead and overwrite the preceding
            // task's qPos/bkPos in L1 while Stage8 still reads them.
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                KDA_FINALIZE_TASK_L1_FREE);

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
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_ZV_FREE_BASE + slot);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_ZW_FREE_BASE + slot);
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

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            // BaseFinalize is independent of the current Stage4 Cube result.
            // Publish one free dAkk slot per local head, then immediately use
            // the disjoint [16,248)-KiB working region.
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
                    KDA_FINALIZE_DAKK_FREE_BASE + slot);
                RunStage4(chunk, head, owner, slot, logicalCore, groupGeneration);
            }

            // Stage4 and Stage5 reinterpret the same phase-wide UB working
            // range.  The last Stage4 MTE3 store is asynchronous, so drain
            // it before Stage5 starts loading q/k/exp2_gk/beta into that
            // range.  Return the phase credit for the next work task; the
            // per-slot credits below continue to protect Stage5's UB->L1
            // ping/pong egress independently.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);

            // Stage5 preserves the same four-head ownership: AIV0 handles
            // owners 0/2 and AIV1 handles owners 1/3, each with ping/pong
            // egress.  Start each head as soon as its dAkk and L1 owner slot
            // are both ready; no group-wide AIV input barrier is introduced.
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage5(chunk, head, owner, slot, logicalCore, groupGeneration);
            }

            // Q raw overlaps both dAkk source slots. Finish the complete
            // Stage5 phase before granting either FixPipe destination.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
                    KDA_FINALIZE_DQ_LOCAL_FREE_BASE + slot);
            }

            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(generation & 1U);
                if (aiv != subBlockIdx_) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage7(chunk, head, owner, slot, logicalCore, groupGeneration);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            // Both Stage7 heads share the upcoming K raw range. Publish FREE
            // only after the complete Q phase, including asynchronous egress.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
                    KDA_FINALIZE_KE_READY_BASE + slot);
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage9(chunk, head, static_cast<uint32_t>(head - headBegin),
                          slot, logicalCore, groupGeneration);
                RunStage11(chunk, chunkTask, head);
            }
            generation -= static_cast<uint64_t>(headEnd - headBegin);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                if ((generation & 1U) != subBlockIdx_) continue;
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                RunStage10(chunk, head, static_cast<uint32_t>(head - headBegin),
                           slot, logicalCore, groupGeneration);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        }
        for (uint32_t slot = 0; slot < KDA_FINALIZE_AIV_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

private:
    __aicore__ inline void RunStage11(
        const FinalizeChunkInfo &chunk, int64_t chunkTask, int64_t head)
    {
        // Stage9's dg is still resident. Its K/exp inputs are dead, while
        // both K-delta slots [0,128) KiB and beta deltas at 240 KiB stay live.
        auto dg = UbBytes(176 * 1024).ReinterpretCast<float>();
        auto raw = UbBytes(128 * 1024).ReinterpretCast<float>();
        auto bias = UbBytes(160 * 1024).ReinterpretCast<float>();
        auto log = UbBytes(161 * 1024).ReinterpretCast<DTYPE_A_LOG>();
        auto da = UbBytes(162 * 1024).ReinterpretCast<float>();
        auto db = UbBytes(163 * 1024).ReinterpretCast<float>();
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint32_t elems = chunk.validRows * KDA_FINALIZE_DIM;
        AscendC::DataCopy(raw, rawG_[token], elems);
        AscendC::DataCopy(bias, dtBias_[head * KDA_FINALIZE_DIM], KDA_FINALIZE_DIM);
        AscendC::DataCopyPad(log, aLog_[head],
            AscendC::DataCopyExtParams{1, sizeof(DTYPE_A_LOG), 0, 0, 0},
            AscendC::DataCopyPadExtParams<DTYPE_A_LOG>{false, 0, 0, static_cast<DTYPE_A_LOG>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        FinalizeStage11VF<DTYPE_A_LOG>(
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DTYPE_A_LOG *>(log.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(da.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()), chunk.validRows, tiling_->lowerBound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        const int64_t partial = head * tiling_->totalChunkNum + chunkTask;
        AscendC::DataCopy(dG_[token], dg, elems);
        AscendC::DataCopyPad(workspace_[tiling_->gatePartialOffset].ReinterpretCast<float>()[partial * 8], da,
            AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0});
        AscendC::DataCopy(workspace_[tiling_->dtBiasPartialOffset].ReinterpretCast<float>()[partial * 128], db, 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBytes(uint32_t offset)
    {
        return ub_[offset];
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Bf16(uint32_t offset)
    {
        AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, 0, 512 * 1024);
        return l1[offset].ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline void CopyStage5NdToNzL1(
        const AscendC::LocalTensor<bfloat16_t> &dstNz,
        const AscendC::LocalTensor<bfloat16_t> &srcNd,
        uint32_t rows, uint32_t cols)
    {
        constexpr uint32_t c0Elems = 32 / sizeof(bfloat16_t);
        AscendC::DataCopyEnhancedParams enhanced;
        enhanced.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
        const AscendC::DataCopyParams params{
            static_cast<uint16_t>(rows), 1,
            static_cast<uint16_t>(cols / c0Elems - 1), 0};
        for (uint32_t col = 0; col < cols; col += c0Elems) {
            const uint32_t dstOffset = (col / c0Elems) * KDA_FINALIZE_CHUNK * c0Elems;
            AscendC::DataCopy(dstNz[dstOffset], srcNd[col], params, enhanced);
        }
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
    }

    __aicore__ inline void RunStage2(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner, uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
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
        // 32-byte aligned. DMA pads only to a block boundary; Stage2 masks
        // inactive beta lanes. Padding all the way to 64 can exceed the
        // instruction's per-side padding limit on a short tail.
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> betaPad{
            true, 0,
            0,
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
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage0Mte3ToMte2_);
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
            0,
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

    __aicore__ inline void RunStage4(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        // Stage3 and Stage4 reinterpret one shared UB working range.  Wait for
        // the preceding phase/head's final MTE3 before issuing new MTE2 loads.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
        auto dkState = UbBytes(16 * 1024).ReinterpretCast<float>();
        auto dqBase = UbBytes(48 * 1024).ReinterpretCast<float>();
        auto dKgbRaw = UbBytes(80 * 1024).ReinterpretCast<float>();
        auto exp2Gk = UbBytes(112 * 1024).ReinterpretCast<float>();
        auto q = UbBytes(160 * 1024).ReinterpretCast<bfloat16_t>();
        auto k = UbBytes(176 * 1024).ReinterpretCast<bfloat16_t>();
        auto beta = UbBytes(192 * 1024).ReinterpretCast<bfloat16_t>();
        auto gateState = UbBytes(193 * 1024).ReinterpretCast<float>();
        auto dbV = UbBytes(194 * 1024).ReinterpretCast<float>();

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t betaOffset = FinalizeTokenOffset(*tiling_, chunk, head, 1);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsDk = workspace_[ws + KDA_FINALIZE_WS_DK_STATE_RAW].ReinterpretCast<float>();
        auto wsDq = workspace_[ws + KDA_FINALIZE_WS_DQ_BASE].ReinterpretCast<float>();
        auto wsDkg = workspace_[ws + KDA_FINALIZE_WS_DKGB_RAW].ReinterpretCast<float>();
        auto wsExp = workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>();
        auto wsGate = workspace_[ws + KDA_FINALIZE_WS_GATE_STATE].ReinterpretCast<float>();
        auto wsDb = workspace_[ws + KDA_FINALIZE_WS_DB_V].ReinterpretCast<float>();

        const uint32_t vectorElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        AscendC::DataCopy(dkState, wsDk, vectorElems);
        AscendC::DataCopy(dqBase, wsDq, vectorElems);
        AscendC::DataCopy(dKgbRaw, wsDkg, vectorElems);
        AscendC::DataCopy(exp2Gk, wsExp, vectorElems);
        AscendC::DataCopy(q, q_[token], vectorElems);
        AscendC::DataCopy(k, k_[token], vectorElems);
        AscendC::DataCopy(gateState, wsGate, KDA_FINALIZE_DIM);
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> betaPad{
            true, 0,
            0,
            static_cast<bfloat16_t>(0)};
        AscendC::DataCopyPad(beta, beta_[betaOffset], betaCopy, betaPad);
        AscendC::DataCopyExtParams dbCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> dbPad{
            true, 0,
            0, 0.0f};
        AscendC::DataCopyPad(dbV, wsDb, dbCopy, dbPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage4VF(
            reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dKgbRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(beta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(wsDk, dkState, vectorElems);
        AscendC::DataCopy(wsDkg, dKgbRaw, vectorElems);
        AscendC::DataCopyPad(wsDb, dbV, dbCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage3Mte3ToMte2_);
    }

    __aicore__ inline void RunStage5(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_DAKK_READY_BASE + slot);
        // Protect this ping/pong egress from its previous MTE3 reader while
        // allowing the other slot's MTE3 to overlap the current VF.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        // q/k/exp/beta share one phase-wide input range.  The next head may
        // start MTE2 as soon as this head's VF has completed its final load;
        // it need not wait for the disjoint UB->L1 egress.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);

        auto dAkkRaw = UbBytes(
            KDA_FINALIZE_UB_DAKK_RAW + slot * KDA_FINALIZE_MATRIX_BF16_BYTES)
                               .ReinterpretCast<bfloat16_t>();
        auto kNegNd = UbBytes(
            KDA_FINALIZE_UB_K_NEG + slot * KDA_FINALIZE_VECTOR_BF16_BYTES)
                          .ReinterpretCast<bfloat16_t>();
        auto qPosNd = UbBytes(
            KDA_FINALIZE_UB_Q_POS + slot * KDA_FINALIZE_VECTOR_BF16_BYTES)
                          .ReinterpretCast<bfloat16_t>();
        auto bkPosNd = UbBytes(
            KDA_FINALIZE_UB_BK_POS + slot * KDA_FINALIZE_VECTOR_BF16_BYTES)
                           .ReinterpretCast<bfloat16_t>();
        auto q = UbBytes(112 * 1024).ReinterpretCast<bfloat16_t>();
        auto k = UbBytes(128 * 1024).ReinterpretCast<bfloat16_t>();
        auto exp2Gk = UbBytes(144 * 1024).ReinterpretCast<float>();
        auto beta = UbBytes(176 * 1024).ReinterpretCast<bfloat16_t>();
        // Keep a dedicated dAkk ND egress slot per local head. The old
        // workNd and shared dAkkNd transpose scratch are no longer needed.
        auto dAkkNd = UbBytes(232 * 1024 + slot * KDA_FINALIZE_MATRIX_BF16_BYTES)
                          .ReinterpretCast<bfloat16_t>();
        // The Stage5 phase has a disjoint gap between beta and dAkkNd. Keep
        // dAqk there so all GM inputs share one MTE2->V event cycle.
        auto dAqkFp32 = UbBytes(184 * 1024).ReinterpretCast<float>();
        auto dAqkBf16 = UbBytes(200 * 1024).ReinterpretCast<bfloat16_t>();

        const int64_t token =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t betaOffset =
            FinalizeTokenOffset(*tiling_, chunk, head, 1);
        const uint64_t ws =
            FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsExp = workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>();
        const uint32_t vectorElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        const int64_t matrixToken =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        const uint32_t matrixElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_CHUNK;
        AscendC::DataCopy(q, q_[token], vectorElems);
        AscendC::DataCopy(k, k_[token], vectorElems);
        AscendC::DataCopy(exp2Gk, wsExp, vectorElems);
        AscendC::DataCopy(dAqkFp32, dAqk_[matrixToken], matrixElems);
        AscendC::DataCopyExtParams betaCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> betaPad{
            true, 0, 0,
            static_cast<bfloat16_t>(0)};
        AscendC::DataCopyPad(beta, beta_[betaOffset], betaCopy, betaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage5VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(dAkkNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kNegNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(bkPosNd.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dAkkRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(beta.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        FinalizeStage5DaqkVF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(dAqkBf16.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dAqkFp32.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        if (chunk.validRows < KDA_FINALIZE_CHUNK) {
            FinalizeStage5TailVF(
                reinterpret_cast<__ubuf__ bfloat16_t *>(dAqkBf16.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(dAkkNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(kNegNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(qPosNd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(bkPosNd.GetPhyAddr()), chunk.validRows);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage5VToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);

        auto local = L1Bf16(
            KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES);
        CopyStage5NdToNzL1(
            local[KDA_FINALIZE_LOCAL_DAQK / sizeof(bfloat16_t)], dAqkBf16,
            KDA_FINALIZE_CHUNK, KDA_FINALIZE_CHUNK);
        CopyStage5NdToNzL1(
            local[KDA_FINALIZE_LOCAL_DAKK / sizeof(bfloat16_t)], dAkkNd,
            KDA_FINALIZE_CHUNK, KDA_FINALIZE_CHUNK);
        CopyStage5NdToNzL1(
            local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(bfloat16_t)], kNegNd,
            KDA_FINALIZE_CHUNK, KDA_FINALIZE_DIM);
        CopyStage5NdToNzL1(
            local[KDA_FINALIZE_LOCAL_Q_POS / sizeof(bfloat16_t)], qPosNd,
            KDA_FINALIZE_CHUNK, KDA_FINALIZE_DIM);
        CopyStage5NdToNzL1(
            local[KDA_FINALIZE_LOCAL_BK_POS / sizeof(bfloat16_t)], bkPosNd,
            KDA_FINALIZE_CHUNK, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(stage3Mte3ToV_);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(
            KDA_FINALIZE_LOCAL_READY_BASE + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    __aicore__ inline void RunStage7(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        // Stage5's egress and the preceding Stage7 head both overlap this
        // phase-wide working set. Drain them before changing UB semantics.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(
            KDA_FINALIZE_DQ_LOCAL_READY_BASE + slot);

        auto dqLocalRaw = UbBytes(
            KDA_FINALIZE_UB_DQ_LOCAL_RAW +
            slot * KDA_FINALIZE_VECTOR_FP32_BYTES).ReinterpretCast<float>();
        auto dqBase = UbBytes(KDA_FINALIZE_UB_STAGE7_DQ_BASE).ReinterpretCast<float>();
        auto dgBase = UbBytes(KDA_FINALIZE_UB_STAGE7_DG_BASE).ReinterpretCast<float>();
        auto exp2Gk = UbBytes(KDA_FINALIZE_UB_STAGE7_EXP2_GK).ReinterpretCast<float>();
        auto q = UbBytes(KDA_FINALIZE_UB_STAGE7_Q).ReinterpretCast<bfloat16_t>();
        auto qRstd = UbBytes(KDA_FINALIZE_UB_STAGE7_Q_RSTD).ReinterpretCast<float>();

        const int64_t token =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws =
            FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsDq = workspace_[ws + KDA_FINALIZE_WS_DQ_BASE].ReinterpretCast<float>();
        auto wsDg = workspace_[ws + KDA_FINALIZE_WS_DG_BASE].ReinterpretCast<float>();
        auto wsExp = workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>();
        const uint32_t vectorElems =
            static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_DIM;
        AscendC::DataCopy(dqBase, wsDq, vectorElems);
        AscendC::DataCopy(dgBase, wsDg, vectorElems);
        AscendC::DataCopy(exp2Gk, wsExp, vectorElems);
        AscendC::DataCopy(q, q_[token], vectorElems);
        if (tiling_->hasQkL2Norm != 0U) {
            AscendC::DataCopyExtParams scalarCopy{
                1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> scalarPad{
                true, 0,
                0, 0.0f};
            AscendC::DataCopyPad(qRstd, qRstd_[token / KDA_FINALIZE_DIM],
                                scalarCopy, scalarPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage7VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dgBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqLocalRaw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dqBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(exp2Gk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()),
            tiling_->hasQkL2Norm, static_cast<uint16_t>(chunk.validRows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(dq_[token], q, vectorElems);
        AscendC::DataCopy(wsDg, dgBase, vectorElems);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    __aicore__ inline void RunStage9(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZV_READY_BASE + slot);
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + slot);
        auto left = UbBytes(slot * 32 * 1024).ReinterpretCast<float>();
        auto right = UbBytes(64 * 1024 + slot * 32 * 1024).ReinterpretCast<float>();
        // Separate scalar scratch survives until all right raw slots die.
        auto db = UbBytes(240 * 1024 + slot * 256).ReinterpretCast<float>();
        auto k = UbBytes(128 * 1024).ReinterpretCast<bfloat16_t>();
        auto e = UbBytes(144 * 1024).ReinterpretCast<float>();
        auto dg = UbBytes(176 * 1024).ReinterpretCast<float>();
        auto beta = UbBytes(208 * 1024).ReinterpretCast<bfloat16_t>();
        const uint32_t elems = chunk.validRows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        auto wsDg = workspace_[ws + KDA_FINALIZE_WS_DG_BASE].ReinterpretCast<float>();
        AscendC::DataCopy(k, k_[token], elems);
        AscendC::DataCopy(e, workspace_[ws + KDA_FINALIZE_WS_EXP2_GK].ReinterpretCast<float>(), elems);
        AscendC::DataCopy(dg, wsDg, elems);
        AscendC::DataCopyPad(beta, beta_[token / KDA_FINALIZE_DIM],
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<bfloat16_t>{false, 0, 0, static_cast<bfloat16_t>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage9VF(
            reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(e.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(beta.GetPhyAddr()), chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2_);
    }

    __aicore__ inline void RunStage10(
        const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
        uint32_t slot, int64_t coreIdx, uint64_t groupGeneration)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
        auto delta = UbBytes(slot * 32 * 1024).ReinterpretCast<float>();
        auto dbDelta = UbBytes(240 * 1024 + slot * 256).ReinterpretCast<float>();
        auto base = UbBytes(64 * 1024).ReinterpretCast<float>();
        auto k = UbBytes(96 * 1024).ReinterpretCast<bfloat16_t>();
        auto db = UbBytes(112 * 1024).ReinterpretCast<float>();
        auto rs = UbBytes(113 * 1024).ReinterpretCast<float>();
        auto dbOut = UbBytes(114 * 1024).ReinterpretCast<bfloat16_t>();
        const uint32_t elems = chunk.validRows * KDA_FINALIZE_DIM;
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        AscendC::DataCopy(base, workspace_[ws + KDA_FINALIZE_WS_DK_BASE].ReinterpretCast<float>(), elems);
        AscendC::DataCopy(db, workspace_[ws + KDA_FINALIZE_WS_DB_BASE].ReinterpretCast<float>(), KDA_FINALIZE_CHUNK);
        if (tiling_->hasQkL2Norm != 0U) {
            AscendC::DataCopy(k, k_[token], elems);
            AscendC::DataCopyPad(rs, kRstd_[token / KDA_FINALIZE_DIM],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        FinalizeStage10VF(
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(dbOut.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(delta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(base.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(rs.GetPhyAddr()),
            tiling_->hasQkL2Norm, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopy(dk_[token], k, elems);
        AscendC::DataCopyPad(dBeta_[token / KDA_FINALIZE_DIM], dbOut,
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(bfloat16_t)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage7Mte3ToMte2_);
    }

    AscendC::GlobalTensor<float> kRstd_;
    AscendC::GlobalTensor<float> rawG_, dtBias_, dG_;
    AscendC::GlobalTensor<DTYPE_A_LOG> aLog_;
    AscendC::GlobalTensor<bfloat16_t> dk_;
    AscendC::GlobalTensor<bfloat16_t> dBeta_;
    AscendC::GlobalTensor<bfloat16_t> q_;
    AscendC::GlobalTensor<bfloat16_t> k_;
    AscendC::GlobalTensor<bfloat16_t> v_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<bfloat16_t> beta_;
    AscendC::GlobalTensor<bfloat16_t> h_;
    AscendC::GlobalTensor<bfloat16_t> dh_;
    AscendC::GlobalTensor<float> dAqk_;
    AscendC::GlobalTensor<float> dqRaw_;
    AscendC::GlobalTensor<float> qRstd_;
    AscendC::GlobalTensor<bfloat16_t> dq_;
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
    AscendC::TEventID stage5VToMte2_;
    AscendC::TEventID stage7Mte3ToMte2_;
    AscendC::TEventID gateVToMte2_;
    uint32_t zBPublishCount_[KDA_FINALIZE_AIV_SLOTS];
    uint32_t subBlockNum_ = KDA_FINALIZE_AIV_COUNT;
    uint32_t subBlockIdx_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_VECTOR_H
