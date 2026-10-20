/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef KDA_FINALIZE_CATLASS_STAGE1_H
#define KDA_FINALIZE_CATLASS_STAGE1_H

#include "kernel_operator.h"
#include <cstdint>

namespace KdaFinalizeCatlass {
using namespace AscendC;

struct VectorInputs {
    uint64_t q, k, gk, raw, aLog, bias, beta, qRstd, kRstd, h, dh, v, dqRaw, daq;
};

struct VectorOutputs {
    uint64_t dq, dk, dv, dg, db, partial;
    uint32_t vectorStride, headStride;
};

class VectorStage1 {
public:
    __aicore__ inline void Init(const VectorInputs &input, TPipe &pipe)
    {
        pipe_ = &pipe;
        pipe.InitBuffer(buffer_, 188 * 1024);
        ub_ = buffer_.Get<uint8_t>();
        auto diagonalIndices = ub_[128 * 1024 + 936 * 4].ReinterpretCast<int32_t>();
        CreateVecIndex(diagonalIndices, 0, 32);
        PipeBarrier<PIPE_V>();
        Muls(diagonalIndices, diagonalIndices, int32_t(65 * 4), 32);
        PipeBarrier<PIPE_V>();
        Bind(input, {});
        scalarReady_ = pipe.AllocEventID<HardEvent::V_S>();
        scalarWritten_ = pipe.AllocEventID<HardEvent::S_V>();
        auto lowerMask = ub_[185 * 1024].ReinterpretCast<uint64_t>();
        for (uint32_t row = 0; row < 32; ++row) {
            lowerMask.SetValue(row, (uint64_t(1) << (GetSubBlockIdx() * 32 + row)) - 1);
        }
        SetFlag<HardEvent::S_V>(scalarWritten_);
        WaitFlag<HardEvent::S_V>(scalarWritten_);
        Duplicate(ub_[185 * 1024 + 256].ReinterpretCast<float>(), 0.0f, 32);
        PipeBarrier<PIPE_V>();
        phaseFree_ = pipe.AllocEventID<HardEvent::MTE3_MTE2>();
        vectorPhaseFree_ = pipe.AllocEventID<HardEvent::V_MTE2>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            inputReady_[slot] = pipe.AllocEventID<HardEvent::MTE2_V>();
            inputFree_[slot] = pipe.AllocEventID<HardEvent::V_MTE2>();
            outputReady_[slot] = pipe.AllocEventID<HardEvent::V_MTE3>();
            outputFree_[slot] = pipe.AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    __aicore__ inline void Bind(const VectorInputs &input, const VectorOutputs &output)
    {
        outputs_ = output;
        hasNorm_ = input.qRstd != 0;
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input.q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input.k));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input.h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input.dh));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input.v));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.gk));
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.raw));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.aLog));
        bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.bias));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.beta));
        qr_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.qRstd));
        kr_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.kRstd));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.dqRaw));
        daq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input.daq));
    }

    __aicore__ inline void Run(uint32_t rows, float lowerBound, GM_ADDR relay, GM_ADDR snapshot)
    {
        rows_ = rows;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t first = sub * 32;
        const uint32_t active = rows > first ? (rows - first < 32 ? rows - first : 32) : 0;
        auto q = ub_.ReinterpretCast<bfloat16_t>();
        auto k = ub_[8 * 1024].ReinterpretCast<bfloat16_t>();
        auto g = ub_[16 * 1024].ReinterpretCast<float>();
        auto e = ub_[32 * 1024].ReinterpretCast<float>();
        auto factor = ub_[96 * 1024].ReinterpretCast<float>();
        auto x = ub_[112 * 1024].ReinterpretCast<float>();
        auto small = ub_[128 * 1024].ReinterpretCast<float>();
        auto rh = small[160];
        auto gLast = small[288];
        auto bias = small[416]; // Replaced by band center in Stage 7.
        auto scalar = small[544]; // Suffix storage is not live yet.

        // Initialize aligned rows before DMA; a tail may end between 32-byte blocks.
        Duplicate(small, 0.0f, 32);
        Duplicate(small[64], 0.0f, 32);
        Duplicate(small[96], 0.0f, 32);
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
        if (active != 0) {
            DataCopy(q, q_[uint64_t(first) * 128], active * 128);
            DataCopy(k, k_[uint64_t(first) * 128], active * 128);
            DataCopy(g, g_[uint64_t(first) * 128], active * 128);
            DataCopyExtParams copy{1, active * 4, 0, 0, 0};
            DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>((8 - active % 8) % 8), 0};
            DataCopyPad(small, beta_[first], copy, padding);
            if (hasNorm_) {
                DataCopyPad(small[64], qr_[first], copy, padding);
                DataCopyPad(small[96], kr_[first], copy, padding);
            }
        }
        DataCopy(gLast, g_[uint64_t(rows - 1) * 128], 128);
        DataCopy(bias, bias_, 128);
        DataCopyPad(scalar, aLog_, {1, 4, 0, 0, 0}, {true, 0, 7, 0});
        SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
        SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
        if (active < 32) {
            Duplicate(q[active * 128].ReinterpretCast<uint16_t>(), uint16_t(0), (32 - active) * 128);
            Duplicate(k[active * 128].ReinterpretCast<uint16_t>(), uint16_t(0), (32 - active) * 128);
            Duplicate(g[active * 128], 0.0f, (32 - active) * 128);
        }
        Duplicate(e, 0.0f, 32 * 128);
        Exp(scalar, scalar, 8);
        SetFlag<HardEvent::V_S>(scalarReady_);
        WaitFlag<HardEvent::V_S>(scalarReady_);
        const float ea = scalar.GetValue(0);
        // One 32-byte block per row, reused by all beta-weighted Vector stages.
        Brcb(ub_[186 * 1024].ReinterpretCast<float>(), small, 4, {1, 8});

        GlobalTensor<bfloat16_t> ke;
        ke.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(relay + 80 * 1024));
        auto value = ub_[132 * 1024].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t row = slot * 16;
            const uint32_t count = active > row ? (active - row < 16 ? active - row : 16) * 128 : 0;
            auto output = ub_[140 * 1024 + slot * 4096].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            Duplicate(output.ReinterpretCast<uint16_t>(), uint16_t(0), 16 * 128);
            if (count != 0) {
                Cast(value, k[row * 128], RoundMode::CAST_NONE, count);
                Muls(e[row * 128], g[row * 128], 0.6931471805599453f, count);
                PipeBarrier<PIPE_V>();
                Exp(e[row * 128], e[row * 128], count);
                PipeBarrier<PIPE_V>();
                Mul(value, value, e[row * 128], count);
                PipeBarrier<PIPE_V>();
                Cast(output, value, RoundMode::CAST_RINT, count);
            }
            SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            DataCopy(ke[(first + row) * 128], output, 16 * 128);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Both halves publish even when one half is empty.
        CrossCoreSetFlag<2, PIPE_MTE3>(1);
        SetFlag<HardEvent::MTE3_MTE2>(phaseFree_);

        // Gate inputs use the two disjoint resident 16-row regions as DMA slots.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t row = slot * 16;
            const uint32_t count = active > row ? (active - row < 16 ? active - row : 16) * 128 : 0;
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            if (count != 0) {
                DataCopy(x[row * 128], raw_[uint64_t(first + row) * 128], count);
            }
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t row = slot * 16;
            const uint32_t valid = active > row ? (active - row < 16 ? active - row : 16) : 0;
            const uint32_t count = valid * 128;
            auto raw = x[row * 128];
            auto derivative = factor[row * 128];
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            if (valid != 0) {
                Add(raw, raw, bias, 64, valid, {1, 1, 1, 16, 16, 0});
                Add(raw[64], raw[64], bias[64], 64, valid, {1, 1, 1, 16, 16, 0});
                PipeBarrier<PIPE_V>();
                Muls(derivative, raw, ea, count);
                PipeBarrier<PIPE_V>();
                Abs(derivative, derivative, count);
                PipeBarrier<PIPE_V>();
                Muls(derivative, derivative, -1.0f, count);
                PipeBarrier<PIPE_V>();
                Exp(derivative, derivative, count);
                PipeBarrier<PIPE_V>();
                Adds(value, derivative, 1.0f, count);
                Muls(derivative, derivative, lowerBound * ea, count);
                PipeBarrier<PIPE_V>();
                Mul(value, value, value, count);
                PipeBarrier<PIPE_V>();
                Div(derivative, derivative, value, count);
            }
            if (valid < 16) {
                Duplicate(raw[count], 0.0f, (16 - valid) * 128);
                Duplicate(derivative[count], 0.0f, (16 - valid) * 128);
            }
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
        }

        // rH shares scratch addresses with the retired kE output slots.
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        StateRows(rh, sub * 64);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        GlobalTensor<float> sharedRh;
        sharedRh.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay + 153 * 1024));
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(sharedRh[sub * 64], rh, 64);
        if (snapshot != nullptr) {
            GlobalTensor<float> capture;
            capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot));
            DataCopy(capture[first * 128], e, 32 * 128);
            DataCopy(capture[8448 + first * 128], factor, 32 * 128);
            DataCopy(capture[16640 + first * 128], x, 32 * 128);
            DataCopy(capture[8192 + sub * 64], rh, 64);
            if (sub == 0) {
                DataCopy(capture[8320], gLast, 128);
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    __aicore__ inline void Stage3(uint32_t rows, GM_ADDR relay, GM_ADDR snapshot)
    {
        // Stage 1 rH still reads this scratch until its Vector work completes.
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        const uint32_t first = GetSubBlockIdx() * 32;
        auto small = ub_[128 * 1024].ReinterpretCast<float>();
        auto betaFull = ub_[156 * 1024].ReinterpretCast<float>();
        GlobalTensor<float> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        GlobalTensor<bfloat16_t> zb;
        zb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(relay + 144 * 1024));
        WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
        LoadRows(betaFull, beta_, 0, 64, 1);
        SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
        SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, source[64 * 1024 / 4 + (first + slot * 16) * 64], 16 * 64);
            DataCopy(input[1024], source[128 * 1024 / 4 + (first + slot * 16) * 64], 16 * 64);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        auto result = ub_[148 * 1024].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<float>();
            auto output = ub_[152 * 1024 + slot * 2048].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            PipeBarrier<PIPE_V>();
            Sub(result, input, input[1024], 16 * 64);
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            PipeBarrier<PIPE_V>();
            Mul(result, result, betaFull, 64, 16, {1, 1, 1, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < 16; ++row) {
                const uint32_t globalRow = first + slot * 16 + row;
                uint64_t mask[2] = {globalRow < rows ? (~uint64_t(0) << globalRow) : ~uint64_t(0), 0};
                Duplicate(result[row * 64], 0.0f, mask, 1, 1, 8);
            }
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            PipeBarrier<PIPE_V>();
            Cast(output, result, RoundMode::CAST_RINT, 16 * 64);
            SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            DataCopy(zb[(first + slot * 16) * 64], output, 16 * 64);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(3);
        // The following dv pipeline reuses the Zb staging area only after both readers finish.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        SetFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 12288].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, source[32 * 1024 / 4 + (first + slot * 16) * 128], 16 * 128);
            LoadRows(input[2048].ReinterpretCast<bfloat16_t>(), v_, first + slot * 16, 16, 128);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        GlobalTensor<bfloat16_t> dv;
        dv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(snapshot != nullptr ?
            snapshot + 99328 : reinterpret_cast<GM_ADDR>(outputs_.dv)));
        auto value = ub_[164 * 1024].ReinterpretCast<float>();
        auto product = ub_[172 * 1024].ReinterpretCast<float>();
        auto partial = ub_[180 * 1024].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 12288].ReinterpretCast<float>();
            auto output = ub_[156 * 1024 + slot * 4096].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            Cast(value, input[2048].ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, 16 * 128);
            PipeBarrier<PIPE_V>();
            Mul(product, value, input, 16 * 128);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(partial, product, 64, 16, 1, 1, 16);
            WholeReduceSum(partial[64], product[64], 64, 16, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(small[32 + slot * 16], partial, partial[64], 16);
            auto betaRows = ub_[186 * 1024].ReinterpretCast<float>()[slot * 16 * 8];
            Mul(value, input, betaRows, 64, 16, {1, 1, 0, 16, 16, 1});
            Mul(value[64], input[64], betaRows, 64, 16, {1, 1, 0, 16, 16, 1});
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            PipeBarrier<PIPE_V>();
            Cast(output, value, RoundMode::CAST_RINT, 16 * 128);
            SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            const uint32_t start = first + slot * 16;
            if (snapshot != nullptr) {
                DataCopy(dv[start * 128], output, 16 * 128);
            } else if (start < rows) {
                const uint16_t count = rows - start < 16 ? rows - start : 16;
                DataCopyPad(dv[uint64_t(start) * outputs_.vectorStride], output,
                            {count, 256, 0, (outputs_.vectorStride - 128) * 2, 0});
            }
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        if (snapshot != nullptr) {
            GlobalTensor<float> db;
            db.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot + 115712));
            WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
            SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
            DataCopy(db[first], small[32], 32);
            SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
        }
    }

    __aicore__ inline void Stage5(float scale, GM_ADDR relay, GM_ADDR snapshot)
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        const uint32_t first = GetSubBlockIdx() * 32;
        auto g = ub_[16 * 1024].ReinterpretCast<float>();
        auto e = ub_[32 * 1024].ReinterpretCast<float>();
        auto dq = ub_[48 * 1024].ReinterpretCast<float>();
        auto dk = ub_[64 * 1024].ReinterpretCast<float>();
        auto dg = ub_[80 * 1024].ReinterpretCast<float>();
        auto small = ub_[128 * 1024].ReinterpretCast<float>();
        auto stateSum = small[672];
        auto qf = ub_[156 * 1024].ReinterpretCast<float>();
        auto kf = ub_[160 * 1024].ReinterpretCast<float>();
        auto state = ub_[164 * 1024].ReinterpretCast<float>();
        auto term = ub_[168 * 1024].ReinterpretCast<float>();
        auto product = ub_[172 * 1024].ReinterpretCast<float>();
        auto partial = ub_[176 * 1024].ReinterpretCast<float>();
        GlobalTensor<float> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        SetFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        Duplicate(stateSum, 0.0f, 128);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 12288].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, source[(first + slot * 8) * 128], 8 * 128);
            DataCopy(input[1024], source[96 * 1024 / 4 + (first + slot * 8) * 128], 8 * 128);
            LoadRows(input[2048], dqRaw_, first + slot * 8, 8, 128);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t offset = tile * 8 * 128;
            auto input = ub_[132 * 1024 + slot * 12288].ReinterpretCast<float>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            PipeBarrier<PIPE_V>();
            Cast(qf, ub_.ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 8 * 128);
            Cast(kf, ub_[8 * 1024].ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 8 * 128);
            Sub(state, small[288], g[offset], 64, 8, {1, 1, 1, 16, 0, 16});
            Sub(state[64], small[352], g[offset + 64], 64, 8, {1, 1, 1, 16, 0, 16});
            Muls(dq[offset], input[2048], scale, 8 * 128);
            PipeBarrier<PIPE_V>();
            Muls(state, state, 0.6931471805599453f, 8 * 128);
            Mul(dq[offset], dq[offset], e[offset], 8 * 128);
            auto betaRows = ub_[186 * 1024].ReinterpretCast<float>()[tile * 8 * 8];
            Mul(term, e[offset], betaRows, 64, 8, {1, 1, 0, 16, 16, 1});
            Mul(term[64], e[offset + 64], betaRows, 64, 8, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();
            Exp(state, state, 8 * 128);
            Mul(term, term, input[1024], 8 * 128);
            PipeBarrier<PIPE_V>();
            Mul(state, state, input, 8 * 128);
            Mul(product, kf, e[offset], 8 * 128);
            PipeBarrier<PIPE_V>();
            Sub(dk[offset], state, term, 8 * 128);
            Mul(product, product, input[1024], 8 * 128);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(partial, product, 64, 8, 1, 1, 16);
            WholeReduceSum(partial[64], product[64], 64, 8, 1, 1, 16);
            Mul(product, kf, state, 8 * 128);
            Mul(dg[offset], qf, dq[offset], 8 * 128);
            PipeBarrier<PIPE_V>();
            Add(partial, partial, partial[64], 8);
            Sub(dg[offset], dg[offset], product, 8 * 128);
            for (uint32_t row = 0; row < 8; ++row) {
                Add(stateSum, stateSum, product[row * 128], 128);
                PipeBarrier<PIPE_V>();
            }
            Mul(term, kf, betaRows, 64, 8, {1, 1, 0, 16, 16, 1});
            Mul(term[64], kf[64], betaRows, 64, 8, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();
            Sub(small[32 + tile * 8], small[32 + tile * 8], partial, 8);
            Mul(term, term, e[offset], 8 * 128);
            PipeBarrier<PIPE_V>();
            Mul(term, term, input[1024], 8 * 128);
            PipeBarrier<PIPE_V>();
            Sub(dg[offset], dg[offset], term, 8 * 128);
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < 4) {
                WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                const uint32_t next = (first + (tile + 2) * 8) * 128;
                DataCopy(input, source[next], 8 * 128);
                DataCopy(input[1024], source[96 * 1024 / 4 + next], 8 * 128);
                LoadRows(input[2048], dqRaw_, first + (tile + 2) * 8, 8, 128);
                SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            }
        }
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(source[152 * 1024 / 4 + GetSubBlockIdx() * 128], stateSum, 128);
        if (snapshot != nullptr) {
            GlobalTensor<float> capture;
            capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot + 124160));
            DataCopy(capture[first * 128], dq, 32 * 128);
            DataCopy(capture[8192 + first * 128], dk, 32 * 128);
            DataCopy(capture[16384 + first * 128], dg, 32 * 128);
            DataCopy(capture[24576 + first], small[32], 32);
            DataCopy(capture[24640 + GetSubBlockIdx() * 128], stateSum, 128);
        }
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
    }

    __aicore__ inline void Stage7(uint32_t rows, GM_ADDR relay, GM_ADDR snapshot)
    {
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t first = sub * 32;
        auto small = ub_[128 * 1024].ReinterpretCast<float>();
        auto aq = ub_[132 * 1024].ReinterpretCast<float>();
        auto ak = ub_[140 * 1024].ReinterpretCast<float>();
        auto zv = ub_[148 * 1024].ReinterpretCast<float>();
        GlobalTensor<float> source, capture;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        if (snapshot != nullptr) {
            capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot));
        }
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        SetFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
        LoadRows(aq, daq_, first, 32, 64);
        DataCopy(ak, source[160 * 1024 / 4 + first * 64], 32 * 64);
        if (snapshot != nullptr) {
            DataCopy(zv, source[64 * 1024 / 4 + first * 64], 32 * 64);
        }
        SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
        SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        if (snapshot != nullptr) {
            DataCopy(capture[223744 / 4 + first * 64], ak, 32 * 64);
            DataCopy(capture[240128 / 4 + first * 64], zv, 32 * 64);
        }
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        auto diagonalIndices = small[936].ReinterpretCast<uint32_t>();
        Gather(small[128], aq, diagonalIndices, first * 4, 32);
        PipeBarrier<PIPE_V>();
        Brcb(ub_[187 * 1024].ReinterpretCast<float>(), small[128], 4, {1, 8});
        Muls(ak, ak, -1.0f, 32 * 64);
        PipeBarrier<PIPE_V>();
        SetCmpMask(ub_[185 * 1024 + 256].ReinterpretCast<float>());
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(64);
        Select(aq, ub_[185 * 1024], aq, 32, {1, 1, 1, 8, 8, 8});
        Select(ak, ub_[185 * 1024], ak, 32, {1, 1, 1, 8, 8, 8});
        ResetMask();
        const uint32_t active = rows > first ? (rows - first < 32 ? rows - first : 32) : 0;
        if (active < 32) {
            PipeBarrier<PIPE_V>();
            Duplicate(aq[active * 64], 0.0f, (32 - active) * 64);
            Duplicate(ak[active * 64], 0.0f, (32 - active) * 64);
        }
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(source[64 * 1024 / 4 + first * 64], aq, 32 * 64);
        DataCopy(source[160 * 1024 / 4 + first * 64], ak, 32 * 64);
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);

        auto center = ub_[180 * 1024].ReinterpretCast<float>();
        auto edge = ub_[180 * 1024 + 512].ReinterpretCast<float>();
        // S5 consumed e; reuse its 16 KiB for the larger S7 tile.
        auto kf = ub_[32 * 1024].ReinterpretCast<float>();
        auto delta = ub_[40 * 1024].ReinterpretCast<float>();
        auto g = ub_[16 * 1024].ReinterpretCast<float>();
        for (int band = 1; band >= 0; --band) {
            const uint32_t bandFirst = band * 32;
            const bool validBand = rows > bandFirst;
            const uint32_t last = validBand ? (rows < bandFirst + 32 ? rows - 1 : bandFirst + 31) : 0;
            WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
            if (validBand) {
                DataCopy(center, g_[uint64_t(bandFirst) * 128], 128);
                DataCopy(edge, g_[uint64_t(last) * 128], 128);
            }
            SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
            WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
            if (validBand) {
                Add(center, center, edge, 128);
                PipeBarrier<PIPE_V>();
                Muls(center, center, 0.5f, 128);
            } else {
                Duplicate(center, 0.0f, 128);
            }
            PipeBarrier<PIPE_V>();
            if (band == sub) {
                Adds(small[416], center, 0.0f, 128);
            }
            for (uint32_t tile = 0; tile < 2; ++tile) {
                const uint32_t slot = tile & 1;
                const uint32_t offset = tile * 16 * 128;
                auto kn = ub_[132 * 1024 + slot * 24576].ReinterpretCast<float>();
                auto qp = kn[2048];
                auto bp = kn[4096];
                WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
                PipeBarrier<PIPE_V>();
                const uint32_t tileFirst = first + tile * 16;
                const uint32_t negativeRows = validBand && tileFirst <= last
                    ? (last + 1 - tileFirst < 16 ? last + 1 - tileFirst : 16) : 0;
                const uint32_t positiveRows = validBand && tileFirst >= bandFirst && tileFirst < rows
                    ? (rows - tileFirst < 16 ? rows - tileFirst : 16) : 0;
                Cast(kf, ub_[8 * 1024].ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 16 * 128);
                if (negativeRows != 0) {
                    Sub(delta, center, g[offset], 64, negativeRows, {1, 1, 1, 16, 0, 16});
                    Sub(delta[64], center[64], g[offset + 64], 64, negativeRows, {1, 1, 1, 16, 0, 16});
                    PipeBarrier<PIPE_V>();
                    Muls(delta, delta, 0.6931471805599453f, negativeRows * 128);
                    PipeBarrier<PIPE_V>();
                    Exp(delta, delta, negativeRows * 128);
                    PipeBarrier<PIPE_V>();
                    Mul(kn, kf, delta, negativeRows * 128);
                }
                if (negativeRows < 16) {
                    Duplicate(kn[negativeRows * 128], 0.0f, (16 - negativeRows) * 128);
                }
                PipeBarrier<PIPE_V>();
                // The positive range extends to the chunk end, including the other half for band 0.
                if (positiveRows != 0) {
                    Cast(qp, ub_.ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, positiveRows * 128);
                    Sub(delta, g[offset], center, 64, positiveRows, {1, 1, 1, 16, 16, 0});
                    Sub(delta[64], g[offset + 64], center[64], 64, positiveRows, {1, 1, 1, 16, 16, 0});
                    auto betaRows = ub_[186 * 1024].ReinterpretCast<float>()[tile * 16 * 8];
                    Mul(bp, kf, betaRows, 64, positiveRows, {1, 1, 0, 16, 16, 1});
                    Mul(bp[64], kf[64], betaRows, 64, positiveRows, {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Muls(delta, delta, 0.6931471805599453f, positiveRows * 128);
                    PipeBarrier<PIPE_V>();
                    Exp(delta, delta, positiveRows * 128);
                    PipeBarrier<PIPE_V>();
                    Mul(qp, qp, delta, positiveRows * 128);
                    Mul(bp, bp, delta, positiveRows * 128);
                }
                if (positiveRows < 16) {
                    Duplicate(qp[positiveRows * 128], 0.0f, (16 - positiveRows) * 128);
                    Duplicate(bp[positiveRows * 128], 0.0f, (16 - positiveRows) * 128);
                }
                SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
                WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
                const uint32_t base = (208 + (1 - band) * 96) * 1024 / 4 + (first + tile * 16) * 128;
                DataCopy(source[base], kn, 16 * 128);
                DataCopy(source[base + 8192], qp, 16 * 128);
                DataCopy(source[base + 16384], bp, 16 * 128);
                SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            }
            SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
            CrossCoreSetFlag<2, PIPE_MTE3>(band == 1 ? 5 : 6);
        }
    }

    __aicore__ inline void Stage9(uint32_t rows, bool norm, uint32_t betaMode, GM_ADDR relay, GM_ADDR snapshot)
    {
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t first = sub * 32;
        const uint32_t active = rows > first ? (rows - first < 32 ? rows - first : 32) : 0;
        auto g = ub_[16 * 1024].ReinterpretCast<float>();
        auto dq = ub_[48 * 1024].ReinterpretCast<float>();
        auto dk = ub_[64 * 1024].ReinterpretCast<float>();
        auto dg = ub_[80 * 1024].ReinterpretCast<float>();
        auto small = ub_[128 * 1024].ReinterpretCast<float>();
        auto suffix = small[544];
        auto partial = small[800];
        auto qf = ub_[16 * 1024].ReinterpretCast<float>();
        auto kf = ub_[24 * 1024].ReinterpretCast<float>();
        auto term = ub_[32 * 1024].ReinterpretCast<float>();
        auto product = ub_[40 * 1024].ReinterpretCast<float>();
        auto reduced = ub_[180 * 1024].ReinterpretCast<float>();
        auto remotePartial = ub_[184 * 1024].ReinterpretCast<float>();
        GlobalTensor<float> source, capture;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        if (snapshot != nullptr) {
            capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot));
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        // Apply the state correction before the local terms, matching CPU association.
        if (sub == (rows > 32 ? 1U : 0U)) {
            auto stateInput = ub_[132 * 1024].ReinterpretCast<float>();
            auto stateSum = ub_[134 * 1024].ReinterpretCast<float>();
            auto factor = ub_[135 * 1024].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
            DataCopy(stateInput, source[152 * 1024 / 4], 384);
            SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
            WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
            Add(stateSum, stateInput, stateInput[128], 128);
            Muls(factor, small[288], 0.6931471805599453f, 128);
            PipeBarrier<PIPE_V>();
            Exp(factor, factor, 128);
            PipeBarrier<PIPE_V>();
            Mul(factor, factor, stateInput[256], 128);
            PipeBarrier<PIPE_V>();
            Add(stateSum, stateSum, factor, 128);
            PipeBarrier<PIPE_V>();
            Add(dg[(active - 1) * 128], dg[(active - 1) * 128], stateSum, 128);
            SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
        }
        // Release state scratch before the two 16-row input slots are filled.
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        const uint32_t base = (400 + (1 - sub) * 48) * 1024 / 4;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 24576].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            for (uint32_t item = 0; item < 3; ++item) {
                DataCopy(input[item * 2048], source[base + item * 4096 + slot * 2048], 2048);
            }
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        Sub(g, g, small[416], 64, 32, {1, 1, 1, 16, 16, 0});
        Sub(g[64], g[64], small[480], 64, 32, {1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
        if (active < 32) {
            Duplicate(g[active * 128], 0.0f, (32 - active) * 128);
        }
        PipeBarrier<PIPE_V>();
        Muls(g, g, 0.6931471805599453f, 4096);
        PipeBarrier<PIPE_V>();
        Exp(g, g, 4096);
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 24576].ReinterpretCast<float>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            Mul(input, input, g[slot * 2048], 2048);
            Mul(input[2048], input[2048], g[slot * 2048], 2048);
            Div(input[4096], input[4096], g[slot * 2048], 2048);
        }
        // Both slots consumed the factors; g is now available for qf/kf.
        PipeBarrier<PIPE_V>();
        for (uint32_t tile = 0; tile < 2; ++tile) {
            const uint32_t offset = tile * 2048;
            auto input = ub_[132 * 1024 + tile * 24576].ReinterpretCast<float>();
            auto localQ = input;
            auto left = input[2048];
            auto right = input[4096];
            PipeBarrier<PIPE_V>();
            Cast(qf, ub_.ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 2048);
            Cast(kf, ub_[8 * 1024].ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            Add(dq[offset], dq[offset], localQ, 2048);
            Mul(product, qf, localQ, 2048);
            auto betaRows = ub_[186 * 1024].ReinterpretCast<float>()[tile * 16 * 8];
            Mul(term, left, betaRows, 64, 16, {1, 1, 0, 16, 16, 1});
            Mul(term[64], left[64], betaRows, 64, 16, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();
            Add(dg[offset], dg[offset], product, 2048);
            Add(dk[offset], dk[offset], term, 2048);
            Sub(term, term, right, 2048);
            Mul(product, kf, left, 2048);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(reduced, product, 64, 16, 1, 1, 16);
            WholeReduceSum(reduced[64], product[64], 64, 16, 1, 1, 16);
            Mul(term, kf, term, 2048);
            Add(dk[offset], dk[offset], right, 2048);
            PipeBarrier<PIPE_V>();
            Add(reduced, reduced, reduced[64], 16);
            Add(dg[offset], dg[offset], term, 2048);
            auto diagRows = ub_[187 * 1024].ReinterpretCast<float>()[tile * 16 * 8];
            Mul(term, kf, diagRows, 64, 16, {1, 1, 0, 16, 16, 1});
            Mul(term[64], kf[64], diagRows, 64, 16, {1, 1, 0, 16, 16, 1});
            Mul(product, qf, diagRows, 64, 16, {1, 1, 0, 16, 16, 1});
            Mul(product[64], qf[64], diagRows, 64, 16, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();
            Add(small[32 + tile * 16], small[32 + tile * 16], reduced, 16);
            Add(dq[offset], dq[offset], term, 2048);
            Add(dk[offset], dk[offset], product, 2048);
            SetFlag<HardEvent::V_MTE2>(inputFree_[tile]);
        }
        Duplicate(suffix, 0.0f, 128);
        if (sub == 0) {
            CrossCoreSetFlag<1, PIPE_MTE3>(11);
            CrossCoreSetFlag<1, PIPE_MTE3>(12);
            CrossCoreWaitFlag(11);
            SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
            WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
            WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
            DataCopy(suffix, source[496 * 1024 / 4], 128);
            SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
            WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
            SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
        }
        PipeBarrier<PIPE_V>();
        if (active != 0) {
            Add(dg[(active - 1) * 128], suffix, dg[(active - 1) * 128], 128);
            PipeBarrier<PIPE_V>();
            for (int row = int(active) - 2; row >= 0; --row) {
                Add(dg[row * 128], dg[(row + 1) * 128], dg[row * 128], 128);
                PipeBarrier<PIPE_V>();
            }
            Adds(suffix, dg, 0.0f, 128);
            PipeBarrier<PIPE_V>();
        }
        if (sub == 1) {
            SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
            DataCopy(source[496 * 1024 / 4], suffix, 128);
            CrossCoreSetFlag<1, PIPE_MTE3>(11);
        }
        Mul(dg, dg, ub_[96 * 1024].ReinterpretCast<float>(), 32 * 128);
        Duplicate(partial, 0.0f, 136);
        PipeBarrier<PIPE_V>();
        auto gateProduct = ub_[132 * 1024].ReinterpretCast<float>();
        auto rowSums = ub_[148 * 1024].ReinterpretCast<float>();
        Mul(gateProduct, dg, ub_[112 * 1024].ReinterpretCast<float>(), 32 * 128);
        // e is retired here; preserve dg for writeback while reducing 32 rows.
        auto biasSum = ub_[32 * 1024].ReinterpretCast<float>();
        Add(biasSum, dg, dg[2048], 2048);
        PipeBarrier<PIPE_V>();
        for (uint32_t count = 1024; count >= 256; count /= 2) {
            Add(biasSum, biasSum, biasSum[count], count);
            PipeBarrier<PIPE_V>();
        }
        Add(partial[8], biasSum, biasSum[128], 128);
        WholeReduceSum(rowSums, gateProduct, 64, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(partial, rowSums, 64, 1, 1, 1, 8);
        if (sub == 1) {
            CrossCoreWaitFlag(11);
            SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
            DataCopy(source[496 * 1024 / 4 + 128], partial, 136);
            CrossCoreSetFlag<1, PIPE_MTE3>(12);
            CrossCoreWaitFlag(12);
        } else {
            CrossCoreWaitFlag(12);
            SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
            WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
            WaitFlag<HardEvent::V_MTE2>(inputFree_[0]);
            DataCopy(remotePartial, source[496 * 1024 / 4 + 128], 136);
            SetFlag<HardEvent::MTE2_V>(inputReady_[0]);
            WaitFlag<HardEvent::MTE2_V>(inputReady_[0]);
            SetFlag<HardEvent::V_MTE2>(inputFree_[0]);
            PipeBarrier<PIPE_V>();
            Add(partial, partial, remotePartial, 136);
            SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
            if (snapshot != nullptr) {
                DataCopy(capture[322304 / 4], partial, 136);
            } else {
                GlobalTensor<float> workPartial;
                workPartial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputs_.partial));
                DataCopy(workPartial, partial, 136);
            }
        }
        // Normalize in FP32, then round once at the final BF16 writeback.
        GlobalTensor<bfloat16_t> finalQ, finalK;
        finalQ.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(snapshot != nullptr ?
            snapshot + 256512 : reinterpret_cast<GM_ADDR>(outputs_.dq)));
        finalK.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(snapshot != nullptr ?
            snapshot + 272896 : reinterpret_cast<GM_ADDR>(outputs_.dk)));
        for (uint32_t tile = 0; tile < 2; ++tile) {
            const uint32_t offset = tile * 2048;
            const uint32_t slot = tile & 1;
            PipeBarrier<PIPE_V>();
            if (norm) {
                Cast(qf, ub_.ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 2048);
                Cast(kf, ub_[8 * 1024].ReinterpretCast<bfloat16_t>()[offset], RoundMode::CAST_NONE, 2048);
                PipeBarrier<PIPE_V>();
                for (uint32_t which = 0; which < 2; ++which) {
                    auto grad = which == 0 ? dq[offset] : dk[offset];
                    auto normalized = which == 0 ? qf : kf;
                    Mul(product, grad, normalized, 2048);
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(reduced, product, 64, 16, 1, 1, 16);
                    WholeReduceSum(reduced[64], product[64], 64, 16, 1, 1, 16);
                    PipeBarrier<PIPE_V>();
                    Add(reduced, reduced, reduced[64], 16);
                    auto rowBroadcast = ub_[182 * 1024].ReinterpretCast<float>();
                    PipeBarrier<PIPE_V>();
                    Brcb(rowBroadcast, reduced, 2, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul(term, normalized, rowBroadcast, 64, 16, {1, 1, 0, 16, 16, 1});
                    Mul(term[64], normalized[64], rowBroadcast, 64, 16, {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Sub(grad, grad, term, 2048);
                    PipeBarrier<PIPE_V>();
                    Brcb(rowBroadcast, small[64 + which * 32 + tile * 16], 2, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul(grad, grad, rowBroadcast, 64, 16, {1, 1, 0, 16, 16, 1});
                    Mul(grad[64], grad[64], rowBroadcast, 64, 16, {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                }
            }
            auto output = ub_[132 * 1024 + slot * 16384].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            Cast(output, dq[offset], RoundMode::CAST_RINT, 2048);
            Cast(output[2048], dk[offset], RoundMode::CAST_RINT, 2048);
            SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            const uint32_t start = first + tile * 16;
            if (snapshot != nullptr) {
                DataCopy(finalQ[first * 128 + offset], output, 2048);
                DataCopy(finalK[first * 128 + offset], output[2048], 2048);
            } else if (start < rows) {
                const uint16_t count = rows - start < 16 ? rows - start : 16;
                const DataCopyExtParams write{count, 256, 0, (outputs_.vectorStride - 128) * 2, 0};
                DataCopyPad(finalQ[uint64_t(start) * outputs_.vectorStride], output, write);
                DataCopyPad(finalK[uint64_t(start) * outputs_.vectorStride], output[2048], write);
            }
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        if (betaMode != 0) {
            Muls(reduced, small, betaMode == 2 ? -0.5f : -1.0f, 32);
            PipeBarrier<PIPE_V>();
            Adds(reduced, reduced, 1.0f, 32);
            PipeBarrier<PIPE_V>();
            Mul(reduced, reduced, small, 32);
            PipeBarrier<PIPE_V>();
            Mul(small[32], small[32], reduced, 32);
        }
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        if (snapshot != nullptr) {
            DataCopy(capture[289280 / 4 + first * 128], dg, 4096);
            DataCopy(capture[322048 / 4 + first], small[32], 32);
        } else if (active != 0) {
            GlobalTensor<float> finalG, finalBeta;
            finalG.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputs_.dg));
            finalBeta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputs_.db));
            DataCopyPad(finalG[uint64_t(first) * outputs_.headStride * 128], dg,
                        {static_cast<uint16_t>(active), 512, 0, (outputs_.headStride - 1) * 512, 0});
            auto packedBeta = ub_[144 * 1024].ReinterpretCast<float>();
            Brcb(packedBeta, small[32], 4, {1, 8});
            SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
            DataCopyPad(finalBeta[uint64_t(first) * outputs_.headStride], packedBeta,
                        {static_cast<uint16_t>(active), 4, 0, (outputs_.headStride - 1) * 4, 0});
        }
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(inputFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outputReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::V_S>(scalarReady_);
        pipe_->ReleaseEventID<HardEvent::S_V>(scalarWritten_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(phaseFree_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vectorPhaseFree_);
    }

private:
    template <typename T>
    __aicore__ inline void LoadRows(LocalTensor<T> destination, GlobalTensor<T> source,
                                   uint32_t first, uint32_t tileRows, uint32_t columns)
    {
        const uint32_t valid = rows_ > first ? (rows_ - first < tileRows ? rows_ - first : tileRows) : 0;
        if (valid != tileRows) {
            // The free token orders DMA; Vector zeroing also needs the previous Vector readers retired.
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, float>::value) {
                Duplicate(destination, 0.0f, tileRows * columns);
            } else {
                Duplicate(destination.template ReinterpretCast<uint16_t>(), uint16_t(0), tileRows * columns);
            }
            SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
            WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        }
        if (valid != 0) {
            const uint32_t count = valid * columns;
            const uint8_t right = (32 / sizeof(T) - count % (32 / sizeof(T))) % (32 / sizeof(T));
            DataCopyPad(destination, source[uint64_t(first) * columns],
                        {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0}, {true, 0, right, 0});
        }
    }

    __aicore__ inline void StateRows(LocalTensor<float> result, uint32_t first)
    {
        constexpr uint32_t ELEMENTS = 16 * 128;
        auto hf = ub_[148 * 1024].ReinterpretCast<float>();
        auto df = ub_[156 * 1024].ReinterpretCast<float>();
        auto product = ub_[164 * 1024].ReinterpretCast<float>();
        auto partial = ub_[172 * 1024].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, h_[uint64_t(first) * 128 + slot * ELEMENTS], ELEMENTS);
            DataCopy(input[ELEMENTS], dh_[uint64_t(first) * 128 + slot * ELEMENTS], ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint32_t slot = tile & 1;
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            Cast(hf, input, RoundMode::CAST_NONE, ELEMENTS);
            Cast(df, input[ELEMENTS], RoundMode::CAST_NONE, ELEMENTS);
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < 4) {
                WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                DataCopy(input, h_[uint64_t(first) * 128 + (tile + 2) * ELEMENTS], ELEMENTS);
                DataCopy(input[ELEMENTS], dh_[uint64_t(first) * 128 + (tile + 2) * ELEMENTS], ELEMENTS);
                SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            }
            PipeBarrier<PIPE_V>();
            Mul(product, hf, df, ELEMENTS);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(partial, product, 64, 16, 1, 1, 16);
            WholeReduceSum(partial[64], product[64], 64, 16, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(result[tile * 16], partial, partial[64], 16);
        }
    }

    TPipe *pipe_;
    VectorOutputs outputs_{};
    uint32_t rows_ = 0;
    bool hasNorm_ = false;
    TBuf<TPosition::VECCALC> buffer_;
    LocalTensor<uint8_t> ub_;
    GlobalTensor<bfloat16_t> q_, k_, h_, dh_, v_;
    GlobalTensor<float> g_, raw_, aLog_, bias_, beta_, qr_, kr_, dqRaw_, daq_;
    TEventID inputReady_[2], inputFree_[2], outputReady_[2], outputFree_[2];
    TEventID scalarReady_, scalarWritten_, phaseFree_, vectorPhaseFree_;
};
} // namespace KdaFinalizeCatlass
#endif
