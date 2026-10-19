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
    uint64_t q, k, gk, raw, aLog, bias, beta, qRstd, kRstd, h, dh, v, dqRaw;
};

class VectorStage1 {
public:
    __aicore__ inline void Init(const VectorInputs &input, TPipe &pipe)
    {
        pipe_ = &pipe;
        pipe.InitBuffer(buffer_, 188 * 1024);
        ub_ = buffer_.Get<uint8_t>();
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
        scalarReady_ = pipe.AllocEventID<HardEvent::V_S>();
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

    __aicore__ inline void Run(uint32_t rows, float lowerBound, GM_ADDR relay, GM_ADDR snapshot)
    {
        const uint32_t sub = GetSubBlockIdx();
        const uint32_t first = sub * 32;
        const uint32_t active = rows > first ? (rows - first < 32 ? rows - first : 32) : 0;
        const uint32_t lastOwner = rows > 32 ? 1 : 0;
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
            DataCopyPad(small[64], qr_[first], copy, padding);
            DataCopyPad(small[96], kr_[first], copy, padding);
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
        if (sub == lastOwner) {
            StateRows(rh);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        GlobalTensor<float> capture;
        capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot));
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(capture[first * 128], e, 32 * 128);
        DataCopy(capture[8448 + first * 128], factor, 32 * 128);
        DataCopy(capture[16640 + first * 128], x, 32 * 128);
        if (sub == lastOwner) {
            DataCopy(capture[8192], rh, 128);
        }
        if (sub == 0) {
            DataCopy(capture[8320], gLast, 128);
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
        DataCopy(betaFull, beta_, 64);
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
        SetFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(phaseFree_);
        SetFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        WaitFlag<HardEvent::V_MTE2>(vectorPhaseFree_);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 12288].ReinterpretCast<float>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, source[32 * 1024 / 4 + (first + slot * 16) * 128], 16 * 128);
            DataCopy(input[2048].ReinterpretCast<bfloat16_t>(), v_[uint64_t(first + slot * 16) * 128], 16 * 128);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        GlobalTensor<bfloat16_t> dv;
        dv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(snapshot + 99328));
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
            for (uint32_t row = 0; row < 16; ++row) {
                Muls(value[row * 128], input[row * 128], small.GetValue(slot * 16 + row), 128);
            }
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
            PipeBarrier<PIPE_V>();
            Cast(output, value, RoundMode::CAST_RINT, 16 * 128);
            SetFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[slot]);
            DataCopy(dv[(first + slot * 16) * 128], output, 16 * 128);
            SetFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        }
        GlobalTensor<float> db;
        db.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot + 115712));
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(db[first], small[32], 32);
        SetFlag<HardEvent::MTE3_V>(outputFree_[0]);
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
            DataCopy(input[2048], dqRaw_[uint64_t(first + slot * 8) * 128], 8 * 128);
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
            for (uint32_t row = 0; row < 8; ++row) {
                Muls(term[row * 128], e[offset + row * 128], small.GetValue(tile * 8 + row), 128);
            }
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
                Muls(term[row * 128], kf[row * 128], small.GetValue(tile * 8 + row), 128);
            }
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
                DataCopy(input[2048], dqRaw_[uint64_t(next)], 8 * 128);
                SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            }
        }
        GlobalTensor<float> capture;
        capture.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(snapshot + 124160));
        WaitFlag<HardEvent::MTE3_V>(outputFree_[0]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputReady_[0]);
        DataCopy(capture[first * 128], dq, 32 * 128);
        DataCopy(capture[8192 + first * 128], dk, 32 * 128);
        DataCopy(capture[16384 + first * 128], dg, 32 * 128);
        DataCopy(capture[24576 + first], small[32], 32);
        DataCopy(source[152 * 1024 / 4 + GetSubBlockIdx() * 128], stateSum, 128);
        DataCopy(capture[24640 + GetSubBlockIdx() * 128], stateSum, 128);
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
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(phaseFree_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vectorPhaseFree_);
    }

private:
    __aicore__ inline void StateRows(LocalTensor<float> result)
    {
        constexpr uint32_t ELEMENTS = 16 * 128;
        auto hf = ub_[148 * 1024].ReinterpretCast<float>();
        auto df = ub_[156 * 1024].ReinterpretCast<float>();
        auto product = ub_[164 * 1024].ReinterpretCast<float>();
        auto partial = ub_[172 * 1024].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            DataCopy(input, h_[slot * ELEMENTS], ELEMENTS);
            DataCopy(input[ELEMENTS], dh_[slot * ELEMENTS], ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < 8; ++tile) {
            const uint32_t slot = tile & 1;
            auto input = ub_[132 * 1024 + slot * 8192].ReinterpretCast<bfloat16_t>();
            WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
            Cast(hf, input, RoundMode::CAST_NONE, ELEMENTS);
            Cast(df, input[ELEMENTS], RoundMode::CAST_NONE, ELEMENTS);
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < 8) {
                WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                DataCopy(input, h_[(tile + 2) * ELEMENTS], ELEMENTS);
                DataCopy(input[ELEMENTS], dh_[(tile + 2) * ELEMENTS], ELEMENTS);
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
    TBuf<TPosition::VECCALC> buffer_;
    LocalTensor<uint8_t> ub_;
    GlobalTensor<bfloat16_t> q_, k_, h_, dh_, v_;
    GlobalTensor<float> g_, raw_, aLog_, bias_, beta_, qr_, kr_, dqRaw_;
    TEventID inputReady_[2], inputFree_[2], outputReady_[2], outputFree_[2];
    TEventID scalarReady_, phaseFree_, vectorPhaseFree_;
};
} // namespace KdaFinalizeCatlass
#endif
