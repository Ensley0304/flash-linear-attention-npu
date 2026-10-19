/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef KDA_FINALIZE_CATLASS_STAGE0_H
#define KDA_FINALIZE_CATLASS_STAGE0_H

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace KdaFinalizeCatlass {

using namespace AscendC;
using Bf16Matrix = Catlass::Gemm::GemmType<bfloat16_t, Catlass::layout::RowMajor>;
constexpr FixpipeConfig FIX_NZ = {CO2Layout::NZ, false};
constexpr uint32_t KIB = 1024;

// Stage 0 owns the formulas and handoffs; helpers only load and multiply tiles.
class CubeStage0 {
public:
    __aicore__ inline void Init(TPipe &pipe)
    {
        pipe_ = &pipe;
        pipe.InitBuffer(l1Buffer_, 400 * KIB);
        pipe.InitBuffer(l0ABuffer_, 64 * KIB);
        pipe.InitBuffer(l0BBuffer_, 64 * KIB);
        pipe.InitBuffer(l0CBuffer_, 64 * KIB);
        inputReady_ = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        residentReady_ = pipe.AllocEventID<HardEvent::FIX_MTE1>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            operandReady_[slot] = pipe.AllocEventID<HardEvent::MTE1_M>();
            operandFree_[slot] = pipe.AllocEventID<HardEvent::M_MTE1>();
            resultReady_[slot] = pipe.AllocEventID<HardEvent::M_FIX>();
            resultFree_[slot] = pipe.AllocEventID<HardEvent::FIX_M>();
            SetFlag<HardEvent::M_MTE1>(operandFree_[slot]);
            SetFlag<HardEvent::FIX_M>(resultFree_[slot]);
        }
    }

    __aicore__ inline void Run(GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
                              GM_ADDR dvScan, GM_ADDR relay, GM_ADDR dwCapture, uint32_t rows)
    {
        auto l1 = l1Buffer_.Get<uint8_t>();
        auto a = l1[0].ReinterpretCast<bfloat16_t>();
        auto dw = l1[16 * KIB].ReinterpretCast<bfloat16_t>();
        auto hh = l1[64 * KIB].ReinterpretCast<bfloat16_t>();
        auto ddh = l1[96 * KIB].ReinterpretCast<bfloat16_t>();
        auto vn = l1[128 * KIB].ReinterpretCast<bfloat16_t>();
        auto ds = l1[144 * KIB].ReinterpretCast<bfloat16_t>();
        auto vv = l1[160 * KIB].ReinterpretCast<bfloat16_t>();
        Load(a, akk, 64, 64, rows, rows);
        Load(hh, h, 128, 128, 128, 128);
        Load(ddh, dh, 128, 128, 128, 128);
        Load(vn, vNew, 64, 128, rows, 128);
        Load(ds, dvScan, 64, 128, rows, 128);
        Load(vv, v, 64, 128, rows, 128);
        SetFlag<HardEvent::MTE2_MTE1>(inputReady_);
        WaitFlag<HardEvent::MTE2_MTE1>(inputReady_);

        GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        GlobalTensor<bfloat16_t> capture;
        capture.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dwCapture));
        auto vectorFix = FixpipeParamsV220(128, 64, 64, 128, false);
        vectorFix.quantPre = QuantMode_t::NoQuant;

        // Independent GEMMs use separate L0A/B/C slots.
        Operands<false, true>(vn, ddh, 64, 128, 128, 0);
        Operands<true, false>(a, ds, 64, 128, 64, 1);
        auto state = Multiply(64, 128, 128, 0);
        auto dv = Multiply(64, 128, 64, 1);
        Operands<false, true>(ds, hh, 64, 128, 128, 0);
        WaitFlag<HardEvent::M_FIX>(resultReady_[0]);
        Fixpipe<float, float, CFG_ROW_MAJOR>(output, state, vectorFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[0]);

        auto dwResult = Multiply(64, 128, 128, 0);
        Operands<false, true>(ds, vv, 64, 64, 128, 1);
        WaitFlag<HardEvent::M_FIX>(resultReady_[1]);
        Fixpipe<float, float, CFG_ROW_MAJOR>(output[32 * KIB / sizeof(float)], dv, vectorFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[1]);

        auto zv = Multiply(64, 64, 128, 1);
        WaitFlag<HardEvent::M_FIX>(resultReady_[0]);
        auto dwFix = FixpipeParamsV220(128, 64, 64, 64, false);
        dwFix.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, FIX_NZ>(dw, dwResult, dwFix);
        // Development-only snapshot; production dW remains in BF16 NZ L1.
        auto dumpFix = vectorFix;
        dumpFix.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(capture, dwResult, dumpFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[0]);

        WaitFlag<HardEvent::M_FIX>(resultReady_[1]);
        auto squareFix = FixpipeParamsV220(64, 64, 64, 64, false);
        squareFix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(output[64 * KIB / sizeof(float)], zv, squareFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[1]);
    }

    __aicore__ inline void Stage2(GM_ADDR relay)
    {
        auto l1 = l1Buffer_.Get<uint8_t>();
        auto a = l1.ReinterpretCast<bfloat16_t>();
        auto dw = l1[16 * KIB].ReinterpretCast<bfloat16_t>();
        auto ke = l1[32 * KIB].ReinterpretCast<bfloat16_t>();
        // kE is already padded by both AIVs; dW stays in its Stage 0 L1 slot.
        Load(ke, relay + 80 * KIB, 64, 128, 64, 128);
        SetFlag<HardEvent::MTE2_MTE1>(inputReady_);
        WaitFlag<HardEvent::MTE2_MTE1>(inputReady_);
        SetFlag<HardEvent::FIX_MTE1>(residentReady_);
        WaitFlag<HardEvent::FIX_MTE1>(residentReady_);
        Operands<true, false>(a, dw, 64, 128, 64, 0);
        Operands<false, true>(dw, ke, 64, 64, 128, 1);
        auto dkg = Multiply(64, 128, 64, 0);
        auto zw = Multiply(64, 64, 128, 1);
        GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay));
        WaitFlag<HardEvent::M_FIX>(resultReady_[0]);
        auto vectorFix = FixpipeParamsV220(128, 64, 64, 128, false);
        vectorFix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(output[96 * KIB / sizeof(float)], dkg, vectorFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[0]);
        WaitFlag<HardEvent::M_FIX>(resultReady_[1]);
        auto squareFix = FixpipeParamsV220(64, 64, 64, 64, false);
        squareFix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(output[128 * KIB / sizeof(float)], zw, squareFix);
        SetFlag<HardEvent::FIX_M>(resultFree_[1]);
    }

    __aicore__ inline void Stage4(GM_ADDR relay, GM_ADDR capture)
    {
        auto l1 = l1Buffer_.Get<uint8_t>();
        auto a = l1.ReinterpretCast<bfloat16_t>();
        auto zb = l1[48 * KIB].ReinterpretCast<bfloat16_t>();
        auto tza = l1[56 * KIB].ReinterpretCast<bfloat16_t>();
        Load(zb, relay + 144 * KIB, 64, 64, 64, 64);
        SetFlag<HardEvent::MTE2_MTE1>(inputReady_);
        WaitFlag<HardEvent::MTE2_MTE1>(inputReady_);
        Operands<false, true>(zb, a, 64, 64, 64, 0);
        auto result = Multiply(64, 64, 64, 0);
        WaitFlag<HardEvent::M_FIX>(resultReady_[0]);
        auto fix = FixpipeParamsV220(64, 64, 64, 64, false);
        fix.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, FIX_NZ>(tza, result, fix);
        GlobalTensor<bfloat16_t> dump;
        dump.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(capture));
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(dump, result, fix);
        SetFlag<HardEvent::FIX_M>(resultFree_[0]);
    }

    __aicore__ inline void Stage6(GM_ADDR relay)
    {
        auto l1 = l1Buffer_.Get<uint8_t>();
        auto a = l1.ReinterpretCast<bfloat16_t>();
        auto tza = l1[56 * KIB].ReinterpretCast<bfloat16_t>();
        SetFlag<HardEvent::FIX_MTE1>(residentReady_);
        WaitFlag<HardEvent::FIX_MTE1>(residentReady_);
        Operands<true, false>(a, tza, 64, 64, 64, 1);
        auto result = Multiply(64, 64, 64, 1);
        WaitFlag<HardEvent::M_FIX>(resultReady_[1]);
        GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(relay + 160 * KIB));
        auto fix = FixpipeParamsV220(64, 64, 64, 64, false);
        fix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<float, float, CFG_ROW_MAJOR>(output, result, fix);
        SetFlag<HardEvent::FIX_M>(resultFree_[1]);
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::M_MTE1>(operandFree_[slot]);
            WaitFlag<HardEvent::FIX_M>(resultFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE1_M>(operandReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::M_MTE1>(operandFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::M_FIX>(resultReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::FIX_M>(resultFree_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(inputReady_);
        pipe_->ReleaseEventID<HardEvent::FIX_MTE1>(residentReady_);
    }

private:
    __aicore__ inline void Load(LocalTensor<bfloat16_t> destination, GM_ADDR source, uint32_t rows,
                               uint32_t columns, uint32_t validRows, uint32_t validColumns)
    {
        GlobalTensor<bfloat16_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(source));
        if (validRows != rows || validColumns != columns) {
            InitConstValue(destination, InitConstValueParams<bfloat16_t>(
                                            1, rows * columns * sizeof(bfloat16_t) / 32, 0, bfloat16_t(0)));
            PipeBarrier<PIPE_MTE2>();
        }
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = validRows;
        copy.dValue = validColumns;
        copy.srcDValue = columns;
        copy.dstNzC0Stride = rows;
        copy.dstNzNStride = 1;
        DataCopy(destination, input, copy);
    }

    template <bool TRANSPOSE_A, bool TRANSPOSE_B>
    __aicore__ inline void Operands(LocalTensor<bfloat16_t> a, LocalTensor<bfloat16_t> b,
                                   uint32_t m, uint32_t n, uint32_t k, uint32_t slot)
    {
        auto aa = l0ABuffer_.Get<bfloat16_t>()[slot * 32 * KIB / sizeof(bfloat16_t)];
        auto bb = l0BBuffer_.Get<bfloat16_t>()[slot * 32 * KIB / sizeof(bfloat16_t)];
        WaitFlag<HardEvent::M_MTE1>(operandFree_[slot]);
        LoadData2DParams load{};
        if constexpr (TRANSPOSE_A) {
            load.repeatTimes = (m / 16) * (k / 16);
            load.srcStride = 1;
            load.ifTranspose = true;
            LoadData(aa, a, load);
        } else {
            load.repeatTimes = k / 16;
            load.srcStride = m / 16;
            for (uint32_t row = 0; row < m / 16; ++row) {
                LoadData(aa[row * (k / 16) * 256], a[row * 256], load);
            }
        }
        if constexpr (TRANSPOSE_B) {
            load.repeatTimes = (k / 16) * (n / 16);
            load.srcStride = 1;
            load.ifTranspose = false;
            LoadData(bb, b, load);
        } else {
            load.repeatTimes = n / 16;
            load.srcStride = k / 16;
            load.ifTranspose = true;
            for (uint32_t row = 0; row < k / 16; ++row) {
                LoadData(bb[row * (n / 16) * 256], b[row * 256], load);
            }
        }
        SetFlag<HardEvent::MTE1_M>(operandReady_[slot]);
    }

    __aicore__ inline LocalTensor<float> Multiply(uint32_t m, uint32_t n, uint32_t k, uint32_t slot)
    {
        auto a = l0ABuffer_.Get<bfloat16_t>()[slot * 32 * KIB / sizeof(bfloat16_t)];
        auto b = l0BBuffer_.Get<bfloat16_t>()[slot * 32 * KIB / sizeof(bfloat16_t)];
        auto c = l0CBuffer_.Get<float>()[slot * 32 * KIB / sizeof(float)];
        WaitFlag<HardEvent::MTE1_M>(operandReady_[slot]);
        WaitFlag<HardEvent::FIX_M>(resultFree_[slot]);
        Catlass::Gemm::Tile::TileMmad<Catlass::Arch::AtlasA2, Bf16Matrix, Bf16Matrix, void> multiply;
        multiply(c, a, b, m, n, k);
        SetFlag<HardEvent::M_MTE1>(operandFree_[slot]);
        SetFlag<HardEvent::M_FIX>(resultReady_[slot]);
        return c;
    }

    TPipe *pipe_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::A2> l0ABuffer_;
    TBuf<TPosition::B2> l0BBuffer_;
    TBuf<TPosition::CO1> l0CBuffer_;
    TEventID inputReady_, residentReady_;
    TEventID operandReady_[2], operandFree_[2], resultReady_[2], resultFree_[2];
};

} // namespace KdaFinalizeCatlass
#endif
