// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace KDA::Arch22::Fused32 {
constexpr uint64_t CUBE_INPUT_BYTES=128*128*4;
constexpr uint64_t CUBE_SERVICE_BYTES=32+3*CUBE_INPUT_BYTES;
__aicore__ inline void CubeObserver(uint64_t commands) {
    using namespace AscendC;
    for(uint64_t i=0;i<commands;++i) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(3);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);
    }
}
__aicore__ inline void CubeService(GM_ADDR commandAddress) {
    using namespace AscendC;
    GlobalTensor<float> command;command.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(commandAddress));
    GM_ADDR a=commandAddress+32;
    GM_ADDR b=a+CUBE_INPUT_BYTES;
    GM_ADDR c=b+CUBE_INPUT_BYTES;
    TPipe pipe;
    TBuf<TPosition::A1> l1;
    TBuf<TPosition::A2> ab;
    TBuf<TPosition::B2> bb;
    TBuf<TPosition::CO1> cb;
    pipe.InitBuffer(l1, 2 * 128 * 128 * sizeof(float));
    pipe.InitBuffer(ab, 128 * 128 * sizeof(float));
    pipe.InitBuffer(bb, 128 * 128 * sizeof(float));
    pipe.InitBuffer(cb, 128 * 128 * sizeof(float));
    GlobalTensor<float> ga, gb, gc;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    auto la = l1.Get<float>();
    auto lb = la[128 * 128];
    auto aa = ab.Get<float>();
    auto ba = bb.Get<float>();
    auto ca = cb.Get<float>();
    ga.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    gb.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SetHF32Mode(false);
    while (true) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);
        DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(command);
        const uint32_t m=command.GetValue(0), n=command.GetValue(1), k=command.GetValue(2), mode=command.GetValue(3);
        if (!m) { CrossCoreSetFlag<0x2, PIPE_FIX>(2); break; }
        // Deliberately serial primitive baseline; no claimed pipeline overlap.
        Nd2NzParams cp{};
        const bool ta = mode & 1, tb = mode & 2;
        cp.ndNum = 1; cp.nValue = ta ? k : m; cp.dValue = ta ? m : k;
        cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue; cp.dstNzNStride = 1;
        DataCopy(la, ga, cp);
        cp.nValue = tb ? n : k; cp.dValue = tb ? k : n;
        cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue;
        DataCopy(lb, gb, cp);
        SetFlag<HardEvent::MTE2_MTE1>(0); WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<float> tr{};
        tr.l1H = 1;
        tr.strideW = tr.strideH = tr.filterW = tr.filterH = 1;
        tr.dilationFilterW = tr.dilationFilterH = 1; tr.enTranspose = true;
        LoadData2DParams ld{};
        if (ta) {
            tr.l1W = k; tr.channelSize = m;
            tr.kExtension = m; tr.mExtension = k;
            LoadData(aa, la, tr);
        } else {
            ld.repeatTimes = k / 8; ld.srcStride = m / 16;
            for (uint32_t row = 0; row < m / 16; ++row) {
                LoadData(aa[row * k * 16], la[row * 128], ld);
                PipeBarrier<PIPE_MTE1>();
            }
        }
        if (tb) {
            ld.repeatTimes = k * n / 128; ld.srcStride = 1;
            LoadData(ba, lb, ld);
        } else {
            tr.l1W = k; tr.channelSize = n;
            tr.kExtension = n; tr.mExtension = k;
            LoadData(ba, lb, tr);
        }
        SetFlag<HardEvent::MTE1_M>(0); WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams mm{};
        mm.m = m; mm.n = n; mm.k = k; mm.cmatrixInitVal = true;
        mm.kDirectionAlign = ta;
        Mmad(ca, aa, ba, mm);
        SetFlag<HardEvent::M_FIX>(0); WaitFlag<HardEvent::M_FIX>(0);
        auto fix = FixpipeParamsV220(n, m, m, n, false);
        Fixpipe<float, float, CFG_ROW_MAJOR>(gc, ca, fix);
        CrossCoreSetFlag<0x2, PIPE_FIX>(2);
        // Last consumers release result and input slots before the next iteration.
        SetFlag<HardEvent::FIX_M>(0); WaitFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(0); WaitFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::MTE1_MTE2>(0); WaitFlag<HardEvent::MTE1_MTE2>(0);
    }
    PipeBarrier<PIPE_ALL>();

}
}
