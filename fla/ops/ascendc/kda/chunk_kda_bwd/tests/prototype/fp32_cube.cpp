// Native FP32 Cube feasibility probe. Not routed by the public operator.
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

// Packed ND inputs: A[m,k], B[k,n]. Probe dimensions are aligned and <=128.
template<bool MIXED>
__aicore__ inline void CubeProbe(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR out,
                                 uint32_t m, uint32_t n, uint32_t k, uint32_t repeats, uint32_t mode)
{
    if constexpr (MIXED) {
        if ASCEND_IS_AIV {
            TPipe pipe;
            TQue<TPosition::VECIN, 1> input;
            TQue<TPosition::VECOUT, 1> output;
            pipe.InitBuffer(input, 1, 128 * 128 * 4 / 2);
            pipe.InitBuffer(output, 1, 128 * 128 * 4 / 2);
            GlobalTensor<float> src, dst;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
            src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            const uint32_t count = m * n / 2;
            const uint32_t offset = GetSubBlockIdx() * count;
            CrossCoreSetFlag<0x2, PIPE_MTE2>(3);
            for (uint32_t iter = 0; iter < repeats; ++iter) {
                CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);
                auto x = input.AllocTensor<float>();
                DataCopy(x, src[offset], count);
                input.EnQue(x);
                x = input.DeQue<float>();
                auto y = output.AllocTensor<float>();
                Adds(y, x, 0.0F, count);
                output.EnQue(y);
                input.FreeTensor(x);
                y = output.DeQue<float>();
                DataCopy(dst[iter * m * n + offset], y, count);
                output.FreeTensor(y);
                // Both AIVs retire their reads before the AIC can overwrite c.
                if (iter + 1 < repeats) CrossCoreSetFlag<0x2, PIPE_MTE3>(3);
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    if ASCEND_IS_AIC {
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
    SetHF32Mode(false);
    for (uint32_t iter = 0; iter < repeats; ++iter) {
        if constexpr (MIXED) CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);
        // Deliberately serial primitive baseline; no claimed pipeline overlap.
        Nd2NzParams cp{};
        const bool ta = mode & 1, tb = mode & 2;
        cp.ndNum = 1; cp.nValue = ta ? k : m; cp.dValue = ta ? m : k;
        cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue; cp.dstNzNStride = 1;
        DataCopy(la, ga[MIXED ? iter * m * k : 0], cp);
        cp.nValue = tb ? n : k; cp.dValue = tb ? k : n;
        cp.srcDValue = cp.dValue; cp.dstNzC0Stride = cp.nValue;
        DataCopy(lb, gb[MIXED ? iter * k * n : 0], cp);
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
        if constexpr (MIXED) CrossCoreSetFlag<0x2, PIPE_FIX>(2);
        // Last consumers release result and input slots before the next iteration.
        SetFlag<HardEvent::FIX_M>(0); WaitFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(0); WaitFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::MTE1_MTE2>(0); WaitFlag<HardEvent::MTE1_MTE2>(0);
    }
    PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ void fp32_cube(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                    uint32_t m, uint32_t n, uint32_t k, uint32_t repeats, uint32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CubeProbe<false>(a, b, c, nullptr, m, n, k, repeats, mode);
}

__global__ __aicore__ void fp32_mixed(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR out,
                                     uint32_t m, uint32_t n, uint32_t k, uint32_t repeats, uint32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    CubeProbe<true>(a, b, c, out, m, n, k, repeats, mode);
}

extern "C" void run_fp32_cube(void* a, void* b, void* c, uint32_t m, uint32_t n,
                               uint32_t k, uint32_t repeats, uint32_t mode, void* stream)
{
    fp32_cube<<<1, nullptr, stream>>>(static_cast<uint8_t*>(a), static_cast<uint8_t*>(b),
                                    static_cast<uint8_t*>(c), m, n, k, repeats, mode);
}

extern "C" void run_fp32_mixed(void* a, void* b, void* c, void* out, uint32_t m, uint32_t n,
                                uint32_t k, uint32_t repeats, uint32_t mode, void* stream)
{
    fp32_mixed<<<1, nullptr, stream>>>(static_cast<uint8_t*>(a), static_cast<uint8_t*>(b),
                                     static_cast<uint8_t*>(c), static_cast<uint8_t*>(out), m, n, k, repeats, mode);
}
