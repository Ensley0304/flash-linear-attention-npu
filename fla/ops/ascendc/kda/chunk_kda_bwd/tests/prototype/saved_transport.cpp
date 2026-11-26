// Exercises the production saved-backward Cube transport, not the KDA formula.
#include "acl/acl.h"
#include "kernel_operator.h"
#include "../../op_kernel/arch22/fused32/saved_cube.h"
#include "../../op_kernel/arch22/fused32/saved_pack.h"
using namespace AscendC;
using namespace KDA::Arch22::Fused32::Saved;

__aicore__ inline void ConsumeResult(TBuf<TPosition::VECCALC> &result,
    GlobalTensor<float> &gr, GlobalTensor<float> &go, uint32_t j,
    uint32_t m, uint32_t n, uint32_t count, uint32_t lane) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(ResultReady(j));
        auto x = result.Get<float>();
        DataCopy(x, gr[(j % BANKS) * RESULT_BANK_BYTES / 4 + lane * m * n / 2], m * n / 2);
        if (j + BANKS < count) CrossCoreSetFlag<0x2, PIPE_MTE2>(ResultFree(j));

        // This copy retains every generation, rather than checking only the last.
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(go[uint64_t(j) * m * n + lane * m * n / 2], x, m * n / 2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    }

__aicore__ inline void VectorTransport(GM_ADDR a, GM_ADDR b, GM_ADDR scratch, GM_ADDR out,
                                      uint32_t m, uint32_t n, uint32_t k, uint32_t count, uint32_t mode) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> input;
    TBuf<TPosition::VECCALC> result;
    pipe.InitBuffer(input, 32 * 1024);
    pipe.InitBuffer(result, 16 * 1024);
    GlobalTensor<float> ga, gb, gi, gr, go;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
    gi.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
    gr.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + BANKS * INPUT_BANK_BYTES));
    go.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
    gi.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    gr.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    const uint32_t lane = GetSubBlockIdx();
    uint64_t packet = 0;
    for (uint32_t j = 0; j < count; ++j) {
        if (j >= BANKS) ConsumeResult(result, gr, go, j - BANKS, m, n, count, lane);
        for (uint32_t begin = 0; begin < k; begin += 128) {
            const uint32_t kr = k - begin < 128 ? k - begin : 128;
            if (packet >= BANKS) CrossCoreWaitFlag<0x2, PIPE_MTE3>(InputFree(packet));
            auto x = input.Get<float>();
            const bool ta = mode & 1, tb = mode & 2;
            const Matrix av{uint64_t(j) * m * k * 4, ta ? 1U : k, ta ? m : 1U};
            const Matrix bv{uint64_t(j) * k * n * 4, tb ? 1U : n, tb ? k : 1U};
            LoadPacketSlice(x, ga, Slice(av, 0, begin, m, kr, m, kr, ta, lane));
            LoadPacketSlice(x[4096], gb, Slice(bv, begin, 0, kr, n, kr, n, tb, lane));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            const uint64_t base = (packet % BANKS) * INPUT_BANK_BYTES / 4;
            DataCopy(gi[base + lane * m * kr / 2], x, m * kr / 2);
            DataCopy(gi[base + 8192 + lane * kr * n / 2], x[4096], kr * n / 2);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(InputReady(packet));
            // LoadPacketSlice begins with Vector zero-fill. Releasing only
            // MTE2 permits that fill to overwrite a still-running MTE3 read.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            ++packet;
        }
    }
    for (uint32_t j = count > BANKS ? count - BANKS : 0; j < count; ++j) ConsumeResult(result, gr, go, j, m, n, count, lane);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(RETIRE_READY);
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
}

extern "C" __global__ __aicore__ void saved_transport(GM_ADDR a, GM_ADDR b, GM_ADDR scratch,
    GM_ADDR out, uint32_t m, uint32_t n, uint32_t k, uint32_t count, uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        Cube cube;
        cube.Init(scratch, scratch + BANKS * INPUT_BANK_BYTES,
                  uint64_t(count) * ReductionPackets(k), count);
        for (uint32_t j = 0; j < count; ++j) cube.Multiply(m, n, k, mode & 1, mode & 2);
        cube.End();
        cube.RetireRecord();
    }
    if ASCEND_IS_AIV { VectorTransport(a, b, scratch, out, m, n, k, count, mode); }
}
extern "C" void run_saved_transport(void *a, void *b, void *scratch, void *out,
    uint32_t m, uint32_t n, uint32_t k, uint32_t count, uint32_t mode, void *stream) {
    saved_transport<<<1, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)scratch,
                                           (GM_ADDR)out, m, n, k, count, mode);
}

__global__ __aicore__ void saved_handshake() {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
        CrossCoreSetFlag<0x2, PIPE_FIX>(4);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_READY);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
    }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(4);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(RETIRE_READY);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
    }
}
extern "C" void run_saved_handshake(void *stream) {
    saved_handshake<<<1, nullptr, stream>>>();
}

__global__ __aicore__ void saved_io(GM_ADDR a, GM_ADDR b, GM_ADDR scratch,
    GM_ADDR out, uint32_t m, uint32_t n, uint32_t k, uint32_t count, uint32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        const uint64_t total = uint64_t(count) * ReductionPackets(k);
        uint64_t packet = 0;
        for (uint32_t j = 0; j < count; ++j) {
            for (uint32_t begin = 0; begin < k; begin += 128, ++packet) {
                CrossCoreWaitFlag<0x2, PIPE_MTE2>(InputReady(packet));
                if (packet + BANKS < total) CrossCoreSetFlag<0x2, PIPE_MTE2>(InputFree(packet));
            }
            if (j >= BANKS) CrossCoreWaitFlag<0x2, PIPE_FIX>(ResultFree(j));
            CrossCoreSetFlag<0x2, PIPE_FIX>(ResultReady(j));
        }
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(RETIRE_READY);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(RETIRE_ACK);
    }
    if ASCEND_IS_AIV { VectorTransport(a, b, scratch, out, m, n, k, count, mode); }
}
extern "C" void run_saved_io(void *a, void *b, void *scratch, void *out,
    uint32_t m, uint32_t n, uint32_t k, uint32_t count, uint32_t mode, void *stream) {
    saved_io<<<1, nullptr, stream>>>(static_cast<uint8_t *>(a), static_cast<uint8_t *>(b),
        static_cast<uint8_t *>(scratch), static_cast<uint8_t *>(out), m, n, k, count, mode);
}
