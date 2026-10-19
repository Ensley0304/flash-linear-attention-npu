/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef KDA_FINALIZE_CATLASS_FINALIZE_H
#define KDA_FINALIZE_CATLASS_FINALIZE_H

#include "stage0.h"
#include "stage1.h"

namespace KdaFinalizeCatlass {
struct Configuration {
    uint64_t batches, heads, tokens, chunks, variable, tokenMajor, norm, betaMode;
    float scale, lowerBound;
};

struct Arguments {
    VectorInputs input;
    VectorOutputs output;
    uint64_t akk, vNew, dvScan, metadata, relay, partial, syncBase;
    Configuration config;
};
} // namespace KdaFinalizeCatlass

__global__ __aicore__ void FinalizeCatlass(KdaFinalizeCatlass::Arguments args)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC;
    using namespace KdaFinalizeCatlass;
    SetSyncBaseAddr(args.syncBase);
    const auto cfg = args.config;
    GlobalTensor<uint64_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(args.metadata));
    if ASCEND_IS_AIC {
        const uint64_t core = GetBlockIdx();
        TPipe pipe;
        CubeStage0 cube;
        cube.Init(pipe);
        auto relay = reinterpret_cast<GM_ADDR>(args.relay + core * 512 * 1024);
        for (uint64_t work = core; work < cfg.chunks * cfg.heads; work += GetBlockNum()) {
            const uint64_t chunk = work / cfg.heads;
            const uint64_t head = work % cfg.heads;
            const uint64_t seq = cfg.variable ? 0 : metadata.GetValue(chunk * 4);
            const uint64_t first = metadata.GetValue(chunk * 4 + 1);
            const uint32_t rows = metadata.GetValue(chunk * 4 + 2);
            const uint64_t token = (seq * cfg.heads + head) * cfg.tokens + first;
            const uint64_t state = (chunk * cfg.heads + head) * 128 * 128 * 2;
            cube.Run(reinterpret_cast<GM_ADDR>(args.input.v + token * 256),
                     reinterpret_cast<GM_ADDR>(args.akk + token * 128),
                     reinterpret_cast<GM_ADDR>(args.vNew + token * 256),
                     reinterpret_cast<GM_ADDR>(args.input.h + state),
                     reinterpret_cast<GM_ADDR>(args.input.dh + state),
                     reinterpret_cast<GM_ADDR>(args.dvScan + token * 256), relay, nullptr, rows);
            // Stage 0 reuses only the retired early relay and L1 regions.
            // Later stages still wait until both AIVs finish the previous work.
            if (work != core) {
                CrossCoreWaitFlag(9);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(0);
            CrossCoreWaitFlag(1);
            cube.Stage2(relay);
            CrossCoreSetFlag<2, PIPE_FIX>(2);
            CrossCoreWaitFlag(3);
            cube.Stage4(relay, nullptr);
            cube.Stage6(relay);
            CrossCoreSetFlag<2, PIPE_FIX>(4);
            CrossCoreWaitFlag(5);
            cube.Stage8(relay, 1);
            CrossCoreSetFlag<2, PIPE_FIX>(7);
            CrossCoreWaitFlag(6);
            cube.Stage8(relay, 0);
            CrossCoreSetFlag<2, PIPE_FIX>(8);
        }
        if (core < cfg.chunks * cfg.heads) {
            CrossCoreWaitFlag(9);
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        const uint64_t core = GetBlockIdx() / GetSubBlockNum();
        const uint32_t sub = GetSubBlockIdx();
        TPipe pipe;
        VectorStage1 vector;
        vector.Init(args.input, pipe);
        auto relay = reinterpret_cast<GM_ADDR>(args.relay + core * 512 * 1024);
        for (uint64_t work = core; work < cfg.chunks * cfg.heads; work += GetBlockNum()) {
            const uint64_t chunk = work / cfg.heads;
            const uint64_t head = work % cfg.heads;
            const uint64_t seq = cfg.variable ? 0 : metadata.GetValue(chunk * 4);
            const uint64_t first = metadata.GetValue(chunk * 4 + 1);
            const uint32_t rows = metadata.GetValue(chunk * 4 + 2);
            const uint64_t token = (seq * cfg.heads + head) * cfg.tokens + first;
            const uint64_t state = (chunk * cfg.heads + head) * 128 * 128 * 2;
            auto input = args.input;
            input.q += token * 256;
            input.k += token * 256;
            input.v += token * 256;
            input.gk += token * 512;
            input.raw += token * 512;
            input.dqRaw += token * 512;
            input.daq += token * 256;
            input.beta += token * 4;
            input.qRstd = cfg.norm ? input.qRstd + token * 4 : 0;
            input.kRstd = cfg.norm ? input.kRstd + token * 4 : 0;
            input.h += state;
            input.dh += state;
            input.aLog += head * 4;
            input.bias += head * 512;
            const uint64_t tokenMajor = (seq * cfg.tokens + first) * cfg.heads + head;
            const uint64_t vectorToken = cfg.tokenMajor ? tokenMajor : token;
            auto output = args.output;
            output.dq += vectorToken * 256;
            output.dk += vectorToken * 256;
            output.dv += vectorToken * 256;
            output.dg += tokenMajor * 512;
            output.db += tokenMajor * 4;
            output.partial = args.partial + work * 544;
            output.vectorStride = cfg.tokenMajor ? cfg.heads * 128 : 128;
            output.headStride = cfg.heads;
            vector.Bind(input, output);
            vector.Run(rows, cfg.lowerBound, relay, nullptr);
            CrossCoreWaitFlag(0);
            CrossCoreWaitFlag(2);
            vector.Stage3(rows, relay, nullptr);
            vector.Stage5(cfg.scale, relay, nullptr);
            CrossCoreWaitFlag(4);
            vector.Stage7(rows, relay, nullptr);
            CrossCoreWaitFlag(7);
            if (sub == 0) {
                CrossCoreWaitFlag(8);
            }
            vector.Stage9(rows, cfg.norm != 0, cfg.betaMode, relay, nullptr);
            if (sub == 1) {
                CrossCoreWaitFlag(8);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        vector.End();
    }
}

__global__ __aicore__ void FinalizeCatlassReduce(GM_ADDR partial, GM_ADDR da, GM_ADDR dbias,
                                              uint64_t heads, uint64_t chunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, sumBuffer;
    pipe.InitBuffer(inputBuffer, 2 * 32 * 544);
    pipe.InitBuffer(sumBuffer, 544);
    auto input = inputBuffer.Get<float>();
    auto sum = sumBuffer.Get<float>();
    GlobalTensor<float> source, outA, outBias;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(partial));
    outA.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(da));
    outBias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbias));
    const uint64_t head = GetBlockIdx();
    const uint64_t tiles = (chunks + 31) / 32;
    TEventID ready[2], free[2];
    for (uint32_t slot = 0; slot < 2; ++slot) {
        ready[slot] = pipe.AllocEventID<HardEvent::MTE2_V>();
        free[slot] = pipe.AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(free[slot]);
        if (slot < tiles) {
            const uint16_t count = chunks - slot * 32 < 32 ? chunks - slot * 32 : 32;
            WaitFlag<HardEvent::V_MTE2>(free[slot]);
            DataCopyPad(input[slot * 32 * 136], source[(uint64_t(slot) * 32 * heads + head) * 136],
                        {count, 544, static_cast<uint32_t>((heads - 1) * 544), 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(ready[slot]);
        }
    }
    Duplicate(sum, 0.0f, 136);
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        const uint32_t slot = tile & 1;
        const uint32_t count = chunks - tile * 32 < 32 ? chunks - tile * 32 : 32;
        WaitFlag<HardEvent::MTE2_V>(ready[slot]);
        for (uint32_t row = 0; row < count; ++row) {
            PipeBarrier<PIPE_V>();
            Add(sum, sum, input[slot * 32 * 136 + row * 136], 136);
        }
        SetFlag<HardEvent::V_MTE2>(free[slot]);
        if (tile + 2 < tiles) {
            const uint16_t next = chunks - (tile + 2) * 32 < 32 ? chunks - (tile + 2) * 32 : 32;
            WaitFlag<HardEvent::V_MTE2>(free[slot]);
            DataCopyPad(input[slot * 32 * 136], source[((tile + 2) * 32 * heads + head) * 136],
                        {next, 544, static_cast<uint32_t>((heads - 1) * 544), 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(ready[slot]);
        }
    }
    auto writeReady = pipe.AllocEventID<HardEvent::V_MTE3>();
    auto writeDone = pipe.AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::V_MTE3>(writeReady);
    WaitFlag<HardEvent::V_MTE3>(writeReady);
    DataCopyPad(outA[head], sum, {1, 4, 0, 0, 0});
    DataCopy(outBias[head * 128], sum[8], 128);
    SetFlag<HardEvent::MTE3_V>(writeDone);
    WaitFlag<HardEvent::MTE3_V>(writeDone);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        WaitFlag<HardEvent::V_MTE2>(free[slot]);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(ready[slot]);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(free[slot]);
    }
    pipe.ReleaseEventID<HardEvent::V_MTE3>(writeReady);
    pipe.ReleaseEventID<HardEvent::MTE3_V>(writeDone);
}
#endif
