#ifndef CHUNK_KDA_BWD_WY_VECTOR_H
#define CHUNK_KDA_BWD_WY_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "chunk_kda_bwd_wy_common.h"

namespace KDA {

template <typename BetaT>
class ChunkKdaBwdWyVectorProcess {
public:
    __aicore__ ChunkKdaBwdWyVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
        GM_ADDR db, GM_ADDR dg, GM_ADDR dAkk, GM_ADDR workspace)
        : q_(q), k_(k), v_(v), gk_(gk), beta_(beta), h_(h), dh_(dh),
          dvScan_(dvScan), dq_(dq), dk_(dk), dv_(dv), db_(db), dg_(dg), dAkk_(dAkk),
          workspace_(workspace) {}

    __aicore__ inline void Init(
        const ChunkKdaBwdWyTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q_));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k_));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v_));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk_));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta_));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h_));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh_));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq_));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk_));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv_));
        dvScanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dvScan_));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db_));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        dgBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dg_));
        dAkkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk_));
        dAkkBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dAkk_));
        wsFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_));
        wsBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_));

        // Two entries let MTE2/MTE3 alternate buffers while Vector consumes
        // the preceding tile.  RowReduce uses the unused tail of its output
        // plane, so the full 96 KiB UB budget remains available to IO ping-pong.
        pipe_->InitBuffer(inputQueue_, 2, kIoBytes);
        pipe_->InitBuffer(outputQueue_, 2, kIoBytes);
        pipe_->InitBuffer(arena_, kArenaBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_.stage == kWyFusedStage) {
            ProcessFused();
            return;
        }
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount = (headNum + 1U) / 2U;
        const uint64_t taskGroupCount = static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;

        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(WyWorkspaceFreeFlag(0));
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(WyWorkspaceFreeFlag(1));
        uint32_t localGeneration = 0;
        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, ++localGeneration) {
            const uint32_t taskIdx = static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindow = static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase = headWindow * kWyHeadsPerWindow;
            const uint32_t headCount = headBase + 1U < headNum ? 2U : 1U;
            const WyChunkTask task = GetWyChunkTask(tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;

            if (tiling_.stage == 3) {
                // Keep both heads in the same dependency phase, matching the
                // proven PR190/K4 pipeline rather than completing one head's
                // four-phase handshake before starting the next head.
                for (uint32_t lane = 0; lane < headCount; ++lane) {
                    const uint64_t slot = WyWorkspaceSlotBase(
                        tiling_, coreIdx, localGeneration, lane);
                    const uint32_t head = headBase + lane;
                    BuildKE(task, head, validLen, slot, subBlockIdx, subBlockNum);
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    NegateDW(validLen, slot, subBlockIdx, subBlockNum);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        WyVectorToCubeFlag(lane));
                }
                for (uint32_t lane = 0; lane < headCount; ++lane) {
                    const uint64_t slot = WyWorkspaceSlotBase(
                        tiling_, coreIdx, localGeneration, lane);
                    const uint32_t head = headBase + lane;
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    BuildZbStage(
                        task, head, validLen, slot, subBlockIdx, subBlockNum);
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                    WyWorkspaceFreeFlag(localGeneration));
                continue;
            }

            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(tiling_, coreIdx, localGeneration, lane);
                const uint32_t head = headBase + lane;
                if (tiling_.stage <= 2) {
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    FinishBaseStage(
                        task, head, validLen, slot, subBlockIdx,
                        subBlockNum, static_cast<uint32_t>(tiling_.stage));
                } else if (tiling_.stage == 4) {
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    PersistNegatedDW(
                        task, head, validLen, slot, subBlockIdx, subBlockNum);
                } else if (tiling_.stage == 5) {
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    if (subBlockIdx == 0) {
                        FinishGradientRows(task, head, validLen, slot, 0, validLen);
                        PrepareStateGate(
                            task, head, validLen, slot, subBlockNum);
                        AddPreparedStateGate(task, head, validLen, slot);
                    }
                } else if (tiling_.stage == 6) {
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    PersistTza(
                        task, head, validLen, slot, subBlockIdx, subBlockNum);
                } else if (tiling_.stage == 7) {
                    AscendC::CrossCoreWaitFlag(WyCubeToVectorFlag(lane));
                    FinishDA(task, head, validLen, slot, subBlockIdx, subBlockNum);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(
                WyWorkspaceFreeFlag(localGeneration));
        }
        if (tiling_.stage == 6 || tiling_.stage == 7) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                WyVectorToCubeFlag(0));
        }
    }

private:
    __aicore__ inline void SignalVectorDependency(uint32_t flag)
    {
        // Both AIV sub-blocks own disjoint rows.  Release AIC only after all
        // MTE3 writes required by the actual dependency are globally visible.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    __aicore__ inline void ProcessFused()
    {
        // Four producer flags allow AIC to publish independent S0/S1/S2/S3a
        // results without waiting for AIV after every stage.  They are reused
        // only after the S3a acknowledgement proves that AIV consumed the
        // first wave.  The four acknowledgement flags correspond exactly to
        // the true data dependencies: -dW, Zb, Tza, and final workspace free.
        constexpr uint32_t kS0Ready = 0;
        constexpr uint32_t kS1Ready = 1;
        constexpr uint32_t kS2Ready = 4;
        constexpr uint32_t kS3aReady = 5;
        constexpr uint32_t kNegDwReady = 2;
        constexpr uint32_t kZbReady = 3;
        constexpr uint32_t kTzaReady = 6;
        constexpr uint32_t kTaskDone = 7;

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kWyFusedHeadsPerWindow - 1U) /
            kWyFusedHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;

        uint32_t localGeneration = 0;
        for (uint64_t taskGroupIdx = coreIdx;
             taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, ++localGeneration) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindow =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase =
                headWindow * kWyFusedHeadsPerWindow;
            const uint32_t headCount =
                headBase + kWyFusedHeadsPerWindow <= headNum ?
                kWyFusedHeadsPerWindow : headNum - headBase;
            const WyChunkTask task = GetWyChunkTask(tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;

            // kE has no AIC dependency.  Build it while AIC produces the
            // independent base GEMMs instead of serializing it behind S2.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                BuildKE(task, headBase + lane, validLen, slot,
                        subBlockIdx, subBlockNum);
            }

            // S0: consume dq_raw and finish dq_base.
            AscendC::CrossCoreWaitFlag(kS0Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 0);
            }

            // S1: consume dk_raw and finish dk_state.
            AscendC::CrossCoreWaitFlag(kS1Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 1);
            }

            // S2: consume dVb and write dv/db_base.
            AscendC::CrossCoreWaitFlag(kS2Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 2);
            }

            // S3a: consume dW and publish the negated BF16 tile required by
            // the dependent zW GEMM.
            AscendC::CrossCoreWaitFlag(kS3aReady);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                NegateDW(validLen, slot, subBlockIdx, subBlockNum);
            }
            SignalVectorDependency(kNegDwReady);

            // The S1 state partials are now complete on both AIV sub-blocks.
            // Build the expensive h*dh state-gate term while AIC executes the
            // dependent zW GEMMs, then keep the 128-element result in the
            // otherwise-unused fused dk_raw workspace until S5 writes dg.
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            for (uint32_t lane = subBlockIdx;
                 lane < headCount; lane += subBlockNum) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                PrepareStateGate(
                    task, headBase + lane, validLen, slot, subBlockNum);
            }

            // S3b: consume zW and form the saved BF16 Zb tile.
            AscendC::CrossCoreWaitFlag(kS0Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                BuildZbStage(task, headBase + lane, validLen, slot,
                             subBlockIdx, subBlockNum);
            }
            SignalVectorDependency(kZbReady);

            // Gradient rows remain evenly split across both AIV sub-blocks.
            // Only the lightweight final state add remains on the S5 path;
            // its h*dh reduction was overlapped with AIC above.
            AscendC::CrossCoreWaitFlag(kS1Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                uint32_t begin = 0;
                uint32_t end = 0;
                NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
                FinishGradientRows(task, headBase + lane, validLen,
                                   slot, begin, end);
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            for (uint32_t lane = subBlockIdx;
                 lane < headCount; lane += subBlockNum) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                AddPreparedStateGate(
                    task, headBase + lane, validLen, slot);
            }

            // S6: consume Tza and persist it into dead dv_scan storage.
            AscendC::CrossCoreWaitFlag(kS2Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                PersistTza(task, headBase + lane, validLen, slot,
                           subBlockIdx, subBlockNum);
            }
            SignalVectorDependency(kTzaReady);

            // S7: final dAkk mask/sign/writeback.
            AscendC::CrossCoreWaitFlag(kS3aReady);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishDA(task, headBase + lane, validLen, slot,
                         subBlockIdx, subBlockNum);
            }
            SignalVectorDependency(kTaskDone);

        }
    }

    static constexpr uint32_t kRows = 16;
    static constexpr uint32_t kPlaneElements = kRows * kWyKeyDim;
    static constexpr uint32_t kIoBytes = kPlaneElements * sizeof(float);
    // FinishGradients has the largest live set and uses Plane(0)..Plane(7).
    // Reserving 24 planes exceeded the conservative per-AIV UB budget of the
    // 1AIC:2AIV MIX kernel without providing any reusable live storage.
    static constexpr uint32_t kArenaBytes = 8 * kPlaneElements * sizeof(float);
    static constexpr float kLn2 = 0.69314718055994530942f;
    static_assert(4 * kIoBytes + kArenaBytes <= 96 * 1024,
                   "ChunkKdaBwdWy AIV buffers exceed the A2/A3 UB budget");

    __aicore__ inline AscendC::LocalTensor<float> Plane(uint32_t idx)
    {
        return arena_.Get<float>()[idx * kPlaneElements];
    }

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
        uint32_t count)
    {
        auto in = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(
            in, src, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<float>();
        AscendC::Adds(dst, ready, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<bfloat16_t> src,
        uint32_t count)
    {
        auto in = inputQueue_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(
            in, src,
            {1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<bfloat16_t>();
        AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        auto out = outputQueue_.AllocTensor<float>();
        AscendC::Adds(out, src, 0.0f, count);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyPad(
            dst, ready, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<bfloat16_t> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        auto out = outputQueue_.AllocTensor<bfloat16_t>();
        AscendC::Cast(out, src, AscendC::RoundMode::CAST_RINT, count);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<bfloat16_t>();
        AscendC::DataCopyPad(
            dst, ready,
            {1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void StoreStrided(
        AscendC::GlobalTensor<bfloat16_t> dst,
        AscendC::LocalTensor<float> src, uint32_t rows, uint32_t cols,
        uint32_t physicalCols)
    {
        auto out = outputQueue_.AllocTensor<bfloat16_t>();
        AscendC::Cast(
            out, src, AscendC::RoundMode::CAST_RINT, rows * cols);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
            0,
            static_cast<uint32_t>((physicalCols - cols) * sizeof(bfloat16_t)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Exp2(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        AscendC::Muls(dst, src, kLn2, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBeta(
        AscendC::LocalTensor<float> dst, uint64_t offset, uint32_t count)
    {
        Load(dst, betaGm_[offset], count);
    }

    __aicore__ inline void BroadcastRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> scalar,
        uint32_t rows)
    {
        AscendC::Brcb(dst, scalar, static_cast<uint8_t>((rows + 7U) / 8U), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulRowsByScalar(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        AscendC::LocalTensor<float> broadcast, uint32_t rows, uint32_t cols)
    {
        const uint8_t stride = static_cast<uint8_t>(cols * sizeof(float) / 32);
        for (uint32_t col = 0; col < cols; col += 64) {
            const uint32_t mask = cols - col < 64 ? cols - col : 64;
            AscendC::Mul(dst[col], src[col], broadcast, mask, rows,
                         {1, 1, 0, stride, stride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduce128(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows)
    {
        // dst contains at most kRows scalar outputs.  Its remaining plane is
        // dead during the reduction and provides aligned 8-float partial
        // slots for every row, avoiding a dedicated 512-byte TBuf.
        auto partial = dst[kRows];
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::WholeReduceSum(
                partial[row * 8], src[row * 128], 64, 2, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(dst, partial, 2, rows, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalRows(
        uint32_t validLen, uint32_t subBlockIdx, uint32_t subBlockNum,
        uint32_t &begin, uint32_t &end)
    {
        begin = validLen * subBlockIdx / subBlockNum;
        end = validLen * (subBlockIdx + 1U) / subBlockNum;
    }

    __aicore__ inline void BuildKE(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t ws = (slot + tiling_.kEOffset) / sizeof(bfloat16_t);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto k = Plane(0);
            auto e = Plane(1);
            auto out = Plane(2);
            Load(k, kGm_[tokenBase + row * 128], rows * 128);
            Load(e, gkGm_[tokenBase + row * 128], rows * 128);
            Exp2(e, e, rows * 128);
            AscendC::Mul(out, k, e, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(wsBf16_[ws + row * 128], out, rows * 128);
        }
    }

    __aicore__ inline void FinishBaseStage(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum, uint32_t stage)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t scalarBase =
            (static_cast<uint64_t>(task.batchIdx) * tiling_.headNum + head) *
            tiling_.seqlen + task.begin;
        const uint64_t dqRaw = (slot + tiling_.dqRawOffset) / sizeof(float);
        const uint64_t dkRaw = (slot + tiling_.dkRawOffset) / sizeof(float);
        const uint64_t dVb = (slot + tiling_.dVbOffset) / sizeof(float);
        auto statePartial = Plane(6);
        auto gkLast = Plane(7);
        if (stage == 1) {
            AscendC::Duplicate(statePartial, 0.0f, 128);
            // The chunk anchor is shared by every row tile.  Keep one copy in
            // UB instead of issuing the same 512-byte GM load per tile.
            Load(gkLast, gkGm_[tokenBase + (validLen - 1U) * 128], 128);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (stage == 2) {
            // beta and db are chunk-row scalars.  Keep the complete rows owned
            // by this sub-block in otherwise-dead planes and perform one GM
            // transaction instead of one transaction per matrix tile.
            LoadBeta(Plane(6), scalarBase + begin, end - begin);
        }

        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto x = Plane(0);
            auto y = Plane(1);
            auto z = Plane(2);
            auto aux = Plane(3);
            auto scalar = Plane(4);
            auto brcb = Plane(5);

            if (stage == 0) {
                Load(x, dqGm_[tokenBase + row * 128], rows * 128);
                Load(y, gkGm_[tokenBase + row * 128], rows * 128);
                Exp2(y, y, rows * 128);
                AscendC::Mul(z, x, y, rows * 128);
                AscendC::Muls(z, z, tiling_.scale, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Store(dqGm_[tokenBase + row * 128], z, rows * 128);
            } else if (stage == 1) {
                Load(x, dkGm_[tokenBase + row * 128], rows * 128);
                for (uint32_t r = 0; r < rows; ++r) {
                    AscendC::Adds(z[r * 128], gkLast, 0.0f, 128);
                }
                AscendC::PipeBarrier<PIPE_V>();
                Load(y, gkGm_[tokenBase + row * 128], rows * 128);
                AscendC::Sub(z, z, y, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Exp2(z, z, rows * 128);
                AscendC::Mul(z, x, z, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Store(dkGm_[tokenBase + row * 128], z, rows * 128);

                // GPU keeps dk_state live and immediately accumulates
                // sum_t(k * dk_state).  Do the same here instead of reading
                // final dk back from GM in S5 and reconstructing dk_state.
                Load(x, kGm_[tokenBase + row * 128], rows * 128);
                AscendC::Mul(y, x, z, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t stride = 1; stride < rows; stride <<= 1U) {
                    for (uint32_t r = 0; r + stride < rows;
                         r += stride << 1U) {
                        AscendC::Add(
                            y[r * 128], y[r * 128],
                            y[(r + stride) * 128], 128);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(statePartial, statePartial, y, 128);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                Load(x, wsFp32_[dVb + row * 128], rows * 128);
                Load(y, vGm_[tokenBase + row * 128], rows * 128);
                AscendC::Mul(aux, x, y, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                RowReduce128(z, aux, rows);
                AscendC::Adds(Plane(7)[row - begin], z, 0.0f, rows);
                AscendC::PipeBarrier<PIPE_V>();
                BroadcastRows(brcb, Plane(6)[row - begin], rows);
                MulRowsByScalar(z, x, brcb, rows, 128);
                Store(dvGm_[tokenBase + row * 128], z, rows * 128);
            }
        }
        if (stage == 1) {
            const uint64_t stateBase =
                (slot + tiling_.dkRawOffset) / sizeof(float) +
                subBlockIdx * 128U;
            Store(wsFp32_[stateBase], statePartial, 128);
        } else if (stage == 2) {
            Store(dbGm_[scalarBase + begin], Plane(7), end - begin);
        }
    }

    __aicore__ inline void NegateDW(
        uint32_t validLen, uint64_t slot, uint32_t subBlockIdx,
        uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t dW = (slot + tiling_.dWOffset) / sizeof(bfloat16_t);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto value = Plane(0);
            Load(value, wsBf16_[dW + row * 128], rows * 128);
            AscendC::Muls(value, value, -1.0f, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(wsBf16_[dW + row * 128], value, rows * 128);
        }
    }

    __aicore__ inline void PersistNegatedDW(
        const WyChunkTask &task, uint32_t head, uint32_t validLen,
        uint64_t slot, uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t dW = (slot + tiling_.dWOffset) / sizeof(bfloat16_t);
        const uint64_t packedBase = 2U * WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto value = Plane(0);
            Load(value, wsBf16_[dW + row * 128], rows * 128);
            AscendC::Muls(value, value, -1.0f, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dgBf16Gm_[packedBase + row * 128], value, rows * 128);
        }
    }

    __aicore__ inline void FinishGradientRows(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end)
    {
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t scalarBase =
            (static_cast<uint64_t>(task.batchIdx) * tiling_.headNum + head) *
            tiling_.seqlen + task.begin;
        const uint64_t kE = (slot + tiling_.kEOffset) / sizeof(bfloat16_t);
        const uint64_t dKgb = (slot + tiling_.dKgbOffset) / sizeof(float);
        const uint32_t ownedRows = end - begin;
        // Plane(5)'s first 16 scalars and aligned reduction scratch occupy at
        // most 144 elements.  Its tail safely holds the sub-block's beta/db
        // vectors across all row tiles.
        auto scalarStorage = Plane(5);
        auto betaRows = scalarStorage[256];
        auto dbRows = scalarStorage[256 + 64];
        LoadBeta(betaRows, scalarBase + begin, ownedRows);
        Load(dbRows, dbGm_[scalarBase + begin], ownedRows);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto dkg = Plane(0);
            auto ke = Plane(1);
            auto e = Plane(2);
            auto tmp = Plane(3);
            auto acc = Plane(4);
            auto scalar = Plane(5);
            auto brcb = Plane(6);
            auto qk = Plane(7);

            Load(dkg, wsFp32_[dKgb + row * 128], rows * 128);
            Load(ke, wsBf16_[kE + row * 128], rows * 128);

            // db_k and db_base.
            AscendC::Mul(tmp, dkg, ke, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            RowReduce128(scalar, tmp, rows);
            AscendC::Add(
                dbRows[row - begin], dbRows[row - begin], scalar, rows);
            AscendC::PipeBarrier<PIPE_V>();

            BroadcastRows(brcb, betaRows[row - begin], rows);

            // dk_base = dk_state + dKgb * beta * exp2(gk).
            Load(e, gkGm_[tokenBase + row * 128], rows * 128);
            Exp2(e, e, rows * 128);
            // qk is dead until the gate expression below.  Use it here so
            // tmp keeps kE*dKgb resident for the gate_w contribution.
            AscendC::Mul(qk, dkg, e, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            MulRowsByScalar(qk, qk, brcb, rows, 128);
            Load(acc, dkGm_[tokenBase + row * 128], rows * 128);
            AscendC::Add(e, acc, qk, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dkGm_[tokenBase + row * 128], e, rows * 128);

            // gate_qk + gate_w.  Keep dk_state resident in acc while e is
            // used for the final-dk writeback; no subtractive reconstruction.
            Load(qk, qGm_[tokenBase + row * 128], rows * 128);
            Load(e, dqGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(qk, qk, e, rows * 128);
            Load(e, kGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(acc, e, acc, rows * 128);
            AscendC::Sub(qk, qk, acc, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            // tmp still contains kE*dKgb from the db reduction above.
            MulRowsByScalar(tmp, tmp, brcb, rows, 128);
            AscendC::Add(qk, qk, tmp, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dgGm_[tokenBase + row * 128], qk, rows * 128);
        }
        Store(dbGm_[scalarBase + begin], dbRows, ownedRows);
    }

    __aicore__ inline void BuildZb(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end)
    {
        const uint64_t scalarBase =
            (static_cast<uint64_t>(task.batchIdx) * tiling_.headNum + head) *
            tiling_.seqlen + task.begin;
        const uint64_t zV = (slot + tiling_.zVOffset) / sizeof(bfloat16_t);
        const uint64_t zW = (slot + tiling_.zWOffset) / sizeof(bfloat16_t);
        const uint64_t zB = WyDAkkBf16TaskBase(
            tiling_, task.batchIdx, head, task.begin);
        auto beta = Plane(0);
        LoadBeta(beta, scalarBase, validLen);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto zv = Plane(1);
            auto zw = Plane(2);
            auto out = Plane(3);
            Load(zv, wsBf16_[zV + row * 64], rows * 64);
            Load(zw, wsBf16_[zW + row * 64], rows * 64);
            AscendC::Add(out, zv, zw, rows * 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(
                out, out, beta, 64, static_cast<uint8_t>(rows),
                {1, 1, 1, 8, 8, 0});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t logicalRow = row + r;
                if (logicalRow == 0) {
                    AscendC::Duplicate(out[r * 64], 0.0f, 64);
                } else if (logicalRow < 64) {
                    uint64_t upperMask[1] = {0xffffffffffffffffULL};
                    upperMask[0] <<= logicalRow;
                    AscendC::Duplicate(
                        out[r * 64], 0.0f, upperMask, 1, 1, 8);
                }
            }
            // Store starts with an AIV Adds into the output queue.  Preserve
            // the Duplicate -> Adds dependency explicitly, as in mature
            // vector post-processing paths.
            AscendC::PipeBarrier<PIPE_V>();
            Store(dAkkBf16Gm_[zB + row * 64], out, rows * 64);
        }
    }

    __aicore__ inline void PrepareStateGate(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockNum)
    {
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t hBase = WySavedHOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        const uint64_t dhBase = WyDhOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        const uint32_t last = validLen - 1U;
        const uint64_t stateBase =
            (slot + tiling_.dkRawOffset) / sizeof(float);
        auto state = Plane(0);
        auto partial = Plane(1);
        auto gateAnchor = Plane(7);
        Load(state, wsFp32_[stateBase], 128);
        for (uint32_t part = 1; part < subBlockNum; ++part) {
            Load(partial, wsFp32_[stateBase + part * 128U], 128);
            AscendC::Add(state, state, partial, 128);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // gk at the final token is invariant across all eight 16-row state
        // tiles.  Load and exponentiate the complete 128-column anchor once.
        Load(gateAnchor, gkGm_[tokenBase + last * 128], 128);
        Exp2(gateAnchor, gateAnchor, 128);

        for (uint32_t col = 0; col < 128; col += kRows) {
            auto x = Plane(2);
            auto y = Plane(3);
            auto product = Plane(4);
            auto reduced = Plane(5);
            Load(x, hGm_[hBase + col * 128], kRows * 128);
            Load(y, dhGm_[dhBase + col * 128], kRows * 128);
            AscendC::Mul(product, x, y, kRows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            RowReduce128(reduced, product, kRows);
            AscendC::Mul(reduced, reduced, gateAnchor[col], kRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(state[col], state[col], reduced, kRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Store(wsFp32_[stateBase], state, 128);
    }

    __aicore__ inline void AddPreparedStateGate(
        const WyChunkTask &task, uint32_t head, uint32_t validLen,
        uint64_t slot)
    {
        const uint64_t tokenBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t stateBase =
            (slot + tiling_.dkRawOffset) / sizeof(float);
        const uint32_t last = validLen - 1U;
        auto state = Plane(0);
        auto gradient = Plane(1);
        Load(state, wsFp32_[stateBase], 128);
        Load(gradient, dgGm_[tokenBase + last * 128], 128);
        AscendC::Add(gradient, gradient, state, 128);
        AscendC::PipeBarrier<PIPE_V>();
        Store(dgGm_[tokenBase + last * 128], gradient, 128);
    }

    __aicore__ inline void BuildZbStage(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        BuildZb(task, head, validLen, slot, begin, end);
    }

    __aicore__ inline void PersistTza(
        const WyChunkTask &task, uint32_t head, uint32_t validLen,
        uint64_t slot, uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        const uint64_t src =
            (slot + tiling_.zaOutputOffset) / sizeof(bfloat16_t);
        // dv_scan is dead after stage 4.  Reuse its first 64 columns per row
        // for T_za so stage 7 does not read BF16 input from the same physical
        // dAkk allocation that it overwrites with FP32 output.
        const uint64_t dst = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows =
                row + kRows <= end ? kRows : end - row;
            auto value = Plane(0);
            Load(value, wsBf16_[src + row * 64U], rows * 64U);
            StoreStrided(
                dvScanGm_[dst + row * 128U], value, rows, 64U, 128U);
        }
        // The next L0 consumes T_za through AIC MTE2, so use the same
        // producer/consumer direction as kda_gate_cumsum and causal_conv1d.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void FinishDA(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t raw = (slot + tiling_.zaInputOffset) / sizeof(float);
        // dAkk is [B,H,T,64] without chunk padding.  Use the actual token
        // stride so a tail chunk does not shift the following head.
        const uint64_t out = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 64);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto value = Plane(0);
            Load(value, wsFp32_[raw + row * 64], rows * 64);
            AscendC::Muls(value, value, -1.0f, rows * 64);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t logicalRow = row + r;
                if (logicalRow == 0) {
                    AscendC::Duplicate(value[r * 64], 0.0f, 64);
                } else {
                    uint64_t upperMask[1] = {0xffffffffffffffffULL};
                    upperMask[0] <<= logicalRow;
                    AscendC::Duplicate(
                        value[r * 64], 0.0f, upperMask, 1, 1, 8);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            Store(dAkkGm_[out + row * 64], value, rows * 64);
        }
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR gk_;
    GM_ADDR beta_;
    GM_ADDR h_;
    GM_ADDR dh_;
    GM_ADDR dvScan_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dv_;
    GM_ADDR db_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR workspace_;
    ChunkKdaBwdWyTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<bfloat16_t> qGm_;
    AscendC::GlobalTensor<bfloat16_t> kGm_;
    AscendC::GlobalTensor<bfloat16_t> vGm_;
    AscendC::GlobalTensor<float> gkGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;
    AscendC::GlobalTensor<bfloat16_t> hGm_;
    AscendC::GlobalTensor<bfloat16_t> dhGm_;
    AscendC::GlobalTensor<float> dqGm_;
    AscendC::GlobalTensor<float> dkGm_;
    AscendC::GlobalTensor<bfloat16_t> dvGm_;
    AscendC::GlobalTensor<bfloat16_t> dvScanGm_;
    AscendC::GlobalTensor<float> dbGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<bfloat16_t> dgBf16Gm_;
    AscendC::GlobalTensor<float> dAkkGm_;
    AscendC::GlobalTensor<bfloat16_t> dAkkBf16Gm_;
    AscendC::GlobalTensor<float> wsFp32_;
    AscendC::GlobalTensor<bfloat16_t> wsBf16_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_WY_VECTOR_H
