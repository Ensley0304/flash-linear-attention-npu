#ifndef CHUNK_KDA_BWD_C_VECTOR_H
#define CHUNK_KDA_BWD_C_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "chunk_kda_bwd_c_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils/vector/regbase.hpp"
#endif

namespace KDA {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
static __simd_vf__ inline void KdaBwdCRowDotAccA5(
    __ubuf__ float *dst, __ubuf__ float *lhs, __ubuf__ float *rhs,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        Duplicate(acc, 0.0f, fullMask);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            RegTensor<float> lhsReg;
            RegTensor<float> rhsReg;
            RegTensor<float> product;
            DataCopy(lhsReg, lhs + row * cols + col);
            DataCopy(rhsReg, rhs + row * cols + col);
            Mul(product, lhsReg, rhsReg, fullMask);
            Add(acc, acc, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, dst + row);
        Add(sum, sum, current, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dst + row, sum, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCDvDbA5(
    __ubuf__ float *dvDst, __ubuf__ float *rowAcc,
    __ubuf__ float *dvb, __ubuf__ float *v, __ubuf__ float *beta,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        RegTensor<float> betaReg;
        Duplicate(acc, 0.0f, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> dvbReg;
            RegTensor<float> vReg;
            RegTensor<float> product;
            DataCopy(dvbReg, dvb + offset);
            DataCopy(vReg, v + offset);
            Mul(product, dvbReg, vReg, fullMask);
            Add(acc, acc, product, fullMask);
            Mul(product, dvbReg, betaReg, fullMask);
            DataCopy(dvDst + offset, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, rowAcc + row);
        Add(sum, sum, current, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            rowAcc + row, sum, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCMulRowDotSubA5(
    __ubuf__ float *productDst, __ubuf__ float *rowAcc,
    __ubuf__ float *lhs, __ubuf__ float *rhs,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        Duplicate(acc, 0.0f, fullMask);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> lhsReg;
            RegTensor<float> rhsReg;
            RegTensor<float> product;
            DataCopy(lhsReg, lhs + offset);
            DataCopy(rhsReg, rhs + offset);
            Mul(product, lhsReg, rhsReg, fullMask);
            Add(acc, acc, product, fullMask);
            DataCopy(productDst + offset, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, rowAcc + row);
        Sub(current, current, sum, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            rowAcc + row, current, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCFinishDkDgA5(
    __ubuf__ float *dkDst, __ubuf__ float *dgDst,
    __ubuf__ float *dkg, __ubuf__ float *expG,
    __ubuf__ float *beta, __ubuf__ float *dkState,
    __ubuf__ float *q, __ubuf__ float *dq,
    __ubuf__ float *k, __ubuf__ float *gateW,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> betaReg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> dkgReg;
            RegTensor<float> expReg;
            RegTensor<float> stateReg;
            RegTensor<float> qReg;
            RegTensor<float> dqReg;
            RegTensor<float> kReg;
            RegTensor<float> gateWReg;
            RegTensor<float> tmp0;
            RegTensor<float> tmp1;
            DataCopy(dkgReg, dkg + offset);
            DataCopy(expReg, expG + offset);
            DataCopy(stateReg, dkState + offset);
            DataCopy(qReg, q + offset);
            DataCopy(dqReg, dq + offset);
            DataCopy(kReg, k + offset);
            DataCopy(gateWReg, gateW + offset);

            Mul(tmp0, dkgReg, expReg, fullMask);
            Mul(tmp0, tmp0, betaReg, fullMask);
            Sub(tmp0, stateReg, tmp0, fullMask);
            DataCopy(dkDst + offset, tmp0, fullMask);

            Mul(tmp0, qReg, dqReg, fullMask);
            Mul(tmp1, kReg, stateReg, fullMask);
            Sub(tmp0, tmp0, tmp1, fullMask);
            Mul(tmp1, gateWReg, betaReg, fullMask);
            Sub(tmp0, tmp0, tmp1, fullMask);
            DataCopy(dgDst + offset, tmp0, fullMask);
        }
    }
}
#endif

template <typename DataT, uint32_t V_DIM, typename BetaT>
class ChunkKdaBwdCVectorProcess {
public:
    __aicore__ ChunkKdaBwdCVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh, GM_ADDR dqRaw,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
        GM_ADDR db, GM_ADDR dg, GM_ADDR dAkk,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace)
        : q_(q), k_(k), v_(v), gk_(gk), beta_(beta), h_(h), dh_(dh),
          dqRaw_(dqRaw), dq_(dq), dk_(dk), dv_(dv),
          db_(db), dg_(dg), dAkk_(dAkk), cuSeqlens_(cuSeqlens),
          chunkIndices_(chunkIndices),
          workspace_(workspace) {}

    __aicore__ inline void Init(
        const ChunkKdaBwdCTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(q_));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(k_));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(v_));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk_));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta_));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(h_));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dh_));
        dqRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw_));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq_));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk_));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dv_));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db_));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        dAkkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk_));
        dAkkBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dAkk_));
        wsFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_));
        wsBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(workspace_));

        // Two entries let MTE2/MTE3 alternate buffers while Vector consumes
        // the preceding tile.  RowReduce uses the unused tail of its output
        // plane, so the full 96 KiB UB budget remains available to IO ping-pong.
        pipe_->InitBuffer(inputQueue_, 2, kIoBytes);
        pipe_->InitBuffer(outputQueue_, 2, kIoBytes);
        pipe_->InitBuffer(arena_, kArenaBytes);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Keep the two heads' owned dk_state rows in A5's larger UB until
        // FinishGradientRows forms final dk.  This removes one full FP32
        // write/read round trip through GM for every owner.
        pipe_->InitBuffer(dkStateBuffer_, kDkStateBytes);
#endif
    }

    __aicore__ inline void Process()
    {
        ProcessFused();
    }


private:
    __aicore__ inline void SignalVectorDependency(uint32_t flag)
    {
        // Both AIV sub-blocks own disjoint rows.  Release AIC only after all
        // MTE3 writes required by the actual dependency are globally visible.
        // Match the mature Kernel-A/PR190 protocol: both AIV sub-blocks
        // participate in the collective notification consumed by one AIC.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    __aicore__ inline void ProcessFused()
    {
        // Four producer flags allow AIC to publish independent S0/S1/S2/S3a
        // results without waiting for AIV after every stage.  They are reused
        // only after the S3a acknowledgement proves that AIV consumed the
        // first wave.  The four acknowledgement flags correspond exactly to
        // the true data dependencies: -dW, Zb, and final workspace free.
        constexpr uint32_t kS0Ready = 0;
        constexpr uint32_t kS1Ready = 1;
        constexpr uint32_t kS2Ready = 4;
        constexpr uint32_t kS3aReady = 5;
        constexpr uint32_t kZbReady = 3;
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
            const WyChunkTask task = GetWyChunkTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;

            // kE has no AIC dependency.  Build it while AIC produces the
            // independent base GEMMs instead of serializing it behind S2.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // S0 is published before AIC starts S1.  Consume it here so kE
            // and dq share the same resident exp2(gk) tile instead of
            // loading gk and evaluating exp twice.
            AscendC::CrossCoreWaitFlag(kS0Ready);
#endif
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                BuildKE<true>(task, headBase + lane, validLen, slot,
                              subBlockIdx, subBlockNum);
#else
                BuildKE<false>(task, headBase + lane, validLen, slot,
                               subBlockIdx, subBlockNum);
#endif
            }

            // S0: consume dq_raw and finish dq_base.
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            AscendC::CrossCoreWaitFlag(kS0Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 0);
            }
#endif
            // S1: consume dk_raw and finish dk_state.
            AscendC::CrossCoreWaitFlag(kS1Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 1);
            }
            // S2 is ready, but dv_scan is also the final dv storage in the
            // fused contract.  Do not overwrite it until S3a has completed
            // every final read of the original dv_scan values.
            AscendC::CrossCoreWaitFlag(kS2Ready);
            AscendC::CrossCoreWaitFlag(kS3aReady);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, 2);
            }

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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // Ascend950's larger per-AIV UB allows a 32-row WY tile.  This halves
    // row-loop, queue and event overhead on the scalar-heavy A5 path while
    // retaining the proven 16-row footprint on A2/A3.
    static constexpr uint32_t kRows = 32;
    static constexpr uint32_t kUbBudgetBytes = 248 * 1024;
    static constexpr uint32_t kDkStateRowsPerHead = 32;
    static constexpr uint32_t kDkStateBytes =
        kWyFusedHeadsPerWindow * kDkStateRowsPerHead * kWyKeyDim *
        sizeof(float);
#else
    static constexpr uint32_t kRows = 16;
    static constexpr uint32_t kUbBudgetBytes = 96 * 1024;
#endif
    static constexpr uint32_t kPlaneElements = kRows * kWyKeyDim;
    static constexpr uint32_t kIoBytes = kPlaneElements * sizeof(float);
    // FinishGradients has the largest live set and uses Plane(0)..Plane(7).
    // Reserving 24 planes exceeded the conservative per-AIV UB budget of the
    // 1AIC:2AIV MIX kernel without providing any reusable live storage.
    static constexpr uint32_t kArenaBytes = 8 * kPlaneElements * sizeof(float);
    static constexpr float kLn2 = 0.69314718055994530942f;
    static_assert(4 * kIoBytes + kArenaBytes
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                      + kDkStateBytes
#endif
                      <= kUbBudgetBytes,
                   "ChunkKdaBwdC AIV buffers exceed the architecture UB budget");

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline AscendC::LocalTensor<float> DkStateTile(
        uint32_t head, uint32_t ownedRow)
    {
        const uint32_t lane = head % kWyFusedHeadsPerWindow;
        const uint32_t offset =
            (lane * kDkStateRowsPerHead + ownedRow) * kWyKeyDim;
        return dkStateBuffer_.Get<float>()[offset];
    }
#endif

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

    template <typename SrcT>
    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<SrcT> src,
        uint32_t count)
    {
        auto in = inputQueue_.AllocTensor<SrcT>();
        AscendC::DataCopyPad(
            in, src,
            {1, static_cast<uint32_t>(count * sizeof(SrcT)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<SrcT>();
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

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src, uint32_t rows,
        uint32_t cols, uint32_t physicalCols)
    {
        auto in = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((physicalCols - cols) * sizeof(float)),
            0, 0};
        AscendC::DataCopyPad(in, src, params, {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<float>();
        AscendC::Adds(dst, ready, 0.0f, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<DataT> src, uint32_t rows,
        uint32_t cols, uint32_t physicalCols)
    {
        auto in = inputQueue_.AllocTensor<DataT>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(DataT)),
            static_cast<uint32_t>((physicalCols - cols) * sizeof(DataT)),
            0, 0};
        AscendC::DataCopyPad(in, src, params, {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<DataT>();
        AscendC::Cast(
            dst, ready, AscendC::RoundMode::CAST_NONE, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<DataT> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        auto out = outputQueue_.AllocTensor<DataT>();
        AscendC::Cast(out, src, AscendC::RoundMode::CAST_RINT, count);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<DataT>();
        AscendC::DataCopyPad(
            dst, ready,
            {1, static_cast<uint32_t>(count * sizeof(DataT)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void StoreStrided(
        AscendC::GlobalTensor<DataT> dst,
        AscendC::LocalTensor<float> src, uint32_t rows, uint32_t cols,
        uint32_t physicalCols)
    {
        auto out = outputQueue_.AllocTensor<DataT>();
        AscendC::Cast(
            out, src, AscendC::RoundMode::CAST_RINT, rows * cols);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<DataT>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(DataT)),
            0,
            static_cast<uint32_t>((physicalCols - cols) * sizeof(DataT)),
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

    template <bool FINISH_DQ>
    __aicore__ inline void BuildKE(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t tokenBaseV = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, V_DIM);
        const uint64_t ws = (slot + tiling_.kEOffset) / sizeof(DataT);
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
            if constexpr (FINISH_DQ) {
                // e still contains exp2(gk).  Finish dq before this tile is
                // reused, eliminating one gk GM load and one Exp2 pass.
                Load(k, dqRawGm_[tokenBase + row * 128], rows * 128);
                AscendC::Mul(out, k, e, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(out, out, tiling_.scale, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Store(dqGm_[tokenBase + row * 128], out, rows * 128);
            }
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
        const uint64_t tokenBaseV = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, V_DIM);
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t dqRaw = (slot + tiling_.dqRawOffset) / sizeof(float);
        const uint64_t dkRaw = (slot + tiling_.dkRawOffset) / sizeof(float);
        const uint64_t dVb = (slot + tiling_.dVbOffset) / sizeof(float);
        auto statePartial = Plane(6);
        auto gkLast = Plane(7);
        if (begin == end) {
            if (stage == 1) {
                AscendC::Duplicate(statePartial, 0.0f, 128);
                AscendC::PipeBarrier<PIPE_V>();
                const uint64_t stateBase =
                    (slot + tiling_.dkRawOffset) / sizeof(float) +
                    subBlockIdx * 128U;
                Store(wsFp32_[stateBase], statePartial, 128);
            }
            return;
        }
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
            AscendC::Duplicate(Plane(7), 0.0f, end - begin);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto x = Plane(0);
            auto y = Plane(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto z = stage == 1 ? DkStateTile(head, row - begin) : Plane(2);
#else
            auto z = Plane(2);
#endif
            auto aux = Plane(3);
            auto scalar = Plane(4);
            auto brcb = Plane(5);

            if (stage == 0) {
                Load(x, dqRawGm_[tokenBase + row * 128], rows * 128);
                Load(y, gkGm_[tokenBase + row * 128], rows * 128);
                Exp2(y, y, rows * 128);
                AscendC::Mul(z, x, y, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
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
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                Store(dkGm_[tokenBase + row * 128], z, rows * 128);
#endif

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
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                BroadcastRows(brcb, Plane(6)[row - begin], rows);
#endif
                for (uint32_t v0 = 0; v0 < V_DIM; v0 += 128) {
                    LoadRows(
                        x, wsFp32_[dVb + row * V_DIM + v0],
                        rows, 128, V_DIM);
                    LoadRows(
                        y, vGm_[tokenBaseV + row * V_DIM + v0],
                        rows, 128, V_DIM);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    KdaBwdCDvDbA5(
                        (__ubuf__ float *)z.GetPhyAddr(),
                        (__ubuf__ float *)Plane(7)[row - begin].GetPhyAddr(),
                        (__ubuf__ float *)x.GetPhyAddr(),
                        (__ubuf__ float *)y.GetPhyAddr(),
                        (__ubuf__ float *)Plane(6)[row - begin].GetPhyAddr(),
                        static_cast<uint16_t>(rows), 128);
                    AscendC::PipeBarrier<PIPE_V>();
#else
                    AscendC::Mul(aux, x, y, rows * 128);
                    AscendC::PipeBarrier<PIPE_V>();
                    RowReduce128(z, aux, rows);
                    AscendC::Add(
                        Plane(7)[row - begin],
                        Plane(7)[row - begin], z, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    MulRowsByScalar(z, x, brcb, rows, 128);
#endif
                    StoreStrided(
                        dvGm_[tokenBaseV + row * V_DIM + v0],
                        z, rows, 128, V_DIM);
                }
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

    __aicore__ inline void FinishGradientRows(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end)
    {
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t kE = (slot + tiling_.kEOffset) / sizeof(DataT);
        const uint64_t dKgb = (slot + tiling_.dKgbOffset) / sizeof(float);
        const uint32_t ownedRows = end - begin;
        if (ownedRows == 0) {
            return;
        }
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto acc = DkStateTile(head, row - begin);
#else
            auto acc = Plane(4);
#endif
            auto scalar = Plane(5);
            auto brcb = Plane(6);
            auto qk = Plane(7);

            Load(dkg, wsFp32_[dKgb + row * 128], rows * 128);
            Load(ke, wsBf16_[kE + row * 128], rows * 128);

            // dKgb_raw has the opposite sign of mathematical dKgb, so fold
            // the minus sign into each consumer instead of materializing a
            // second full matrix.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaBwdCMulRowDotSubA5(
                (__ubuf__ float *)tmp.GetPhyAddr(),
                (__ubuf__ float *)dbRows[row - begin].GetPhyAddr(),
                (__ubuf__ float *)dkg.GetPhyAddr(),
                (__ubuf__ float *)ke.GetPhyAddr(),
                static_cast<uint16_t>(rows), 128);
#else
            AscendC::Mul(tmp, dkg, ke, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            RowReduce128(scalar, tmp, rows);
            AscendC::Sub(
                dbRows[row - begin], dbRows[row - begin], scalar, rows);
            AscendC::PipeBarrier<PIPE_V>();
#endif

            // dk_base = dk_state - dKgb_raw * beta * exp2(gk).
            Load(e, gkGm_[tokenBase + row * 128], rows * 128);
            Exp2(e, e, rows * 128);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Keep the complete dk/dg elementwise chain in registers.  The
            // two outputs share beta and dk_state, while tmp already holds
            // dKgb_raw*kE from the db reduction above.
            Load(qk, qGm_[tokenBase + row * 128], rows * 128);
            Load(brcb, dqGm_[tokenBase + row * 128], rows * 128);
            Load(ke, kGm_[tokenBase + row * 128], rows * 128);
            KdaBwdCFinishDkDgA5(
                (__ubuf__ float *)e.GetPhyAddr(),
                (__ubuf__ float *)qk.GetPhyAddr(),
                (__ubuf__ float *)dkg.GetPhyAddr(),
                (__ubuf__ float *)e.GetPhyAddr(),
                (__ubuf__ float *)betaRows[row - begin].GetPhyAddr(),
                (__ubuf__ float *)acc.GetPhyAddr(),
                (__ubuf__ float *)qk.GetPhyAddr(),
                (__ubuf__ float *)brcb.GetPhyAddr(),
                (__ubuf__ float *)ke.GetPhyAddr(),
                (__ubuf__ float *)tmp.GetPhyAddr(),
                static_cast<uint16_t>(rows), 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dkGm_[tokenBase + row * 128], e, rows * 128);
            Store(dgGm_[tokenBase + row * 128], qk, rows * 128);
#else
            BroadcastRows(brcb, betaRows[row - begin], rows);
            // qk is dead until the gate expression below.  Use it here so
            // tmp keeps kE*dKgb resident for the gate_w contribution.
            AscendC::Mul(qk, dkg, e, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            MulRowsByScalar(qk, qk, brcb, rows, 128);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Load(acc, dkGm_[tokenBase + row * 128], rows * 128);
#endif
            AscendC::Sub(e, acc, qk, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dkGm_[tokenBase + row * 128], e, rows * 128);

            // gate_qk + gate_w algebraically; dKgb_raw carries the opposite
            // sign, so subtract its contribution.  Keep dk_state resident in acc while e is
            // used for the final-dk writeback; no subtractive reconstruction.
            Load(qk, qGm_[tokenBase + row * 128], rows * 128);
            Load(e, dqGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(qk, qk, e, rows * 128);
            Load(e, kGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(acc, e, acc, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(qk, qk, acc, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            // tmp still contains kE*dKgb from the db reduction above.
            MulRowsByScalar(tmp, tmp, brcb, rows, 128);
            AscendC::Sub(qk, qk, tmp, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dgGm_[tokenBase + row * 128], qk, rows * 128);
#endif
        }
        Store(dbGm_[scalarBase + begin], dbRows, ownedRows);
    }

    __aicore__ inline void BuildZb(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end)
    {
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t zV = (slot + tiling_.zVOffset) / sizeof(DataT);
        const uint64_t zW = (slot + tiling_.zWOffset) / sizeof(DataT);
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
            // zW was formed from dW_raw; subtracting it is equivalent to
            // adding the original zW formed from -dW_raw.
            AscendC::Sub(out, zv, zw, rows * 64);
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
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            auto product = Plane(4);
#endif
            auto reduced = Plane(5);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            auto tileReduced = Plane(6);
#endif
            AscendC::Duplicate(reduced, 0.0f, kRows);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t v0 = 0; v0 < V_DIM; v0 += 128) {
                LoadRows(
                    x, hGm_[hBase + col * V_DIM + v0],
                    kRows, 128, V_DIM);
                LoadRows(
                    y, dhGm_[dhBase + col * V_DIM + v0],
                    kRows, 128, V_DIM);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                KdaBwdCRowDotAccA5(
                    (__ubuf__ float *)reduced.GetPhyAddr(),
                    (__ubuf__ float *)x.GetPhyAddr(),
                    (__ubuf__ float *)y.GetPhyAddr(),
                    static_cast<uint16_t>(kRows), 128);
#else
                AscendC::Mul(product, x, y, kRows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                RowReduce128(tileReduced, product, kRows);
                AscendC::Add(reduced, reduced, tileReduced, kRows);
                AscendC::PipeBarrier<PIPE_V>();
#endif
            }
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
    GM_ADDR dqRaw_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dv_;
    GM_ADDR db_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<DataT> qGm_;
    AscendC::GlobalTensor<DataT> kGm_;
    AscendC::GlobalTensor<DataT> vGm_;
    AscendC::GlobalTensor<float> gkGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;
    AscendC::GlobalTensor<DataT> hGm_;
    AscendC::GlobalTensor<DataT> dhGm_;
    AscendC::GlobalTensor<float> dqRawGm_;
    AscendC::GlobalTensor<float> dqGm_;
    AscendC::GlobalTensor<float> dkGm_;
    AscendC::GlobalTensor<DataT> dvGm_;
    AscendC::GlobalTensor<float> dbGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> dAkkGm_;
    AscendC::GlobalTensor<DataT> dAkkBf16Gm_;
    AscendC::GlobalTensor<float> wsFp32_;
    AscendC::GlobalTensor<DataT> wsBf16_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    AscendC::TBuf<AscendC::TPosition::VECCALC> dkStateBuffer_;
#endif
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_VECTOR_H
