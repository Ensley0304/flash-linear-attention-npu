#ifndef CHUNK_KDA_BWD_DAV_VECTOR_H
#define CHUNK_KDA_BWD_DAV_VECTOR_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_dav_common.h"

namespace KDA {

class ChunkKdaBwdDAvVectorProcess {
public:
    __aicore__ explicit ChunkKdaBwdDAvVectorProcess(GM_ADDR dAqk) : dAqk_(dAqk) {}

    __aicore__ inline void Init(
        const ChunkKdaBwdDAvTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        dAqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk_));
        const uint32_t rowCapacity =
            tiling_.stage == 2 ? kDavChunkSize / 2 : kDavChunkSize;
        pipe_->InitBuffer(
            matrixBuf_, rowCapacity * kDavChunkSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tiling_.stage == 2) {
            ProcessMix();
            return;
        }
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = tiling_.usedCoreNum;
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kDavHeadsPerWindow - 1) / kDavHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindowIdx =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase = headWindowIdx * kDavHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const DavChunkTask task = GetDavChunkTask(tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                FinishHead(task, headBase + headInWindow, 0, validLen);
            }
        }
    }

private:
    __aicore__ inline void ProcessMix()
    {
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        if (subBlockNum == 0) {
            return;
        }
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = tiling_.usedCoreNum;
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kDavHeadsPerWindow - 1) / kDavHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        const uint32_t rowsPerSubBlock =
            (kDavChunkSize + subBlockNum - 1) / subBlockNum;
        const uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;

        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindowIdx =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase = headWindowIdx * kDavHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const DavChunkTask task = GetDavChunkTask(tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;

            // Mirror chunk_kda_bwd_intra exactly: publish the two heads'
            // Vector-ready generations first, so Cube(head0) can overlap
            // the remaining Vector-side scheduling work.  The flag uses a
            // two-AIV broadcast/count because both sub-blocks participate.
            for (uint32_t headInWindow = 0; headInWindow < headCount;
                 ++headInWindow) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                    kDavVectorToCubeStartFlag);
            }
            for (uint32_t headInWindow = 0; headInWindow < headCount;
                 ++headInWindow) {
                // AIC broadcasts one ready event to both AIV sub-blocks.
                // Both consumers must wait even when a tail gives one of
                // them zero rows, otherwise the flag generations diverge.
                AscendC::CrossCoreWaitFlag(kDavCubeToVectorReadyFlag);
                const uint32_t rowCount = rowBegin >= validLen
                    ? 0
                    : (rowBegin + rowsPerSubBlock <= validLen
                           ? rowsPerSubBlock
                           : validLen - rowBegin);
                FinishHead(
                    task, headBase + headInWindow, rowBegin, rowCount);
            }
        }
    }

    __aicore__ inline void FinishHead(
        const DavChunkTask &task, uint32_t headIdx, uint32_t rowBegin,
        uint32_t rowCount)
    {
        if (rowCount == 0) {
            return;
        }
        const uint64_t matrixOffset =
            DavMatrixOffset(tiling_, task.batchIdx, headIdx, task.begin + rowBegin);
        AscendC::LocalTensor<float> matrix = matrixBuf_.Get<float>();
        const uint32_t elementCount = rowCount * kDavChunkSize;
        AscendC::DataCopy(matrix, dAqkGm_[matrixOffset], elementCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(matrix, matrix, tiling_.scale, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
            const uint32_t causalRow = rowBegin + localRow;
            const uint32_t zeroBegin = causalRow + 1;
            if (zeroBegin < kDavChunkSize) {
                // Vector destinations must start on a 32-byte boundary.  A
                // scalar offset at zeroBegin is unaligned for most rows and
                // raises 507015 on A2.  Start at the aligned 64-float row and
                // select the strict upper-triangular lanes with a bit mask.
                uint64_t upperMask[1] = {0xffffffffffffffffULL};
                upperMask[0] <<= zeroBegin;
                AscendC::Duplicate(
                    matrix[localRow * kDavChunkSize], 0.0f,
                    upperMask, 1, 1, 8);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(dAqkGm_[matrixOffset], matrix, elementCount);
        // The same UB tile is reused by the next head in this 2-head window.
        // Wait for CopyOut to finish before the next CopyIn may overwrite it.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

    GM_ADDR dAqk_;
    ChunkKdaBwdDAvTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<float> dAqkGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixBuf_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_DAV_VECTOR_H
