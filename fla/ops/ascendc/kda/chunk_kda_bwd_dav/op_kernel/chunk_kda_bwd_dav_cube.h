#ifndef CHUNK_KDA_BWD_DAV_CUBE_H
#define CHUNK_KDA_BWD_DAV_CUBE_H

#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_kda_bwd_dav_common.h"

namespace KDA {

class ChunkKdaBwdDAvCubeProcess {
public:
    __aicore__ ChunkKdaBwdDAvCubeProcess(
        GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR dAqk, GM_ADDR dv)
        : aqk_(aqk), vNew_(vNew), dO_(dO), dAqk_(dAqk), dv_(dv)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdDAvTilingData &tiling)
    {
        tiling_ = tiling;
    }

    __aicore__ inline void Process(bool publishVectorReady = false)
    {
        AscendC::SetHF32Mode(false);
        using ArchTag = Catlass::Arch::AtlasA2;
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        using Bf16 = bfloat16_t;

        using DATileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, float, RowMajor>;
        using DAL1Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<128>>;
        using DAL0Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using DAMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, DAL1Shape, DAL0Shape, Bf16, Bf16, float, void, DATileCopy>;

        using DvTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, ColumnMajor, Bf16, RowMajor, Bf16, RowMajor>;
        using DvL1Shape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>;
        using DvL0Shape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>;
        using DvMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, DvL1Shape, DvL0Shape, Bf16, Bf16, Bf16, void, DvTileCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
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

            // Keep the two heads stage-grouped: AIC(head0), AIC(head1), then
            // the two AIV sub-blocks consume each corresponding ready event.
            for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                if (publishVectorReady) {
                    // One Vector->Cube generation per head, matching the
                    // proven Intra/PR190 producer-consumer protocol.  Both
                    // AIV sub-blocks publish this generation before AIC
                    // starts writing the GM tile consumed by Vector.
                    AscendC::CrossCoreWaitFlag(kDavVectorToCubeStartFlag);
                }
                const uint32_t headIdx = headBase + headInWindow;
                RunDA<DAMmad, RowMajor, ColumnMajor>(
                    resource, task, headIdx, validLen);
                if (publishVectorReady) {
                    // dAqk is consumed by both AIV sub-blocks.  Publish only
                    // after the FP32 FixPipe store is globally visible; the
                    // following dv MMADs can then overlap Vector postprocess.
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        kDavCubeToVectorReadyFlag);
                }
                RunDv<DvMmad, ColumnMajor, RowMajor>(
                    resource, task, headIdx, validLen);
            }

        }
    }

private:
    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunDA(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        const DavChunkTask &task, uint32_t headIdx, uint32_t validLen)
    {
        AscendC::GlobalTensor<bfloat16_t> dO;
        AscendC::GlobalTensor<bfloat16_t> vNew;
        AscendC::GlobalTensor<float> dAqk;
        const uint64_t valueOffset = DavValueOffset(tiling_, task.batchIdx, headIdx, task.begin);
        const uint64_t matrixOffset = DavMatrixOffset(tiling_, task.batchIdx, headIdx, task.begin);
        dO.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dO_) + valueOffset);
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(vNew_) + valueOffset);
        dAqk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk_) + matrixOffset);

        auto layoutA = tla::MakeLayout<bfloat16_t, RowMajor>(kDavChunkSize, kDavValueDim);
        auto layoutB = tla::MakeLayout<bfloat16_t, ColumnMajor>(kDavValueDim, kDavChunkSize);
        auto layoutC = tla::MakeLayout<float, RowMajor>(kDavChunkSize, kDavChunkSize);
        auto tensorA = tla::MakeTensor(dO, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(vNew, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(dAqk, layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                              tla::MakeShape(validLen, kDavValueDim));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(kDavValueDim, validLen));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0),
                              tla::MakeShape(validLen, validLen));
        Catlass::GemmCoord shape{validLen, validLen, kDavValueDim};
        Mmad mmad(resource);
        mmad(blockA, blockB, blockC, shape);
    }

    template <typename Mmad, typename ColumnMajor, typename RowMajor>
    __aicore__ inline void RunDv(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        const DavChunkTask &task, uint32_t headIdx, uint32_t validLen)
    {
        AscendC::GlobalTensor<bfloat16_t> aqk;
        AscendC::GlobalTensor<bfloat16_t> dO;
        AscendC::GlobalTensor<bfloat16_t> dv;
        const uint64_t valueOffset = DavValueOffset(tiling_, task.batchIdx, headIdx, task.begin);
        const uint64_t matrixOffset = DavMatrixOffset(tiling_, task.batchIdx, headIdx, task.begin);
        aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aqk_) + matrixOffset);
        dO.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(dO_) + valueOffset);
        dv.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(dv_) + valueOffset);

        // Aqk is physically [token, local_token] RowMajor, while dv needs
        // Aqk^T @ dO.  The ColumnMajor [local_token, token] view performs the
        // transpose in Cube without an AIV scatter.  Keep the physical leading
        // dimension at 64 so tail chunks retain the exported [T,64] stride.
        auto layoutA = tla::MakeLayout<bfloat16_t, ColumnMajor>(
            kDavChunkSize, validLen);
        auto layoutB = tla::MakeLayout<bfloat16_t, RowMajor>(kDavChunkSize, kDavValueDim);
        auto layoutC = tla::MakeLayout<bfloat16_t, RowMajor>(kDavChunkSize, kDavValueDim);
        auto tensorA = tla::MakeTensor(aqk, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(dO, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(dv, layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                              tla::MakeShape(validLen, kDavChunkSize));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(kDavChunkSize, kDavValueDim));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0),
                              tla::MakeShape(validLen, kDavValueDim));
        Catlass::GemmCoord shape{validLen, kDavValueDim, kDavChunkSize};
        Mmad mmad(resource);
        mmad(blockA, blockB, blockC, shape);
    }

    GM_ADDR aqk_;
    GM_ADDR vNew_;
    GM_ADDR dO_;
    GM_ADDR dAqk_;
    GM_ADDR dv_;
    ChunkKdaBwdDAvTilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_DAV_CUBE_H
