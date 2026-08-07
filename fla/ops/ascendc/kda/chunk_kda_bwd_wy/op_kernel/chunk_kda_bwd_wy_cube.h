#ifndef CHUNK_KDA_BWD_WY_CUBE_H
#define CHUNK_KDA_BWD_WY_CUBE_H

#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_kda_bwd_wy_common.h"

namespace KDA {

// The fused path owns the MM layout and L1/L0 event lifecycle at phase scope.
// Individual GEMMs only bind the shared resource buffers.  BF16/FP32 phase
// boundaries are still fully drained, preserving the proven type-transition
// safety while avoiding constructor/destructor synchronization for every
// 64x64/64x128 tile.
template <class ArchTag_, class ElementC_, class TileCopy_>
struct WyTileGemmDirect {
    using ArchTag = ArchTag_;
    using TileCopy = TileCopy_;
    using ElementA = typename TileCopy::ElementA;
    using ElementB = typename TileCopy::ElementB;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, ElementA, LayoutTagL1A>;

    static constexpr uint32_t kTile = 128;
    static constexpr uint32_t kL1ABytes =
        kTile * kTile * sizeof(ElementA);
    static constexpr auto kL1ALayout =
        tla::MakeLayout<ElementA, LayoutTagL1A>(
            tla::Int<kTile>{}, tla::Int<kTile>{});
    static constexpr auto kL1BLayout =
        tla::MakeLayout<ElementB, LayoutTagL1B>(
            tla::Int<kTile>{}, tla::Int<kTile>{});

    CATLASS_DEVICE
    explicit WyTileGemmDirect(Catlass::Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            l1A_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1B_ = resource.l1Buf.template GetBufferByByte<ElementB>(kL1ABytes);
            l0A_ = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0B_ = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0C_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        }
    }

    CATLASS_DEVICE ~WyTileGemmDirect() = default;

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void operator()(TensorA &a, TensorB &b, TensorC &c,
                    Catlass::GemmCoord const &shape)
    {
        using CopyGmToL1A =
            typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B =
            typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToGm =
            typename TileCopy::template CopyL0CToGm<TensorC>;
        CopyGmToL1A copyA;
        CopyGmToL1B copyB;
        CopyL1ToL0A copyL0A;
        CopyL1ToL0B copyL0B;
        CopyL0CToGm copyC;
        TileMmad mmad;

        uint32_t m = shape.m();
        const uint32_t n = shape.n();
        const uint32_t k = shape.k();
        const uint32_t mActual = m == 1 ? 16 : m;
        auto l1A = tla::MakeTensor(l1A_, kL1ALayout,
                                   Catlass::Arch::PositionL1{});
        auto l1B = tla::MakeTensor(l1B_, kL1BLayout,
                                   Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        copyA(l1A, a);
        copyB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);

        auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mActual, k);
        auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto l0A = tla::MakeTensor(l0A_, l0ALayout,
                                   Catlass::Arch::PositionL0A{});
        auto l0B = tla::MakeTensor(l0B_, l0BLayout,
                                   Catlass::Arch::PositionL0B{});
        auto l1ATile = GetTile(l1A, tla::MakeCoord(0, 0),
                               tla::MakeShape(mActual, k));
        auto l1BTile = GetTile(l1B, tla::MakeCoord(0, 0),
                               tla::MakeShape(k, n));
        copyL0A(l0A, l1ATile);
        copyL0B(l0B, l1BTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        auto l0CLayout = tla::MakeLayoutL0C(mActual, n);
        auto l0C = tla::MakeTensor(l0C_, l0CLayout,
                                   Catlass::Arch::PositionL0C{});
        mmad(l0C, l0A, l0B, mActual, n, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        copyC(c, l0C, 0b11);
    }

private:
    AscendC::LocalTensor<ElementA> l1A_;
    AscendC::LocalTensor<ElementB> l1B_;
    AscendC::LocalTensor<ElementA> l0A_;
    AscendC::LocalTensor<ElementB> l0B_;
    AscendC::LocalTensor<ElementAccumulator> l0C_;
};

class ChunkKdaBwdWyCubeProcess {
public:
    __aicore__ ChunkKdaBwdWyCubeProcess(
        GM_ADDR v, GM_ADDR vNew, GM_ADDR a, GM_ADDR h, GM_ADDR dO,
        GM_ADDR dh, GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dg,
        GM_ADDR dAkk, GM_ADDR workspace)
        : v_(v), vNew_(vNew), a_(a), h_(h), dO_(dO), dh_(dh),
          dvScan_(dvScan), dq_(dq), dk_(dk), dg_(dg), dAkk_(dAkk),
          workspace_(workspace) {}

    __aicore__ inline void Init(const ChunkKdaBwdWyTilingData &tiling)
    {
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
        using ArchTag = Catlass::Arch::AtlasA2;
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
        using SingleDispatchPolicy =
            Catlass::Gemm::MmadPingpong<ArchTag, false, false>;
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        using Bf16 = bfloat16_t;

        using Fp32C128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, float, RowMajor>;
        using Fp32C128Mmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<128>>,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32C128Copy>;

        using Bf16C128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, Bf16, RowMajor>;
        using Bf16C128Mmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<128>>,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            Bf16, Bf16, Bf16, void, Bf16C128Copy>;

        using Fp32C64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, float, RowMajor>;
        using Fp32C64Mmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<128>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32C64Copy>;
        using Bf16C64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, Bf16, RowMajor>;
        using Bf16C64Mmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<128>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, Bf16, void, Bf16C64Copy>;

        using Fp32A64x128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, RowMajor, float, RowMajor>;
        using Fp32A64x128Mmad = Catlass::Gemm::Block::BlockMmadTla<
            SingleDispatchPolicy,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32A64x128Copy>;
        // Akk is physically [token, local_token], while the Triton kernel
        // reads it as b_A[local_token, token].  Reinterpret the same GM bytes
        // as ColumnMajor instead of materializing a transpose.
        using Fp32AT64x128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, ColumnMajor, Bf16, RowMajor, float, RowMajor>;
        using Fp32AT64x128Mmad = Catlass::Gemm::Block::BlockMmadTla<
            SingleDispatchPolicy,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32AT64x128Copy>;
        using ChainedFp32A64x128Mmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32A64x128Copy>;

        using Bf16SquareCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, RowMajor, Bf16, RowMajor>;
        using Bf16SquareMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, Bf16, void, Bf16SquareCopy>;

        using Fp32SquareCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, RowMajor, float, RowMajor>;
        using Fp32SquareMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32SquareCopy>;

        using Bf16SquareRightTransposeCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, RowMajor, Bf16, ColumnMajor, Bf16, RowMajor>;
        using Bf16SquareRightTransposeMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, Bf16, void, Bf16SquareRightTransposeCopy>;

        using Fp32SquareLeftTransposeCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Bf16, ColumnMajor, Bf16, RowMajor, float, RowMajor>;
        using Fp32SquareLeftTransposeMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>,
            Bf16, Bf16, float, void, Fp32SquareLeftTransposeCopy>;

        using FusedFp32C64Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32C64Copy>;
        using FusedBf16C64Mmad =
            WyTileGemmDirect<ArchTag, Bf16, Bf16C64Copy>;
        using FusedFp32AT64x128Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32AT64x128Copy>;
        using FusedBf16SquareRightTransposeMmad =
            WyTileGemmDirect<ArchTag, Bf16, Bf16SquareRightTransposeCopy>;
        using FusedFp32SquareLeftTransposeMmad =
            WyTileGemmDirect<ArchTag, float, Fp32SquareLeftTransposeCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        if (tiling_.stage == kWyFusedStage) {
            ProcessFused<
                FusedFp32C64Mmad, FusedBf16C64Mmad,
                FusedFp32AT64x128Mmad,
                FusedBf16SquareRightTransposeMmad,
                FusedFp32SquareLeftTransposeMmad,
                RowMajor, ColumnMajor>(resource);
            return;
        }
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount = (headNum + 1U) / 2U;
        const uint64_t taskGroupCount = static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;

        uint32_t localGeneration = 0;
        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, ++localGeneration) {
            const uint32_t taskIdx = static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindow = static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase = headWindow * kWyHeadsPerWindow;
            const uint32_t headCount = headBase + 1U < headNum ? 2U : 1U;
            const WyChunkTask task = GetWyChunkTask(tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            AscendC::CrossCoreWaitFlag(
                WyWorkspaceFreeFlag(localGeneration));
            if (tiling_.stage == 3) {
                // PR190/K4-style 2-head stage grouping.  Produce both heads'
                // base tiles before either head enters the dependent GEMM.
                for (uint32_t lane = 0; lane < headCount; ++lane) {
                    const uint64_t slot = WyWorkspaceSlotBase(
                        tiling_, coreIdx, localGeneration, lane);
                    const uint32_t head = headBase + lane;
                    const uint64_t token128 = WyTokenOffset(
                        tiling_, task.batchIdx, head, task.begin, 128);
                    const uint64_t hOffset = WySavedHOffset(
                        tiling_, task.batchIdx, head, task.chunkIdx);
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, token128, h_, hOffset,
                        slot + tiling_.dWOffset, validLen, 64,
                        128, 128);
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, token128,
                        h_, hOffset + 64U * 128U,
                        slot + tiling_.dWOffset + 64U * sizeof(bfloat16_t),
                        validLen, 64, 128, 128);
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, token128, v_, token128,
                        slot + tiling_.zVOffset, validLen, validLen, 64, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                }
                for (uint32_t lane = 0; lane < headCount; ++lane) {
                    const uint64_t slot = WyWorkspaceSlotBase(
                        tiling_, coreIdx, localGeneration, lane);
                    AscendC::CrossCoreWaitFlag(WyVectorToCubeFlag(lane));
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, workspace_,
                        (slot + tiling_.dWOffset) / sizeof(bfloat16_t),
                        workspace_,
                        (slot + tiling_.kEOffset) / sizeof(bfloat16_t),
                        slot + tiling_.zWOffset, validLen, validLen, 64, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                }
                continue;
            }

            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(tiling_, coreIdx, localGeneration, lane);
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t hOffset = WySavedHOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                const uint64_t dhOffset = WyDhOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                if (tiling_.stage == 0) {
                    // Reuse the proven 64x64 FP32 output tile from
                    // chunk_kda_bwd_dav twice.  A single 64x128 FP32 output
                    // only committed one 16-KiB result tile on A2.
                    RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                        resource, dO_, token128, h_, hOffset,
                        dq_, token128, validLen, 64, 128, 128);
                    RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                        resource, dO_, token128,
                        h_, hOffset + 64U * 128U,
                        dq_, token128 + 64U, validLen, 64, 128, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else if (tiling_.stage == 1) {
                    RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                        resource, vNew_, token128, dh_, dhOffset,
                        dk_, token128, validLen, 64, 128, 128);
                    RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                        resource, vNew_, token128,
                        dh_, dhOffset + 64U * 128U,
                        dk_, token128 + 64U, validLen, 64, 128, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else if (tiling_.stage == 2) {
                    RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                        resource, a_, token64, dvScan_, token128,
                        slot + tiling_.dVbOffset, validLen, validLen, 128,
                        64, 128, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else if (tiling_.stage == 4) {
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, token128, h_, hOffset,
                        slot + tiling_.dWOffset, validLen, 64, 128, 128);
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, token128,
                        h_, hOffset + 64U * 128U,
                        slot + tiling_.dWOffset + 64U * sizeof(bfloat16_t),
                        validLen, 64, 128, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else if (tiling_.stage == 5) {
                    // Stage 4 packs the negated BF16 dW into the not-yet-live
                    // dg output storage.  A fresh L0 avoids the unsupported
                    // BF16-MMAD -> FP32-MMAD resource chain seen on A2.
                    RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                        resource, a_, token64, dg_, 2U * token128,
                        slot + tiling_.dKgbOffset, validLen, validLen, 128,
                        64, 128, 128);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else if (tiling_.stage == 6) {
                    const uint64_t zBase = WyDAkkBf16TaskBase(
                        tiling_, task.batchIdx, head, task.begin);
                    RunLayouts<Bf16SquareRightTransposeMmad, RowMajor, ColumnMajor>(
                        resource, dAkk_, zBase, a_, token64,
                        slot + tiling_.zaOutputOffset,
                        validLen, validLen, validLen, 64, 64, 64);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                } else {
                    RunLayouts<Fp32SquareLeftTransposeMmad, ColumnMajor, RowMajor>(
                        resource, a_, token64, dvScan_, token128,
                        slot + tiling_.zaInputOffset, validLen, validLen,
                        validLen, 64, 128, 64);
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        WyCubeToVectorFlag(lane));
                }
            }
        }
        if (tiling_.stage == 6 || tiling_.stage == 7) {
            AscendC::CrossCoreWaitFlag(WyVectorToCubeFlag(0));
        }
    }

private:
    template <typename Fp32C64Mmad, typename Bf16C64Mmad,
              typename Fp32AT64x128Mmad,
              typename Bf16SquareRightTransposeMmad,
              typename Fp32SquareLeftTransposeMmad,
              typename RowMajor, typename ColumnMajor>
    __aicore__ inline void ProcessFused(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource)
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(true);
        }
        // One notification stream is sufficient because both AIC and AIV
        // traverse the same task/head/stage order.  Reverse acknowledgement
        // is mandatory: a long sequence can emit far more than the hardware
        // limit of 15 unacknowledged cross-core notifications.
        constexpr uint32_t kS0Ready = 0;
        constexpr uint32_t kS1Ready = 1;
        constexpr uint32_t kS2Ready = 4;
        constexpr uint32_t kS3aReady = 5;
        constexpr uint32_t kNegDwReady = 2;
        constexpr uint32_t kZbReady = 3;
        constexpr uint32_t kTzaReady = 6;
        constexpr uint32_t kTaskDone = 7;
        // The fused path is task-serial: S7 waits for the final AIV writeback
        // before the next task can enter S0.  Consequently the ping-pong slot
        // is already free here and needs no second workspace-free protocol.
        // Keeping a second reverse-counted protocol made a slot cross the
        // 15-notification depth at long sequence lengths and deadlock.

        const uint32_t coreIdx = AscendC::GetBlockIdx();
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
            BeginFusedMmadPhase();

            // S0: dO @ h^T -> dq_base, followed by gate/scale postprocess.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t hOffset = WySavedHOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, dO_, token128, h_, hOffset,
                    dq_, token128, validLen, 64, 128, 128);
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, dO_, token128, h_, hOffset + 64U * 128U,
                    dq_, token128 + 64U, validLen, 64, 128, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS0Ready);

            // S1: v_new @ dh^T -> dk_base, followed by gate postprocess.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t dhOffset = WyDhOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, vNew_, token128, dh_, dhOffset,
                    dk_, token128, validLen, 64, 128, 128);
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, vNew_, token128, dh_, dhOffset + 64U * 128U,
                    dk_, token128 + 64U, validLen, 64, 128, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS1Ready);

            // S2: A^T @ dv_scan -> dVb; AIV emits dv and db_base.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, dvScan_, token128,
                    slot + tiling_.dVbOffset, validLen, validLen, 128,
                    64, 128, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS2Ready);

            // FP32 S0-S2 complete.  Drain before entering BF16 S3a/S3b.
            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S3a: dW/zV.  AIV independently builds kE, consumes dW, and
            // publishes the negated BF16 dW required by S3b.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t hOffset = WySavedHOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, token128, h_, hOffset,
                    slot + tiling_.dWOffset, validLen, 64, 128, 128);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, token128, h_, hOffset + 64U * 128U,
                    slot + tiling_.dWOffset + 64U * sizeof(bfloat16_t),
                    validLen, 64, 128, 128);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, token128, v_, token128,
                    slot + tiling_.zVOffset, validLen, validLen, 64, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS3aReady);
            AscendC::CrossCoreWaitFlag(kNegDwReady);

            // S3b: zW = (-dW) @ kE^T; AIV forms and stores Zb.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(bfloat16_t),
                    workspace_,
                    (slot + tiling_.kEOffset) / sizeof(bfloat16_t),
                    slot + tiling_.zWOffset, validLen, validLen, 64, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS0Ready);

            // BF16 S3a/S3b complete.  Start a clean FP32 phase for S5.
            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S5: A^T @ (-dW), then the gradient/gate vector stage.  S3a's
            // negated dW remains live in the slot, so the fused path avoids
            // the former S4 recompute and temporary dg round-trip.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(bfloat16_t),
                    slot + tiling_.dKgbOffset, validLen, validLen, 128,
                    64, 128, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS1Ready);
            // S6 consumes Zb, while the S5 AIV gradient path is independent.
            AscendC::CrossCoreWaitFlag(kZbReady);

            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S6: Zb @ A^T -> Tza; AIV persists Tza into dead dv_scan.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                const uint64_t zBase = WyDAkkBf16TaskBase(
                    tiling_, task.batchIdx, head, task.begin);
                RunLayouts<Bf16SquareRightTransposeMmad,
                           RowMajor, ColumnMajor>(
                    resource, dAkk_, zBase, a_, token64,
                    slot + tiling_.zaOutputOffset,
                    validLen, validLen, validLen, 64, 64, 64);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS2Ready);
            AscendC::CrossCoreWaitFlag(kTzaReady);

            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S7: A^T @ Tza and final causal/sign postprocess for dAkk.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token128 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32SquareLeftTransposeMmad,
                           ColumnMajor, RowMajor>(
                    resource, a_, token64, dvScan_, token128,
                    slot + tiling_.zaInputOffset,
                    validLen, validLen, validLen, 64, 128, 64);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS3aReady);
            AscendC::CrossCoreWaitFlag(kTaskDone);

            EndFusedMmadPhase();
            // The final S7 drain prevents workspace reuse by the next task;
            // the next iteration opens a fresh FP32 phase.
        }
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void BeginFusedMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        }
    }

    __aicore__ inline void EndFusedMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        }
    }

    template <typename Mmad, typename RowMajor>
    __aicore__ inline void RunRowMajorToOutput(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        GM_ADDR cAddr, uint64_t cOffset, uint32_t m, uint32_t k, uint32_t n,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<bfloat16_t> a;
        AscendC::GlobalTensor<bfloat16_t> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(cAddr) + cOffset);
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<bfloat16_t, RowMajor>(64, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, RowMajor>(64, bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Mmad, typename LayoutA, typename LayoutB>
    __aicore__ inline void RunLayouts(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t k, uint32_t n,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<bfloat16_t> a;
        AscendC::GlobalTensor<bfloat16_t> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(workspace_) + cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<bfloat16_t, LayoutA>(64, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, LayoutB>(64, bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, Catlass::layout::RowMajor>(64, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunTransposeBToOutput(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        GM_ADDR cAddr, uint64_t cOffset, uint32_t m, uint32_t n,
        uint32_t physicalN, uint32_t k)
    {
        AscendC::GlobalTensor<bfloat16_t> a;
        AscendC::GlobalTensor<bfloat16_t> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(cAddr) + cOffset);
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<bfloat16_t, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, ColumnMajor>(k, physicalN),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, physicalN),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Fp32C128Mmad, typename Bf16C128Mmad,
              typename Bf16C64Mmad, typename Fp32A64x128Mmad,
              typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunS0(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot)
    {
        const uint64_t token128 = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t token64 = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 64);
        const uint64_t hOffset = WySavedHOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        const uint64_t dhOffset = WyDhOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        RunTransposeB<Fp32C128Mmad, RowMajor, ColumnMajor>(
            resource, dO_, token128, h_, hOffset, slot + tiling_.dqRawOffset,
            validLen, 128, validLen, 128);
        RunTransposeB<Bf16C128Mmad, RowMajor, ColumnMajor>(
            resource, dvScan_, token128, h_, hOffset, slot + tiling_.dWOffset,
            validLen, 128, validLen, 128);
        RunTransposeB<Fp32C128Mmad, RowMajor, ColumnMajor>(
            resource, vNew_, token128, dh_, dhOffset, slot + tiling_.dkRawOffset,
            validLen, 128, validLen, 128);
        RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
            resource, dvScan_, token128, v_, token128, slot + tiling_.zVOffset,
            validLen, validLen, 64, 128);
        RunRowMajor<Fp32A64x128Mmad, RowMajor>(
            resource, a_, token64, dvScan_, token128, slot + tiling_.dVbOffset,
            validLen, validLen, 128, 64, 128, 128);
    }

    template <typename Bf16C64Mmad, typename Fp32A64x128Mmad,
              typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunS1(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot)
    {
        const uint64_t token64 = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 64);
        RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
            resource, workspace_, (slot + tiling_.dWOffset) / sizeof(bfloat16_t),
            workspace_, (slot + tiling_.kEOffset) / sizeof(bfloat16_t),
            slot + tiling_.zWOffset, validLen, validLen, 64, 128);
        RunRowMajor<Fp32A64x128Mmad, RowMajor>(
            resource, a_, token64, workspace_,
            (slot + tiling_.dWOffset) / sizeof(bfloat16_t),
            slot + tiling_.dKgbOffset, validLen, validLen, 128, 64, 128, 128);
    }

    template <typename Fp32SquareMmad, typename Bf16SquareMmad, typename RowMajor>
    __aicore__ inline void RunDA(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot)
    {
        const uint64_t token64 = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 64);
        RunRowMajor<Bf16SquareMmad, RowMajor>(
            resource, workspace_,
            (slot + tiling_.zaInputOffset) / sizeof(bfloat16_t), a_, token64,
            slot + tiling_.zaOutputOffset, validLen, validLen, validLen, 64, 64, 64);
        RunRowMajor<Fp32SquareMmad, RowMajor>(
            resource, a_, token64, workspace_,
            (slot + tiling_.zaOutputOffset) / sizeof(bfloat16_t),
            slot + tiling_.zaInputOffset, validLen, validLen, validLen, 64, 64, 64);
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunTransposeB(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t n, uint32_t physicalN,
        uint32_t k)
    {
        AscendC::GlobalTensor<bfloat16_t> a;
        AscendC::GlobalTensor<bfloat16_t> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(workspace_) + cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(64, k), Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, ColumnMajor>(k, physicalN),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, physicalN),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Mmad, typename RowMajor>
    __aicore__ inline void RunRowMajor(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t k, uint32_t n,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<bfloat16_t> a;
        AscendC::GlobalTensor<bfloat16_t> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(workspace_) + cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<bfloat16_t, RowMajor>(64, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, RowMajor>(64, bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    GM_ADDR v_;
    GM_ADDR vNew_;
    GM_ADDR a_;
    GM_ADDR h_;
    GM_ADDR dO_;
    GM_ADDR dh_;
    GM_ADDR dvScan_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR workspace_;
    ChunkKdaBwdWyTilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_WY_CUBE_H
