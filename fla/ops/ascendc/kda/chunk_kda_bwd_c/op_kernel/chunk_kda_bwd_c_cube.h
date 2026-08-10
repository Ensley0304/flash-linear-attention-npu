#ifndef CHUNK_KDA_BWD_C_CUBE_H
#define CHUNK_KDA_BWD_C_CUBE_H

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_kda_bwd_c_common.h"

namespace KDA {

struct WyTileGemmDirectEvent {
    static constexpr int32_t kL1A = 0;
    static constexpr int32_t kL1B = 1;
    static constexpr int32_t kL0A = 0;
    static constexpr int32_t kL0B = 1;
    static constexpr int32_t kL0C = 0;
};

// The fused path owns the MM layout and L1/L0 event lifecycle at phase scope.
// Individual GEMMs only bind the shared resource buffers.  BF16/FP32 phase
// boundaries are still fully drained, preserving the proven type-transition
// safety while avoiding constructor/destructor synchronization for every
// contraction.
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
    static constexpr int32_t kEventL1A = WyTileGemmDirectEvent::kL1A;
    static constexpr int32_t kEventL1B = WyTileGemmDirectEvent::kL1B;
    static constexpr int32_t kEventL0A = WyTileGemmDirectEvent::kL0A;
    static constexpr int32_t kEventL0B = WyTileGemmDirectEvent::kL0B;
    static constexpr int32_t kEventL0C = WyTileGemmDirectEvent::kL0C;

    // P0's largest reduction is V=256.  Its worst L0A/L0B products are
    // 64x256 and 256x128 respectively, both 32/64 KiB in FP16/BF16, so keep
    // the whole reduction in one MMAD.  Besides matching the proven PR190
    // direct-tile pattern, this avoids an unnecessary partial-sum lifecycle.
    static constexpr uint32_t kReductionTile = 256;
    static constexpr uint32_t kL1PlaneBytes =
        128 * 256 * sizeof(ElementA);
    static_assert(2 * kL1PlaneBytes <= 512 * 1024,
                  "Kernel C direct MMAD L1 planes exceed A2/A3 L1");
    static_assert(64 * 256 * sizeof(ElementA) <= ArchTag::L0A_SIZE,
                  "Kernel C direct MMAD L0A exceeds capacity");
    static_assert(128 * 256 * sizeof(ElementB) <= ArchTag::L0B_SIZE,
                  "Kernel C V256 direct MMAD L0B exceeds capacity");
    static_assert(64 * 256 * sizeof(ElementAccumulator) <= ArchTag::L0C_SIZE,
                  "Kernel C V256 FP32 L0C exceeds capacity");

    CATLASS_DEVICE
    explicit WyTileGemmDirect(Catlass::Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            l1A_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1B_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                kL1PlaneBytes);
            l0A_ = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0B_ = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0C_ =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
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
        using CopyL0CToDst =
            typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmA;
        CopyGmToL1B copyGmB;
        CopyL1ToL0A copyL0A;
        CopyL1ToL0B copyL0B;
        CopyL0CToDst copyC;
        TileMmad mm;

        const uint32_t m = shape.m() == 1 ? 16 : shape.m();
        const uint32_t n = shape.n();
        const uint32_t k = shape.k();
        auto l1A = tla::MakeTensor(
            l1A_, tla::MakeLayout<ElementA, LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto l1B = tla::MakeTensor(
            l1B_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        copyGmB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);

        auto l0C = tla::MakeTensor(
            l0C_, tla::MakeLayoutL0C(m, n),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
        for (uint32_t k0 = 0; k0 < k; k0 += kReductionTile) {
            const uint32_t curK =
                k - k0 < kReductionTile ? k - k0 : kReductionTile;
            auto l0A = tla::MakeTensor(
                l0A_,
                tla::MakeLayout<ElementA, LayoutTagL0A>(m, curK),
                Catlass::Arch::PositionL0A{});
            auto l0B = tla::MakeTensor(
                l0B_,
                tla::MakeLayout<ElementB, LayoutTagL0B>(curK, n),
                Catlass::Arch::PositionL0B{});
            auto tileA = GetTile(
                l1A, tla::MakeCoord(0, k0), tla::MakeShape(m, curK));
            auto tileB = GetTile(
                l1B, tla::MakeCoord(k0, 0), tla::MakeShape(curK, n));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            copyL0A(l0A, tileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
            copyL0B(l0B, tileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
            const bool lastK = k0 + curK == k;
            const uint8_t unitFlag = lastK ? 0b11 : 0b10;
            mm(l0C, l0A, l0B, m, n, curK, k0 == 0, unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        copyC(c, l0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
    }

private:
    AscendC::LocalTensor<ElementA> l1A_;
    AscendC::LocalTensor<ElementB> l1B_;
    AscendC::LocalTensor<ElementA> l0A_;
    AscendC::LocalTensor<ElementB> l0B_;
    AscendC::LocalTensor<ElementAccumulator> l0C_;
};

template <typename DataT, uint32_t V_DIM>
class ChunkKdaBwdCCubeProcess {
public:
    __aicore__ ChunkKdaBwdCCubeProcess(
        GM_ADDR v, GM_ADDR vNew, GM_ADDR a, GM_ADDR h, GM_ADDR dh,
        GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dg,
        GM_ADDR dAkk, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR workspace)
        : v_(v), vNew_(vNew), a_(a), h_(h), dh_(dh),
          dvScan_(dvScan), dq_(dq), dk_(dk), dg_(dg), dAkk_(dAkk),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          workspace_(workspace) {}

    __aicore__ inline void Init(const ChunkKdaBwdCTilingData &tiling)
    {
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        using Element = DataT;

        using Fp32C64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, RowMajor, Element, ColumnMajor, float, RowMajor>;
        using ElementC64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, RowMajor, Element, ColumnMajor, Element, RowMajor>;
        using Fp32AT64x128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, ColumnMajor, Element, RowMajor, float, RowMajor>;
        using ElementSquareRightTransposeCopy =
            Catlass::Gemm::Tile::PackedTileCopyTla<
                ArchTag, Element, RowMajor, Element, ColumnMajor, Element, RowMajor>;
        using Fp32SquareLeftTransposeCopy =
            Catlass::Gemm::Tile::PackedTileCopyTla<
                ArchTag, Element, ColumnMajor, Element, RowMajor, float, RowMajor>;

        using Fp32C64Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32C64Copy>;
        using ElementC64Mmad =
            WyTileGemmDirect<ArchTag, Element, ElementC64Copy>;
        using Fp32AT64x128Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32AT64x128Copy>;
        using ElementSquareRightTransposeMmad =
            WyTileGemmDirect<ArchTag, Element,
                             ElementSquareRightTransposeCopy>;
        using Fp32SquareLeftTransposeMmad =
            WyTileGemmDirect<ArchTag, float,
                             Fp32SquareLeftTransposeCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        ProcessFused<
            Fp32C64Mmad, ElementC64Mmad, Fp32AT64x128Mmad,
            ElementSquareRightTransposeMmad,
            Fp32SquareLeftTransposeMmad,
            RowMajor, ColumnMajor>(resource);
    }

    template <typename Fp32C64Mmad, typename Bf16C64Mmad,
              typename Fp32AT64x128Mmad,
              typename Bf16SquareRightTransposeMmad,
              typename Fp32SquareLeftTransposeMmad,
              typename RowMajor, typename ColumnMajor>
    __aicore__ inline void ProcessFused(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource)
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource)
#endif
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
        constexpr uint32_t kZbReady = 3;
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
            const WyChunkTask task = GetWyChunkTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            BeginFusedMmadPhase();

            // S0: dq_raw is produced by Kernel A.  Publish the dependency
            // credit immediately; AIV applies exp2(gk) and scale while AIC
            // starts the independent dk_raw contraction below.
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS0Ready);
            // S1: v_new @ dh^T -> dk_base, followed by gate postprocess.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t tokenK = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t dhOffset = WyDhOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                // dk_base is one [C,128] result.  A5 can retain the complete
                // right operand and FP32 accumulator, so issue one N=128 MMAD
                // instead of two adjacent N=64 contractions.
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, vNew_, tokenV, dh_, dhOffset,
                    dk_, tokenK, validLen, 128, 128, V_DIM);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS1Ready);

            // S2: A^T @ dv_scan -> dVb; AIV emits dv and db_base.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, dvScan_, tokenV,
                    slot + tiling_.dVbOffset, validLen, validLen, V_DIM,
                    64, V_DIM, V_DIM);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS2Ready);

            // FP32 S0-S2 complete.  Drain before entering BF16 S3a/S3b.
            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S3a: dW_raw/zV.  Keep the shared dW operand unnegated so S3b
            // and S5 can consume it immediately.  Their downstream Vector
            // epilogues fold in the mathematical minus sign, eliminating an
            // AIV GM read/write and the corresponding AIC wait.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t hOffset = WySavedHOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                // A5 L0B can hold the complete Vx128 right operand and L0C
                // can hold the complete 64x128 FP32 accumulator.  Compute
                // dW in one direct tile instead of launching two adjacent
                // N=64 tiles; this also avoids reloading dvScan for the
                // second half.
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, h_, hOffset,
                    slot + tiling_.dWOffset, validLen, 128, 128, V_DIM);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, v_, tokenV,
                    slot + tiling_.zVOffset, validLen, validLen, 64, V_DIM);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS3aReady);

            // S3b produces zW_raw = dW_raw @ kE^T; AIV forms
            // Zb = tril(zV - zW_raw) * beta.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(DataT),
                    workspace_,
                    (slot + tiling_.kEOffset) / sizeof(DataT),
                    slot + tiling_.zWOffset, validLen, validLen, 64, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS0Ready);

            // BF16 S3a/S3b complete.  Start a clean FP32 phase for S5.
            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S5 produces dKgb_raw = A^T @ dW_raw.  The gradient/gate Vector
            // stage consumes it with a negative sign, avoiding a materialized
            // negated dW while preserving the original formulas.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(DataT),
                    slot + tiling_.dKgbOffset, validLen, validLen, 128,
                    64, 128, 128);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS1Ready);
            // S6 consumes Zb, while the S5 AIV gradient path is independent.
            AscendC::CrossCoreWaitFlag(kZbReady);

            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S6: Zb @ A^T -> Tza.  Keep Tza in the current slot; S7 is an
            // AIC consumer, so an AIC->GM->AIV->GM round trip is unnecessary.
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
            EndFusedMmadPhase();
            BeginFusedMmadPhase();

            // S7: A^T @ Tza and final causal/sign postprocess for dAkk.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32SquareLeftTransposeMmad,
                           ColumnMajor, RowMajor>(
                    resource, a_, token64, workspace_,
                    (slot + tiling_.zaOutputOffset) / sizeof(DataT),
                    slot + tiling_.zaInputOffset,
                    validLen, validLen, validLen, 64, 64, 64);
            }
            EndFusedMmadPhase();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS3aReady);
            AscendC::CrossCoreWaitFlag(kTaskDone);
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
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1A);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1B);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0B);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                WyTileGemmDirectEvent::kL0C);
        }
    }

    __aicore__ inline void EndFusedMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1A);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1B);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0A);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0B);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                WyTileGemmDirectEvent::kL0C);
        }
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunTransposeB(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t n,
        uint32_t physicalN, uint32_t k)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, physicalN),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(
            ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(
            tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(
            tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }


    template <typename Mmad, typename LayoutA, typename LayoutB>
    __aicore__ inline void RunLayouts(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t k, uint32_t n,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(workspace_) + cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, LayoutA>(64, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, LayoutB>(64, bPhysicalCols),
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        GM_ADDR cAddr, uint64_t cOffset, uint32_t m, uint32_t n,
        uint32_t physicalN, uint32_t k)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(cAddr) + cOffset);
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN),
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

    GM_ADDR v_;
    GM_ADDR vNew_;
    GM_ADDR a_;
    GM_ADDR h_;
    GM_ADDR dh_;
    GM_ADDR dvScan_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_CUBE_H
