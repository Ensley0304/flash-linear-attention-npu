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

// A5-local two-output path for contractions that share the complete left
// operand.  The left tile is copied to L1/L0A once.  The two right tiles use
// independent L1B buffers, so MTE2 can prefetch the second right operand while
// MTE1/MMAD consumes the first one.  L0B remains single-buffered.  Two compact
// L0C result planes let FIX copy the first output while M computes the second.
template <class ArchTag_, class ElementC_, class TileCopy_>
struct WyTileGemmSharedLeftDualRightDirect {
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
    static constexpr int32_t kEventL1B0 = WyTileGemmDirectEvent::kL1B;
    static constexpr int32_t kEventL1B1 = 2;
    static constexpr int32_t kEventL0A = WyTileGemmDirectEvent::kL0A;
    static constexpr int32_t kEventL0B = WyTileGemmDirectEvent::kL0B;
    static constexpr int32_t kEventL0C0 = WyTileGemmDirectEvent::kL0C;
    static constexpr int32_t kEventL0C1 = 1;
    static constexpr uint32_t kL1PlaneBytes =
        128 * 256 * sizeof(ElementA);
    static constexpr uint32_t kL0CPlaneBytes =
        64 * 128 * sizeof(ElementAccumulator);
    static_assert(3 * kL1PlaneBytes <= 512 * 1024,
                  "Kernel C shared-left L1 buffers exceed capacity");
    static_assert(2 * kL0CPlaneBytes <= ArchTag::L0C_SIZE,
                  "Kernel C dual-output L0C buffers exceed capacity");

    CATLASS_DEVICE
    explicit WyTileGemmSharedLeftDualRightDirect(
        Catlass::Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            l1A_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1B0_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                kL1PlaneBytes);
            l1B1_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                2 * kL1PlaneBytes);
            l0A_ = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0B_ = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0C0_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            l0C1_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                kL0CPlaneBytes);
        }
    }

    template <class TensorA, class TensorB0, class TensorC0,
              class TensorB1, class TensorC1>
    CATLASS_DEVICE
    void operator()(TensorA &a, TensorB0 &b0, TensorC0 &c0,
                    Catlass::GemmCoord const &shape0,
                    TensorB1 &b1, TensorC1 &c1,
                    Catlass::GemmCoord const &shape1)
    {
        using CopyGmToL1A =
            typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B0 =
            typename TileCopy::template CopyGmToL1B<TensorB0>;
        using CopyGmToL1B1 =
            typename TileCopy::template CopyGmToL1B<TensorB1>;
        using CopyL0CToDst0 =
            typename TileCopy::template CopyL0CToDst<TensorC0>;
        using CopyL0CToDst1 =
            typename TileCopy::template CopyL0CToDst<TensorC1>;
        CopyGmToL1A copyGmA;
        CopyGmToL1B0 copyGmB0;
        CopyGmToL1B1 copyGmB1;
        CopyL1ToL0A copyL0A;
        CopyL1ToL0B copyL0B;
        CopyL0CToDst0 copyC0;
        CopyL0CToDst1 copyC1;
        TileMmad mm;

        const uint32_t m = shape0.m() == 1 ? 16 : shape0.m();
        const uint32_t k = shape0.k();
        const uint32_t n0 = shape0.n();
        const uint32_t n1 = shape1.n();
        auto l1A = tla::MakeTensor(
            l1A_, tla::MakeLayout<ElementA, LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto l1B0 = tla::MakeTensor(
            l1B0_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n0),
            Catlass::Arch::PositionL1{});
        auto l1B1 = tla::MakeTensor(
            l1B1_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n1),
            Catlass::Arch::PositionL1{});
        auto tileL1A = GetTile(
            l1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B0 = GetTile(
            l1B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n0));
        auto tileL1B1 = GetTile(
            l1B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n1));

        // All three L1 events are owned by the surrounding S3 MMAD phase.
        // Hoisting event-2 initialization/drain out of the per-head call is
        // essential: otherwise the synchronization cost erases the saved
        // left-operand copy.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B0);
        copyGmB0(l1B0, b0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B1);
        copyGmB1(l1B1, b1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B1);

        auto l0A = tla::MakeTensor(
            l0A_, tla::MakeLayout<ElementA, LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        copyL0A(l0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);

        auto l0B0 = tla::MakeTensor(
            l0B_, tla::MakeLayout<ElementB, LayoutTagL0B>(k, n0),
            Catlass::Arch::PositionL0B{});
        auto l0C0 = tla::MakeTensor(
            l0C0_, tla::MakeLayoutL0C(m, n0),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        copyL0B(l0B0, tileL1B0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C0);
        mm(l0C0, l0A, l0B0, m, n0, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);
        copyC0(c0, l0C0, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);

        auto l0B1 = tla::MakeTensor(
            l0B_, tla::MakeLayout<ElementB, LayoutTagL0B>(k, n1),
            Catlass::Arch::PositionL0B{});
        auto l0C1 = tla::MakeTensor(
            l0C1_, tla::MakeLayoutL0C(m, n1),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        copyL0B(l0B1, tileL1B1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C1);
        mm(l0C1, l0A, l0B1, m, n1, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
        copyC1(c1, l0C1, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
    }

private:
    AscendC::LocalTensor<ElementA> l1A_;
    AscendC::LocalTensor<ElementB> l1B0_;
    AscendC::LocalTensor<ElementB> l1B1_;
    AscendC::LocalTensor<ElementA> l0A_;
    AscendC::LocalTensor<ElementB> l0B_;
    AscendC::LocalTensor<ElementAccumulator> l0C0_;
    AscendC::LocalTensor<ElementAccumulator> l0C1_;
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ElementC64DualRightMmad =
            WyTileGemmSharedLeftDualRightDirect<
                ArchTag, Element, ElementC64Copy>;
#else
        using ElementC64DualRightMmad = ElementC64Mmad;
#endif
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
            Fp32C64Mmad, ElementC64Mmad, ElementC64DualRightMmad,
            Fp32AT64x128Mmad,
            ElementSquareRightTransposeMmad,
            Fp32SquareLeftTransposeMmad,
            RowMajor, ColumnMajor>(resource);
    }

    template <typename Fp32C64Mmad, typename Bf16C64Mmad,
              typename Bf16C64DualRightMmad,
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

        // Every direct MMAD call returns its L1/L0 event credits to the
        // reusable state before it returns.  Initialize the complete event
        // set once for the WY phase instead of draining and recreating it at
        // every formula boundary of every owner.  The final drain still
        // protects the following Intra phase, which reuses the same local
        // storage and event ids with a different layout.
        BeginSharedLeftMmadPhase();
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                RunSharedLeftDualTransposeB<
                    Bf16C64DualRightMmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV,
                    h_, hOffset, slot + tiling_.dWOffset,
                    validLen, 128, 128, V_DIM,
                    v_, tokenV, slot + tiling_.zVOffset,
                    validLen, 64);
#else
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, h_, hOffset,
                    slot + tiling_.dWOffset, validLen, 128, 128, V_DIM);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, v_, tokenV,
                    slot + tiling_.zVOffset, validLen, validLen, 64, V_DIM);
#endif
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
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kS3aReady);
            AscendC::CrossCoreWaitFlag(kTaskDone);
        }
        EndSharedLeftMmadPhase();
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

    __aicore__ inline void BeginSharedLeftMmadPhase()
    {
        BeginFusedMmadPhase();
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        }
    }

    __aicore__ inline void EndSharedLeftMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        }
        EndFusedMmadPhase();
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

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunSharedLeftDualTransposeB(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset,
        GM_ADDR b0Addr, uint64_t b0Offset, uint64_t c0ByteOffset,
        uint32_t m, uint32_t n0, uint32_t physicalN0, uint32_t k,
        GM_ADDR b1Addr, uint64_t b1Offset, uint64_t c1ByteOffset,
        uint32_t n1, uint32_t physicalN1)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b0;
        AscendC::GlobalTensor<DataT> b1;
        a.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b0.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(b0Addr) + b0Offset);
        b1.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(b1Addr) + b1Offset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c0;
        AscendC::GlobalTensor<CType> c1;
        c0.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            c0ByteOffset / sizeof(CType));
        c1.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            c1ByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb0 = tla::MakeTensor(
            b0, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN0),
            Catlass::Arch::PositionGM{});
        auto tb1 = tla::MakeTensor(
            b1, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN1),
            Catlass::Arch::PositionGM{});
        auto tc0 = tla::MakeTensor(
            c0, tla::MakeLayout<CType, RowMajor>(64, physicalN0),
            Catlass::Arch::PositionGM{});
        auto tc1 = tla::MakeTensor(
            c1, tla::MakeLayout<CType, RowMajor>(64, physicalN1),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(
            ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB0 = GetTile(
            tb0, tla::MakeCoord(0, 0), tla::MakeShape(k, n0));
        auto tileB1 = GetTile(
            tb1, tla::MakeCoord(0, 0), tla::MakeShape(k, n1));
        auto tileC0 = GetTile(
            tc0, tla::MakeCoord(0, 0), tla::MakeShape(m, n0));
        auto tileC1 = GetTile(
            tc1, tla::MakeCoord(0, 0), tla::MakeShape(m, n1));
        Catlass::GemmCoord shape0{m, n0, k};
        Catlass::GemmCoord shape1{m, n1, k};
        Mmad mm(resource);
        mm(tileA, tileB0, tileC0, shape0,
           tileB1, tileC1, shape1);
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
