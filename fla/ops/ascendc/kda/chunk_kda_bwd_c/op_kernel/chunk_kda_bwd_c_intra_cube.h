#ifndef CHUNK_KDA_BWD_C_INTRA_CUBE_H
#define CHUNK_KDA_BWD_C_INTRA_CUBE_H

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

#include "chunk_kda_bwd_c_intra_common.h"

namespace KDA {

template <typename LayoutA, uint32_t L1_M, uint32_t L1_K>
class CIntraSingleTileMmad {
private:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Catlass::Arch::Ascend950;
#else
    using ArchTag = Catlass::Arch::AtlasA2;
#endif
    using RowMajor = Catlass::layout::RowMajor;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, float, LayoutA, float, RowMajor, float, RowMajor>;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, float, typename Copy::LayoutTagL1A>;
    static constexpr int32_t kEventL1A = 0;
    static constexpr int32_t kEventL1B = 1;
    static constexpr int32_t kEventL0A = 0;
    static constexpr int32_t kEventL0B = 1;
    static constexpr int32_t kEventL0C = 0;

public:
    __aicore__ inline explicit CIntraSingleTileMmad(
        Catlass::Arch::Resource<ArchTag> &resource)
        : resource_(resource)
    {
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA &a, TensorB &b, TensorC &c,
        const Catlass::GemmCoord &shape)
    {
        constexpr uint32_t kL1PlaneBytes = 128 * 128 * sizeof(float);
        auto l1AStorage =
            resource_.l1Buf.template GetBufferByByte<float>(0);
        auto l1BStorage =
            resource_.l1Buf.template GetBufferByByte<float>(kL1PlaneBytes);
        auto l0AStorage =
            resource_.l0ABuf.template GetBufferByByte<float>(0);
        auto l0BStorage =
            resource_.l0BBuf.template GetBufferByByte<float>(0);
        auto l0CStorage =
            resource_.l0CBuf.template GetBufferByByte<float>(0);

        const uint32_t m = shape.m() == 1 ? 16 : shape.m();
        // A5 packed L1 layouts require compile-time physical extents.  Use
        // the proven maximum lower/upper shapes and take the runtime tile
        // from that base.  A generic 128x128 base changes the blocked mapping
        // and permutes N columns.
        auto l1ABase = tla::MakeTensor(
            l1AStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL1A>(
                tla::Int<L1_M>{}, tla::Int<L1_K>{}),
            Catlass::Arch::PositionL1{});
        auto l1BBase = tla::MakeTensor(
            l1BStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL1B>(
                tla::Int<L1_K>{}, tla::Int<128>{}),
            Catlass::Arch::PositionL1{});
        auto l1A = GetTile(
            l1ABase, tla::MakeCoord(0, 0),
            tla::MakeShape(m, shape.k()));
        auto l1B = GetTile(
            l1BBase, tla::MakeCoord(0, 0),
            tla::MakeShape(shape.k(), shape.n()));
        typename Copy::template CopyGmToL1A<TensorA> copyGmA;
        typename Copy::template CopyGmToL1B<TensorB> copyGmB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        copyGmB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);

        auto l0A = tla::MakeTensor(
            l0AStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL0A>(
                m, shape.k()),
            Catlass::Arch::PositionL0A{});
        auto l0B = tla::MakeTensor(
            l0BStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL0B>(
                shape.k(), shape.n()),
            Catlass::Arch::PositionL0B{});
        typename Copy::CopyL1ToL0A copyA;
        typename Copy::CopyL1ToL0B copyB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        copyA(l0A, l1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        copyB(l0B, l1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);

        auto l0C = tla::MakeTensor(
            l0CStorage, tla::MakeLayoutL0C(m, shape.n()),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
        TileMmad mm;
        // The A5 dynamic tile view does not carry sufficient static shape
        // information for the short overload to recover the physical N
        // mapping.  Pass M/N/K explicitly, as in the proven PR294 pipeline.
        mm(l0C, l0A, l0B,
           m, shape.n(), shape.k(), true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C);

        typename Copy::template CopyL0CToDst<TensorC> fix;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        fix(c, l0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
    }

private:
    Catlass::Arch::Resource<ArchTag> &resource_;
};

class ChunkKdaBwdCIntraCubeProcess {
private:
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

public:
    __aicore__ ChunkKdaBwdCIntraCubeProcess(
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace)
        : cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdCTilingData &tiling)
    {
        tiling_ = tiling;
        AscendC::SetHF32Mode(false);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Process()
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        Catlass::Arch::Resource<ArchTag> resource;
        CIntraSingleTileMmad<RowMajor, 64, kCIntraChunkSize>
            lower(resource);
        CIntraSingleTileMmad<ColumnMajor, 32, 2 * kCIntraChunkSize>
            upper(resource);

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t headWindows =
            (static_cast<uint32_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t groups =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        uint64_t window = 0;
        for (uint64_t group = core; group < groups;
             group += tiling_.usedCoreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(group / headWindows);
            const uint32_t headBase =
                static_cast<uint32_t>(group % headWindows) * 2U;
            const uint32_t headCount =
                headBase + 1U < static_cast<uint32_t>(tiling_.headNum) ? 2U : 1U;
            const CIntraTask task = GetCIntraTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validC = task.end - task.begin;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t processRowBlock =
                tiling_.isVarLen == 0 ? 32U : kRowBlock;
#else
            const uint32_t processRowBlock = kRowBlock;
#endif
            for (uint32_t rowStart = 0; rowStart < validC;
                 rowStart += processRowBlock, ++window) {
                const uint32_t validRows =
                    rowStart + processRowBlock <= validC ?
                        processRowBlock : validC - rowStart;
                const uint32_t prefix = rowStart + validRows;
                const uint32_t lowerK = (prefix + 15U) & ~15U;
                const uint32_t future = validC - rowStart;
                for (uint32_t lane = 0; lane < headCount; ++lane) {
                    AscendC::CrossCoreWaitFlag(kCIntraVecReadyFlag);
                    const uint32_t slot =
                        CIntraWorkspaceSlot(window, lane);
                    const uint64_t slotBase =
                        static_cast<uint64_t>(core) *
                            tiling_.workspaceCoreSize +
                        static_cast<uint64_t>(slot) *
                            tiling_.workspaceSlotSize;
                    RunLower(lower, slotBase, lowerK, processRowBlock);
                    // Upper-A/B are padded to the physical row tile.  Keep M
                    // at that tile size even for a varlen tail; Vector-Post
                    // consumes only validRows.  A non-aligned M (for example
                    // six rows) is not a valid Cube tile and corrupts the
                    // final tail chunk.
                    RunUpper(upper, slotBase, future, processRowBlock);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        kCIntraCubeReadyFlag);
                }
            }
        }
    }

private:
    template <typename Mmad>
    __aicore__ inline void RunLower(
        Mmad &mm, uint64_t slotBase, uint32_t lowerK,
        uint32_t processRowBlock)
    {
        Run<RowMajor>(
            mm, slotBase + tiling_.intraALowerOffset,
            slotBase + tiling_.intraBLowerOffset,
            slotBase + tiling_.intraResultRegionOffset +
                tiling_.intraResultDqOffset,
            2 * processRowBlock, tiling_.keyDim, lowerK,
            lowerK, tiling_.keyDim, tiling_.keyDim);
    }

    template <typename Mmad>
    __aicore__ inline void RunUpper(
        Mmad &mm, uint64_t slotBase, uint32_t future,
        uint32_t processRowBlock)
    {
        const uint32_t reduction = 2 * future;
        Run<ColumnMajor>(
            mm, slotBase + tiling_.intraAUpperOffset,
            slotBase + tiling_.intraBUpperOffset,
            slotBase + tiling_.intraResultRegionOffset +
                tiling_.intraResultDkUpperOffset,
            processRowBlock, tiling_.keyDim, reduction,
            reduction, tiling_.keyDim, tiling_.keyDim);
    }

    template <typename LayoutA, typename Mmad>
    __aicore__ inline void Run(
        Mmad &mm, uint64_t aByte, uint64_t bByte, uint64_t cByte,
        uint32_t m, uint32_t n, uint32_t k,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        using RowMajor = Catlass::layout::RowMajor;
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + aByte));
        b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + bByte));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + cByte));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<float, LayoutA>(m, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<float, RowMajor>(k, bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<float, RowMajor>(m, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto ba = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bb = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto bc = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        mm(ba, bb, bc, Catlass::GemmCoord{m, n, k});
    }

    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_INTRA_CUBE_H
