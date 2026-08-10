#ifndef CHUNK_KDA_BWD_A_CUBE_H
#define CHUNK_KDA_BWD_A_CUBE_H

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
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_kda_bwd_a_common.h"

namespace KDA {

template <typename T, uint32_t V_DIM>
class ChunkKdaBwdACube {
private:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Catlass::Arch::Ascend950;
#else
    using ArchTag = Catlass::Arch::AtlasA2;
#endif
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

    using DvCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, ColumnMajor, T, RowMajor, T, RowMajor>;
    using Q0Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, ColumnMajor, T, RowMajor, float, RowMajor>;
    using DqCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, RowMajor, T, ColumnMajor, float, RowMajor>;
    using DACopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, RowMajor, T, ColumnMajor, float, RowMajor>;

    using DvMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DvCopy::LayoutTagL1A>;
    using Q0Mmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename Q0Copy::LayoutTagL1A>;
    using DqMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DqCopy::LayoutTagL1A>;
    using DAMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DACopy::LayoutTagL1A>;

    static constexpr uint32_t DO_BYTES = KDA_BWD_A_C * V_DIM * sizeof(T);
    static constexpr uint32_t SCRATCH_BYTES =
        V_DIM * KDA_BWD_A_K * sizeof(T);
    static constexpr uint32_t SCRATCH_OFFSET = DO_BYTES;
    static constexpr uint32_t L1_USED_BYTES = SCRATCH_OFFSET + SCRATCH_BYTES;
    static constexpr uint32_t L0A_BYTES = KDA_BWD_A_K * 64 * sizeof(T);
    static constexpr uint32_t L0B_BYTES = 64 * V_DIM * sizeof(T);
    static constexpr uint32_t L0C_BYTES =
        KDA_BWD_A_K * V_DIM * sizeof(float);
    static_assert(L1_USED_BYTES <= 512 * 1024,
                  "Kernel A L1 resident/scratch exceeds 512 KiB.");
    static_assert(L0A_BYTES <= ArchTag::L0A_SIZE,
                  "Kernel A L0A tile exceeds capacity.");
    static_assert(L0B_BYTES <= ArchTag::L0B_SIZE,
                  "Kernel A L0B tile exceeds capacity.");
    static_assert(L0C_BYTES <= ArchTag::L0C_SIZE,
                  "Kernel A FP32 L0C tile exceeds capacity.");

    static constexpr int32_t EVENT_DO = 0;
    static constexpr int32_t EVENT_SCRATCH = 1;
    static constexpr int32_t EVENT_L0A = 0;
    static constexpr int32_t EVENT_L0B = 1;
    static constexpr int32_t EVENT_L0_READY = 0;
    static constexpr int32_t EVENT_L0C = 0;

public:
    __aicore__ ChunkKdaBwdACube(
        GM_ADDR aqk, GM_ADDR qg, GM_ADDR vNew, GM_ADDR h, GM_ADDR dO,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR dv0, GM_ADDR dqRaw, GM_ADDR dAqk, GM_ADDR workspace)
        : aqk_(aqk), qg_(qg), vNew_(vNew), h_(h), dO_(dO),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          dv0_(dv0), dqRaw_(dqRaw), dAqk_(dAqk), workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdATilingData &tiling)
    {
        tiling_ = tiling;
        AscendC::SetHF32Mode(false);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint64_t ownerCount =
            static_cast<uint64_t>(tiling_.chunkNum) * tiling_.headNum;
        uint32_t generation = 0;
        for (uint64_t owner = core; owner < ownerCount;
             owner += tiling_.usedCoreNum, ++generation) {
            const uint32_t slot = generation & 1U;
            const uint32_t freeFlag =
                slot == 0 ? KDA_BWD_A_FREE_FLAG0 : KDA_BWD_A_FREE_FLAG1;
            const uint32_t readyFlag =
                slot == 0 ? KDA_BWD_A_READY_FLAG0 : KDA_BWD_A_READY_FLAG1;
            // The first use of each ping-pong slot cannot overwrite live AIV
            // data, so it needs no reverse credit.  Waiting only on reuse
            // also matches the mature producer-first MIX protocol: AIC
            // publishes ready, AIV consumes, then AIV returns slot-free.
            if (generation >= KDA_BWD_A_POST_SLOTS) {
                WaitSlotFree(slot, freeFlag);
            }

            const uint32_t taskIdx =
                static_cast<uint32_t>(owner / tiling_.headNum);
            const uint32_t head =
                static_cast<uint32_t>(owner % tiling_.headNum);
            ChunkKdaBwdATask task;
            GetChunkKdaBwdATask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
            GM_ADDR slotBase =
                KdaBwdAPostSlot(workspace_, core, slot, tiling_);
            LoadDo(task, head);
            RunDv(task, taskIdx, head, true);
            RunQ0(task, head, slotBase);
            RunDq(task, taskIdx, head);
            RunDAqk(task, head);
            AscendC::PipeBarrier<PIPE_FIX>();
            PublishSlotReady(slot, readyFlag);
        }

        // PR190-style pipeline drain.  The final generation leaves the
        // reusable L1/L0 credits set; consuming them before kernel return is
        // required for all outstanding MTE1/MMAD/Fixpipe work to retire.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

        // Drain the final one or two producer-first generations only after
        // Fixpipe has retired the corresponding ready notifications.  Reuse
        // waits inside the loop consume the previous occupant of a slot; the
        // last occupant(s) still owe AIV->AIC slot-free credits.  Returning
        // with those credits pending leaves stale state for the next launch.
        const uint32_t drainCount =
            generation < KDA_BWD_A_POST_SLOTS ?
                generation : KDA_BWD_A_POST_SLOTS;
        for (uint32_t i = 0; i < drainCount; ++i) {
            const uint32_t completedGeneration = generation - drainCount + i;
            const uint32_t slot = completedGeneration & 1U;
            const uint32_t freeFlag =
                slot == 0 ? KDA_BWD_A_FREE_FLAG0 : KDA_BWD_A_FREE_FLAG1;
            WaitSlotFree(slot, freeFlag);
        }
    }

private:
    __aicore__ inline void WaitSlotFree(
        uint32_t slot, uint32_t fallbackFlag)
    {
        (void)fallbackFlag;
        if (slot == 0) {
            Catlass::Arch::CrossCoreWaitFlag(freeFlag0_);
        } else {
            Catlass::Arch::CrossCoreWaitFlag(freeFlag1_);
        }
    }

    __aicore__ inline void PublishSlotReady(
        uint32_t slot, uint32_t fallbackFlag)
    {
        (void)fallbackFlag;
        if (slot == 0) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(readyFlag0_);
        } else {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(readyFlag1_);
        }
    }

    __aicore__ inline void LoadDo(
        const ChunkKdaBwdATask &task, uint32_t head)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        auto resident = resource.l1Buf.template GetBufferByByte<T>(0);
        auto l1Layout = tla::MakeLayout<T, typename DvCopy::LayoutTagL1B>(
            KDA_BWD_A_C, V_DIM);
        auto l1Tensor =
            tla::MakeTensor(resident, l1Layout, Catlass::Arch::PositionL1{});
        AscendC::GlobalTensor<T> gm;
        const uint64_t offset =
            KdaBwdAHeadTokenOffset(tiling_, task, head, V_DIM);
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dO_) + offset);
        auto gmLayout = tla::MakeLayout<T, RowMajor>(KDA_BWD_A_C, V_DIM);
        auto gmTensor =
            tla::MakeTensor(gm, gmLayout, Catlass::Arch::PositionGM{});
        auto block = GetTile(
            gmTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, V_DIM));
        typename DvCopy::template CopyGmToL1B<decltype(block)> copy;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
        copy(l1Tensor, block);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO);
    }

    __aicore__ inline void RunDv(
        const ChunkKdaBwdATask &task, uint32_t taskIdx, uint32_t head,
        bool waitDoReady)
    {
        (void)taskIdx;
        Catlass::Arch::Resource<ArchTag> resource;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(SCRATCH_OFFSET);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(0);
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

        AscendC::GlobalTensor<T> gmA;
        const uint64_t aOffset =
            KdaBwdAHeadTokenOffset(tiling_, task, head, KDA_BWD_A_C);
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aqk_) + aOffset);
        auto gmALayout =
            tla::MakeLayout<T, ColumnMajor>(KDA_BWD_A_C, KDA_BWD_A_C);
        auto gmATensor =
            tla::MakeTensor(gmA, gmALayout, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            gmATensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, task.validC));
        auto l1ALayout =
            tla::MakeLayout<T, typename DvCopy::LayoutTagL1A>(
                KDA_BWD_A_C, KDA_BWD_A_C);
        auto l1ATensor =
            tla::MakeTensor(l1A, l1ALayout, Catlass::Arch::PositionL1{});
        typename DvCopy::template CopyGmToL1A<decltype(blockA)> copyGmA;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        copyGmA(l1ATensor, blockA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);

        const uint32_t m = task.validC == 1 ? 16 : task.validC;
        auto l0ALayout =
            tla::MakeLayout<T, typename DvCopy::LayoutTagL0A>(m, task.validC);
        auto l0BLayout =
            tla::MakeLayout<T, typename DvCopy::LayoutTagL0B>(
                task.validC, V_DIM);
        auto l0ATensor =
            tla::MakeTensor(l0A, l0ALayout, Catlass::Arch::PositionL0A{});
        auto l0BTensor =
            tla::MakeTensor(l0B, l0BLayout, Catlass::Arch::PositionL0B{});
        auto l1BLayout =
            tla::MakeLayout<T, typename DvCopy::LayoutTagL1B>(
                KDA_BWD_A_C, V_DIM);
        auto l1BTensor =
            tla::MakeTensor(l1B, l1BLayout, Catlass::Arch::PositionL1{});
        auto tileL1A = GetTile(
            l1ATensor, tla::MakeCoord(0, 0),
            tla::MakeShape(m, task.validC));
        auto tileL1B = GetTile(
            l1BTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, V_DIM));
        typename DvCopy::CopyL1ToL0A copyA;
        typename DvCopy::CopyL1ToL0B copyB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        copyA(l0ATensor, tileL1A);
        if (waitDoReady) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        copyB(l0BTensor, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);

        auto l0CLayout = tla::MakeLayoutL0C(m, V_DIM);
        auto l0CTensor =
            tla::MakeTensor(l0C, l0CLayout, Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
        DvMmad mmad;
        mmad(l0CTensor, l0ATensor, l0BTensor, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

        AscendC::GlobalTensor<T> gmOut;
        const uint64_t outOffset =
            KdaBwdAHeadTokenOffset(tiling_, task, head, V_DIM);
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv0_) + outOffset);
        auto outLayout =
            tla::MakeLayout<T, RowMajor>(KDA_BWD_A_C, V_DIM);
        auto outTensor =
            tla::MakeTensor(gmOut, outLayout, Catlass::Arch::PositionGM{});
        auto blockOut = GetTile(
            outTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, V_DIM));
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        typename DvCopy::template CopyL0CToDst<decltype(blockOut)> fix;
#else
        typename DvCopy::template CopyL0CToGm<decltype(blockOut)> fix;
#endif
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

    }

    __aicore__ inline void RunQ0(
        const ChunkKdaBwdATask &task, uint32_t head, GM_ADDR slotBase)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(SCRATCH_OFFSET);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(0);
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::GlobalTensor<T> gmA;
        gmA.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(qg_) +
            KdaBwdAHeadTokenOffset(tiling_, task, head, KDA_BWD_A_K));
        auto gmATensor = tla::MakeTensor(
            gmA, tla::MakeLayout<T, ColumnMajor>(
                     KDA_BWD_A_K, KDA_BWD_A_C),
            Catlass::Arch::PositionGM{});
        auto blockA = GetTile(
            gmATensor, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_BWD_A_K, task.validC));
        auto l1ATensor = tla::MakeTensor(
            l1A, tla::MakeLayout<T, typename Q0Copy::LayoutTagL1A>(
                     KDA_BWD_A_K, KDA_BWD_A_C),
            Catlass::Arch::PositionL1{});
        typename Q0Copy::template CopyGmToL1A<decltype(blockA)> copyGmA;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        copyGmA(l1ATensor, blockA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);

        auto l1BTensor = tla::MakeTensor(
            l1B, tla::MakeLayout<T, typename Q0Copy::LayoutTagL1B>(
                     KDA_BWD_A_C, V_DIM),
            Catlass::Arch::PositionL1{});
        auto l0ATensor = tla::MakeTensor(
            l0A, tla::MakeLayout<T, typename Q0Copy::LayoutTagL0A>(
                     KDA_BWD_A_K, task.validC),
            Catlass::Arch::PositionL0A{});
        auto l0BTensor = tla::MakeTensor(
            l0B, tla::MakeLayout<T, typename Q0Copy::LayoutTagL0B>(
                     task.validC, V_DIM),
            Catlass::Arch::PositionL0B{});
        auto tileA = GetTile(
            l1ATensor, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_BWD_A_K, task.validC));
        auto tileB = GetTile(
            l1BTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, V_DIM));
        typename Q0Copy::CopyL1ToL0A copyA;
        typename Q0Copy::CopyL1ToL0B copyB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        copyA(l0ATensor, tileA);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        copyB(l0BTensor, tileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
        auto l0CTensor = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(KDA_BWD_A_K, V_DIM),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
        Q0Mmad mmad;
        mmad(l0CTensor, l0ATensor, l0BTensor, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

        AscendC::GlobalTensor<float> gmOut;
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            slotBase + tiling_.q0RawOffset));
        auto outTensor = tla::MakeTensor(
            gmOut, tla::MakeLayout<float, RowMajor>(KDA_BWD_A_K, V_DIM),
            Catlass::Arch::PositionGM{});
        auto blockOut = GetTile(
            outTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_BWD_A_K, V_DIM));
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        typename Q0Copy::template CopyL0CToDst<decltype(blockOut)> fix;
#else
        typename Q0Copy::template CopyL0CToGm<decltype(blockOut)> fix;
#endif
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

    }

    template <typename Copy, typename Mmad>
    __aicore__ inline void RunDoLeft(
        GM_ADDR rightBase, uint32_t rightCols,
        GM_ADDR outBase, uint32_t outStride, uint32_t outCols,
        uint32_t validC, bool releaseDo)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(0);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(SCRATCH_OFFSET);
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

        AscendC::GlobalTensor<T> gmRight;
        gmRight.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rightBase));
        auto gmRightTensor = tla::MakeTensor(
            gmRight, tla::MakeLayout<T, ColumnMajor>(V_DIM, rightCols),
            Catlass::Arch::PositionGM{});
        auto rightBlock = GetTile(
            gmRightTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(V_DIM, rightCols));
        auto l1BTensor = tla::MakeTensor(
            l1B, tla::MakeLayout<T, typename Copy::LayoutTagL1B>(
                     V_DIM, rightCols),
            Catlass::Arch::PositionL1{});
        typename Copy::template CopyGmToL1B<decltype(rightBlock)> loadRight;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
        loadRight(l1BTensor, rightBlock);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);

        auto l1ATensor = tla::MakeTensor(
            l1A, tla::MakeLayout<T, typename Copy::LayoutTagL1A>(
                     KDA_BWD_A_C, V_DIM),
            Catlass::Arch::PositionL1{});
        const uint32_t m = validC == 1 ? 16 : validC;
        auto l0CTensor = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(m, outCols),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
        typename Copy::CopyL1ToL0A copyA;
        typename Copy::CopyL1ToL0B copyB;
        Mmad mmad;
        for (uint32_t k0 = 0; k0 < V_DIM; k0 += 64) {
            const bool lastK = k0 + 64 >= V_DIM;
            auto l0ATensor = tla::MakeTensor(
                l0A, tla::MakeLayout<T, typename Copy::LayoutTagL0A>(m, 64),
                Catlass::Arch::PositionL0A{});
            auto l0BTensor = tla::MakeTensor(
                l0B, tla::MakeLayout<T, typename Copy::LayoutTagL0B>(
                         64, outCols),
                Catlass::Arch::PositionL0B{});
            auto tileA = GetTile(
                l1ATensor, tla::MakeCoord(0, k0),
                tla::MakeShape(m, 64));
            auto tileB = GetTile(
                l1BTensor, tla::MakeCoord(k0, 0),
                tla::MakeShape(64, outCols));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
            copyA(l0ATensor, tileA);
            if (releaseDo && lastK) {
                // This flag is issued by MTE1 after the final resident-do
                // read has completed.  The next owner may not overwrite
                // the L1 region before this point.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
            }
            if (k0 == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                    EVENT_SCRATCH);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
            copyB(l0BTensor, tileB);
            if (lastK) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    EVENT_SCRATCH);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
            const uint8_t unitFlag = lastK ? 0b11 : 0b10;
            mmad(l0CTensor, l0ATensor, l0BTensor, k0 == 0, unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

        AscendC::GlobalTensor<float> gmOut;
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outBase));
        auto outTensor = tla::MakeTensor(
            gmOut, tla::MakeLayout<float, RowMajor>(
                       KDA_BWD_A_C, outStride),
            Catlass::Arch::PositionGM{});
        auto blockOut = GetTile(
            outTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(validC, outCols));
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        typename Copy::template CopyL0CToDst<decltype(blockOut)> fix;
#else
        typename Copy::template CopyL0CToGm<decltype(blockOut)> fix;
#endif
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }

    __aicore__ inline void RunDq(
        const ChunkKdaBwdATask &task, uint32_t taskIdx, uint32_t head)
    {
        const uint64_t hOffset = KdaBwdAChunkHeadOffset(
            tiling_, task, taskIdx, head, KDA_BWD_A_K * V_DIM);
        const uint64_t outOffset =
            KdaBwdAHeadTokenOffset(tiling_, task, head, KDA_BWD_A_K);
        RunDoLeft<DqCopy, DqMmad>(
            reinterpret_cast<GM_ADDR>(
                reinterpret_cast<__gm__ T *>(h_) + hOffset),
            KDA_BWD_A_K,
            reinterpret_cast<GM_ADDR>(
                reinterpret_cast<__gm__ float *>(dqRaw_) + outOffset),
            KDA_BWD_A_K, KDA_BWD_A_K, task.validC, false);
    }


    __aicore__ inline void RunDAqk(
        const ChunkKdaBwdATask &task, uint32_t head)
    {
        const uint64_t vOffset =
            KdaBwdAHeadTokenOffset(tiling_, task, head, V_DIM);
        const uint64_t outOffset =
            KdaBwdAHeadTokenOffset(
                tiling_, task, head, KDA_BWD_A_C);
        RunDoLeft<DACopy, DAMmad>(
            reinterpret_cast<GM_ADDR>(
                reinterpret_cast<__gm__ T *>(vNew_) + vOffset),
            KDA_BWD_A_C,
            reinterpret_cast<GM_ADDR>(
                reinterpret_cast<__gm__ float *>(dAqk_) + outOffset),
            KDA_BWD_A_C, task.validC, task.validC, true);
    }

    GM_ADDR aqk_ = nullptr;
    GM_ADDR qg_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dO_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dv0_ = nullptr;
    GM_ADDR dqRaw_ = nullptr;
    GM_ADDR dAqk_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    ChunkKdaBwdATilingData tiling_{};
    Catlass::Arch::CrossCoreFlag readyFlag0_{KDA_BWD_A_READY_FLAG0};
    Catlass::Arch::CrossCoreFlag readyFlag1_{KDA_BWD_A_READY_FLAG1};
    Catlass::Arch::CrossCoreFlag freeFlag0_{KDA_BWD_A_FREE_FLAG0};
    Catlass::Arch::CrossCoreFlag freeFlag1_{KDA_BWD_A_FREE_FLAG1};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_CUBE_H
