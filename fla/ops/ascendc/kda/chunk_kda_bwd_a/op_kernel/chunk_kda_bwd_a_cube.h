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
    using DqCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, RowMajor, T, ColumnMajor, float, RowMajor>;
    using DACopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, RowMajor, T, ColumnMajor, float, RowMajor>;

    using DvMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DvCopy::LayoutTagL1A>;
    using DqMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DqCopy::LayoutTagL1A>;
    using DAMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, T, typename DACopy::LayoutTagL1A>;

    static constexpr uint32_t DO_BYTES = KDA_BWD_A_C * V_DIM * sizeof(T);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t DO_SLOT_COUNT = 2;
#else
    static constexpr uint32_t DO_SLOT_COUNT = 1;
#endif
    static constexpr uint32_t RIGHT0_BYTES =
        V_DIM * KDA_BWD_A_K * sizeof(T);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t AQK_BYTES =
        KDA_BWD_A_C * KDA_BWD_A_C * sizeof(T);
    static constexpr uint32_t OWNER_DO_OFFSET = 0;
    static constexpr uint32_t OWNER_AQK_OFFSET =
        OWNER_DO_OFFSET + DO_BYTES;
    static constexpr uint32_t OWNER_H_OFFSET =
        OWNER_AQK_OFFSET + AQK_BYTES;
    static constexpr uint32_t OWNER_V_OFFSET =
        OWNER_H_OFFSET + RIGHT0_BYTES;
    static constexpr uint32_t OWNER_SLOT_BYTES =
        OWNER_V_OFFSET + V_DIM * KDA_BWD_A_C * sizeof(T);
    // Keep the legacy names valid for template bodies that are not selected
    // by the A5 path.  The live A5 layout is the two owner slots above.
    static constexpr uint32_t SCRATCH_OFFSET = OWNER_AQK_OFFSET;
    static constexpr uint32_t RIGHT1_OFFSET = OWNER_V_OFFSET;
    static constexpr uint32_t RIGHT1_BYTES =
        V_DIM * KDA_BWD_A_C * sizeof(T);
    static constexpr uint32_t L1_USED_BYTES =
        2 * OWNER_SLOT_BYTES;
#else
    static constexpr uint32_t SCRATCH_OFFSET = DO_SLOT_COUNT * DO_BYTES;
    static constexpr uint32_t RIGHT1_OFFSET = SCRATCH_OFFSET + RIGHT0_BYTES;
    static constexpr uint32_t RIGHT1_BYTES =
        V_DIM * KDA_BWD_A_C * sizeof(T);
    static constexpr uint32_t L1_USED_BYTES = RIGHT1_OFFSET + RIGHT1_BYTES;
#endif
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
    static constexpr int32_t EVENT_DO1 = 3;
    static constexpr int32_t EVENT_SCRATCH = 1;
    static constexpr int32_t EVENT_SCRATCH1 = 2;
    static constexpr int32_t EVENT_L0A = 0;
    static constexpr int32_t EVENT_L0B = 1;
    static constexpr int32_t EVENT_L0_READY = 0;
    static constexpr int32_t EVENT_L0C = 0;
    static constexpr int32_t EVENT_L0C1 = 1;

public:
    __aicore__ ChunkKdaBwdACube(
        GM_ADDR aqk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dO,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR dv0, GM_ADDR dqRaw, GM_ADDR dAqk)
        : aqk_(aqk), vNew_(vNew), h_(h), dO_(dO),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          dv0_(dv0), dqRaw_(dqRaw), dAqk_(dAqk)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdATilingData &tiling)
    {
        tiling_ = tiling;
        AscendC::SetHF32Mode(false);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO1);
#else
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
#endif
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C1);
#endif
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint64_t ownerCount =
            static_cast<uint64_t>(tiling_.chunkNum) * tiling_.headNum;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Two complete owner generations live in independent L1 regions.
        // Queue the next owner's do/Aqk/h/v_new before consuming the current
        // generation so MTE2 can overlap current MTE1/MMAD/FixPipe work.
        uint32_t ownerSlot = 0;
        if (core < ownerCount) {
            const uint32_t firstTaskIdx = core / tiling_.headNum;
            const uint32_t firstHead = core % tiling_.headNum;
            ChunkKdaBwdATask firstTask;
            GetChunkKdaBwdATask(
                cuSeqlens_, chunkIndices_, tiling_, firstTaskIdx, firstTask);
            LoadOwner(firstTask, firstTaskIdx, firstHead, ownerSlot);
        }
        for (uint64_t owner = core; owner < ownerCount;
             owner += tiling_.usedCoreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(owner / tiling_.headNum);
            const uint32_t head =
                static_cast<uint32_t>(owner % tiling_.headNum);
            ChunkKdaBwdATask task;
            GetChunkKdaBwdATask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);

            const uint64_t nextOwner = owner + tiling_.usedCoreNum;
            ChunkKdaBwdATask nextTask{};
            uint32_t nextHead = 0;
            const bool hasNext = nextOwner < ownerCount;
            if (hasNext) {
                const uint32_t nextTaskIdx =
                    static_cast<uint32_t>(nextOwner / tiling_.headNum);
                nextHead = static_cast<uint32_t>(nextOwner % tiling_.headNum);
                GetChunkKdaBwdATask(
                    cuSeqlens_, chunkIndices_, tiling_, nextTaskIdx, nextTask);
                LoadOwner(
                    nextTask, nextTaskIdx, nextHead, ownerSlot ^ 1U);
            }
            WaitDoReady(ownerSlot);
            RunDv(task, taskIdx, head, ownerSlot, false);
            RunDqDAqk(task, taskIdx, head, ownerSlot);
            SetDoWritable(ownerSlot);
            ownerSlot ^= 1U;
        }
#else
        for (uint64_t owner = core; owner < ownerCount;
             owner += tiling_.usedCoreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(owner / tiling_.headNum);
            const uint32_t head =
                static_cast<uint32_t>(owner % tiling_.headNum);
            ChunkKdaBwdATask task;
            GetChunkKdaBwdATask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx, task);
            LoadDo(task, head, 0);
            RunDv(task, taskIdx, head, 0, true);
            RunDq(task, taskIdx, head);
            RunDAqk(task, head);
        }
#endif

        // PR190-style pipeline drain.  The final generation leaves the
        // reusable L1/L0 credits set; consuming them before kernel return is
        // required for all outstanding MTE1/MMAD/Fixpipe work to retire.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO1);
#else
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
#endif
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C1);
#endif

    }

private:
    __aicore__ inline void WaitDoWritable(uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot != 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO1);
            return;
        }
#else
        (void)slot;
#endif
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
    }

    __aicore__ inline void SetDoReady(uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot != 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO1);
            return;
        }
#else
        (void)slot;
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO);
    }

    __aicore__ inline void WaitDoReady(uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot != 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO1);
            return;
        }
#else
        (void)slot;
#endif
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_DO);
    }

    __aicore__ inline void SetDoWritable(uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (slot != 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO1);
            return;
        }
#else
        (void)slot;
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_DO);
    }

    __aicore__ inline void LoadDo(
        const ChunkKdaBwdATask &task, uint32_t head, uint32_t slot)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        auto resident = resource.l1Buf.template GetBufferByByte<T>(
            slot * DO_BYTES);
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
        WaitDoWritable(slot);
        copy(l1Tensor, block);
        SetDoReady(slot);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void LoadOwner(
        const ChunkKdaBwdATask &task, uint32_t taskIdx,
        uint32_t head, uint32_t slot)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        const uint32_t slotBase = slot * OWNER_SLOT_BYTES;
        auto doL1 = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + OWNER_DO_OFFSET);
        auto aqkL1 = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + OWNER_AQK_OFFSET);
        auto hL1 = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + OWNER_H_OFFSET);
        auto vL1 = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + OWNER_V_OFFSET);

        AscendC::GlobalTensor<T> doGm;
        AscendC::GlobalTensor<T> aqkGm;
        AscendC::GlobalTensor<T> hGm;
        AscendC::GlobalTensor<T> vGm;
        doGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(dO_) +
            KdaBwdAHeadTokenOffset(tiling_, task, head, V_DIM));
        aqkGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(aqk_) +
            KdaBwdAHeadTokenOffset(
                tiling_, task, head, KDA_BWD_A_C));
        hGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(h_) +
            KdaBwdAChunkHeadOffset(
                tiling_, task, taskIdx, head,
                KDA_BWD_A_K * V_DIM));
        vGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(vNew_) +
            KdaBwdAHeadTokenOffset(tiling_, task, head, V_DIM));

        auto doTensor = tla::MakeTensor(
            doGm, tla::MakeLayout<T, RowMajor>(KDA_BWD_A_C, V_DIM),
            Catlass::Arch::PositionGM{});
        auto aqkTensor = tla::MakeTensor(
            aqkGm,
            tla::MakeLayout<T, ColumnMajor>(
                KDA_BWD_A_C, KDA_BWD_A_C),
            Catlass::Arch::PositionGM{});
        auto hTensor = tla::MakeTensor(
            hGm,
            tla::MakeLayout<T, ColumnMajor>(V_DIM, KDA_BWD_A_K),
            Catlass::Arch::PositionGM{});
        auto vTensor = tla::MakeTensor(
            vGm,
            tla::MakeLayout<T, ColumnMajor>(V_DIM, KDA_BWD_A_C),
            Catlass::Arch::PositionGM{});
        auto doBlock = GetTile(
            doTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, V_DIM));
        auto aqkBlock = GetTile(
            aqkTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, task.validC));
        auto hBlock = GetTile(
            hTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(V_DIM, KDA_BWD_A_K));
        auto vBlock = GetTile(
            vTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(V_DIM, task.validC));

        auto doL1Tensor = tla::MakeTensor(
            doL1,
            tla::MakeLayout<T, typename DvCopy::LayoutTagL1B>(
                KDA_BWD_A_C, V_DIM),
            Catlass::Arch::PositionL1{});
        auto aqkL1Tensor = tla::MakeTensor(
            aqkL1,
            tla::MakeLayout<T, typename DvCopy::LayoutTagL1A>(
                KDA_BWD_A_C, KDA_BWD_A_C),
            Catlass::Arch::PositionL1{});
        auto hL1Tensor = tla::MakeTensor(
            hL1,
            tla::MakeLayout<T, typename DqCopy::LayoutTagL1B>(
                V_DIM, KDA_BWD_A_K),
            Catlass::Arch::PositionL1{});
        auto vL1Tensor = tla::MakeTensor(
            vL1,
            tla::MakeLayout<T, typename DqCopy::LayoutTagL1B>(
                V_DIM, KDA_BWD_A_C),
            Catlass::Arch::PositionL1{});

        typename DvCopy::template CopyGmToL1B<decltype(doBlock)> loadDo;
        typename DvCopy::template CopyGmToL1A<decltype(aqkBlock)> loadAqk;
        typename DqCopy::template CopyGmToL1B<decltype(hBlock)> loadH;
        typename DqCopy::template CopyGmToL1B<decltype(vBlock)> loadV;
        WaitDoWritable(slot);
        loadDo(doL1Tensor, doBlock);
        loadAqk(aqkL1Tensor, aqkBlock);
        loadH(hL1Tensor, hBlock);
        loadV(vL1Tensor, vBlock);
        SetDoReady(slot);
    }
#endif

    __aicore__ inline void RunDv(
        const ChunkKdaBwdATask &task, uint32_t taskIdx, uint32_t head,
        uint32_t doSlot, bool waitDoReady)
    {
        (void)taskIdx;
        Catlass::Arch::Resource<ArchTag> resource;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const uint32_t ownerBase = doSlot * OWNER_SLOT_BYTES;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(
            ownerBase + OWNER_AQK_OFFSET);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(
            ownerBase + OWNER_DO_OFFSET);
#else
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(SCRATCH_OFFSET);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(
            doSlot * DO_BYTES);
#endif
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
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
#else
        auto l1ALayout =
            tla::MakeLayout<T, typename DvCopy::LayoutTagL1A>(
                KDA_BWD_A_C, KDA_BWD_A_C);
        auto l1ATensor =
            tla::MakeTensor(l1A, l1ALayout, Catlass::Arch::PositionL1{});
#endif

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
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_SCRATCH);
#endif
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        copyA(l0ATensor, tileL1A);
        if (waitDoReady) {
            WaitDoReady(doSlot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        copyB(l0BTensor, tileL1B);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_SCRATCH);
#endif
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
        typename DvCopy::template CopyL0CToDst<decltype(blockOut)> fix;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

    }

#if 0 // Q0 is intentionally computed inside PR291 Kernel B.
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
        typename Q0Copy::template CopyL0CToDst<decltype(blockOut)> fix;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

    }

#endif
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
        typename Copy::template CopyL0CToDst<decltype(blockOut)> fix;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fix(blockOut, l0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // A5-local two-output path for the two contractions that share do as
    // their complete left operand:
    //   dq_raw = do @ h^T, dAqk = do @ v_new^T.
    // Both right operands are prefetched into independent L1 regions.  For
    // each K=64 reduction slice, do is copied to L0A once and consumed by
    // both MMADs before L0A is released.  Independent L0C accumulators keep
    // the two outputs live across the V=128/256 reduction loop.
    __aicore__ inline void RunDqDAqk(
        const ChunkKdaBwdATask &task, uint32_t taskIdx, uint32_t head,
        uint32_t ownerSlot)
    {
        (void)taskIdx;
        Catlass::Arch::Resource<ArchTag> resource;
        const uint32_t ownerBase = ownerSlot * OWNER_SLOT_BYTES;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(
            ownerBase + OWNER_DO_OFFSET);
        auto l1B0 = resource.l1Buf.template GetBufferByByte<T>(
            ownerBase + OWNER_H_OFFSET);
        auto l1B1 = resource.l1Buf.template GetBufferByByte<T>(
            ownerBase + OWNER_V_OFFSET);
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        constexpr uint32_t kPackedN = KDA_BWD_A_K + KDA_BWD_A_C;
        static_assert(
            KDA_BWD_A_C * kPackedN * sizeof(float) <= ArchTag::L0C_SIZE,
            "Kernel A packed N=192 FP32 L0C exceeds capacity.");
        static_assert(
            64 * kPackedN * sizeof(T) <= ArchTag::L0B_SIZE,
            "Kernel A packed N=192 L0B exceeds capacity.");

        auto l1B0Tensor = tla::MakeTensor(
            l1B0, tla::MakeLayout<T, typename DqCopy::LayoutTagL1B>(
                      V_DIM, KDA_BWD_A_K),
            Catlass::Arch::PositionL1{});
        auto l1B1Tensor = tla::MakeTensor(
            l1B1, tla::MakeLayout<T, typename DqCopy::LayoutTagL1B>(
                      V_DIM, KDA_BWD_A_C),
            Catlass::Arch::PositionL1{});

        auto l1ATensor = tla::MakeTensor(
            l1A, tla::MakeLayout<T, typename DqCopy::LayoutTagL1A>(
                     KDA_BWD_A_C, V_DIM),
            Catlass::Arch::PositionL1{});
        const uint32_t m = task.validC == 1 ? 16 : task.validC;
        auto l0CTensor = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(m, kPackedN),
            Catlass::Arch::PositionL0C{});
        const uint32_t packedN = KDA_BWD_A_K + task.validC;
        auto tileL0C = GetTile(
            l0CTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(m, packedN));
        auto tileL0CDq = GetTile(
            l0CTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(m, KDA_BWD_A_K));
        auto tileL0CDA = GetTile(
            l0CTensor, tla::MakeCoord(0, KDA_BWD_A_K),
            tla::MakeShape(m, task.validC));
        typename DqCopy::CopyL1ToL0A copyA;
        typename DqCopy::CopyL1ToL0B copyB;
        DqMmad mm;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
        for (uint32_t k0 = 0; k0 < V_DIM; k0 += 64) {
            const bool lastK = k0 + 64 >= V_DIM;
            auto l0ATensor = tla::MakeTensor(
                l0A, tla::MakeLayout<T, typename DqCopy::LayoutTagL0A>(m, 64),
                Catlass::Arch::PositionL0A{});
            auto l0BTensor = tla::MakeTensor(
                l0B, tla::MakeLayout<T, typename DqCopy::LayoutTagL0B>(
                         64, kPackedN),
                Catlass::Arch::PositionL0B{});
            auto tileA = GetTile(
                l1ATensor, tla::MakeCoord(0, k0), tla::MakeShape(m, 64));
            auto tileB0 = GetTile(
                l1B0Tensor, tla::MakeCoord(k0, 0),
                tla::MakeShape(64, KDA_BWD_A_K));
            auto tileB1 = GetTile(
                l1B1Tensor, tla::MakeCoord(k0, 0),
                tla::MakeShape(64, task.validC));
            auto tileL0B0 = GetTile(
                l0BTensor, tla::MakeCoord(0, 0),
                tla::MakeShape(64, KDA_BWD_A_K));
            auto tileL0B1 = GetTile(
                l0BTensor, tla::MakeCoord(0, KDA_BWD_A_K),
                tla::MakeShape(64, task.validC));
            auto tileL0B = GetTile(
                l0BTensor, tla::MakeCoord(0, 0),
                tla::MakeShape(64, packedN));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
            copyA(l0ATensor, tileA);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
            copyB(tileL0B0, tileB0);
            copyB(tileL0B1, tileB1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_L0_READY);
            mm(tileL0C, l0ATensor, tileL0B, k0 == 0,
               lastK ? 0b11 : 0b10);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);

        const uint64_t dqOffset = KdaBwdAHeadTokenOffset(
            tiling_, task, head, KDA_BWD_A_K);
        const uint64_t dAOffset = KdaBwdAHeadTokenOffset(
            tiling_, task, head, KDA_BWD_A_C);
        AscendC::GlobalTensor<float> dqGm;
        AscendC::GlobalTensor<float> dAGm;
        dqGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dqRaw_) + dqOffset);
        dAGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dAqk_) + dAOffset);
        auto dqTensor = tla::MakeTensor(
            dqGm, tla::MakeLayout<float, RowMajor>(
                      KDA_BWD_A_C, KDA_BWD_A_K),
            Catlass::Arch::PositionGM{});
        auto dATensor = tla::MakeTensor(
            dAGm, tla::MakeLayout<float, RowMajor>(
                      KDA_BWD_A_C, KDA_BWD_A_C),
            Catlass::Arch::PositionGM{});
        auto dqBlock = GetTile(
            dqTensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, KDA_BWD_A_K));
        auto dABlock = GetTile(
            dATensor, tla::MakeCoord(0, 0),
            tla::MakeShape(task.validC, task.validC));
        typename DqCopy::template CopyL0CToDst<decltype(dqBlock)> fixDq;
        typename DqCopy::template CopyL0CToDst<decltype(dABlock)> fixDA;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_L0C);
        fixDq(dqBlock, tileL0CDq, 0b11);
        fixDA(dABlock, tileL0CDA, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }
#endif

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
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dO_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dv0_ = nullptr;
    GM_ADDR dqRaw_ = nullptr;
    GM_ADDR dAqk_ = nullptr;
    ChunkKdaBwdATilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_CUBE_H
