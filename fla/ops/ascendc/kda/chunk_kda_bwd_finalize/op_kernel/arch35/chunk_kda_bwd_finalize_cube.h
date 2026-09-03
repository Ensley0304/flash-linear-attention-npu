/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "chunk_kda_bwd_finalize_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace KDA {

class ChunkKdaBwdFinalizeCubeStage03 {
public:
    __aicore__ inline void Init(
        GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
        GM_ADDR dvScan, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR workspace, const ChunkKdaBwdFinalizeTilingData *tiling)
    {
        v_ = v;
        akk_ = akk;
        vNew_ = vNew;
        h_ = h;
        dh_ = dh;
        dvScan_ = dvScan;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        workspace_ = workspace;
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetMMLayoutTransform(true);
        Catlass::Arch::Resource<ArchTag> resource;
        InitEvents();
        const int64_t coreIdx = AscendC::GetBlockIdx();
        const int64_t coreNum = AscendC::GetBlockNum();
        uint64_t groupGeneration = 0;
        uint64_t headGeneration = 0;
        bool stage0Prefetched = false;
        for (int64_t workTask = coreIdx; workTask < tiling_->workTaskNum;
             workTask += coreNum, ++groupGeneration) {
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
            const int64_t headEnd = FinalizeMin(
                headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
            FinalizeChunkInfo chunk;
            ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                stage0Prefetched = false;
                continue;
            }

            // Stage0 has two 112-KiB L1 streams.  Prime both streams, then
            // refill the stream just released by head n with head n+2 before
            // computing head n+1.  This keeps the MTE2 queue ahead of MTE1 /
            // MMAD / FixPipe in the same pattern as KernelA's resident-L1
            // preload, without increasing the L1 footprint.
            const int64_t preloadEnd = FinalizeMin(headBegin + 2, headEnd);
            if (!stage0Prefetched) {
                for (int64_t head = headBegin; head < preloadEnd; ++head) {
                    const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                    LoadStage0(resource, chunk, head, owner);
                }
            }
            stage0Prefetched = false;
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                ComputeStage0(resource, chunk, owner, aiv, slot,
                              coreIdx, groupGeneration);
                const int64_t nextHead = head + 2;
                if (nextHead < headEnd) {
                    LoadStage0(resource, chunk, nextHead,
                               static_cast<uint32_t>(nextHead - headBegin));
                }
            }

            // The two Stage0 stream buffers are dead after the four heads.
            // Prefetch the next work task's stream-only operands while the
            // current task runs Stage1/Stage3.  Akk remains resident for the
            // current task and is replaced only after Stage3 consumes it.
            FinalizeChunkInfo nextChunk;
            int64_t nextHeadBegin = 0;
            int64_t nextHeadEnd = 0;
            const int64_t nextWorkTask = workTask + coreNum;
            bool prefetchNext = false;
            if (nextWorkTask < tiling_->workTaskNum) {
                const int64_t nextHeadWindow = nextWorkTask / tiling_->chunkTaskNum;
                const int64_t nextChunkTask =
                    nextWorkTask - nextHeadWindow * tiling_->chunkTaskNum;
                nextHeadBegin = nextHeadWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
                nextHeadEnd = FinalizeMin(
                    nextHeadBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
                ResolveFinalizeChunk(
                    nextChunkTask, cuSeqlens_, chunkIndices_, *tiling_, nextChunk);
                prefetchNext = nextChunk.valid && nextHeadBegin < nextHeadEnd;
                if (prefetchNext) {
                    const int64_t nextPreloadEnd =
                        FinalizeMin(nextHeadBegin + 2, nextHeadEnd);
                    for (int64_t head = nextHeadBegin;
                         head < nextPreloadEnd; ++head) {
                        LoadStage0Stream(
                            resource, nextChunk, head,
                            static_cast<uint32_t>(head - nextHeadBegin));
                    }
                }
            }

            headGeneration -= static_cast<uint64_t>(headEnd - headBegin);
            // Stage1 starts per head as soon as both dW and kE are ready; it
            // does not wait for a group-wide Vector barrier.
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                RunStage1(resource, chunk, owner, aiv, slot,
                          coreIdx, groupGeneration);
            }
            headGeneration -= static_cast<uint64_t>(headEnd - headBegin);
            // Stage3 consumes each Zb as soon as its owner AIV publishes the
            // L1 slot.  Akk^T is the same nZ resident copy loaded by Stage0.
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                RunStage3(resource, chunk, owner, aiv, slot);
            }
            if (prefetchNext) {
                const int64_t nextPreloadEnd =
                    FinalizeMin(nextHeadBegin + 2, nextHeadEnd);
                for (int64_t head = nextHeadBegin;
                     head < nextPreloadEnd; ++head) {
                    LoadStage0Akk(
                        resource, nextChunk, head,
                        static_cast<uint32_t>(head - nextHeadBegin));
                }
                stage0Prefetched = true;
            }
        }
        DrainEvents();
        AscendC::SetMMLayoutTransform(false);
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    using DT = bfloat16_t;
    using Acc = float;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using CopyTransB = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutCM, Acc, LayoutRM>;
    using CopyTransA = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutCM, DT, LayoutRM, Acc, LayoutRM>;
    using CopyRegular = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutRM, Acc, LayoutRM>;
    using CopyStage3 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutCM, DT, LayoutCM, Acc, LayoutRM>;
    using CopyTransBToUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRM, DT, LayoutCM, Acc, LayoutRM>;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename CopyRegular::LayoutTagL1A>;

    enum class ResultPath : uint32_t { GM_FP32, L1_BF16, AIV_UB_FP32 };

    static constexpr uint32_t L1_AKK = 0;
    static constexpr uint32_t L1_DW = 32 * 1024;
    static constexpr uint32_t L1_KE = 96 * 1024;
    static constexpr uint32_t L1_ZB = 160 * 1024;
    static constexpr uint32_t L1_STREAM = 192 * 1024;
    static constexpr uint32_t STREAM_BYTES = 112 * 1024;
    static constexpr uint32_t STREAM_VNEW = 0;
    static constexpr uint32_t STREAM_DH = 16 * 1024;
    static constexpr uint32_t STREAM_DVSCAN = 48 * 1024;
    static constexpr uint32_t STREAM_H = 64 * 1024;
    static constexpr uint32_t STREAM_V = 96 * 1024;
    static constexpr uint32_t L1_TZA = 416 * 1024;
    static constexpr uint32_t STAGE0_READY_COUNT = 4;
    static constexpr uint32_t L0_BYTES = 32 * 1024;
    static constexpr uint32_t L0C_BYTES = 128 * 1024;
    static constexpr AscendC::FixpipeConfig FIX_NZ_L1 = {
        AscendC::CO2Layout::NZ, false};

    template <typename TileCopy, typename GmLayout>
    __aicore__ inline void LoadGmToL1A(
        AscendC::LocalTensor<DT> dst, GM_ADDR src,
        uint32_t m, uint32_t k)
    {
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        auto tensorGm = tla::MakeTensor(
            gm, tla::MakeLayout<DT, GmLayout>(m, k), Catlass::Arch::PositionGM{});
        auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tensorL1 = tla::MakeTensor(
            dst, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1A<decltype(block)>{}(tensorL1, block);
    }

    template <typename TileCopy, typename GmLayout>
    __aicore__ inline void LoadGmToL1B(
        AscendC::LocalTensor<DT> dst, GM_ADDR src,
        uint32_t k, uint32_t n)
    {
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        auto tensorGm = tla::MakeTensor(
            gm, tla::MakeLayout<DT, GmLayout>(k, n), Catlass::Arch::PositionGM{});
        auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tensorL1 = tla::MakeTensor(
            dst, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1B<decltype(block)>{}(tensorL1, block);
    }

    template <typename TileCopy, ResultPath PATH>
    __aicore__ inline void RunGemm(
        Catlass::Arch::Resource<ArchTag> &resource,
        AscendC::LocalTensor<DT> l1A, AscendC::LocalTensor<DT> l1B,
        uint32_t m, uint32_t n, uint32_t k, uint32_t slot,
        GM_ADDR gmDst, AscendC::LocalTensor<DT> l1Dst,
        uint32_t aiv, uint32_t aivSlot, uint32_t aivUbOffset,
        uint64_t freeFlag, uint64_t readyFlag)
    {
        auto l0A = resource.l0ABuf.template GetBufferByByte<DT>(slot * L0_BYTES);
        auto l0B = resource.l0BBuf.template GetBufferByByte<DT>(slot * L0_BYTES);
        auto l0C = resource.l0CBuf.template GetBufferByByte<Acc>(slot * L0C_BYTES);
        auto tensorL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<DT, typename TileCopy::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<DT, typename TileCopy::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        auto tileA = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        typename TileCopy::CopyL1ToL0A{}(tensorL0A, tileA);
        typename TileCopy::CopyL1ToL0B{}(tensorL0B, tileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
        Catlass::Gemm::Tile::TileMmadTla<
            ArchTag, DT, typename TileCopy::LayoutTagL1A>{}(tileC, tensorL0A, tensorL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);

        if constexpr (PATH == ResultPath::GM_FP32) {
            AscendC::GlobalTensor<float> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmDst));
            auto params = AscendC::FixpipeParamsV220(n, m, (m + 15U) / 16U * 16U, n, false);
            params.quantPre = QuantMode_t::NoQuant;
            params.unitFlag = 0;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dst, l0C, params);
        } else if constexpr (PATH == ResultPath::L1_BF16) {
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> params;
            params.nSize = n;
            params.mSize = m;
            params.srcStride = (m + 15U) / 16U * 16U;
            params.dstStride = KDA_FINALIZE_CHUNK * 16U;
            params.quantPre = QuantMode_t::F322BF16;
            params.unitFlag = 0;
            AscendC::Fixpipe<DT, Acc, FIX_NZ_L1>(l1Dst, l0C, params);
        } else {
            const uint64_t flagOffset =
                static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                freeFlag + flagOffset + aivSlot);
            auto remoteUb = resource.ubBuf.template GetBufferByByte<float>(aivUbOffset);
            auto tensorUb = tla::MakeTensor(
                remoteUb, tla::MakeLayout<float, LayoutRM>(m, KDA_FINALIZE_CHUNK),
                Catlass::Arch::PositionUB{});
            auto blockUb = tla::GetTile(
                tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
            typename CopyTransBToUb::template CopyL0CToDst<decltype(blockUb)>{}(
                blockUb, tensorL0C, static_cast<uint8_t>(aiv), 0);
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                readyFlag + flagOffset + aivSlot);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
    }

    __aicore__ inline AscendC::TEventID Stage0Ready(
        uint32_t stream, uint32_t stage) const
    {
        return static_cast<AscendC::TEventID>(
            stream * STAGE0_READY_COUNT + stage);
    }

    __aicore__ inline void LoadStage0(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t stream = owner & 1U;
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t akkToken = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        const int64_t state = FinalizeStateOffset(*tiling_, chunk, head);
        LoadGmToL1A<CopyTransB, LayoutRM>(
            vNewL1, vNew_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            dhL1, dh_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));

        LoadGmToL1A<CopyTransA, LayoutCM>(
            akkL1, akk_ + akkToken * sizeof(DT), rows, rows);
        LoadGmToL1A<CopyRegular, LayoutRM>(
            dvScanL1, dvScan_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));

        LoadGmToL1B<CopyTransB, LayoutCM>(
            hL1, h_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        LoadGmToL1B<CopyTransB, LayoutCM>(
            vL1, v_ + token * sizeof(DT), KDA_FINALIZE_DIM, rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
    }

    __aicore__ inline void LoadStage0Stream(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t stream = owner & 1U;
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t state = FinalizeStateOffset(*tiling_, chunk, head);

        LoadGmToL1A<CopyTransB, LayoutRM>(
            vNewL1, vNew_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            dhL1, dh_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));
        LoadGmToL1A<CopyRegular, LayoutRM>(
            dvScanL1, dvScan_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            hL1, h_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        LoadGmToL1B<CopyTransB, LayoutCM>(
            vL1, v_ + token * sizeof(DT), KDA_FINALIZE_DIM, rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
    }

    __aicore__ inline void LoadStage0Akk(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t stream = owner & 1U;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        const int64_t akkToken =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        LoadGmToL1A<CopyTransA, LayoutCM>(
            akkL1, akk_ + akkToken * sizeof(DT), rows, rows);
        // dvScan was queued by LoadStage0Stream.  This event is emitted only
        // after Akk joins the same MTE2 stream, so MTE1 observes both inputs.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));
    }

    __aicore__ inline void ComputeStage0(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t stream = owner & 1U;
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto dwL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);

        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        GM_ADDR wsBase = workspace_ + ws;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));
        RunGemm<CopyTransB, ResultPath::GM_FP32>(
            resource, vNewL1, dhL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DK_STATE_RAW, {}, 0, 0, 0, 0, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));
        RunGemm<CopyTransA, ResultPath::GM_FP32>(
            resource, akkL1, dvScanL1, rows, KDA_FINALIZE_DIM, rows,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DVB, {}, 0, 0, 0, 0, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        RunGemm<CopyTransB, ResultPath::L1_BF16>(
            resource, dvScanL1, hL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, dwL1, 0, 0, 0, 0, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
        RunGemm<CopyTransB, ResultPath::AIV_UB_FP32>(
            resource, dvScanL1, vL1, rows, rows, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, {}, aiv, aivSlot,
            KDA_FINALIZE_UB_ZV + aivSlot * KDA_FINALIZE_MATRIX_FP32_BYTES,
            KDA_FINALIZE_ZV_FREE_BASE, KDA_FINALIZE_ZV_READY_BASE);
    }

    __aicore__ inline void RunStage1(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto dwL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        auto kEL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_KE + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        const uint64_t flagOffset =
            static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_KE_READY_BASE + flagOffset + aivSlot);
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        GM_ADDR wsBase = workspace_ + ws;
        RunGemm<CopyRegular, ResultPath::GM_FP32>(
            resource, akkL1, dwL1, rows, KDA_FINALIZE_DIM, rows,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DKGB_RAW, {}, 0, 0, 0, 0, 0);
        RunGemm<CopyTransB, ResultPath::AIV_UB_FP32>(
            resource, dwL1, kEL1, rows, rows, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, {}, aiv, aivSlot,
            KDA_FINALIZE_UB_ZW + aivSlot * KDA_FINALIZE_MATRIX_FP32_BYTES,
            KDA_FINALIZE_ZW_FREE_BASE, KDA_FINALIZE_ZW_READY_BASE);
    }

    __aicore__ inline void RunStage3(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint64_t flagOffset =
            static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_ZB_READY_BASE + flagOffset + aivSlot);
        auto zBL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_ZB + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto tzaL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_TZA + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        RunGemm<CopyStage3, ResultPath::L1_BF16>(
            resource, zBL1, akkL1, rows, rows, rows,
            NextSlot(), nullptr, tzaL1, 0, 0, 0, 0, 0);
        // PIPE_FIX is deliberately used for the return credit, matching the
        // mature fwd_h L1 producer/consumer protocol.  It is conservative:
        // Zb is certainly no longer read when the Tza FixPipe completes.
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
            KDA_FINALIZE_ZB_FREE_BASE + flagOffset + aivSlot);
    }

    __aicore__ inline uint32_t NextSlot()
    {
        const uint32_t current = l0Slot_;
        l0Slot_ ^= 1U;
        return current;
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
        }
    }

    GM_ADDR v_ = nullptr;
    GM_ADDR akk_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR dvScan_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    uint32_t l0Slot_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H
