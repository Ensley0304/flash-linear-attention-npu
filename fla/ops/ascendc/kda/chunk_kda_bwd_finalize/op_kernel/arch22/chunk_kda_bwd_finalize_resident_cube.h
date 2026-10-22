/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_CUBE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_CUBE_H

#include "chunk_kda_bwd_finalize_resident_common.h"
#include "../../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_finalize_wy.h"

namespace KDA {

// Resident A2 Cube chain. A work owns one head/chunk and exchanges two
// 32-row bands with its AIV pair through a private workspace slot.
template <typename Element = bfloat16_t>
class ChunkKdaBwdFinalizeA2ResidentCubeStage1a {
private:
    using ArchTag = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;
    using DkCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, RowMajor, Element, ColumnMajor, float, RowMajor>;
    using DvbCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, ColumnMajor, Element, RowMajor, float, RowMajor>;
    using DkMmad = WyTileGemmDirect<ArchTag, float, DkCopy>;
    using DkBf16Mmad = WyTileGemmDirect<ArchTag, Element, DkCopy>;
    using DvbMmad = WyTileGemmDirect<ArchTag, float, DvbCopy>;
    using DvbBf16Mmad = WyTileGemmDirect<ArchTag, Element, DvbCopy>;
    using Fp32Copy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, float, RowMajor, float, RowMajor, float, RowMajor>;
    using Fp32CopyTransA =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, float, ColumnMajor, float, RowMajor, float, RowMajor>;
    static constexpr uint32_t kL0Bytes = 32 * 1024;
    static constexpr uint32_t kL0CBytes = 64 * 1024;
    static constexpr uint32_t kIntraL0APlaneBytes = 32 * KDA_FINALIZE_CHUNK * sizeof(float);
    static constexpr uint32_t kIntraL0BPlaneBytes = KDA_FINALIZE_CHUNK * KDA_FINALIZE_DIM * sizeof(float);
    static constexpr int32_t kL0A1Event = 2;
    static constexpr int32_t kL0B1Event = 3;
    static constexpr int32_t kL0Ready1Event = 1;
    static_assert(2 * kIntraL0APlaneBytes <= ArchTag::L0A_SIZE, "Intra L0A ping-pong exceeds capacity.");
    static_assert(2 * kIntraL0BPlaneBytes <= ArchTag::L0B_SIZE, "Intra L0B ping-pong exceeds capacity.");
    static constexpr AscendC::FixpipeConfig kFixNzL1 = {AscendC::CO2Layout::NZ, false};

public:
    __aicore__ inline void Init(GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
                                const ChunkKdaBwdFinalizeTilingData *tiling)
    {
        v_ = v;
        akk_ = akk;
        vNew_ = vNew;
        h_ = h;
        dh_ = dh;
        dvScan_ = dvScan;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        workspaceAddr_ = workspace;
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetMMLayoutTransform(true);
        AscendC::SetHF32Mode(false);
        Catlass::Arch::Resource<ArchTag> resource;

        const int64_t logicalCore = AscendC::GetBlockIdx();
        uint64_t workOrdinal = 0;
        bool stage1aReady = false;
        bool stage1bReady = false;
        for (int64_t head = 0; head < tiling_->NV; ++head) {
            const FinalizeA2ResidentChunkRange range = ResolveFinalizeA2ResidentChunkRange(logicalCore, head, *tiling_);
            for (int64_t local = 0; local < range.count; ++local) {
                const int64_t chunkTask = range.begin + local;
                FinalizeChunkInfo chunk;
                ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
                if (!chunk.valid) {
                    continue;
                }
                const uint64_t slotBase = FinalizeA2ResidentSlotBase(
                    logicalCore, static_cast<uint32_t>(workOrdinal % KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH));
                if (stage1aReady) {
                    stage1aReady = false;
                } else {
                    BeginMmadPhase();
                    RunStage1a(resource, chunk, head, slotBase);
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_BASE0_READY);
                }
                if (stage1bReady) {
                    stage1bReady = false;
                } else {
                    WaitVectorStage(KDA_FINALIZE_A2_FLAG_KE_READY);
                    BeginMmadPhase();
                    RunStage1b(resource, chunk, head, slotBase);
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_BASE1_READY);
                }
                WaitVectorStage(KDA_FINALIZE_A2_FLAG_ZB_READY);
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3) {
                    BeginMmadPhase();
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3_COMPUTE) {
                        RunStage3(resource, chunk, head, slotBase);
                    }
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_DAKK_READY);
                }
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5) {
                    WaitVectorStage(KDA_FINALIZE_A2_FLAG_INTRA_INPUT_READY);
                    BeginMmadPhase();
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5_COMPUTE) {
                        RunStage5Dq(resource, slotBase);
                    }
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_INTRA_DQ_READY);
                    WaitVectorStage(KDA_FINALIZE_A2_FLAG_INTRA_DK_INPUT_READY);
                    BeginMmadPhase();
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5_COMPUTE) {
                        RunStage5Dk(resource, slotBase);
                    }
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_INTRA_DK_READY);
                    WaitVectorStage(KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_INPUT_READY);
                    BeginMmadPhase();
                    if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE5_COMPUTE) {
                        RunStage5Right(resource, slotBase);
                    }
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_INTRA_RIGHT_READY);
                }
                FinalizeChunkInfo nextChunk{};
                int64_t nextHead = 0;
                bool nextStage1bComputed = false;
                if (FindNextWork(logicalCore, head, local, nextChunk, nextHead)) {
                    const uint64_t nextSlotBase = FinalizeA2ResidentSlotBase(
                        logicalCore, static_cast<uint32_t>((workOrdinal + 1U) % KDA_FINALIZE_A2_RESIDENT_SLOT_DEPTH));
                    BeginMmadPhase();
                    RunStage1a(resource, nextChunk, nextHead, nextSlotBase);
                    EndMmadPhase();
                    FenceFixToMte2();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_BASE0_READY);
                    stage1aReady = true;
                    WaitVectorStage(KDA_FINALIZE_A2_FLAG_KE_READY);
                    BeginMmadPhase();
                    RunStage1b(resource, nextChunk, nextHead, nextSlotBase);
                    EndMmadPhase();
                    FenceFixToMte2();
                    nextStage1bComputed = true;
                }
                WaitVectorStage(KDA_FINALIZE_A2_FLAG_WORK_DONE);
                if (nextStage1bComputed) {
                    // BASE1_READY reuses the current work's RIGHT_READY event.
                    // Publish only after WORK_DONE proves both AIVs consumed it.
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FINALIZE_A2_FLAG_BASE1_READY);
                    stage1bReady = true;
                }
                ++workOrdinal;
            }
        }

        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline bool FindNextWork(int64_t logicalCore, int64_t currentHead, int64_t currentLocal,
                                        FinalizeChunkInfo &nextChunk, int64_t &nextHead)
    {
        for (int64_t head = currentHead; head < tiling_->NV; ++head) {
            const FinalizeA2ResidentChunkRange range = ResolveFinalizeA2ResidentChunkRange(logicalCore, head, *tiling_);
            const int64_t firstLocal = head == currentHead ? currentLocal + 1 : 0;
            for (int64_t local = firstLocal; local < range.count; ++local) {
                ResolveFinalizeChunk(range.begin + local, cuSeqlens_, chunkIndices_, *tiling_, nextChunk);
                if (nextChunk.valid) {
                    nextHead = head;
                    return true;
                }
            }
        }
        return false;
    }

    template <typename Copy, typename GmLayout>
    __aicore__ inline void LoadGmToL1A(AscendC::LocalTensor<Element> dst, GM_ADDR src, uint32_t m, uint32_t k)
    {
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(src));
        auto tensorGm = tla::MakeTensor(gm, tla::MakeLayout<Element, GmLayout>(m, k), Catlass::Arch::PositionGM{});
        auto tensorL1 = tla::MakeTensor(dst, tla::MakeLayout<Element, typename Copy::LayoutTagL1A>(m, k),
                                        Catlass::Arch::PositionL1{});
        typename Copy::template CopyGmToL1A<decltype(tensorGm)>{}(tensorL1, tensorGm);
    }

    template <typename Copy, typename GmLayout>
    __aicore__ inline void LoadGmToL1B(AscendC::LocalTensor<Element> dst, GM_ADDR src, uint32_t k, uint32_t n)
    {
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(src));
        auto tensorGm = tla::MakeTensor(gm, tla::MakeLayout<Element, GmLayout>(k, n), Catlass::Arch::PositionGM{});
        auto tensorL1 = tla::MakeTensor(dst, tla::MakeLayout<Element, typename Copy::LayoutTagL1B>(k, n),
                                        Catlass::Arch::PositionL1{});
        typename Copy::template CopyGmToL1B<decltype(tensorGm)>{}(tensorL1, tensorGm);
    }

    template <typename Copy, bool RUN_MMAD = true>
    __aicore__ inline void RunL1Mmad(Catlass::Arch::Resource<ArchTag> &resource, AscendC::LocalTensor<Element> l1A,
                                     AscendC::LocalTensor<Element> l1B, uint32_t m, uint32_t n, uint32_t k)
    {
        auto l0A = resource.l0ABuf.template GetBufferByByte<Element>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<Element>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        auto tensorL1A = tla::MakeTensor(l1A, tla::MakeLayout<Element, typename Copy::LayoutTagL1A>(m, k),
                                         Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, tla::MakeLayout<Element, typename Copy::LayoutTagL1B>(k, n),
                                         Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, tla::MakeLayout<Element, typename Copy::LayoutTagL0A>(m, k),
                                         Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, tla::MakeLayout<Element, typename Copy::LayoutTagL0B>(k, n),
                                         Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});

        if constexpr (RUN_MMAD) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0B);
        typename Copy::CopyL1ToL0A{}(tensorL0A, tensorL1A);
        typename Copy::CopyL1ToL0B{}(tensorL0B, tensorL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        if constexpr (RUN_MMAD) {
            Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, typename Copy::LayoutTagL1A>{}(
                tensorL0C, tensorL0A, tensorL0B, m, n, k, true, 0b11);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0B);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0B);
        }
    }

    template <typename Copy>
    __aicore__ inline void RunL1GemmToGm(Catlass::Arch::Resource<ArchTag> &resource, AscendC::LocalTensor<Element> l1A,
                                         AscendC::LocalTensor<Element> l1B, GM_ADDR dst, uint32_t m, uint32_t n,
                                         uint32_t k)
    {
        RunL1Mmad<Copy>(resource, l1A, l1B, m, n, k);
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        auto params = AscendC::FixpipeParamsV220(n, m, (m + 15U) / 16U * 16U, n, false);
        params.quantPre = QuantMode_t::NoQuant;
        params.unitFlag = 0b11;
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm, l0C, params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
    }

    template <typename Copy>
    __aicore__ inline void RunL1GemmToGmBf16(Catlass::Arch::Resource<ArchTag> &resource,
                                             AscendC::LocalTensor<Element> l1A, AscendC::LocalTensor<Element> l1B,
                                             GM_ADDR dst, uint32_t m, uint32_t n, uint32_t k)
    {
        RunL1Mmad<Copy>(resource, l1A, l1B, m, n, k);
        auto params = AscendC::FixpipeParamsV220(n, m, (m + 15U) / 16U * 16U, n, false);
        params.quantPre = QuantMode_t::F322BF16;
        params.unitFlag = 0b11;
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::GlobalTensor<Element> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(dst));
        AscendC::Fixpipe<Element, float, AscendC::CFG_ROW_MAJOR>(gm, l0C, params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
    }

    template <typename Copy>
    __aicore__ inline void RunL1GemmToL1Bf16(Catlass::Arch::Resource<ArchTag> &resource,
                                             AscendC::LocalTensor<Element> l1A, AscendC::LocalTensor<Element> l1B,
                                             AscendC::LocalTensor<Element> dst, uint32_t m, uint32_t n, uint32_t k)
    {
        RunL1Mmad<Copy>(resource, l1A, l1B, m, n, k);
        const uint32_t alignedM = (m + 15U) / 16U * 16U;
        auto params = AscendC::FixpipeParamsV220(n, m, alignedM, KDA_FINALIZE_CHUNK, false);
        params.quantPre = QuantMode_t::F322BF16;
        params.unitFlag = 0b11;
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::Fixpipe<Element, float, kFixNzL1>(dst, l0C, params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
    }

    __aicore__ inline void WaitL1LoadCredits()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1B);
    }

    __aicore__ inline void PublishL1Operands()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1B);
    }

    __aicore__ inline void RunStage1b(Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
                                      int64_t head, uint64_t slotBase)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint64_t token = static_cast<uint64_t>(FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM));
        const uint64_t matrixToken =
            static_cast<uint64_t>(FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK));
        const uint64_t hState = static_cast<uint64_t>(FinalizeHOffset(*tiling_, chunk, head));

        auto inputA = resource.l1Buf.template GetBufferByByte<Element>(0);
        auto inputB = resource.l1Buf.template GetBufferByByte<Element>(64 * 1024);

        if (KDA_FINALIZE_A2_RESIDENT_STAGE1B_L1 && rows == KDA_FINALIZE_CHUNK) {
            GM_ADDR ws = workspaceAddr_ + slotBase;
            const uint64_t elementBytes = sizeof(Element);
            auto dwL1 = resource.l1Buf.template GetBufferByByte<Element>(KDA_FINALIZE_A2_L1_DW);
            auto keL1 = resource.l1Buf.template GetBufferByByte<Element>(KDA_FINALIZE_A2_L1_KE);
            auto akkL1 = resource.l1Buf.template GetBufferByByte<Element>(KDA_FINALIZE_A2_L1_ZB);

            // Zv and DW share dVscan. Keep it in L1A while replacing only
            // the right operand between the two contractions.
            WaitL1LoadCredits();
            LoadGmToL1A<DkCopy, RowMajor>(inputA, dvScan_ + token * elementBytes, rows, KDA_FINALIZE_DIM);
            LoadGmToL1B<DkCopy, ColumnMajor>(inputB, v_ + token * elementBytes, KDA_FINALIZE_DIM, rows);
            PublishL1Operands();
            if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZV) {
                RunL1GemmToGm<DkCopy>(resource, inputA, inputB, ws + KDA_FINALIZE_A2_WS_ZV, rows, rows,
                                      KDA_FINALIZE_DIM);
            } else {
                RunL1Mmad<DkCopy, false>(resource, inputA, inputB, rows, rows, KDA_FINALIZE_DIM);
            }

            WaitL1LoadCredits();
            LoadGmToL1B<DkCopy, ColumnMajor>(inputB, h_ + hState * elementBytes, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
            PublishL1Operands();
            if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_DW) {
                RunL1GemmToGmBf16<DkCopy>(resource, inputA, inputB, ws + KDA_FINALIZE_A2_WS_DW_RELAY, rows,
                                          KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
            } else {
                RunL1Mmad<DkCopy, false>(resource, inputA, inputB, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
            }
            if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZW || KDA_FINALIZE_A2_RESIDENT_RUN_DKGB) {
                FenceFixToMte2();
                WaitL1LoadCredits();
                LoadGmToL1A<DkCopy, RowMajor>(dwL1, ws + KDA_FINALIZE_A2_WS_DW_RELAY, rows, KDA_FINALIZE_DIM);
                if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZW) {
                    LoadGmToL1B<DkCopy, ColumnMajor>(keL1, ws + KDA_FINALIZE_A2_WS_KE, KDA_FINALIZE_DIM, rows);
                }
                PublishL1Operands();
            }
            if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZW) {
                RunL1GemmToGm<DkCopy>(resource, dwL1, keL1, ws + KDA_FINALIZE_A2_WS_ZW, rows, rows, KDA_FINALIZE_DIM);
            }
            if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_DKGB) {
                WaitL1LoadCredits();
                LoadGmToL1A<DvbCopy, ColumnMajor>(akkL1, akk_ + matrixToken * elementBytes, rows, rows);
                PublishL1Operands();
                RunL1GemmToGm<DvbCopy>(resource, akkL1, dwL1, ws + KDA_FINALIZE_A2_WS_DKGB, rows, KDA_FINALIZE_DIM,
                                       rows);
            }
            return;
        }

        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZV) {
            RunGemm<DkMmad, RowMajor, ColumnMajor>(resource, dvScan_, token, v_, token,
                                                   slotBase + KDA_FINALIZE_A2_WS_ZV, rows, KDA_FINALIZE_DIM, rows,
                                                   KDA_FINALIZE_DIM, rows, KDA_FINALIZE_CHUNK);
        }

        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_DW) {
            RunGemmBf16<DkBf16Mmad, RowMajor, ColumnMajor>(
                resource, dvScan_, token, h_, hState, slotBase + KDA_FINALIZE_A2_WS_DW_RELAY, rows, KDA_FINALIZE_DIM,
                KDA_FINALIZE_DIM, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        }

        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_DKGB || KDA_FINALIZE_A2_RESIDENT_RUN_ZW) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        }
        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_ZW) {
            RunGemm<DkMmad, RowMajor, ColumnMajor>(
                resource, workspaceAddr_, (slotBase + KDA_FINALIZE_A2_WS_DW_RELAY) / sizeof(Element), workspaceAddr_,
                (slotBase + KDA_FINALIZE_A2_WS_KE) / sizeof(Element), slotBase + KDA_FINALIZE_A2_WS_ZW, rows,
                KDA_FINALIZE_DIM, rows, KDA_FINALIZE_DIM, rows, KDA_FINALIZE_CHUNK);
        }
        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_DKGB) {
            RunGemm<DvbMmad, ColumnMajor, RowMajor>(resource, akk_, matrixToken, workspaceAddr_,
                                                    (slotBase + KDA_FINALIZE_A2_WS_DW_RELAY) / sizeof(Element),
                                                    slotBase + KDA_FINALIZE_A2_WS_DKGB, rows, rows, KDA_FINALIZE_DIM,
                                                    KDA_FINALIZE_CHUNK, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        }
    }

    __aicore__ inline void RunStage3(Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
                                     int64_t head, uint64_t slotBase)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint64_t matrixToken =
            static_cast<uint64_t>(FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK));
        auto zb = resource.l1Buf.template GetBufferByByte<Element>(0);
        auto at = resource.l1Buf.template GetBufferByByte<Element>(64 * 1024);
        auto tza = resource.l1Buf.template GetBufferByByte<Element>(KDA_FINALIZE_A2_L1_TZA);
        WaitL1LoadCredits();
        LoadGmToL1A<DkCopy, RowMajor>(zb, workspaceAddr_ + slotBase + KDA_FINALIZE_A2_WS_ZB, 64, 64);
        // Keep the full NZ stride for tails. Zb already has zero padding;
        // Akk only has valid rows in GM, so pad its L1 destination here.
        if (rows < KDA_FINALIZE_CHUNK) {
            AscendC::InitConstValue(at, AscendC::InitConstValueParams<Element>(1, 256, 0, static_cast<Element>(0)));
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::GlobalTensor<Element> akk;
        akk.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(akk_) + matrixToken);
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = KDA_FINALIZE_CHUNK;
        copy.srcDValue = KDA_FINALIZE_CHUNK;
        copy.dstNzC0Stride = KDA_FINALIZE_CHUNK;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(at, akk, copy);
        PublishL1Operands();
        RunL1GemmToL1Bf16<DkCopy>(resource, zb, at, tza, 64, 64, 64);
        if constexpr (KDA_FINALIZE_A2_RESIDENT_RUN_STAGE3_DAKK) {
            // Akk^T and BF16 Tza remain in L1 for the second contraction.
            WaitL1LoadCredits();
            PublishL1Operands();
            RunL1GemmToGm<DvbCopy>(resource, at, tza, workspaceAddr_ + slotBase + KDA_FINALIZE_A2_WS_INTRA_DAKK, 64, 64,
                                   64);
        }
    }

    template <typename Copy, bool MATRIX_A, typename GmLayout>
    __aicore__ inline void LoadIntraFp32(AscendC::LocalTensor<float> dst, GM_ADDR src, uint32_t rows, uint32_t cols)
    {
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src));
        auto source = tla::MakeTensor(gm, tla::MakeLayout<float, GmLayout>(rows, cols), Catlass::Arch::PositionGM{});
        using L1Layout =
            typename std::conditional<MATRIX_A, typename Copy::LayoutTagL1A, typename Copy::LayoutTagL1B>::type;
        auto target = tla::MakeTensor(dst, tla::MakeLayout<float, L1Layout>(rows, cols), Catlass::Arch::PositionL1{});
        if constexpr (MATRIX_A) {
            typename Copy::template CopyGmToL1A<decltype(source)>{}(target, source);
        } else {
            typename Copy::template CopyGmToL1B<decltype(source)>{}(target, source);
        }
    }

    template <typename Copy, bool TWO_TERMS>
    __aicore__ inline void RunIntraBand(Catlass::Arch::Resource<ArchTag> &resource, uint32_t band,
                                        AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> b,
                                        AscendC::LocalTensor<float> secondA, AscendC::LocalTensor<float> secondB,
                                        GM_ADDR dst)
    {
        constexpr uint32_t m = 32, n = 128, k = 64;
        auto c = resource.l0CBuf.template GetBufferByByte<float>(0);
        auto tensorC = tla::MakeTensor(c, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WyTileGemmDirectEvent::kL1B);
        for (uint32_t term = 0; term < (TWO_TERMS ? 2U : 1U); ++term) {
            auto l1A =
                tla::MakeTensor(term == 0 ? a : secondA, tla::MakeLayout<float, typename Copy::LayoutTagL1A>(64, k),
                                Catlass::Arch::PositionL1{});
            auto tileA = tla::GetTile(l1A, tla::MakeCoord(band * m, 0), tla::MakeShape(m, k));
            auto l1B =
                tla::MakeTensor(term == 0 ? b : secondB, tla::MakeLayout<float, typename Copy::LayoutTagL1B>(k, n),
                                Catlass::Arch::PositionL1{});
            auto l0A = resource.l0ABuf.template GetBufferByByte<float>(term * m * k * sizeof(float));
            auto l0B = resource.l0BBuf.template GetBufferByByte<float>(term * k * n * sizeof(float));
            auto tensorA = tla::MakeTensor(l0A, tla::MakeLayout<float, typename Copy::LayoutTagL0A>(m, k),
                                           Catlass::Arch::PositionL0A{});
            auto tensorB = tla::MakeTensor(l0B, tla::MakeLayout<float, typename Copy::LayoutTagL0B>(k, n),
                                           Catlass::Arch::PositionL0B{});
            const int32_t eventA = term == 0 ? WyTileGemmDirectEvent::kL0A : kL0A1Event;
            const int32_t eventB = term == 0 ? WyTileGemmDirectEvent::kL0B : kL0B1Event;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventA);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventB);
            typename Copy::CopyL1ToL0A{}(tensorA, tileA);
            typename Copy::CopyL1ToL0B{}(tensorB, l1B);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(term);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(term);
            Catlass::Gemm::Tile::TileMmadTla<ArchTag, float, typename Copy::LayoutTagL1A>{}(
                tensorC, tensorA, tensorB, m, n, k, term == 0, term + 1 == (TWO_TERMS ? 2U : 1U) ? 0b11 : 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventA);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        auto params = AscendC::FixpipeParamsV220(n, m, m, n, false);
        params.unitFlag = 0b11;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gm, c, params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
    }

    __aicore__ inline void RunStage5Dq(Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase)
    {
        auto a = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_DAQK);
        auto kNeg = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_K_NEG);
        GM_ADDR ws = workspaceAddr_ + slotBase;
        for (uint32_t band = 0; band < 2; ++band) {
            WaitL1LoadCredits();
            if (band == 0) {
                LoadIntraFp32<Fp32Copy, true, RowMajor>(a, ws + KDA_FINALIZE_A2_WS_INTRA_DAQK, 64, 64);
            }
            LoadIntraFp32<Fp32Copy, false, RowMajor>(kNeg[band * 8192],
                                                     ws + KDA_FINALIZE_A2_WS_INTRA_K_NEG0 + band * 32768, 64, 128);
            PublishL1Operands();
            RunIntraBand<Fp32Copy, false>(resource, band, a, kNeg[band * 8192], {}, {},
                                          ws + KDA_FINALIZE_A2_WS_INTRA_DQ_EARLY + band * 16384);
        }
    }

    __aicore__ inline void RunStage5Dk(Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase)
    {
        auto a = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_DAKK);
        auto kNeg = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_K_NEG);
        GM_ADDR ws = workspaceAddr_ + slotBase;
        for (uint32_t band = 0; band < 2; ++band) {
            WaitL1LoadCredits();
            if (band == 0) {
                LoadIntraFp32<Fp32Copy, true, RowMajor>(a, ws + KDA_FINALIZE_A2_WS_INTRA_DAKK, 64, 64);
            }
            PublishL1Operands();
            RunIntraBand<Fp32Copy, false>(resource, band, a, kNeg[band * 8192], {}, {},
                                          ws + KDA_FINALIZE_A2_WS_INTRA_DK + band * 16384);
        }
    }

    __aicore__ inline void RunStage5Right(Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase)
    {
        auto a = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_DAQK_T);
        auto b = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_DAKK_T);
        auto q = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_Q_POS);
        auto bk = resource.l1Buf.template GetBufferByByte<float>(KDA_FINALIZE_A2_L1_INTRA_BK_POS);
        GM_ADDR ws = workspaceAddr_ + slotBase;
        for (uint32_t band = 0; band < 2; ++band) {
            WaitL1LoadCredits();
            if (band == 0) {
                LoadIntraFp32<Fp32CopyTransA, true, ColumnMajor>(a, ws + KDA_FINALIZE_A2_WS_INTRA_DAQK, 64, 64);
                LoadIntraFp32<Fp32CopyTransA, true, ColumnMajor>(b, ws + KDA_FINALIZE_A2_WS_INTRA_DAKK, 64, 64);
            }
            LoadIntraFp32<Fp32CopyTransA, false, RowMajor>(q, ws + KDA_FINALIZE_A2_WS_INTRA_Q_POS + band * 32768, 64,
                                                           128);
            LoadIntraFp32<Fp32CopyTransA, false, RowMajor>(bk, ws + KDA_FINALIZE_A2_WS_INTRA_BK_POS + band * 32768, 64,
                                                           128);
            PublishL1Operands();
            RunIntraBand<Fp32CopyTransA, true>(resource, band, a, q, b, bk,
                                               ws + KDA_FINALIZE_A2_WS_INTRA_RIGHT + band * 16384);
        }
    }

private:
    __aicore__ inline void RunStage1a(Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
                                      int64_t head, uint64_t slotBase)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint64_t token = static_cast<uint64_t>(FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM));
        const uint64_t matrixToken =
            static_cast<uint64_t>(FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK));
        const uint64_t state = static_cast<uint64_t>(FinalizeDhOffset(*tiling_, chunk, head));

        constexpr uint32_t kMTile = KDA_FINALIZE_CHUNK;
        for (uint32_t row = 0; row < rows; row += kMTile) {
            const uint32_t curM = FinalizeMin(kMTile, rows - row);
            const uint64_t outputByteOffset = static_cast<uint64_t>(row) * KDA_FINALIZE_DIM * sizeof(float);
            RunGemm<DkMmad, RowMajor, ColumnMajor>(resource, vNew_, token + row * KDA_FINALIZE_DIM, dh_, state,
                                                   KDA_FINALIZE_A2_WS_DK_STATE + slotBase + outputByteOffset, curM,
                                                   KDA_FINALIZE_DIM, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
                                                   KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        }
        for (uint32_t row = 0; row < rows; row += kMTile) {
            const uint32_t curM = FinalizeMin(kMTile, rows - row);
            const uint64_t outputByteOffset = static_cast<uint64_t>(row) * KDA_FINALIZE_DIM * sizeof(float);
            RunGemm<DvbMmad, ColumnMajor, RowMajor>(
                resource, akk_, matrixToken + row, dvScan_, token, KDA_FINALIZE_A2_WS_DVB + slotBase + outputByteOffset,
                curM, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_CHUNK, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        }
    }

    template <typename Mmad, typename LayoutA, typename LayoutB>
    __aicore__ inline void RunGemm(Catlass::Arch::Resource<ArchTag> &resource, GM_ADDR aAddr, uint64_t aOffset,
                                   GM_ADDR bAddr, uint64_t bOffset, uint64_t cByteOffset, uint32_t m, uint32_t k,
                                   uint32_t n, uint32_t aPhysicalCols, uint32_t bPhysicalCols, uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<Element> a;
        AscendC::GlobalTensor<Element> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(bAddr) + bOffset);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspaceAddr_) + cByteOffset / sizeof(float));

        auto tensorA = tla::MakeTensor(a, tla::MakeLayout<Element, LayoutA>(KDA_FINALIZE_CHUNK, aPhysicalCols),
                                       Catlass::Arch::PositionGM{});
        auto tensorB =
            tla::MakeTensor(b,
                            tla::MakeLayout<Element, LayoutB>(
                                AscendC::IsSameType<LayoutB, ColumnMajor>::value ? aPhysicalCols : k, bPhysicalCols),
                            Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, tla::MakeLayout<float, RowMajor>(KDA_FINALIZE_CHUNK, cPhysicalCols),
                                       Catlass::Arch::PositionGM{});
        auto tileA = tla::GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = tla::GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord shape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, shape);
    }

    template <typename Mmad, typename LayoutA, typename LayoutB>
    __aicore__ inline void RunGemmBf16(Catlass::Arch::Resource<ArchTag> &resource, GM_ADDR aAddr, uint64_t aOffset,
                                       GM_ADDR bAddr, uint64_t bOffset, uint64_t cByteOffset, uint32_t m, uint32_t k,
                                       uint32_t n, uint32_t aPhysicalCols, uint32_t bPhysicalCols,
                                       uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<Element> a;
        AscendC::GlobalTensor<Element> b;
        AscendC::GlobalTensor<Element> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(bAddr) + bOffset);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(workspaceAddr_) + cByteOffset / sizeof(Element));

        auto tensorA = tla::MakeTensor(a, tla::MakeLayout<Element, LayoutA>(KDA_FINALIZE_CHUNK, aPhysicalCols),
                                       Catlass::Arch::PositionGM{});
        auto tensorB =
            tla::MakeTensor(b,
                            tla::MakeLayout<Element, LayoutB>(
                                AscendC::IsSameType<LayoutB, ColumnMajor>::value ? aPhysicalCols : k, bPhysicalCols),
                            Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, tla::MakeLayout<Element, RowMajor>(KDA_FINALIZE_CHUNK, cPhysicalCols),
                                       Catlass::Arch::PositionGM{});
        auto tileA = tla::GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = tla::GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord shape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, shape);
    }

    __aicore__ inline void BeginMmadPhase()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1B);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kL0A1Event);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kL0B1Event);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
    }

    __aicore__ inline void EndMmadPhase()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WyTileGemmDirectEvent::kL1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(WyTileGemmDirectEvent::kL0B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kL0A1Event);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kL0B1Event);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(WyTileGemmDirectEvent::kL0C);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(WyTileGemmDirectEvent::kL0C);
    }

    __aicore__ inline void FenceFixToMte2()
    {
        constexpr int32_t kFixToMte2Event = 0;
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(kFixToMte2Event);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(kFixToMte2Event);
    }

    __aicore__ inline void WaitVectorStage(uint32_t flag)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag);
    }

    GM_ADDR v_ = nullptr;
    GM_ADDR akk_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR dvScan_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspaceAddr_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_RESIDENT_CUBE_H
