/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <algorithm>
#include <tuple>
#include <vector>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_host/op_tiling/chunk_kda_bwd_finalize_stage01_tiling.h"
#include "fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize_cube.h"
#include "fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize_vector.h"

#ifndef KDA_FINALIZE_STAGE01_CAPTURE
#define KDA_FINALIZE_STAGE01_CAPTURE 1
#endif

namespace ascend_ops::KdaFinalizeStage01 {

constexpr bool CAPTURE = KDA_FINALIZE_STAGE01_CAPTURE != 0;

using namespace KDA::Arch22;

// These two flags belong only to this development harness, not to Stage 2.
constexpr uint16_t CAPTURE_READY = 9;
constexpr uint16_t CAPTURE_DONE = 10;

__aicore__ inline ChunkInfo ResolveChunk(const Stage01TilingData &tiling, const __gm__ ChunkInfo *chunks,
                                         int64_t chunkIndex)
{
    ChunkInfo chunk;
    if (tiling.isVariable) {
        chunk = {chunks[chunkIndex].sequence, chunks[chunkIndex].localChunk, chunks[chunkIndex].tokenStart,
                 chunks[chunkIndex].validRows};
    } else {
        const int64_t chunksPerSequence = (tiling.tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
        chunk.sequence = chunkIndex / chunksPerSequence;
        chunk.localChunk = chunkIndex % chunksPerSequence;
        chunk.tokenStart = chunk.localChunk * CHUNK_SIZE;
        const int64_t remaining = tiling.tokens - chunk.tokenStart;
        chunk.validRows = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
    }
    return chunk;
}

template <uint32_t LAST_STAGE>
__global__ __aicore__ void Stage01Kernel(GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h,
                                         GM_ADDR dh, GM_ADDR dvScan, GM_ADDR metadata, GM_ADDR workspace,
                                         GM_ADDR capture, GM_ADDR trace, Stage01TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    const int64_t workCount = ((tiling.heads + 1) / HEADS_PER_WORK) * tiling.chunks;
    const auto *chunks = reinterpret_cast<const __gm__ ChunkInfo *>(metadata);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        ChunkKdaBwdFinalizeCube cube;
        cube.Init(v, akk, vNew, h, dh, dvScan, workspace, &pipe);
        for (int64_t work = core; work < workCount; work += tiling.usedCoreNum) {
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t firstHead = (work / tiling.chunks) * HEADS_PER_WORK;
            // Fixed-length metadata stays in registers; variable lengths use the packed table.
            const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(WORK_OPEN);
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.LoadStage0Inputs(tiling, chunk, chunkIndex, firstHead + owner, owner, work == core);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.Stage0<CAPTURE>(tiling, chunk, chunkIndex, firstHead + owner, owner, core, capture);
                    // Stage 2 is absent: explicitly retire its dW ready event.
                    cube.WaitDw(owner);
                    // These five inputs end their L1 lifetime in Stage 0.
                    // Keep akk and dW untouched for their later consumers.
                    const int64_t nextWork = work + tiling.usedCoreNum;
                    if (nextWork < workCount) {
                        const int64_t nextChunkIndex = nextWork % tiling.chunks;
                        const int64_t nextHead = (nextWork / tiling.chunks) * HEADS_PER_WORK + owner;
                        if (nextHead < tiling.heads) {
                            const ChunkInfo nextChunk = ResolveChunk(tiling, chunks, nextChunkIndex);
                            cube.PreloadState(tiling, nextChunk, nextChunkIndex, nextHead, owner);
                        }
                    }
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(CAPTURE_READY);
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KE_READY);
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(CAPTURE_DONE);
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetSubBlockIdx();
        AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, UB_BYTES);
        auto ub = ubBuf.Get<uint8_t>();
        auto record = ub[85 * 1024].ReinterpretCast<int64_t>();
        ChunkKdaBwdFinalizeVector vector;
        if constexpr (LAST_STAGE == 1) {
            vector.Init(k, gk, h, dh, workspace, ub, &pipe);
        }
        AscendC::GlobalTensor<int64_t> traceGm;
        traceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace));
        const auto traceReady = pipe.AllocEventID<AscendC::HardEvent::S_MTE3>();
        const auto traceFree = pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
        for (int64_t work = core; work < workCount; work += tiling.usedCoreNum) {
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t head = (work / tiling.chunks) * HEADS_PER_WORK + owner;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(WORK_OPEN);
            if constexpr (LAST_STAGE == 1) {
                if (head < tiling.heads) {
                    const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
                    vector.Stage1<CAPTURE>(tiling, chunk, chunkIndex, head, owner, core, capture);
                }
            }
            // The absent owner in an odd-head work also closes the collective.
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(KE_READY);
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(CAPTURE_READY);
            if constexpr (CAPTURE) {
                record.SetValue(0, work);
                record.SetValue(1, core);
                record.SetValue(2, owner);
                record.SetValue(3, head);
                record.SetValue(4, chunkIndex);
                record.SetValue(5, work / tiling.usedCoreNum + 1);
                record.SetValue(6, tiling.usedCoreNum);
                record.SetValue(7, chunks[chunkIndex].validRows);
                record.SetValue(8, head < tiling.heads ? 1 : 0);
                record.SetValue(9, LAST_STAGE);
                record.SetValue(10, 7); // WORK_OPEN waited, KE_READY sent, CAPTURE_READY waited.
                record.SetValue(11, AscendC::GetBlockNum());
                record.SetValue(12, AscendC::GetSubBlockNum());
                record.SetValue(13, AscendC::GetBlockIdx());
                record.SetValue(14, static_cast<int64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES);
                record.SetValue(15, 0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::DataCopy(traceGm[(work * HEADS_PER_WORK + owner) * 16], record, 16);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(CAPTURE_DONE);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
        }
        pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(traceReady);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(traceFree);
        if constexpr (LAST_STAGE == 1) {
            vector.End();
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
RunStage01(const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk, const at::Tensor &akk,
           const at::Tensor &vNew, const at::Tensor &h, const at::Tensor &dh, const at::Tensor &dvScan,
           at::IntArrayRef cuSeqlens, int64_t coreLimit, int64_t lastStage)
{
    TORCH_CHECK(lastStage == 0 || lastStage == 1, "last_stage must be 0 or 1");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "expected NPU inputs");
    c10::DeviceGuard guard(k.device());
    TORCH_CHECK(k.dim() == 3 || k.dim() == 4, "expected [B,H,T,128] or [H,T,128]");
    const bool variable = k.dim() == 3;
    const int64_t batch = variable ? 1 : k.size(0);
    const int64_t heads = k.size(k.dim() - 3);
    const int64_t tokens = k.size(k.dim() - 2);
    TORCH_CHECK(k.size(-1) == HEAD_DIM, "Stage 0/1 requires K=V=128");
    for (const auto *tensor : {&k, &v, &vNew, &h, &dh, &dvScan, &akk}) {
        TORCH_CHECK(tensor->device() == k.device() && tensor->is_contiguous() && tensor->scalar_type() == at::kBFloat16,
                    "expected contiguous BF16 inputs on one device");
    }
    TORCH_CHECK(gk.device() == k.device() && gk.is_contiguous() && gk.scalar_type() == at::kFloat &&
                    gk.sizes() == k.sizes(),
                "gk must be contiguous FP32 with k's shape");
    TORCH_CHECK(v.sizes() == k.sizes() && vNew.sizes() == k.sizes() && dvScan.sizes() == k.sizes(),
                "v, v_new and dv_scan must have k's shape");
    auto akkShape = k.sizes().vec();
    akkShape.back() = CHUNK_SIZE;
    TORCH_CHECK(akk.sizes() == at::IntArrayRef(akkShape), "akk must have token layout with width 64");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot query Ascend platform");
    const uint32_t available = platform->GetCoreNumAic();
    TORCH_CHECK(coreLimit >= 0 && coreLimit <= available, "invalid core_limit");
    const auto plan = optiling::BuildFinalizeStage01Plan(batch, heads, tokens, variable,
                                                         std::vector<int64_t>(cuSeqlens.begin(), cuSeqlens.end()),
                                                         coreLimit == 0 ? available : static_cast<uint32_t>(coreLimit));
    const auto tiling = plan.tiling;
    const std::vector<int64_t> stateShape =
        variable ? std::vector<int64_t>{tiling.chunks, heads, HEAD_DIM, HEAD_DIM} :
                   std::vector<int64_t>{batch, tiling.chunks / batch, heads, HEAD_DIM, HEAD_DIM};
    TORCH_CHECK(h.sizes() == at::IntArrayRef(stateShape) && dh.sizes() == h.sizes(),
                "h/dh must use chunk-major state layout");

    std::vector<int64_t> metadata;
    metadata.reserve(plan.chunks.size() * 4);
    for (const auto &chunk : plan.chunks) {
        metadata.insert(metadata.end(), {chunk.sequence, chunk.localChunk, chunk.tokenStart, chunk.validRows});
    }
    auto chunkMetadata = at::tensor(metadata, k.options().dtype(at::kLong));
    auto workspace = CAPTURE ?
                         at::full({static_cast<int64_t>(plan.workspaceBytes)}, 0x5A, k.options().dtype(at::kByte)) :
                         at::empty({static_cast<int64_t>(plan.workspaceBytes)}, k.options().dtype(at::kByte));
    auto capture = CAPTURE ? at::full({heads, tiling.chunks, CAPTURE_BYTES}, 0xA5, k.options().dtype(at::kByte)) :
                             at::empty({1}, k.options().dtype(at::kByte));
    auto trace = CAPTURE ?
                     at::full({static_cast<int64_t>(plan.traceBytes / 128), 16}, -1, k.options().dtype(at::kLong)) :
                     at::empty({1}, k.options().dtype(at::kLong));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [=]() -> int {
        if (lastStage == 0) {
            Stage01Kernel<0><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(), (GM_ADDR)capture.data_ptr(),
                (GM_ADDR)trace.data_ptr(), tiling);
        } else {
            Stage01Kernel<1><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(), (GM_ADDR)capture.data_ptr(),
                (GM_ADDR)trace.data_ptr(), tiling);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("KdaFinalizeStage01", launch);
    return {capture, trace, workspace};
}

TORCH_LIBRARY_FRAGMENT(ascend_ops, m)
{
    m.def("kda_finalize_stage01(Tensor k, Tensor v, Tensor gk, Tensor akk, Tensor v_new, Tensor h, Tensor dh, "
          "Tensor dv_scan, int[] cu_seqlens, int core_limit=0, int last_stage=1) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("kda_finalize_stage01", RunStage01);
}

} // namespace ascend_ops::KdaFinalizeStage01
