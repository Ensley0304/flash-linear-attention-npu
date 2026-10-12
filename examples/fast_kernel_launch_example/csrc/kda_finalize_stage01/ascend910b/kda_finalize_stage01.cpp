/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <algorithm>
#include <cmath>
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

// Stage 0-6 harness: two GM generations overlap Stage 3 with the next Cube work.
// Later stages must retain exp2_gk/rH or defer this release until their last UB read.
constexpr uint16_t VECTOR_RELEASE = 9;
constexpr uint16_t VECTOR_ACK = 10;
constexpr uint16_t S02_READY = 4;
constexpr uint16_t S5_DONE = 12;

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
                                         GM_ADDR dh, GM_ADDR dvScan, GM_ADDR beta, GM_ADDR dv, GM_ADDR q, GM_ADDR dqRaw,
                                         float scale, GM_ADDR metadata, GM_ADDR workspace, GM_ADDR capture,
                                         GM_ADDR trace, Stage01TilingData tiling)
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
            const uint32_t bank = LAST_STAGE >= 3 ? (work / tiling.usedCoreNum) & 1 : 0;
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t firstHead = (work / tiling.chunks) * HEADS_PER_WORK;
            // Fixed-length metadata stays in registers; variable lengths use the packed table.
            const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
            if (LAST_STAGE < 2 || work == core) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(WORK_OPEN);
            }
            if constexpr (LAST_STAGE >= 5) {
                if (work >= core + 2 * tiling.usedCoreNum) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + bank);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.LoadStage0Inputs(tiling, chunk, chunkIndex, firstHead + owner, owner, work == core,
                                          LAST_STAGE >= 4 ? bank : 0);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.Stage0<CAPTURE>(tiling, chunk, chunkIndex, firstHead + owner, owner, core, capture, bank,
                                         LAST_STAGE >= 4 ? bank : 0);
                    if constexpr (LAST_STAGE < 2) {
                        cube.WaitDw(owner);
                        cube.ReleaseAkk(owner);
                    }
                    if constexpr (LAST_STAGE >= 2) {
                        if (owner + 1 == HEADS_PER_WORK || firstHead + owner + 1 == tiling.heads) {
                            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KE_READY);
                            for (uint32_t keOwner = 0; keOwner < HEADS_PER_WORK; ++keOwner) {
                                if (firstHead + keOwner < tiling.heads) {
                                    cube.LoadKe(keOwner, core, bank);
                                }
                            }
                            // kE is now protected in L1. Vector can prepare the next work
                            // while Cube consumes the current one; akk has its own free event.
                            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
                            if (work + tiling.usedCoreNum < workCount) {
                                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(WORK_OPEN);
                            }
                        }
                    }
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
            if constexpr (LAST_STAGE < 2) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KE_READY);
            }
            if constexpr (LAST_STAGE >= 2) {
                for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                    if (firstHead + owner < tiling.heads) {
                        cube.Stage2<CAPTURE>(tiling, chunkIndex, firstHead + owner, owner, core, capture, bank,
                                             LAST_STAGE >= 4);
                    }
                }
            }
            if constexpr (LAST_STAGE < 2) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(VECTOR_RELEASE);
            }
            if constexpr (LAST_STAGE >= 3) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S02_READY + bank);
                // Two GM generations: retire the previous consumer before reusing its bank.
                if (work != core) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK + (bank ^ 1));
                    if constexpr (LAST_STAGE >= 4) {
                        const int64_t readyWork = work - tiling.usedCoreNum;
                        const int64_t readyHead = (readyWork / tiling.chunks) * HEADS_PER_WORK;
                        const uint32_t readyBank = (readyWork / tiling.usedCoreNum) & 1;
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.LoadZb(owner, core, readyBank);
                            }
                        }
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.Stage4<CAPTURE, (LAST_STAGE >= 6)>(tiling, readyWork % tiling.chunks,
                                                                        readyHead + owner, owner, readyBank, capture);
                            }
                        }
                        if constexpr (LAST_STAGE >= 6) {
                            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                                if (readyHead + owner < tiling.heads) {
                                    cube.Stage6<CAPTURE>(tiling, readyWork % tiling.chunks, readyHead + owner, owner,
                                                         core, readyBank, capture);
                                }
                            }
                        }
                    }
                }
                if (work + tiling.usedCoreNum >= workCount) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK + bank);
                    if constexpr (LAST_STAGE >= 4) {
                        const int64_t readyWork = work;
                        const int64_t readyHead = (readyWork / tiling.chunks) * HEADS_PER_WORK;
                        const uint32_t readyBank = (readyWork / tiling.usedCoreNum) & 1;
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.LoadZb(owner, core, readyBank);
                            }
                        }
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.Stage4<CAPTURE, (LAST_STAGE >= 6)>(tiling, readyWork % tiling.chunks,
                                                                        readyHead + owner, owner, readyBank, capture);
                            }
                        }
                        if constexpr (LAST_STAGE >= 6) {
                            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                                if (readyHead + owner < tiling.heads) {
                                    cube.Stage6<CAPTURE>(tiling, readyWork % tiling.chunks, readyHead + owner, owner,
                                                         core, readyBank, capture);
                                }
                            }
                        }
                    }
                }
            } else {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK);
            }
        }
        if constexpr (LAST_STAGE >= 5) {
            const int64_t last = core + ((workCount - 1 - core) / tiling.usedCoreNum) * tiling.usedCoreNum;
            if (last > core) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + (((last / tiling.usedCoreNum) & 1) ^ 1));
            }
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + ((last / tiling.usedCoreNum) & 1));
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
        if constexpr (LAST_STAGE >= 1) {
            vector.Init(k, gk, h, dh, workspace, ub, &pipe);
            if constexpr (LAST_STAGE >= 3) {
                vector.InitStage3(v, beta, dv);
            }
            if constexpr (LAST_STAGE >= 5) {
                vector.InitStage5(q, dqRaw, scale);
            }
        }
        AscendC::GlobalTensor<int64_t> traceGm;
        traceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace));
        const auto traceReady = pipe.AllocEventID<AscendC::HardEvent::S_MTE3>();
        const auto traceFree = pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
        if constexpr (CAPTURE && LAST_STAGE >= 3) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
        }
        for (int64_t work = core; work < workCount; work += tiling.usedCoreNum) {
            const uint32_t bank = LAST_STAGE >= 3 ? (work / tiling.usedCoreNum) & 1 : 0;
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t head = (work / tiling.chunks) * HEADS_PER_WORK + owner;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(WORK_OPEN);
            if constexpr (LAST_STAGE >= 1) {
                if (head < tiling.heads) {
                    const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
                    vector.Stage1<CAPTURE, (LAST_STAGE >= 5)>(tiling, chunk, chunkIndex, head, owner, core, capture,
                                                              bank);
                }
            }
            // The absent owner in an odd-head work also closes the collective.
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(KE_READY);
            if constexpr (LAST_STAGE >= 3) {
                // Send the next kE first; retire the preceding Stage 3 while Cube runs Stage 2.
                if (work != core) {
                    const int64_t previous = work - tiling.usedCoreNum;
                    const int64_t previousChunk = previous % tiling.chunks;
                    const int64_t previousHead = (previous / tiling.chunks) * HEADS_PER_WORK + owner;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S02_READY + (bank ^ 1));
                    if (previousHead < tiling.heads) {
                        const ChunkInfo chunk = ResolveChunk(tiling, chunks, previousChunk);
                        vector.Stage3<CAPTURE>(tiling, chunk, previousChunk, previousHead, owner, core, capture,
                                               bank ^ 1);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK + (bank ^ 1));
                    if constexpr (LAST_STAGE >= 5) {
                        if (previousHead < tiling.heads) {
                            const ChunkInfo chunk = ResolveChunk(tiling, chunks, previousChunk);
                            vector.Stage5<CAPTURE>(tiling, chunk, previousChunk, previousHead, owner, core, capture,
                                                   bank ^ 1);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(S5_DONE + (bank ^ 1));
                    }
                }
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
                if (work + tiling.usedCoreNum >= workCount) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S02_READY + bank);
                    if (head < tiling.heads) {
                        const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
                        vector.Stage3<CAPTURE>(tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK + bank);
                    if constexpr (LAST_STAGE >= 5) {
                        if (head < tiling.heads) {
                            const ChunkInfo chunk = ResolveChunk(tiling, chunks, chunkIndex);
                            vector.Stage5<CAPTURE>(tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(S5_DONE + (bank));
                    }
                }
            } else {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
            }
            if constexpr (CAPTURE) {
                if constexpr (LAST_STAGE >= 3) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                }
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
                record.SetValue(10, 7); // WORK_OPEN waited, KE_READY sent, VECTOR_RELEASE waited.
                record.SetValue(11, AscendC::GetBlockNum());
                record.SetValue(12, AscendC::GetSubBlockNum());
                record.SetValue(13, AscendC::GetBlockIdx());
                record.SetValue(14, static_cast<int64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES +
                                        bank * WS_BANK_BYTES);
                record.SetValue(15, 0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::DataCopy(traceGm[(work * HEADS_PER_WORK + owner) * 16], record, 16);
                if constexpr (LAST_STAGE >= 3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                }
            }
            if constexpr (LAST_STAGE < 3) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK);
            }
            if constexpr (LAST_STAGE < 3) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
            }
        }
        if constexpr (CAPTURE && LAST_STAGE >= 3) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
        }
        pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(traceReady);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(traceFree);
        if constexpr (LAST_STAGE >= 1) {
            vector.End();
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
RunStage01(const at::Tensor &k, const at::Tensor &v, const at::Tensor &gk, const at::Tensor &akk,
           const at::Tensor &vNew, const at::Tensor &h, const at::Tensor &dh, const at::Tensor &dvScan,
           at::IntArrayRef cuSeqlens, int64_t coreLimit, int64_t lastStage, const c10::optional<at::Tensor> &beta,
           const c10::optional<at::Tensor> &dv, const c10::optional<at::Tensor> &q,
           const c10::optional<at::Tensor> &dqRaw, double scale)
{
    TORCH_CHECK(lastStage >= 0 && lastStage <= 6, "last_stage must be 0, 1, 2, 3, 4, 5 or 6");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "expected NPU inputs");
    c10::DeviceGuard guard(k.device());
    TORCH_CHECK(k.dim() == 3 || k.dim() == 4, "expected [B,H,T,128] or [H,T,128]");
    const bool variable = k.dim() == 3;
    const int64_t batch = variable ? 1 : k.size(0);
    const int64_t heads = k.size(k.dim() - 3);
    const int64_t tokens = k.size(k.dim() - 2);
    TORCH_CHECK(k.size(-1) == HEAD_DIM, "Stage 0-6 requires K=V=128");
    for (const auto *tensor : {&k, &v, &vNew, &h, &dh, &dvScan, &akk}) {
        TORCH_CHECK(tensor->device() == k.device() && tensor->is_contiguous() && tensor->scalar_type() == at::kBFloat16,
                    "expected contiguous BF16 inputs on one device");
    }
    TORCH_CHECK(gk.device() == k.device() && gk.is_contiguous() && gk.scalar_type() == at::kFloat &&
                    gk.sizes() == k.sizes(),
                "gk must be contiguous FP32 with k's shape");
    TORCH_CHECK(v.sizes() == k.sizes() && vNew.sizes() == k.sizes() && dvScan.sizes() == k.sizes(),
                "v, v_new and dv_scan must have k's shape");
    if (lastStage >= 3) {
        TORCH_CHECK(beta.has_value() && dv.has_value(), "Stage 3 requires beta and dv_out");
        auto betaShape = k.sizes().vec();
        betaShape.pop_back();
        TORCH_CHECK(beta->device() == k.device() && beta->is_contiguous() && beta->scalar_type() == at::kFloat &&
                        beta->sizes() == at::IntArrayRef(betaShape),
                    "beta must be contiguous FP32 [B,H,T] or [H,T]");
        TORCH_CHECK(dv->device() == k.device() && dv->is_contiguous() && dv->scalar_type() == at::kBFloat16 &&
                        dv->sizes() == k.sizes(),
                    "dv_out must be contiguous BF16 with k's shape");
    }
    if (lastStage >= 5) {
        TORCH_CHECK(q.has_value() && dqRaw.has_value(), "Stage 5 requires q and dq_raw");
        TORCH_CHECK(q->device() == k.device() && q->is_contiguous() && q->scalar_type() == at::kBFloat16 &&
                        q->sizes() == k.sizes(),
                    "q must be contiguous BF16 with k's shape");
        TORCH_CHECK(dqRaw->device() == k.device() && dqRaw->is_contiguous() && dqRaw->scalar_type() == at::kFloat &&
                        dqRaw->sizes() == k.sizes(),
                    "dq_raw must be contiguous FP32 with k's shape");
        TORCH_CHECK(std::isfinite(static_cast<float>(scale)), "scale must be finite");
    }
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
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else if (lastStage == 1) {
            Stage01Kernel<1><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else if (lastStage == 2) {
            Stage01Kernel<2><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else if (lastStage == 3) {
            Stage01Kernel<3><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else if (lastStage == 4) {
            Stage01Kernel<4><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else if (lastStage == 5) {
            Stage01Kernel<5><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        } else {
            Stage01Kernel<6><<<tiling.usedCoreNum, nullptr, stream>>>(
                (GM_ADDR)k.data_ptr(), (GM_ADDR)v.data_ptr(), (GM_ADDR)gk.data_ptr(), (GM_ADDR)akk.data_ptr(),
                (GM_ADDR)vNew.data_ptr(), (GM_ADDR)h.data_ptr(), (GM_ADDR)dh.data_ptr(), (GM_ADDR)dvScan.data_ptr(),
                (GM_ADDR)(beta ? beta->data_ptr() : nullptr), (GM_ADDR)(dv ? dv->data_ptr() : nullptr),
                (GM_ADDR)(q ? q->data_ptr() : nullptr), (GM_ADDR)(dqRaw ? dqRaw->data_ptr() : nullptr),
                static_cast<float>(scale), (GM_ADDR)chunkMetadata.data_ptr(), (GM_ADDR)workspace.data_ptr(),
                (GM_ADDR)capture.data_ptr(), (GM_ADDR)trace.data_ptr(), tiling);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("KdaFinalizeStage01", launch);
    return {capture, trace, workspace};
}

TORCH_LIBRARY_FRAGMENT(ascend_ops, m)
{
    m.def("kda_finalize_stage01(Tensor k, Tensor v, Tensor gk, Tensor akk, Tensor v_new, Tensor h, Tensor dh, "
          "Tensor dv_scan, int[] cu_seqlens, int core_limit=0, int last_stage=1, Tensor? beta=None, Tensor(a!)? "
          "dv_out=None, Tensor? q=None, Tensor? dq_raw=None, float scale=1.0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("kda_finalize_stage01", RunStage01);
}

} // namespace ascend_ops::KdaFinalizeStage01
