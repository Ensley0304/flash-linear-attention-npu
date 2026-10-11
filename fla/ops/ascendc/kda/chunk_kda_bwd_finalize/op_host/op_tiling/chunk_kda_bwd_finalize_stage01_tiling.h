/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_STAGE01_TILING_H
#define CHUNK_KDA_BWD_FINALIZE_STAGE01_TILING_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include "../../op_kernel/arch22/chunk_kda_bwd_finalize_common.h"

namespace optiling {

// Private Stage 0/1 harness. All task/metadata/workspace sizing has one owner.
struct FinalizeStage01Plan {
    KDA::Arch22::Stage01TilingData tiling{};
    std::vector<KDA::Arch22::ChunkInfo> chunks;
    uint64_t workspaceBytes = 0;
    uint64_t captureBytes = 0;
    uint64_t traceBytes = 0;
};

inline FinalizeStage01Plan BuildFinalizeStage01Plan(int64_t batch, int64_t heads, int64_t tokens, bool variable,
                                                    const std::vector<int64_t> &cuSeqlens, uint32_t availableCores)
{
    using namespace KDA::Arch22;
    if (batch <= 0 || heads <= 0 || tokens <= 0 || availableCores == 0) {
        throw std::invalid_argument("Stage 0/1 requires positive dimensions and an available AIC");
    }
    if (variable && (batch != 1 || cuSeqlens.size() < 2 || cuSeqlens.front() != 0 || cuSeqlens.back() != tokens)) {
        throw std::invalid_argument("varlen requires B=1 and cu_seqlens spanning [0,T]");
    }
    if (!variable && !cuSeqlens.empty()) {
        throw std::invalid_argument("fixed inputs must not provide cu_seqlens");
    }
    FinalizeStage01Plan plan;
    const int64_t sequences = variable ? static_cast<int64_t>(cuSeqlens.size()) - 1 : batch;
    for (int64_t seq = 0; seq < sequences; ++seq) {
        const int64_t begin = variable ? cuSeqlens[seq] : 0;
        const int64_t end = variable ? cuSeqlens[seq + 1] : tokens;
        if (begin < 0 || end < begin || end > tokens) {
            throw std::invalid_argument("cu_seqlens must be nondecreasing and within [0,T]");
        }
        for (int64_t start = begin, local = 0; start < end; ++local) {
            const int64_t rows = std::min<int64_t>(CHUNK_SIZE, end - start);
            plan.chunks.push_back({seq, local, start, rows});
            start += rows;
        }
    }
    const uint64_t count = plan.chunks.size();
    const uint64_t maxSize = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (count == 0 || static_cast<uint64_t>(heads) > maxSize / count / CAPTURE_BYTES) {
        throw std::overflow_error("Stage 0/1 capture/workspace size is invalid");
    }
    const uint64_t work = ((static_cast<uint64_t>(heads) + 1) / HEADS_PER_WORK) * count;
    plan.tiling = {batch,
                   heads,
                   tokens,
                   static_cast<int64_t>(count),
                   static_cast<uint32_t>(std::min<uint64_t>(availableCores, work)),
                   static_cast<uint32_t>(variable)};
    plan.workspaceBytes = static_cast<uint64_t>(plan.tiling.usedCoreNum) * WS_CORE_BYTES;
    plan.captureBytes = static_cast<uint64_t>(heads) * count * CAPTURE_BYTES;
    plan.traceBytes = work * HEADS_PER_WORK * 128;
    return plan;
}

} // namespace optiling

#endif // CHUNK_KDA_BWD_FINALIZE_STAGE01_TILING_H
