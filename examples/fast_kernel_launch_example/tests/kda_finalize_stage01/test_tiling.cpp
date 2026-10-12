// Copyright (c) 2026 Tianjin University, Ltd.
#include <cassert>
#include <functional>
#include <limits>
#include "fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_host/op_tiling/chunk_kda_bwd_finalize_stage01_tiling.h"

int main()
{
    using optiling::BuildFinalizeStage01Plan;
    const auto fixed = BuildFinalizeStage01Plan(2, 3, 65, false, {}, 1);
    assert(fixed.chunks.size() == 4 && fixed.tiling.usedCoreNum == 1);
    assert(fixed.chunks[1].validRows == 1 && fixed.chunks[2].sequence == 1);
    assert(fixed.workspaceBytes == 1280 * 1024);
    assert(fixed.captureBytes == 3 * 4 * KDA::Arch22::CAPTURE_BYTES);
    const auto variable = BuildFinalizeStage01Plan(1, 3, 99, true, {0, 0, 1, 34, 99, 99}, 24);
    assert(variable.chunks.size() == 4 && variable.tiling.usedCoreNum == 8);
    assert(variable.chunks[0].sequence == 1 && variable.chunks[3].tokenStart == 98);
    assert(variable.chunks[3].validRows == 1);
    const auto model = BuildFinalizeStage01Plan(1, 96, 8192, false, {}, 24);
    assert(model.chunks.size() == 128 && model.tiling.usedCoreNum == 24);
    assert(model.workspaceBytes == 30 * 1024 * 1024);
    for (const auto &bad : std::vector<std::function<void()>>{
             [] { BuildFinalizeStage01Plan(1, 0, 64, false, {}, 1); },
             [] { BuildFinalizeStage01Plan(1, 1, 64, false, {}, 0); },
             [] { BuildFinalizeStage01Plan(1, 1, 64, false, {0, 64}, 1); },
             [] { BuildFinalizeStage01Plan(1, 1, 64, true, {1, 64}, 1); },
             [] { BuildFinalizeStage01Plan(1, 1, 64, true, {0, 40, 30, 64}, 1); },
             [] { BuildFinalizeStage01Plan(1, 1, 64, true, {0, 63}, 1); },
             [] { BuildFinalizeStage01Plan(1, std::numeric_limits<int64_t>::max(), 1, false, {}, 1); }}) {
        bool rejected = false;
        try {
            bad();
        } catch (const std::exception &) {
            rejected = true;
        }
        assert(rejected);
    }
}
