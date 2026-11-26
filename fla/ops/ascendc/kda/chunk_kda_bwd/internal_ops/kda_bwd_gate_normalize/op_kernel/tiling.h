// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
namespace KDA::Arch22 {
struct GateNormalizeTiling {
    uint32_t batch,heads,tokens,keys,chunks,useExp2,safeGate;
    float lowerBound;
};
}
