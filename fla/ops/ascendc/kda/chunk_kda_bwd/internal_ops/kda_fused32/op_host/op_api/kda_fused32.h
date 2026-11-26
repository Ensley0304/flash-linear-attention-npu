// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
#include "../../../../op_kernel/arch22/fused32/contract.h"

namespace l0op {
struct KdaFused32Request {
    // Inputs follow kda_fused32_def.cpp exactly; metadata is supplied by L2.
    std::array<const aclTensor*,24> inputs{};
    // o,ht,dq,dk,dv,dg,db,dh0,dA,dbias, then 13 caches, then U.
    std::array<const aclTensor*,24> outputs{};
    KDA::Arch22::Fused32::Config config{};
    const aclIntArray *cu=nullptr,*chunkIndices=nullptr;
    const char *layout=nullptr;
    bool backward=false;
};
// Performs all descriptor/flag checks in ACLNN; no data transforms or fallback.
aclnnStatus KdaFused32(KdaFused32Request &request,aclOpExecutor *executor);
}
