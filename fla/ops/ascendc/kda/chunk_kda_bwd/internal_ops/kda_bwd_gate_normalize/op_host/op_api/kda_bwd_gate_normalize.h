// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
namespace l0op {
const aclTensor *KdaBwdGateNormalize(const aclTensor *raw,const aclTensor *a,const aclTensor *bias,
    const aclIntArray *cu,bool useExp2,bool safeGate,double lower,const aclTensor *out,aclOpExecutor *executor);
}
