/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <type_traits>

#define KDA_FINALIZE_A2_RESIDENT_SOURCE_REVISION 155
#ifndef KDA_FINALIZE_A2_DIAG_SKIP_GATE
#define KDA_FINALIZE_A2_DIAG_SKIP_GATE 0
#endif
#ifndef KDA_FINALIZE_A2_RESIDENT_FUSED_GATE
#define KDA_FINALIZE_A2_RESIDENT_FUSED_GATE 1
#endif

#include "chunk_kda_bwd_finalize_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_finalize_cube.h"
#include "arch35/chunk_kda_bwd_finalize_vector.h"
#else
#include "../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_finalize.h"
#if !defined(TILING_KEY_VAR) || TILING_KEY_VAR == 1
#include "arch35/chunk_kda_bwd_finalize_cube.h"
#include "arch22/chunk_kda_bwd_finalize_pre_vector.h"
#define KDA_FINALIZE_A2_FUSED_PRE_COMPILED 1
#endif
#include "arch22/chunk_kda_bwd_finalize_post.h"
#include "arch22/chunk_kda_bwd_finalize_resident_cube.h"
#include "arch22/chunk_kda_bwd_finalize_resident_vector.h"
#endif

namespace KDA {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
__aicore__ inline void ChunkKdaBwdFinalizeImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR akk,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
    GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
    GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
    const ChunkKdaBwdFinalizeTilingData *tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdFinalizeCubeStage10 cube;
        cube.Init(v, akk, vNew, h, dh, dvScan, cuSeqlens, chunkIndices,
                  workspace, tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdFinalizeVectorStage12 vector;
        vector.Init(q, k, v, gk, beta, h, dh, dAqk, dqRaw, qRstd, dq, dv,
                    kRstd, dk, dBeta,
                    rawG, aLog, dtBias, dG,
                    cuSeqlens, chunkIndices,
                    workspace, tiling, &pipe);
        vector.Process();
    }
}
#endif

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
constexpr bool KDA_FINALIZE_A2_PERF_RUN_PRE = true;
constexpr bool KDA_FINALIZE_A2_PERF_RUN_INTRA = true;
constexpr bool KDA_FINALIZE_A2_PERF_RUN_GATE = true;

template <typename BetaT, bool VARLEN_TND>
__aicore__ inline void ChunkKdaBwdFinalizeA2Impl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR akk,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR dBeta, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
    GM_ADDR dALog, GM_ADDR dDtBias, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR workspace, const ChunkKdaBwdFinalizeTilingData *tiling)
{
    if constexpr (KDA_FINALIZE_A2_RESIDENT_BRINGUP != 0U) {
        bool active = false;
        if ASCEND_IS_AIC {
            active = AscendC::GetBlockIdx() <
                static_cast<uint32_t>(tiling->a2.usedCoreNum);
        }
        if ASCEND_IS_AIV {
            active = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() <
                static_cast<uint32_t>(tiling->a2.usedCoreNum);
        }

        if ASCEND_IS_AIC {
            ChunkKdaBwdFinalizeA2ResidentCubeStage1a<> cube;
            cube.Init(v, akk, vNew, h, dh, dvScan, cuSeqlens, chunkIndices,
                      workspace, tiling);
            cube.Process();
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            ChunkKdaBwdFinalizeA2ResidentVectorStage0<
                BetaT, DTYPE_RAW_G> vector;
            vector.Init(q, k, v, gk, beta, h, dh, dAqk, dqRaw,
                        dq, dk, dv, dBeta, dG,
                        rawG, aLog, dtBias,
                        cuSeqlens, chunkIndices, workspace, tiling, &pipe);
            vector.Process();
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
#if !KDA_FINALIZE_A2_DIAG_SKIP_GATE
            if (active) {
                constexpr bool kRawGateFp32 =
                    AscendC::IsSameType<DTYPE_RAW_G, float>::value;
                const bool fusedGate = KDA_FINALIZE_A2_RESIDENT_FUSED_GATE &&
                    kRawGateFp32 &&
                    tiling->safeGate != 0U &&
                    tiling->a2.useGateInKernel != 0 &&
                    tiling->a2.rawGateIsBf16 == 0 &&
                    tiling->a2.isVarLen == 0 &&
                    tiling->a2.seqlen % KDA_FINALIZE_CHUNK == 0 &&
                    tiling->NV <= 32;
                AscendC::TPipe pipe;
                if (fusedGate) {
                    ChunkKdaBwdFinalizeA2GatePartialReduce reduce;
                    reduce.Init(
                        workspace, dALog, dDtBias, tiling, &pipe);
                    reduce.Process();
                } else {
                    ChunkKdaBwdCGateProcess<true, DTYPE_RAW_G> process(
                        dG, rawG, aLog, dtBias, dALog, dDtBias,
                        cuSeqlens, chunkIndices);
                    process.Init(tiling->a2, &pipe);
                    process.Process();
                }
            }
#endif
        }
        AscendC::SyncAll<false>();
        return;
    }

    GM_ADDR dqFp32 = workspace + tiling->a2DqFp32Offset;
    GM_ADDR dkFp32 = workspace + tiling->a2DkFp32Offset;
    GM_ADDR dBetaFp32 = workspace + tiling->a2DBetaFp32Offset;
    GM_ADDR dAkkFp32 = workspace + tiling->a2DAkkFp32Offset;
    GM_ADDR dBetaOut = VARLEN_TND ? dBetaFp32 : dBeta;
#if defined(KDA_FINALIZE_A2_FUSED_PRE_COMPILED)
    constexpr bool kCanUseFusedPre =
        !VARLEN_TND && AscendC::IsSameType<BetaT, bfloat16_t>::value;
    const bool useFusedPre = kCanUseFusedPre &&
        tiling->T % KDA_FINALIZE_CHUNK == 0 &&
        tiling->K == KDA_FINALIZE_DIM && tiling->V == KDA_FINALIZE_DIM;
#endif
    bool active = false;
    if ASCEND_IS_AIC {
        active = AscendC::GetBlockIdx() <
            static_cast<uint32_t>(tiling->a2.usedCoreNum);
    }
    if ASCEND_IS_AIV {
        active = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum() <
            static_cast<uint32_t>(tiling->a2.usedCoreNum);
    }
#if defined(KDA_FINALIZE_A2_FUSED_PRE_COMPILED)
    if constexpr (KDA_FINALIZE_A2_PERF_RUN_PRE) {
      if (useFusedPre) {
        if ASCEND_IS_AIC {
            if (active) {
                ChunkKdaBwdFinalizeCubeStage10 cube;
                cube.Init(v, akk, vNew, h, dh, dvScan,
                          cuSeqlens, chunkIndices, workspace, tiling);
                cube.Process();
            }
        }
        if ASCEND_IS_AIV {
            if (active) {
                AscendC::TPipe pipe;
                ChunkKdaBwdFinalizeA2PreVector<BetaT> vector;
                vector.Init(q, k, v, gk, beta, h, dh, dqRaw,
                            dqFp32, dkFp32, dv, dBetaFp32, dG, dAkkFp32,
                            cuSeqlens, chunkIndices, workspace, tiling, &pipe);
                vector.Process();
            }
        }
      } else
#endif
      {
        if ASCEND_IS_AIC {
            if (active) {
                ChunkKdaBwdCCubeProcess<bfloat16_t, 128, true> process(
                    v, vNew, akk, h, dh, dvScan, dqFp32, dkFp32, dG,
                    dAkkFp32, cuSeqlens, chunkIndices, workspace);
                process.Init(tiling->a2);
                process.Process();
            }
        }
        if ASCEND_IS_AIV {
            if (active) {
                AscendC::TPipe pipe;
                ChunkKdaBwdCVectorProcess<bfloat16_t, 128, BetaT> process(
                    q, k, v, gk, beta, h, dh, dqRaw,
                    dqFp32, dkFp32, dv, dBetaFp32, dG, dAkkFp32,
                    cuSeqlens, chunkIndices, workspace);
                process.Init(tiling->a2, &pipe);
                process.Process();
            }
        }
      }
#if defined(KDA_FINALIZE_A2_FUSED_PRE_COMPILED)
    }
#endif

    if constexpr (KDA_FINALIZE_A2_PERF_RUN_INTRA) {
        if ASCEND_IS_AIC {
            if (active) {
                ChunkKdaBwdCIntraCubeProcess process(
                    cuSeqlens, chunkIndices, workspace);
                process.Init(tiling->a2);
                process.Process();
            }
        }
        if ASCEND_IS_AIV {
            if (active) {
                AscendC::TPipe pipe;
                ChunkKdaBwdCIntraVectorProcess<
                    128, 64, true, false, VARLEN_TND,
                bfloat16_t, BetaT, DTYPE_RAW_G, bfloat16_t,
                typename std::conditional<VARLEN_TND, float, BetaT>::type,
                false> process(
                        q, k, gk, beta, dAqk, dAkkFp32,
                        dqRaw, dqFp32, dkFp32, dBetaFp32, dG,
                        dq, dk, dBetaOut, dG,
                        cuSeqlens, chunkIndices, rawG, aLog, dtBias,
                        dALog, dDtBias, workspace);
                process.Init(tiling->a2, &pipe);
                process.Process();
            }
        }
    }

    AscendC::SyncAll<false>();
    if constexpr (KDA_FINALIZE_A2_PERF_RUN_GATE) {
        if ASCEND_IS_AIV {
            if (active) {
                AscendC::TPipe pipe;
                ChunkKdaBwdCGateProcess<true, DTYPE_RAW_G> process(
                    dG, rawG, aLog, dtBias, dALog, dDtBias,
                    cuSeqlens, chunkIndices);
                process.Init(tiling->a2, &pipe);
                process.Process();
            }
            if constexpr (VARLEN_TND) {
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                AscendC::TPipe pipe;
                ChunkKdaBwdFinalizeA2Post<BetaT> post;
                post.Init(
                    dqFp32, dkFp32, dBetaFp32,
                    dq, dk, dBeta, tiling, &pipe);
                post.ProcessBeta();
            }
        }
    }

    // Drain the final Gate/Post stores before this fused invocation returns.
    AscendC::SyncAll<false>();
}
#endif

} // namespace KDA

extern "C" __global__ __aicore__ void chunk_kda_bwd_finalize(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
    GM_ADDR rawG, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dAqk, GM_ADDR dqRaw,
    GM_ADDR qRstd, GM_ADDR kRstd,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dBeta,
    GM_ADDR dG, GM_ADDR dALog, GM_ADDR dDtBias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdFinalizeTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2) ||
        TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        GM_ADDR userWorkspace = workspace;
        KDA::ChunkKdaBwdFinalizeImpl(
            q, k, v, gk, beta, akk, vNew, h, dh, dvScan,
            dAqk, dqRaw, qRstd, dq, dv, kRstd, dk, dBeta,
            rawG, aLog, dtBias, dG,
            cuSeqlens, chunkIndices, userWorkspace, &tilingData);
        // Only AIVs produce the partials. AIV-only SyncAll uses flag 14,
        // which no earlier AIC->AIV handoff uses. MIX SyncAll uses 12/13
        // and would consume the final, unspent ZB_FREE credits instead.
        AscendC::SyncAll<true>();
        if ASCEND_IS_AIV {
            KDA::FinalizeStage12(userWorkspace, dALog, dDtBias, tilingData);
        }
#else
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        if (tilingData.isVariable != 0U) {
            KDA::ChunkKdaBwdFinalizeA2Impl<DTYPE_BETA, true>(
                q, k, v, gk, beta, akk, vNew, h, dh, dvScan,
                dAqk, dqRaw, dq, dk, dv, dBeta,
                rawG, aLog, dtBias, dG, dALog, dDtBias,
                cuSeqlens, chunkIndices, userWorkspace, &tilingData);
        } else {
            KDA::ChunkKdaBwdFinalizeA2Impl<DTYPE_BETA, false>(
                q, k, v, gk, beta, akk, vNew, h, dh, dvScan,
                dAqk, dqRaw, dq, dk, dv, dBeta,
                rawG, aLog, dtBias, dG, dALog, dDtBias,
                cuSeqlens, chunkIndices, userWorkspace, &tilingData);
        }
#endif
    }
}
