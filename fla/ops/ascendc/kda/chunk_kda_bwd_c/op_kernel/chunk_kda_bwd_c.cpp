#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_c_struct.h"
#include "chunk_kda_bwd_c_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_c_cube.h"
#include "arch35/chunk_kda_bwd_c_vector.h"
#include "arch35/chunk_kda_bwd_c_intra_cube.h"
#include "arch35/chunk_kda_bwd_c_intra_vector.h"
#include "arch35/chunk_kda_bwd_c_gate.h"
#else
#include "chunk_kda_bwd_c_cube.h"
#include "chunk_kda_bwd_c_vector.h"
#include "chunk_kda_bwd_c_intra_cube.h"
#include "chunk_kda_bwd_c_intra_vector.h"
#include "chunk_kda_bwd_c_gate.h"
#endif

using namespace AscendC;

namespace KDA {

template <typename DataT, uint32_t V_DIM, typename BetaT,
          bool SAFE_GATE, bool VARLEN_TND>
__aicore__ inline void ChunkKdaBwdCImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR a, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dqRaw, GM_ADDR dAqk, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR db, GM_ADDR dg,
    GM_ADDR dAkk, GM_ADDR dA, GM_ADDR dBias, GM_ADDR workspace,
    const ChunkKdaBwdCTilingData *tiling)
{
    if ASCEND_IS_AIC {
        {
            ChunkKdaBwdCCubeProcess<DataT, V_DIM> process(
                v, vNew, a, h, dh, dvScan, dq, dk, dg, dAkk,
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
        {
            ChunkKdaBwdCIntraCubeProcess process(
                cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling);
            process.Process();
        }
    }
    if ASCEND_IS_AIV {
        {
            AscendC::TPipe pipe;
            ChunkKdaBwdCVectorProcess<DataT, V_DIM, BetaT> process(
                q, k, v, gk, beta, h, dh, dqRaw,
                dq, dk, dv, db, dg, dAkk, cuSeqlens, chunkIndices,
                workspace);
            process.Init(*tiling, &pipe);
            process.Process();
        }
        {
            AscendC::TPipe pipe;
            // Kernel C's public varlen ABI is head-major [H,T,D].  The
            // mature standalone Intra template names its alternate layout
            // VARLEN_TND; enabling that branch would apply token-major
            // H*D row strides to these head-major tensors.  Chunk lookup is
            // already handled by GetCIntraTask, so keep the dense row-stride
            // specialization for both dense and varlen Kernel-C layouts.
            ChunkKdaBwdCIntraVectorProcess<
                128, 64, SAFE_GATE, false, VARLEN_TND, DataT, BetaT> process(
                    q, k, gk, beta, dAqk, dAkk,
                    dqRaw, dq, dk, db, dg, dq, dk, db, dg,
                    cuSeqlens, chunkIndices, workspace);
            process.Init(*tiling, &pipe);
            process.Process();
        }
        {
            AscendC::TPipe pipe;
            ChunkKdaBwdCGateProcess<SAFE_GATE, DTYPE_RAW_G> process(
                dg, rawG, aLog, dtBias, dA, dBias,
                cuSeqlens, chunkIndices);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Dense/full-chunk accumulated-gate mode is completed in
            // Intra-Post while dg is resident.  Raw-gate and varlen/tail
            // paths retain the standalone phase.
            if (tiling->useGateInKernel != 0 ||
                tiling->isVarLen != 0 ||
                tiling->seqlen % 64U != 0) {
                process.Init(*tiling, &pipe);
                process.Process();
            }
#else
            process.Init(*tiling, &pipe);
            process.Process();
#endif
        }
    }
}

} // namespace KDA

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_kda_bwd_c(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR v_new, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR Akk, GM_ADDR h, GM_ADDR dh, GM_ADDR dv_scan,
    GM_ADDR dq_raw, GM_ADDR dAqk, GM_ADDR raw_g, GM_ADDR a_log,
    GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR db, GM_ADDR dg, GM_ADDR dAkk, GM_ADDR dA,
    GM_ADDR dbias, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdCTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdCTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 128, DTYPE_BETA, true, false>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(2)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 128, DTYPE_BETA, true, true>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(3)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 128, DTYPE_BETA, false, false>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(4)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 128, DTYPE_BETA, false, true>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(5)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 256, DTYPE_BETA, true, false>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(6)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 256, DTYPE_BETA, true, true>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(7)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 256, DTYPE_BETA, false, false>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(8)) {
        KDA::ChunkKdaBwdCImpl<DTYPE_Q, 256, DTYPE_BETA, false, true>(
            q, k, v, v_new, gk, beta, Akk, h, dh, dv_scan,
            dq_raw, dAqk, cu_seqlens, chunk_indices,
            raw_g, a_log, dt_bias, dq, dk, dv, db, dg, dAkk,
            dA, dbias, userWorkspace, &tilingData);
    }
}
#endif
