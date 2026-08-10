#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_a_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_a_cube.h"
#include "arch35/chunk_kda_bwd_a_vector.h"
#else
#include "chunk_kda_bwd_a_cube.h"
#include "chunk_kda_bwd_a_vector.h"
#endif

namespace KDA {

template <typename T, uint32_t V_DIM>
__aicore__ inline void RunChunkKdaBwdA(
    GM_ADDR aqk, GM_ADDR qg, GM_ADDR vNew, GM_ADDR h, GM_ADDR dO,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dv0, GM_ADDR q0, GM_ADDR dqRaw, GM_ADDR dAqk,
    GM_ADDR workspace, const ChunkKdaBwdATilingData &tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdACube<T, V_DIM> cube(
            aqk, qg, vNew, h, dO, cuSeqlens, chunkIndices,
            dv0, dqRaw, dAqk, workspace);
        cube.Init(tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdAVector<T, V_DIM> vector(
            q0, dAqk, cuSeqlens, chunkIndices, workspace);
        vector.Init(tiling, &pipe);
        vector.Process();
    }
}

} // namespace KDA

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_kda_bwd_a(
    GM_ADDR aqk, GM_ADDR qg, GM_ADDR v_new, GM_ADDR h, GM_ADDR d_o,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dv0, GM_ADDR q0, GM_ADDR dq_raw, GM_ADDR d_aqk,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdATilingData);
    GET_TILING_DATA_WITH_STRUCT(
        KDA::ChunkKdaBwdATilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);

    if (TILING_KEY_IS(1)) {
        KDA::RunChunkKdaBwdA<half, 128>(
            aqk, qg, v_new, h, d_o, cu_seqlens, chunk_indices,
            dv0, q0, dq_raw, d_aqk, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(2)) {
        KDA::RunChunkKdaBwdA<half, 256>(
            aqk, qg, v_new, h, d_o, cu_seqlens, chunk_indices,
            dv0, q0, dq_raw, d_aqk, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(3)) {
        KDA::RunChunkKdaBwdA<bfloat16_t, 128>(
            aqk, qg, v_new, h, d_o, cu_seqlens, chunk_indices,
            dv0, q0, dq_raw, d_aqk, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(4)) {
        KDA::RunChunkKdaBwdA<bfloat16_t, 256>(
            aqk, qg, v_new, h, d_o, cu_seqlens, chunk_indices,
            dv0, q0, dq_raw, d_aqk, userWorkspace, tilingData);
    }
}
#endif
