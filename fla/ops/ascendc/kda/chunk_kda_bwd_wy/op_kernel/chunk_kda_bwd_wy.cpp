#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_wy_struct.h"
#include "chunk_kda_bwd_wy_common.h"
#include "chunk_kda_bwd_wy_cube.h"
#include "chunk_kda_bwd_wy_vector.h"

using namespace AscendC;

namespace KDA {

template <typename BetaT>
__aicore__ inline void ChunkKdaBwdWyImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR a, GM_ADDR h, GM_ADDR dO, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR db,
    GM_ADDR dg, GM_ADDR dAkk, GM_ADDR workspace,
    const ChunkKdaBwdWyTilingData *tiling)
{
    if ASCEND_IS_AIC {
        ChunkKdaBwdWyCubeProcess process(
            v, vNew, a, h, dO, dh, dvScan, dq, dk, dg, dAkk, workspace);
        process.Init(*tiling);
        process.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdWyVectorProcess<BetaT> process(
            q, k, v, gk, beta, h, dh, dvScan, dq, dk, dv, db, dg, dAkk,
            workspace);
        process.Init(*tiling, &pipe);
        process.Process();
    }
}

} // namespace KDA

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_kda_bwd_wy(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR v_new, GM_ADDR gk,
    GM_ADDR beta, GM_ADDR Akk, GM_ADDR h, GM_ADDR d_o, GM_ADDR dh,
    GM_ADDR dv_scan, GM_ADDR dq_base, GM_ADDR dk_base, GM_ADDR dv,
    GM_ADDR db_base, GM_ADDR dg_base, GM_ADDR dAkk, GM_ADDR workspace,
    GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdWyTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdWyTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        KDA::ChunkKdaBwdWyImpl<DTYPE_BETA>(
            q, k, v, v_new, gk, beta, Akk, h, d_o, dh, dv_scan,
            dq_base, dk_base, dv, db_base, dg_base, dAkk, userWorkspace,
            &tilingData);
    }
}
#endif
