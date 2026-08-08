#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "chunk_kda_bwd_dav_struct.h"
#include "chunk_kda_bwd_dav_cube.h"
#include "chunk_kda_bwd_dav_vector.h"

#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void chunk_kda_bwd_dav(
    GM_ADDR aqk, GM_ADDR v_new, GM_ADDR d_o, GM_ADDR d_aqk, GM_ADDR dv,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    AscendC::AscendCUtils::SetOverflow(1);
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdDAvTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdDAvTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIC_ONLY);
        KDA::ChunkKdaBwdDAvCubeProcess cube(aqk, v_new, d_o, d_aqk, dv);
        cube.Init(tilingData);
        cube.Process();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        KDA::ChunkKdaBwdDAvVectorProcess vector(d_aqk);
        vector.Init(tilingData, &pipe);
        vector.Process();
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
            KDA::ChunkKdaBwdDAvCubeProcess cube(
                aqk, v_new, d_o, d_aqk, dv);
            cube.Init(tilingData);
            cube.Process(true);
        }
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            KDA::ChunkKdaBwdDAvVectorProcess vector(d_aqk);
            vector.Init(tilingData, &pipe);
            vector.Process();
        }
    }
}
#endif
