// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include "lib/matmul_intf.h"
#include "owner.h"
#include "tiling.h"
using namespace KDA::Arch22::Fused32;
extern "C" __global__ __aicore__ void kda_fused32(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR g,
    GM_ADDR beta,
    GM_ADDR a_log,
    GM_ADDR bias,
    GM_ADDR h0,
    GM_ADDR dht,
    GM_ADDR d_o,
    GM_ADDR g_cache,
    GM_ADDR aqk,
    GM_ADDR akk,
    GM_ADDR w,
    GM_ADDR qg,
    GM_ADDR kg,
    GM_ADDR v_new,
    GM_ADDR h,
    GM_ADDR qhat,
    GM_ADDR khat,
    GM_ADDR qr,
    GM_ADDR kr,
    GM_ADDR beta_eff,
    GM_ADDR sequence_metadata,
    GM_ADDR o,
    GM_ADDR ht,
    GM_ADDR dq,
    GM_ADDR dk,
    GM_ADDR dv,
    GM_ADDR dg,
    GM_ADDR db,
    GM_ADDR dh0,
    GM_ADDR d_a,
    GM_ADDR d_bias,
    GM_ADDR g_out,
    GM_ADDR aqk_out,
    GM_ADDR akk_out,
    GM_ADDR w_out,
    GM_ADDR qg_out,
    GM_ADDR kg_out,
    GM_ADDR v_new_out,
    GM_ADDR h_out,
    GM_ADDR qhat_out,
    GM_ADDR khat_out,
    GM_ADDR qr_out,
    GM_ADDR kr_out,
    GM_ADDR beta_eff_out,
    GM_ADDR u_out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1,KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(Tiling);
    GET_TILING_DATA_WITH_STRUCT(Tiling,t,tiling);
    if(!(t.outputMask & (1U<<0))) o=nullptr;
    if(!(t.outputMask & (1U<<1))) ht=nullptr;
    if(!(t.outputMask & (1U<<2))) dq=nullptr;
    if(!(t.outputMask & (1U<<3))) dk=nullptr;
    if(!(t.outputMask & (1U<<4))) dv=nullptr;
    if(!(t.outputMask & (1U<<5))) dg=nullptr;
    if(!(t.outputMask & (1U<<6))) db=nullptr;
    if(!(t.outputMask & (1U<<7))) dh0=nullptr;
    if(!(t.outputMask & (1U<<8))) d_a=nullptr;
    if(!(t.outputMask & (1U<<9))) d_bias=nullptr;
    if(!(t.outputMask & (1U<<10))) g_out=nullptr;
    if(!(t.outputMask & (1U<<11))) aqk_out=nullptr;
    if(!(t.outputMask & (1U<<12))) akk_out=nullptr;
    if(!(t.outputMask & (1U<<13))) w_out=nullptr;
    if(!(t.outputMask & (1U<<14))) qg_out=nullptr;
    if(!(t.outputMask & (1U<<15))) kg_out=nullptr;
    if(!(t.outputMask & (1U<<16))) v_new_out=nullptr;
    if(!(t.outputMask & (1U<<17))) h_out=nullptr;
    if(!(t.outputMask & (1U<<18))) qhat_out=nullptr;
    if(!(t.outputMask & (1U<<19))) khat_out=nullptr;
    if(!(t.outputMask & (1U<<20))) qr_out=nullptr;
    if(!(t.outputMask & (1U<<21))) kr_out=nullptr;
    if(!(t.outputMask & (1U<<22))) beta_eff_out=nullptr;
    if(!(t.outputMask & (1U<<23))) u_out=nullptr;
    const bool back=t.backward!=0;
    Buffers b{q,k,v,g,beta,a_log,bias,h0,dht,d_o,
        back?g_cache:g_out,back?aqk:aqk_out,back?akk:akk_out,back?w:w_out,
        back?qg:qg_out,back?kg:kg_out,back?v_new:v_new_out,back?h:h_out,
        back?qhat:qhat_out,back?khat:khat_out,back?qr:qr_out,back?kr:kr_out,
        back?beta_eff:beta_eff_out,o,ht,dq,dk,dv,dg,db,dh0,d_a,d_bias,
        sequence_metadata,AscendC::GetUserWorkspace(workspace)};
    b.uOut=u_out;
    if(TILING_KEY_IS(1)) {
        uint32_t core=AscendC::GetBlockIdx();
        if ASCEND_IS_AIV { core/=AscendC::GetSubBlockNum(); }
        GM_ADDR command=b.workspace+uint64_t(core)*t.workspace.bytes+t.workspace.cube;
        if ASCEND_IS_AIC { CubeService(command); }
        if ASCEND_IS_AIV {
            if(AscendC::GetSubBlockIdx()) {
                const uint64_t heads=(t.config.heads-core+t.blocks-1)/t.blocks;
                const uint64_t vt=(t.config.values+127)/128,kt=(t.config.keys+127)/128;
                const uint64_t stateTiles=t.config.stateVFirst?((t.config.values+31)/32)*kt:((t.config.keys+31)/32)*vt;
                CubeObserver(heads*t.config.chunks*(vt*(2+kt)+vt*kt+2*stateTiles)+1);
            }
            else { Owner<DTYPE_V,DTYPE_G,DTYPE_BETA,true> owner;
                   owner.Init(b,t.config,t.workspace,t.blocks);owner.Process<true>(); }
        }
    } else if(TILING_KEY_IS(0)) {
        Owner<DTYPE_V,DTYPE_G,DTYPE_BETA> owner;owner.Init(b,t.config,t.workspace);
        if(back)owner.Process<true>();else owner.Process<false>();
    }
}
