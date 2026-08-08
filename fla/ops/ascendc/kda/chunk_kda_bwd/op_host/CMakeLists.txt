set(chunk_kda_bwd_depends
    "fla/ops/ascendc/kda/chunk_kda_bwd_dav;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu;fla/ops/ascendc/kda/chunk_kda_bwd_wy;fla/ops/ascendc/kda/chunk_kda_bwd_intra;fla/ops/ascendc/kda/chunk_kda_bwd_gate_post"
    CACHE STRING "Kernel source dependencies for chunk_kda_bwd" FORCE)
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_kda_bwd_def.cpp)
endif()

add_modules_sources(OPTYPE chunk_kda_bwd ACLNNTYPE aclnn_exclude)
