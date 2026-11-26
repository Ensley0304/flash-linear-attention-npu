add_op_to_compiled_list()
set(kda_bwd_gate_normalize_depends "fla/ops/ascendc/kda/chunk_kda_fwd_prepare"
    CACHE STRING "Private C64 gate stage implementation" FORCE)
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE kda_bwd_gate_normalize_def.cpp)
endif()
add_modules_sources(OPTYPE kda_bwd_gate_normalize ACLNNTYPE aclnn_exclude)
add_ops_compile_options(OP_NAME KdaBwdGateNormalize OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
