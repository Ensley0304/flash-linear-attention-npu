#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${FINALIZE_WHEEL_SITE:?set to isolated, validated finalize wheel site}"
export PYTHONPATH="$FINALIZE_WHEEL_SITE:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export ASCEND_CUSTOM_OPP_PATH="$FINALIZE_WHEEL_SITE/fla_npu/opp/vendors/fla_npu_transformer"
export ATK_CUSTOM_OPP_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KDA_FINALIZE_CPU_THREADS=1
cases=${1:-smoke_case.json}
out=${2:-"$PWD/results"}
mkdir -p "$out"
python generate_finalize.py
python -m atk node --backend pyaclnn --devices "${FINALIZE_DEVICE:-0}" --run_modes ascend_use_deterministic_algorithms -o "$out" \
    node --backend cpu --run_modes ascend_use_deterministic_algorithms -o "$out" \
    task -c "$cases" --task accuracy -p "$PWD" -cp --single_process --timeout 14400
