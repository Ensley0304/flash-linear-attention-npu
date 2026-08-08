#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if [[ -n "${KDA_BWD_SITEPKGS:-}" ]]; then
    export PYTHONPATH="${KDA_BWD_SITEPKGS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export KDA_BWD_LAYOUT_ONLY=BNSD
export KDA_BWD_H=2
export KDA_BWD_T=64

output=validation_artifacts/msprof_kda_bwd_fused_h2_t64_20260806
rm -rf "${output}"
msprof \
    --application="python validation_artifacts/kda_bwd_l2_smoke.py" \
    --output="${output}" \
    --aic-metrics=PipeUtilization \
    --task-time=on
