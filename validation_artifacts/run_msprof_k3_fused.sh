#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if [[ -n "${KDA_BWD_SITEPKGS:-}" ]]; then
    export PYTHONPATH="${KDA_BWD_SITEPKGS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export CASE_HEADS=2
export CASE_SEQLEN=64

output=validation_artifacts/msprof_k3_fused_h2_t64_20260806
rm -rf "${output}"
msprof \
    --application="python validation_artifacts/k3_direct_canary.py" \
    --output="${output}" \
    --aic-metrics=PipeUtilization \
    --task-time=on
