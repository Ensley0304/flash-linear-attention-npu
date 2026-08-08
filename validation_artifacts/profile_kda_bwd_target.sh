#!/usr/bin/env bash
set -euo pipefail

export KDA_BWD_H=96
export KDA_BWD_T=18432
export KDA_BWD_LAYOUT_ONLY=BNSD
export KDA_BWD_LAUNCH_ONLY=1

exec python validation_artifacts/kda_bwd_l2_smoke.py
