#!/usr/bin/env bash
set -euo pipefail

export KDA_BWD_KERNEL=K4
export KDA_BWD_H=96
export KDA_BWD_T=18432

exec python validation_artifacts/kda_bwd_kernel_timing.py
