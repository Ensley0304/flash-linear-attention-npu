#!/usr/bin/env bash
set -euo pipefail

export CASE_HEADS=96
export CASE_SEQLEN=18432
export CASE_BETA_DTYPE=bf16

exec python validation_artifacts/kda_bwd_wy_timing.py
