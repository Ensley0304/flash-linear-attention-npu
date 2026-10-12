#!/bin/bash
set -e
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
cd /data/wys/kda-finalize-a2-stage9-20261012
while [ ! -e stage12/r3-build.status ]; do sleep 5; done
python -u stage12/reduce-check.py stage12/r3-capture.so > stage12/r3-reduce-check.log 2>&1
python -u stage12/run.py r3 profile > stage12/r3-profile-run.log 2>&1
