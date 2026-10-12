#!/bin/bash
set -e
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
cd /data/wys/kda-finalize-a2-stage9-20261012
for i in $(seq 1 240); do
    [ -e stage12/r4-build.status ] && break
    sleep 5
done
test -e stage12/r4-build.status
python -u stage12/run.py r4 smoke > stage12/r4-smoke-run.log 2>&1
python -u stage12/run.py r4 profile > stage12/r4-profile-run.log 2>&1
python -u stage12/run.py r4 check > stage12/r4-check-run.log 2>&1
python -u stage12/model.py r4 > stage12/r4-model-capture.log 2>&1
python -u stage12/model.py r4 --perf > stage12/r4-model-perf.log 2>&1
python -u stage12/run.py baseline profile --baseline > stage12/baseline-profile-run.log 2>&1
echo PASS > stage12/final.status
