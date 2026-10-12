#!/bin/bash
set -e
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
cd /data/wys/kda-finalize-a2-stage9-20261012
python -u stage12/matrix.py r4 check --device 2 > stage12/r4-check-batch-capture.log 2>&1
python -u stage12/matrix.py r4 check --perf --device 2 > stage12/r4-check-batch-perf.log 2>&1
python -u stage12/model.py r4 --device 2 > stage12/r4-model-capture.log 2>&1
python -u stage12/model.py r4 --perf --device 2 > stage12/r4-model-perf.log 2>&1
python -u stage12/run.py d2-r3 profile --library-tag r3 --device 2 > stage12/d2-r3-profile-run.log 2>&1
python -u stage12/run.py d2-r4 profile --library-tag r4 --device 2 > stage12/d2-r4-profile-run.log 2>&1
python -u stage12/run.py d2-baseline profile --baseline --device 2 > stage12/d2-baseline-profile-run.log 2>&1
echo PASS > stage12/device2.status
