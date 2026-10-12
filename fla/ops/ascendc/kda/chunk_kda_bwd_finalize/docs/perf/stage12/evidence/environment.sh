#!/bin/bash
set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
hostname
whoami
pwd
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
python -c 'import sys,torch,torch_npu; print(sys.executable); print(torch.__version__,torch_npu.__version__)'
clang-format --version
npu-smi info -l
printf 'CANN=%s
' "$ASCEND_HOME_PATH"
