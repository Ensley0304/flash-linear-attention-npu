set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
python -u bound-opt/run.py "$@"
