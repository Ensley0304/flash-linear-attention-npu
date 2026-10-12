set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
while [ ! -f stage10/r1-build.status ]; do sleep 10; done
python -u stage10/run.py r1 smoke
python -u stage10/run.py r1 check
python -u stage10/run.py r1 profile
python -u stage10/run.py baseline profile --baseline --library-tag r11
echo PASS > stage10/r1-all.status
