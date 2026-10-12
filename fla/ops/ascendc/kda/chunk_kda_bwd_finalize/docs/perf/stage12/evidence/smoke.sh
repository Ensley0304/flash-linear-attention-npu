set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
tar -xf stage12-tests.tar
python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/check_stage01.py --library stage12/r1-capture.so --stage 11 --device 3 --cores 1 --suite smoke --repeat 2 --tail-golden-dir stage12/golden > stage12/r1-smoke11.log 2>&1
python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/check_stage01.py --library stage12/r1-capture.so --stage 12 --device 3 --cores 1 --suite smoke --repeat 2 --tail-golden-dir stage12/golden > stage12/r1-smoke12.log 2>&1
echo PASS > stage12/r1-smoke.status
