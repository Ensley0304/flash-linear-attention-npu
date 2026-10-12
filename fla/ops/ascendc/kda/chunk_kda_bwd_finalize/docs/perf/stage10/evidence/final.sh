set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
while [ ! -f stage10/r1-all.status ]; do sleep 10; done
python -u stage10/model.py --library stage10/r1-capture.so --tag model-capture --device 3 > stage10/model-capture.log 2>&1
python -u stage10/model.py --library stage10/r1-perf.so --tag model-perf --device 3 --perf --gold model-capture.json > stage10/model-perf.log 2>&1
msprof op simulator --soc-version=Ascend910B3 --kernel-name='*Stage01Kernel*' --launch-count=1 --core-id=0 --timeout=5 --output=stage10/simulator-r1 python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/bench_stage01.py --library stage10/r1-perf.so --heads 2 --tokens 192 --cores 1 --stage 10 --warmup 0 --repeat 1 --device 0 > stage10/simulator-r1.log 2>&1
echo PASS > stage10/final.status
