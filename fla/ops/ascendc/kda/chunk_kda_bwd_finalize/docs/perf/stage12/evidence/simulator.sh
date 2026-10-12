#!/bin/bash
set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
for i in $(seq 1 240); do
    [ -e stage12/r4-retry-profile.status ] && break
    sleep 5
done
test -e stage12/r4-retry-profile.status
msprof op simulator --soc-version=Ascend910B3 --kernel-name='*Stage01Kernel*' --launch-count=1 --core-id=0 --timeout=5 --output=stage12/simulator-main-r4 python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/bench_stage01.py --library stage12/r4-perf.so --heads 2 --tokens 192 --cores 1 --stage 11 --warmup 0 --repeat 1 --device 0 > stage12/simulator-main-r4.log 2>&1
msprof op simulator --soc-version=Ascend910B3 --kernel-name='*Stage12Kernel*' --launch-count=1 --core-id=0 --timeout=5 --output=stage12/simulator-reduce-r4 python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/bench_stage01.py --library stage12/r4-perf.so --heads 2 --tokens 2113 --cores 1 --stage 12 --warmup 0 --repeat 1 --device 0 > stage12/simulator-reduce-r4.log 2>&1
echo PASS > stage12/simulator.status
