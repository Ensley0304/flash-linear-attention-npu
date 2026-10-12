cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
tag=$1
msprof op simulator --soc-version=Ascend910B3 --kernel-name='*Stage01Kernel*' --launch-count=1 --core-id=0 --timeout=5 --output=bound-opt/simulator-$tag python -u examples/fast_kernel_launch_example/tests/kda_finalize_stage01/bench_stage01.py --library bound-opt/$tag-perf.so --heads 2 --tokens 192 --cores 1 --stage 9 --warmup 0 --repeat 1 --device 0 > bound-opt/simulator-$tag.log 2>&1
echo $? > bound-opt/simulator-$tag.exitcode
