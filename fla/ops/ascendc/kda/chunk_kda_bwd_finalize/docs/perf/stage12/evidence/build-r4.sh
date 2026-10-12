set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
export NPU_ARCH=ascend910b FAST_KERNEL_OP_NAME=kda_finalize_stage01
mkdir -p stage12
for file in fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/*.h examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp; do
    clang-format -i "$file"
done
g++ -std=c++17 -Wall -Wextra -Werror -I. examples/fast_kernel_launch_example/tests/kda_finalize_stage01/test_tiling.cpp -o stage12/test-tiling
stage12/test-tiling > stage12/host.log
cd examples/fast_kernel_launch_example
KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build > ../../stage12/r4-build-capture.log 2>&1
cp build/_C.abi3.so ../../stage12/r4-capture.so
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build > ../../stage12/r4-build-perf.log 2>&1
cp build/_C.abi3.so ../../stage12/r4-perf.so
echo PASS > ../../stage12/r4-build.status
