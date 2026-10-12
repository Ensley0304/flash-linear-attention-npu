set -e
cd /data/wys/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
export NPU_ARCH=ascend910b FAST_KERNEL_OP_NAME=kda_finalize_stage01
tag=$1
clang-format -i fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize_vector.h
clang-format -i examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp
cp fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize_vector.h bound-opt/$tag-vector.h
cp examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp bound-opt/$tag-entry.cpp
cd examples/fast_kernel_launch_example
KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build > ../../bound-opt/$tag-build-capture.log 2>&1
cp build/_C.abi3.so ../../bound-opt/$tag-capture.so
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build > ../../bound-opt/$tag-build-perf.log 2>&1
cp build/_C.abi3.so ../../bound-opt/$tag-perf.so
echo PASS > ../../bound-opt/$tag-build.status
