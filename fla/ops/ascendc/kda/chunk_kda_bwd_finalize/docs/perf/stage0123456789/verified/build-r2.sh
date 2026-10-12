set -e
cd /data/Ensley0304/kda-finalize-a2-stage9-20261012
source /data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh
export PATH=/data/miniconda3/envs/wys_gdn/bin:$PATH
export NPU_ARCH=ascend910b FAST_KERNEL_OP_NAME=kda_finalize_stage01
clang-format -i fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/*.h examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp
cd examples/fast_kernel_launch_example
KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build > ../../stage9/build-r2-capture.log 2>&1
cp build/_C.abi3.so ../../stage9/r2-capture.so
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build > ../../stage9/build-r2-perf.log 2>&1
cp build/_C.abi3.so ../../stage9/r2-perf.so
echo PASS > ../../stage9/build-r2.status
