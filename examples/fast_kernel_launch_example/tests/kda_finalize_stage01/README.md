# A2 KDA Finalize：S24 开发入口

私有入口保留 `kda_finalize_stage01` 名称，完整计算使用 `last_stage=12`，主流水和参数归约合为单次launch。未接入公开ACLNN/torch接口；当前A5实现不变。

## 构建

在已有torch/torch_npu环境加载CANN后，从仓库根目录执行：

```bash
cd examples/fast_kernel_launch_example
NPU_ARCH=ascend910b FAST_KERNEL_OP_NAME=kda_finalize_stage01 \
  KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build
export KDA_FINALIZE_LIBRARY="$PWD/build/_C.abi3.so"
```

测试显式加载本次动态库，无需覆盖系统OPP。生产性能路径为capture=0；阶段采数另构建capture=1，两者的局部中间表示不同，不能用采数版性能代表生产版。

## Host tiling 检查

从仓库根目录执行：

```bash
g++ -std=c++17 -Wall -Wextra -Werror -I. \
  examples/fast_kernel_launch_example/tests/kda_finalize_stage01/test_tiling.cpp \
  -o /tmp/kda_finalize_stage01_tiling
/tmp/kda_finalize_stage01_tiling
```

覆盖定长/变长元数据、尾块、空序列、核数和非法输入。每核组workspace为1536KiB；24核组为36864KiB。PR同步修正这两项旧断言，不修改tiling实现。

## 精度与性能

完整七输出使用[ATK双标杆工程](../../../../tests/atk/chunk_kda_bwd_finalize/README.md)。最新S24通过九边界和原20case，完整200case三seed待重跑。K=V=128、chunk=64，支持定长/变长、Q/K norm成对开启、三种beta模式和原生output_layout。

同精度参考采用FP32局部GEMM，高精参考独立FP64；不读取设备workspace。CPU参考与输入种子未改，不能通过放宽阈值接受优化。

完整生产入口每次调用仅统计一个 `Stage01Kernel` 的 msprof `Task Duration(us)`，不再累加独立Stage12 kernel。现有 `bench_stage01.py` 与 `check_stage01.py` 等脚本用于开发期阶段定位；正式完整功能与七输出比较使用ATK工程。设备调用超时为60秒，CPU参考计算另计。

## S24 实现范围

- 两head、两代GM workspace和片上双缓冲。
- Stage5输入驻留；Stage7分带居中指数及整数/小数指数分解。
- Stage8/10局部矩阵乘采用FP32，完整块有效半块使用K32。
- Stage11融合dk、dBeta、gate和L2Norm反向；dG/dBeta固定BSND/TND。
- Stage12在AIV同步后复用UB，按固定chunk顺序归约。

详见[当前设计与验收](../../../../fla/ops/ascendc/kda/chunk_kda_bwd_finalize/docs/design.md)。当前PR基线上的设备重编译、完整200case、A3/A5、公开接口和正式CI均待完成。
