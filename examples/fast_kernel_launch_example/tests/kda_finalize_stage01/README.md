# KDA Finalize A2/A3：Stage 0/1 开发验证

仅验证用户确认的 Stage 0/1，不代表完整 Finalize 已支持 A2/A3。公开 A5 入口不变。

## 构建和运行

先进入已有的 torch/torch_npu 环境并加载对应 CANN。仓库根目录执行：

```bash
export NPU_ARCH=ascend910b
export FAST_KERNEL_OP_NAME=kda_finalize_stage01
cd examples/fast_kernel_launch_example
python setup.py cmake_build

python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 0 --device 1 --cores 1 --suite tails
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 1 --device 1 --cores 1 --suite tails
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 1 --device 1 --cores 0 --suite tails
```

测试直接加载本次构建的动态库，无需安装 wheel 或覆盖 OPP。`--cores 1` 强制同核复用 workspace；`--cores 0` 使用平台可用 AIC。测试每次进入设备调用后设置 60 秒超时，超时进程以 124 退出。

Host 元数据和大小检查（仓库根目录，无需 NPU）：

```bash
g++ -std=c++17 -Wall -Wextra -Werror -I. \
  examples/fast_kernel_launch_example/tests/kda_finalize_stage01/test_tiling.cpp \
  -o /tmp/kda_finalize_stage01_tiling
/tmp/kda_finalize_stage01_tiling
```

## 范围

- Stage 0：dk_state_raw、dv_b、BF16 dW、zV。
- Stage 1：rH、FP32 exp2_gk、BF16 kE、gk_last。
- 定长 B=1/2；H=3/96；尾长 1/16/31/32/33/63/64；变长含空序列；默认每例重复两次。
- 逐项对照 FP64/FP32 CPU 标杆，检查 NaN/Inf、capture 哨兵及真实 workspace 最后一代数据。
- trace 记录实际 work/core/owner/head/chunk、槽位使用次数、block 数、子核数、GM 偏移及握手进度；奇数 head 的空 owner 仍参与同步。

开发入口返回 `(capture, trace, workspace)`。capture 每个 `(head,chunk)` 占 160 KiB，具体布局见 kernel common 头；额外采数写回不计入生产 workspace，也不用于性能评估。dW 的 capture 来自同一 L0C 的 BF16 导出，后续 Stage 2 还需验证其 L1 消费链路。

原始输入保持 BNSD/NTD，h/dh 为 chunk-major。变长测试入口接收 CPU `cu_seqlens` 列表并生成完整 chunk 元数据；只接受连续 BF16 主数据、FP32 gk、K=V=128。

## 双缓冲回归与性能

`--suite model --repeat 1` 完整检查 B=1/H=96/T=8192 的定长及变长输入。变长长度为 `[0,1,31,64,65,4096,3935,0]`；全部中间结果对照原有 CPU 标杆，容差不变。

性能版使用编译开关关闭 capture/trace，并缩小对应返回占位张量；计算和同步路径保持相同。仍保留缺失 Stage 2 时的测试收尾握手，因此不代表完整 Finalize 性能。以下命令在 `examples/fast_kernel_launch_example` 下运行：

```bash
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build
msprof --output=./profile_stage01 --ai-core=on --aic-metrics=PipeUtilization \
  --task-time=on --ascendcl=on \
  python tests/kda_finalize_stage01/bench_stage01.py --library build/_C.abi3.so
```

默认定长 H=96/T=8192，每阶段预热 20 次、采样 10 次；加 `--varlen` 测同规模变长输入。只统计 `Stage01Kernel` 最后 10 次记录；初始化、Host wall time 和其它 kernel 不计入。精度版恢复时执行 `KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build`。

性能库也可用 `check_stage01.py --workspace-only --stage 1 --cores 0 --suite model --repeat 1 --library build/_C.abi3.so` 检查各核真实 workspace 最后一代结果，避免只验证带 capture 的同步路径。

小 case 模拟器入口支持 `--heads 2 --tokens 64 --cores 1 --stage 1 --warmup 0 --repeat 1 --device 0`。模拟器使用 CPU 初始化，不读取输出；需要完整的一个 AIC 和两个 AIV 时间线，不能以模拟器耗时替代硬件性能。
