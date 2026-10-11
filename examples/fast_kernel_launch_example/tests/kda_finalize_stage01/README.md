# KDA Finalize A2/A3：Stage 0～2 开发验证

仅验证用户确认的 Stage 0～2，不代表完整 Finalize 已支持 A2/A3。公开 A5 入口不变，开发入口保留原名。

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
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 2 --device 1 --cores 0 --suite model --repeat 1
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
- Stage 2：dKgb_raw=akkᵀ@dW、zW=dW@kEᵀ，BF16 输入、FP32 累加和输出。
- 定长 B=1/2；H=3/96；尾长 1/16/31/32/33/63/64；变长含空序列；默认每例重复两次。
- 逐项对照 FP64/FP32 CPU 标杆，检查 NaN/Inf、capture 哨兵及真实 workspace 最后一代数据。
- trace 记录实际 work/core/owner/head/chunk、槽位使用次数、block 数、子核数、GM 偏移及握手进度；奇数 head 的空 owner 仍参与同步。

开发入口返回 `(capture, trace, workspace)`。capture 每个 `(head,chunk)` 占 224 KiB，新增结果位于 160/192 KiB；真实 workspace 的对应偏移为 80/112 KiB。额外采数不用于性能评估。Stage 2 实际消费驻留 L1 的 dW，验证 L0C→L1 链路。

Stage 0/1 的容差不变。Stage 2 同时检查两层：以设备捕获的 BF16 dW/kE 做 FP64 矩阵乘，使用 `rtol=3e-4, atol=2e-6`；再与独立 CPU 全路径结果比较，传播既定 BF16 门槛 `E(x)=2e-5+8e-3|x|`：

```text
dKgb 误差预算 = |akk|ᵀ @ E(dW)
zW 误差预算   = E(dW) @ |kE|ᵀ + |dW| @ E(kE)ᵀ + E(dW) @ E(kE)ᵀ
```

预算外仍只允许上述 FP32 累加容差，所有结果必须有限。原因是 FP32 累加的微小差异可能使 BF16 舍入跨过相邻值，不能将该传播误差误判为后续 MMAD 错误。`--gate-mode flat` 将 gk 分布设为 [-2,0]，使所有 kE 行保持显著，补充默认累计 gate 场景。

原始输入保持 BNSD/NTD，h/dh 为 chunk-major。变长测试入口接收 CPU `cu_seqlens` 列表并生成完整 chunk 元数据；只接受连续 BF16 主数据、FP32 gk、K=V=128。

## 双缓冲回归与性能

`--suite model --repeat 1` 完整检查 B=1/H=96/T=8192 的定长及变长输入。变长长度为 `[0,1,31,64,65,4096,3935,0]`；全部中间结果对照 CPU 标杆，Stage 0/1 容差不变，Stage 2 使用上述两层门禁。

性能版使用编译开关关闭 capture/trace，并缩小对应返回占位张量；计算和同步路径保持相同。以下命令测 Stage 0～2 开发入口，不代表完整 Finalize 性能，在 `examples/fast_kernel_launch_example` 下运行：

```bash
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build
msprof --output=./profile_stage01 --ai-core=on --aic-metrics=PipeUtilization \
  --task-time=on --ascendcl=on \
  python tests/kda_finalize_stage01/bench_stage01.py --library build/_C.abi3.so --stage 2
```

默认定长 H=96/T=8192，每阶段预热 20 次、采样 10 次；加 `--varlen` 测同规模变长输入。只统计 `Stage01Kernel` 最后 10 次记录；初始化、Host wall time 和其它 kernel 不计入。精度版恢复时执行 `KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build`。

性能库也可用 `check_stage01.py --workspace-only --stage 2 --cores 0 --suite model --repeat 1 --library build/_C.abi3.so` 检查各核真实 workspace 最后一代结果，避免只验证带 capture 的同步路径。

小 case 模拟器入口支持 `--heads 2 --tokens 128 --cores 1 --stage 2 --warmup 0 --repeat 1 --device 0`。两个 work 用于检查跨 work 重叠。模拟器使用 CPU 初始化，不读取输出；需要完整的一个 AIC 和两个 AIV 时间线，不能以模拟器耗时替代硬件性能。
