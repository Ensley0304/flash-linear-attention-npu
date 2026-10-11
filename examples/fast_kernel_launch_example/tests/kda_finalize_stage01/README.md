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
