# A2/A3 Stage 0/1 开发验证

- 分支：`kda-finalize-a2`；首版基线：`aa9289eb`；双缓冲优化基线：`ecb0768b`。
- 工程参考：`CHUNK_INDEPENDENT V1`；用户指定 PR #404，读取其 head `d246019c6f76e5d400ed7117e49b49c912eee1a0`。采用 Stage 顺序展开、集中地址常量、显式事件；不沿用 arch35 指令和直传路径。
- A2 API 参考：当前仓库 `chunk_kda_fwd_prepare/op_kernel/arch22/` 的 ND→NZ、LoadData、Fixpipe 与事件写法。
- 本轮仅实现 Stage 0/1。开发入口 `torch.ops.ascend_ops.kda_finalize_stage01` 与公开 Finalize 隔离，原 A5 注册、tiling、kernel 和 ABI 不变。

## 环境与阶段结果

- 服务器：`a2-server`，hostname=`ubuntu`，用户=`root`；设备 1，910B3。
- 独立目录：`/data/Ensley0304/kda-finalize-a2-stage01-20261011`。
- CANN：`/data/wys/Ascend/cann_9.1.0_910b_0605/cann`（9.1）。全局 CANN 缺少 set_env.sh，本轮不修改全局安装。
- Python：`/data/miniconda3/envs/wys_gdn/bin/python`；torch=`2.7.1+cpu`，torch_npu=`2.7.1.post5`。仅在独立目录构建、加载测试动态库，不安装或覆盖 wheel/OPP。
- Stage 0：设备编译通过；B=1、H=3、定长 T=65，以及变长长度 `[0,1,0,33,65,0]`，单核各重复 2 次通过。四个结果逐项对照 FP64 CPU 标杆；同时检查奇数 head、尾块、工作项追踪和 capture 哨兵。
- Stage 1：设备编译、首轮定长/变长检查通过，随后完成下表回归。后续 Stage 不在本轮范围，尚无整算子精度或性能结论。

最终代码的扩展验证（每组 11 个 case，各重复 2 次）：

| 执行范围 | 核数 | 结果 |
| --- | --- | --- |
| Stage 0 | 1 AIC | 22/22 通过 |
| Stage 0+1 | 1 AIC | 22/22 通过 |
| Stage 0+1 | 按任务数分配，最多 20 AIC | 22/22 通过 |

覆盖 T=1/16/31/32/33/63/64/129、B=2/T=65、H=96/T=65，以及变长长度 `[0,1,31,32,33,63,64,65,0]`。检查八个中间量、NaN/Inf、哨兵、真实 workspace 最后一代内容、运行时核信息和复用次数。单核 H=96/T=65 实际连续执行 96 个 work。

Host 检查通过：固定/变长元数据、空序列跳过、非法输入拒绝、大小溢出，以及目标 B=1/H=96/T=8192 的任务/内存计算。T=8192 本轮未运行设备精度或性能测试。

代码以仓库 `.clang-format`、clang-format 18.1.8 格式化；host 检查使用 `g++ -std=c++17 -Wall -Wextra -Werror`。本地源码与设备测试源码逐文件 SHA256 一致。构建及三个回归日志位于上述独立目录的 `stage01-final-build.log`、`stage0-cores1-tails.log`、`stage1-cores1-tails.log`、`stage1-cores0-tails.log`。

可复现命令见 [Stage 0/1 测试说明](../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。

## 数据与同步追溯

| 设计项 | 代码位置 | 检查方式 |
| --- | --- | --- |
| L1/L0/UB/workspace | `arch22/chunk_kda_bwd_finalize_common.h` | 固定字节偏移及 static_assert |
| Stage 0 四个 GEMM | `arch22/chunk_kda_bwd_finalize_cube.h::Stage0` | 转置、尾块和 BF16 dW 对照 |
| Stage 1 计算与驻留 | `arch22/chunk_kda_bwd_finalize_vector.h::Stage1` | rH/指数/kE/gk_last 对照及 kE GM 实际内容 |
| task/元数据/大小 | `op_tiling/chunk_kda_bwd_finalize_stage01_tiling.h` | 独立 host 检查及运行时 trace |
| 单窗口复用 | 开发入口 Stage01Kernel | 1 AIC 强制处理多 work，记录 work/core/owner/slot 使用次数/实际 blockDim |
| 开发期结束握手 | flag 9/10 | Stage 2 缺席时消费 dW ready；两 AIV 均参与 ready/done，关闭本地事件 |

开发采数区独立于生产 workspace：每 (head,chunk) 一个 160 KiB snapshot；每 work 的两个 AIV 各一个 128 B trace。采数会增加写回，不用于性能比较。dW snapshot 是同一 L0C 的 BF16 ND 导出；L1 中 dW 的后继消费需在 Stage 2 实现时进一步验证。

FP32 矩阵/归约使用 `rtol=3e-4, atol=2e-6`；指数使用 `rtol=3e-5, atol=1e-7`；BF16 导出使用 `rtol=8e-3, atol=2e-5`。这些是开发期中间量检查阈值，不替代最终七输出验收。

## Stage 0/1 双缓冲优化（2026-10-11）

参考用户指定 [cannbot-skills PR #1069](https://gitcode.com/cann/cannbot-skills/pull/1069)，固定 head `4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca`。读取其中 `ops/ascendc-performance-best-practices/references/elementwise/double_buffer_design.md`、`matmul/pingpong_design.md`、`matmul/mte2_preload_design.md`，采用预取、分槽 ready/free 和等待后移的方法；缓冲大小、tile 和事件数量按本算子重新计算。

### 采用的改动

- Stage 1：32 行输入 ping/pong，Cast 后提前释放 BF16 输入，下一块搬运与本块归约重叠；kE 使用两个独立 BF16 输出槽。仅清零 k/exp2_gk 的无效尾行。
- Stage 0：提前载入两个 head 的独立 L1 区，分别管理 ready/free；L0 搬运、MMAD 与 Fixpipe 分开排队。小 case 时间线的 MTE2/Cube 重叠增加，但 Stage 0 单独硬件耗时基本持平，不将其列为主要延时收益。
- 固定 L1/L0/UB/GM workspace 总量不变；W0～W10 为 Stage 1 工作区，W11/W12 空闲；dW 驻留和后续 Stage 地址不变。
- 性能入口增加 `KDA_FINALIZE_STAGE01_CAPTURE=0` 编译开关，关闭额外采数；公开算子接口不变。收尾握手仍属于两阶段开发入口。

### 实验追踪

同一 A2 910B3/device 1、CANN 9.1、B=1/H=96/T=8192/K=V=128，预热 20 次，最后 10 次取中位数。下表为各候选当轮 `Task Duration(us)`，最终同脚本 A/B 数据另见 [performance.csv](perf/stage01_double_buffer/performance.csv)。

| 候选 | Stage 0+1 / us | 选择 |
| --- | ---: | --- |
| 原始串行 Vector tile | 2678.67 | 对照 |
| 16 行双缓冲 | 1854.70 | 有效 |
| 加 Stage 0 预取/排队 | 1844.81 | 保留；主要收益仍在 Stage 1 |
| k/gk 批量提前驻留 | 1941.92 | 回退，未进入最终代码 |
| 32 行双缓冲 | 1798.96 | 采用；随后规整事件闭环、仅清零尾行 |

### 最终精度回归

同一份格式化后的源码重新构建，关闭自动同步、显式管理事件；CPU 参考公式与容差均未改变。测试将逐元素比较按 tensor 批量执行以减少 Python 开销，仍覆盖全部 head/chunk/元素，没有抽样。

| 最终构建检查 | 设备调用数 | 结果 |
| --- | ---: | --- |
| Stage 0，单核 tails | 22 | 通过 |
| Stage 0+1，单核 tails | 22 | 通过 |
| Stage 0+1，多核 tails | 22 | 通过 |
| Stage 0+1，B=1/H=96/T=8192 定长，20 AIC | 1 | 全量通过 |
| Stage 0+1，同规模变长，20 AIC | 1 | 全量通过 |

共 68 次；变长长度 `[0,1,31,64,65,4096,3935,0]`，包含空序列、非对齐序列边界与 tail。仍检查全部八个中间量、真实 workspace 最后一代结果、保护区、实际分核和同核 slot 回绕。dW 的 L1 后继消费与完整七输出仍需后续 Stage 验证。

原始设备目录：`/data/Ensley0304/kda-finalize-a2-stage01-20261011/optimization`。最终精度日志为 `verified-stage0-cores1.log`、`verified-stage1-cores1.log`、`verified-stage1-cores0.log`、`verified-model.log`；实际精度库为 `verified-capture.so`，性能库为 `verified-perf.so`。

性能库另外通过 28 次 `--workspace-only` 检查：Stage 0 单核 smoke 4 次、Stage 0+1 多核 tails 22 次、完整 8K 定长/变长各 1 次。读取各核最后一个 work 的真实 dk_state_raw/dv_b/zV/kE，并对照同一 CPU 标杆，检查关闭 capture 后的同步路径。更新后的测试 CLI 另经 2 次 capture smoke 验证。

最终同脚本 A/B：定长 2678.80 → 1798.35 us，变长 2747.55 → 1859.79 us，短序列 8.09 → 5.64 us。完整统计、精度/性能库身份及实际指令重叠见 [性能记录](perf/stage01_double_buffer/README.md)。
