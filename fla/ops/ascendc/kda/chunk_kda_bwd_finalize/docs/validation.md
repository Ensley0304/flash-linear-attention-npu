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

## 跨 work 预取：90% busy 目标

用户要求继续按 PR #1069 优化，以至少一个主流水线 busy≥90% 且实际耗时下降为目标。基线固定为 `9899448a`；模型、输入布局、预热 20 次、采样 10 次及设备配置沿用上一轮。

Stage 0 每 head 最后一次 L1 读取后，提前装载下一 work 的五个已结束生命周期的输入；akk/dW 保持当前 work 所有权，跨核握手未删。首 work 装载、同核回绕、奇数 head、最后一代排空均通过原精度门禁。后续 Stage 7 的 qPos/bkPos 分区已调整，避免覆盖预取数据；尚未实现的 Stage 不列入验收。

70 次精度检查通过，包括完整 8K 定长/变长及关闭 capture 后的实际 workspace。最终 A/B：定长 1849.70→1574.73 us，MTE2 66.4%→90.55%；变长 1911.00→1615.70 us，MTE2 66.4%→90.7%。达到两阶段开发入口的本轮目标。

取消开发入口跨核握手的诊断候选虽达到 MTE2 92.6%，但耗时高于保留握手的版本，因此回退。数据、候选、复现和产物身份见 [跨 work 性能记录](perf/stage01_cross_work/README.md)。

格式化后重新构建最终性能库，再通过 4 次 workspace smoke；总计 74 次精度检查。该库变长复测 1636.16 us、MTE2 90.6%。msprof 导出后退出 137，已保留异常日志并独立核验完整 30 条 task 数据，详见性能记录；不将此工具异常归为 kernel 精度或执行失败。

## Stage 2：双缓冲与三阶段流水

新增 `dKgb_raw=akkᵀ@dW`、`zW=dW@kEᵀ`，均为 BF16×BF16、FP32 累加/输出。dW 从 L0C 转 BF16 NZ 后直接在 L1 消费；两个 head 的 kE 独立管理 ready/free，复用原有两套 L0A/B/C。L1/UB/生产 workspace 分配不变，capture 扩至 224 KiB。

三阶段入口在 kE 搬入 L1 后释放 GM 槽，允许下一 work 的 Stage 1 与当前 Stage 2 重叠。akk 独立 free 事件防止过早覆盖，退出时排空 Fixpipe。该提前推进仅针对当前开发入口：后续 Stage 仍消费 exp2_gk/rH/akk，扩展时必须重新落实其生命周期，不能直接沿用提前释放点。

首轮定位到 BF16 舍入边界：dW 与 CPU 相差一个 BF16 间隔，导致 dKgb_raw 最大绝对差 3.24e-4；以设备实际 BF16 操作数复算，Stage 2 局部误差约 1e-8。保留原有八个中间量的容差；新增 Stage 2 同时使用严格局部 GEMM 检查和独立 CPU 全路径的 BF16 误差传播预算，规则见[测试入口](../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。预算校准能够拒绝全零、漏掉一半结果和错误 head。

格式化后的 capture 库通过 92 次设备检查：Stage 0/1/2 单核 tails 各 22 次、Stage 2 多核 flat gate tails 22 次，以及累计/flat gate 的完整 8K 定长和变长各 1 次。全部 head/chunk/元素、十个中间量、workspace 最后一代、保护区与 trace 均检查；没有抽样。flat gate 使 kE 所有行保持显著，避免仅检查指数衰减后的近零值。Host tiling 的正常与异常分支检查通过。

性能库另通过累计/flat gate 的 8K 定长/变长各核最后代 workspace 校验，共 4 次，总计 96 次。最终定长 1632.68 μs、MTE2 90.6%，变长 1686.16 μs、MTE2 90.7%，均超过目标；相对同样执行六次 MMAD/head 的首版三阶段耗时降低约 20%。最终变长独立重采正常退出，首轮导出后的 137 异常也完整保留。候选、原始计数、库身份及时间线见[三阶段性能记录](perf/stage012/README.md)。

## Stage 3：四阶段流水

新增 `Zb=tril((zV-zW)*beta[None,:],-1)`、`dv=beta[:,None]*dv_b`、`db_v=sum(v*dv_b,-1)`。FP32 beta/db_v 驻留 UB，Zb/dv 转 BF16；私有入口当前使用 FP32 beta 和 BNSD/NTD dv 输出。

32 行输入/输出双缓冲；每 head 原有 512 KiB GM 分成两代槽，上一 Stage 3 与当前 Cube 重叠。保留 FIX ready、按槽消费确认及输入布局切换保护；去掉重复事件往返和整轮 Scalar 等待。早期版本的跨槽覆盖竞态已修复，未纳入性能基线。

最终格式化源码通过 118 次设备检查：Stage 0/1/2 单核 tails 各 22 次，Stage 3 单核累计 gate tails 和多核 flat gate tails 各 22 次，完整 8K 两种布局、两种 gate 的 capture 检查 4 次，性能库相同四组检查 4 次。全部 dv 和 capture 中十二个量逐元素验证，无抽样；另检查两个 GM 槽最后有效代、保护区和 trace。Host tiling 与 clang-format 检查通过。

既定 Stage 0～2 容差不变。Stage 3 同时通过局部和独立全路径检查；门槛校准拒绝 beta 错轴、缺少下三角、错误 head、全零/减半和 db_v 多乘 beta。采数版只等待 trace 自身复用，避免整轮采数同步掩盖计算竞态。

最终硬件数据、库身份及完整三 lane 时间线见[四阶段性能记录](perf/stage0123/README.md)。这仍是 Stage 0～3 开发入口；完整七输出、ATK 双标杆和 A3 上板验证尚未完成。

## Stage 4：五阶段流水

新增 `Tza=Zb@akk.T`，BF16×BF16、FP32 累加，直接转 BF16 NZ 驻留 L1。两代 akk/Tza 使用每 head 原有预留区，L1 仍为 384 KiB；Zb 双 head 预取，L0A/B/C 沿用双槽。Stage 2 不再提前释放本代 akk，Stage 4 最后读取后才允许覆盖。

精度版检查 Tza 的独立 CPU 全路径和实际 Zb 局部标杆，并通过仅在采数版执行的 `Tza_L1@akk.T` 核对真实 L1 数据。Stage 0～3 容差不变，新增误差预算通过错误转置、串 head、串代、全零和减半校准。性能版不执行该验证乘法，也不额外导出 Tza。

140 次设备检查通过，覆盖 Stage 0～3 回归、Stage 4 单核/多核尾块、两种 gate 的完整 8K 定长/变长及性能库实际 GM 中间量。Host tiling 和 clang-format 通过。最终 Stage 0～4 耗时为定长 2036.11 μs、变长 2102.43 μs，最大流水线 AIC MTE2 分别为 91.2%/91.4%；最后 10 次均超过 90%，msprof 均正常退出。

测试清单、msprof 和完整三 lane 时间线见[五阶段性能记录](perf/stage01234/README.md)。复现入口为 `--stage 4`。这仍是部分 Stage 的开发验证，不能代替完整七输出、ATK 双标杆及 A3 验收；后续 Stage 5/6 须继续延长驻留数据生命周期。
