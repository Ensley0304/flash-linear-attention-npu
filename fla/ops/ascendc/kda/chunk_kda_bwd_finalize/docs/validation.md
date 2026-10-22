# A2/A3 Stage 0/1 开发验证

2026-10-22 后续增量 S1b：raw_g 提前搬运与 dk/扫描重叠，九边界和原 20 case 七输出全部通过。八配置专项门禁通过，无 norm 稳定改善 2.63%～2.86%，norm 指令不变；目标同轮 S1 10.4804→10.1867 ms。详见[compact Stage11](compact-stage11.md)。仍未达 9 ms，未重跑完整 200×3。

2026-10-22 最新增量：S1 的 Stage 11 前带 16 行布局通过九个边界和原 20 case、七输出；无 norm 专项性能门禁通过，同轮原始模型 case 从 R2b 的 10.4700 降到 10.3655 ms。norm 主核指令不变。详见[资源、精度及八配置结果](compact-stage11.md)；9 ms 未达到，完整 200×3、公开入口及 A5 未重跑。以下为历史记录。

最新增量验收（2026-10-21）：H24将Stage 1的rH归约移至Stage 0 Cube，复用L1 h/dh计算四个32×32对角块，Stage 5提取对角线。边界9例和原20case七输出全部通过原CT L1双标杆；FP32局部矩阵乘参考、FP64 golden、种子和阈值不变。累加顺序改变已纳入本轮精度验证，但未重跑完整200case三seed、公开入口或A5。

设备2两轮八配置均改善0.93%～2.47%；目标B1/H96/T8192/K=V128/BF16、无norm/beta反向，从同轮H14的10.70579 ms降至10.60315 ms，尚未达到8 ms。每配置20预热+20采样，统计主核与归约核Task Duration；采集前后设备空闲检查均通过。H23设备0的受干扰性能报告已明确作废并另在设备2复测，未混入验收结果。

验收库SHA256：`4f93391f97e4e4219787eb73a6436a94c14e9efd82300c6faf8a2d7b1f02d281`。完整模拟器三条MIX lane均完成；源码/性能/精度证据见[跨阶段调度实验](cross-stage-scheduling.md)及`hybrid-evidence/h24-*.json`。以下仍按时间顺序保留早期记录。

> 以下按开发阶段保留历史记录；最新 Stage 11/12 结果见本文末尾。

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

## Stage 5：六阶段流水（精度通过，性能目标未达标）

新增 FP32 dq_base/dk_base/dg_base/db_base，按原始 gk 差值计算状态门控，在最后有效 token 加入 gate_state。Stage 1 按代保存 rH；Stage 5 完成通知保护两代 GM 槽，Stage 3 的 ACK 仍允许 Cube Stage 4 与 Stage 5 重叠。生产 workspace 不扩容，采数区扩为 360 KiB/head/chunk。

局部 FP64 检查使用实际捕获的上游量，独立全路径检查传播原有 BF16/FP32 误差预算；未放宽已有门槛。CPU 门槛校准拒绝全零、减半、串 head/代、错误 scale、遗漏 beta 和错误 gate 补偿位置。166 次设备检查、Host tiling 和 clang-format 全部通过，覆盖完整 8K 定长/变长、两种 gate、旧 Stage 回归和缩放边界。

已实验 32 行输入复用、减少核数、循环展开、共用 mask、UB 地址错开及分块行归约。保留 16 行 UB 地址错开版本：最终定长 3003.52 μs、变长 3095.20 μs，两组 msprof 均退出 0。最大流水占比分别为 Vector 75.55%/75.05%，未达到 90%。32 行方案还曾触发 Select 内部临时区覆盖 beta 广播，修复后精度通过，但耗时仍回退，不采用。

2026-10-12 `/data` 满盘导致两次 msprof 空输出，已排除；工作副本复制至 A2 的 `/root/ensley-kda-finalize-a2-stage5-20261012`，原目录与文件保留。所有异常采集日志保留，不能用工具退出码或 benchmark 的 DONE 单独认定性能数据有效。

完整三 lane 模拟器确认 42 次生产 MMAD，两个 AIV 的 Stage 5 均与 MTE2、MTE3 及 Cube Stage 4 重叠。数据与复现方式见[六阶段开发记录](perf/stage012345/README.md)。这仍是部分 Stage 的开发验证，不能代替完整七输出、ATK 双标杆或 A3 验收；90% 性能目标仍待完成。

## Stage 6：七阶段流水（精度通过，最大流水占比未提升）

新增 `dAkk_raw=akk.T@Tza`，BF16×BF16、FP32 累加，实际消费两代 NZ L1 的 Tza。Stage 4 复用已消费的 dW ready 事件发布 Tza，Stage 6 末次 L1 读取后才释放 akk。双 head 沿用 L0 双槽，与 Stage 5 并行；结果写到每代 workspace `[216,232) KiB`，总量不变。capture 扩至 384 KiB。

184 次设备检查全部通过，覆盖 Stage 0～5 回归、Stage 6 单核/多核尾块、两种 gate 的完整 8K 定长/变长以及性能库。局部 FP64 和独立全路径门槛沿用既定规则；校准拒绝全零、减半、错误转置、串 head 和串代。Host tiling、clang-format、源码哈希检查通过。完整模拟器含 48 次生产 MMAD，两个 AIV 的 Stage 5 均与 Stage 6 实际重叠。

| 模型 case | 轮次 / 状态 | 触发证据或预计关键路径 | 结构决策或候选与唯一主要变量 | 精度结果 | 性能前后值 | 瓶颈变化 | 依据或结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 定长 8K | Stage 6 功能接入 | Stage 5 Vector 最大 | L1 驻留与双槽接入 Stage 6 | 通过 | 3009.25 → 3042.35 μs | Vector 75.40% → 74.55% | 新增计算大部分重叠，90% 未达到 |
| 变长 8K | Stage 6 功能接入 | Stage 5 Vector 最大 | 同上 | 通过 | 3084.79 → 3116.44 μs | Vector 75.15% → 74.40% | 同上 |

定长采集分析/导出时异常退出，原始数据离线导出成功且两次完整结果一致；变长最终采集正常退出。详细对照、辅助截断测量和日志身份见[七阶段开发记录](perf/stage0123456/README.md)。本轮不代表完整 Finalize、ATK 或 A3 验收。

## Stage 7：八阶段流水（精度通过，90% 目标未达）

接入 strict-lower 高低位、原始对角线和 32 行逆序分带的居中操作数；16 行输入/输出双缓冲。qPos/bkPos 共用正向指数，Vector 元数据留 UB，仅采数版导出。新增 Stage 6 FIX 通知，进入和退出 Stage 7 均保护 UB 复用，最后一带写完才释放该代 GM。

r3 完成 228 次 CPU 标杆检查；最终 r5 新增 70 次检查，四组完整模型的 capture/trace/dv 与已验证 r3 逐位一致。性能库独立检查 CPU 标杆与最后两代 workspace，未放宽精度门槛。Host、格式、源码哈希及完整三 lane 模拟器检查通过。

| 模型 case | 优化变量 | 精度 | Stage 7 初版 → 最终版 | 最大流水变化 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 定长 8K | 复用正向指数，去掉元数据验证写回 | 通过 | 6580.122 → 6414.639 μs | Vector 74.90% → 74.00% | 耗时降 2.51%，90% 未达到 |
| 变长 8K | 同上 | 通过 | 6662.653 → 6534.861 μs | Vector 75.15% → 74.00% | 耗时降 1.92%，90% 未达到 |

同期 Stage 6 最大占比为 74.70% / 74.60%，新增 Stage 7 未提高该占比。八组正式 msprof 均正常退出；中间候选、原始 CSV 和验证边界见[八阶段开发记录](perf/stage01234567/README.md)。当前未接 Stage 8/10 的带间消费者及 L1 搬运，不代表完整 Finalize、ATK 或 A3 验收。

## Stage 8：九阶段流水（精度通过，90% 目标未达）

接入两个 32×64 @ 64×128 的三项 BF16 残差 GEMM，FP32 累加，L0 双槽。新增独立 L1 `[384,512)` 保留下三角操作数并逐带加载 kNeg，既有 akk/状态预取保持；workspace 每代扩至 320 KiB。采用先生成 kNeg、立即通知 Cube、再生成 qPos/bkPos 的调度，增加真实重叠。

316 次 CPU 标杆检查通过；两版本四组完整 8K 模型在采数/性能库均逐位一致。极小 kNeg 的检查补齐了 Stage 7 已有绝对误差经 A 放大的预算，kernel 与前序门槛不变。Host、格式、源码哈希与完整三 lane 模拟器通过；Stage 8 有 72 次生产 MMAD，L1/L0 搬运和计算存在实际重叠。

| 模型 case | 优化变量 | 精度 | Stage 8 首版 → 最终版 | 最大流水变化 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 定长 8K | 提前发布 kNeg，与 qPos/bkPos 重叠 | 通过 | 7599.582 → 6463.000 μs | Vector 62.80% → 73.55% | 快 14.96%，90% 未达 |
| 变长 8K | 同上 | 通过 | 7704.344 → 6593.072 μs | Vector 62.70% → 73.20% | 快 14.42%，90% 未达 |

同期 Stage 7 最大占比为 74.40% / 74.20%，本轮没有继续提高该占比。变长自动导出异常，离线恢复完整 30 条目标记录，两次测量差异约 0.31%；退出码和复采均归档。详见[九阶段开发记录](perf/stage012345678/README.md)。仍是部分 Stage 验证，不代表完整 Finalize、ATK 或 A3 验收。

## Stage 9：分阶段验证通过

范围：逐带合成 dq、更新 dg，包含可选 Q L2Norm backward。沿用用户确认的一 AIV 一 head 调度和私有分阶段入口，不改变公开 A5 ABI。Stage 10/11 尚未接入。

- Stage 8 的 h0/h1 分别在 Fixpipe 完成后发布 S8_READY=3 / S8_SECOND_READY=14。两 AIV 都按 3→14 消费，owner 在自己的 ready 后执行本带 Stage 9；h0 可与 h1 Cube 重叠。下一带 BAND_READY 形成反压，奇数 head 的空 owner 也闭合两次通知。
- UB 保留 Stage 7 的 q/k/gk、下三角 mask、对角线、带指数与 beta 广播。16 行 tile；输入/原位 FP32 输出双槽 `[88,112)` / `[112,136)` KiB，BF16 输出双槽 `[136,140)` / `[140,144)`，q/k 转换 `[144,160)`，乘积 `[184,192)`，归约 `[178,180)`、行广播 `[180,180.5)`；q_rstd 复用 `[81,81.25)`。进入本带时等待两份旧输出并桥接到 MTE2，再重装原位槽；退出时排空输出，保护 Stage 7 的区间复用。
- dq/dg 更新复用 WS_DQ_BASE/WS_DG_BASE；Stage 5 的旧值在当前 tile 读入后结束生命周期。dq 另转 BF16 写实际输出，capture 的 720/752 KiB 保存 FP32 dq/dg；workspace 不扩容。
- 精度检查：原有 Stage 0～8 不放宽门槛；Stage 9 对照局部 FP64、直接因果指数差公式、实际 BF16 输出和 workspace，覆盖 Q norm、尾块、空序列、奇数 head、回绕、极端 gate 与完整 8K 模型。性能库另行检查。
- 性能实验：同卡重测 Stage 8 和 Stage 9，B=1/H=96/T=8192/K=V=128，定长与既定变长切分，20 次预热、10 次采样；统计 Task Duration 和最大流水占比。90% 是目标，不以增加无效指令提高占比。
- 优化参考：cannbot-skills PR 1069 对应本地参考提交 4950cbd7 的 double_buffer_design / ub_resident_design；先做片上复用及双槽，再根据实测选择广播或 tile 优化。

r2 的 132 次采数版 tails、44 次性能版 tails、18 次前序 smoke 回归通过；四组完整 8K 模型的 Stage 9 精度和版本一致性检查通过。最终定长/变长耗时 7717.991 / 7836.817 μs，最大流水为 Vector 67.70% / 67.85%，90% 目标未达。原始 CSV、源码哈希、模型输出哈希及完整三 lane 模拟器摘要见[十阶段验证记录](perf/stage0123456789/README.md)。工作目录已集中到 `/data/wys/kda-finalize-a2-stage9-20261012`。

## Stage 0～9 bound 优化（2026-10-12）

按用户计划独立测试跨带缓冲、完整 chunk 特化和跨 work 提前计算，均未获得稳定收益。后续保留 r11：Stage 7 输出交替用槽、跳过全零 tile 的高低位拆分；Stage 9 提前 base 搬入和 q/k 转换。192 KiB UB、GM/L1 布局、全部双缓冲、精度门槛和残差项不变。共享 mask 的 repeat 实验也已因回退撤销。

| B=1/H=96/T=8192/K=V=128 | 原基线 | 最终 r11 | 变化 |
| --- | ---: | ---: | ---: |
| 定长耗时 | 7711.310 μs | 7483.497 μs | −2.95% |
| 变长耗时 | 7805.193 μs | 7591.663 μs | −2.73% |
| 定长最大流水占比 | Vector 67.7% | Vector 65.4% | 未到 80% |
| 变长最大流水占比 | Vector 67.8% | Vector 65.2% | 未到 80% |

最终版本通过 132 次 Stage 9 tails、36 次 Stage 0～8 smoke、44 次性能库 tails；四组完整 8K 采数/性能版与已验证原基线输出逐位一致。Host 检查与三 lane 模拟器通过。去掉无效 Vector 计算后耗时和占比同时下降；本轮有实际加速，但 bound 目标仍未完成。

全部候选、同步/布局说明、精度范围、原始 msprof 和复现脚本见[本轮记录](perf/stage09-bound-opt/README.md)。仍为部分 Stage 开发验证，不代表完整 Finalize、ATK 或 A3 验收。

## Stage 10：转置 GEMM 接入

沿用已确认的一 AIV 一 head、逆序 32 行带与私有 fast launch 入口。参考角色仍为 CHUNK_INDEPENDENT V1（PR #404）；不改变公开 A5 ABI。两项转置 GEMM 各保留三项高低位乘积，L0A/B/C 双槽按 head 轮转，FP32 累加后写 right_raw。L1/workspace 复用消费结束的 kNeg 区，UB、workspace 总量不变。

同步按带执行两轮 BAND_READY/BAND_LOADED：kNeg 先读入 L1，qPos/bkPos 再发布，Stage 9 与 Stage 10 并行；第二轮完成后才能写下一带。空 owner 也闭合全部通知。精度检查增加实际六项乘积与独立 FP64 居中公式，前序门槛不变；计划覆盖三种 gate、Q norm、尾块、奇数 head、槽位回绕、性能库与完整 8K 模型，并对比同期 Stage 9 的 msprof。

224 次小 case 检查通过（8 次 Stage 10 smoke、132 次三种 gate×Q norm tails、40 次前序 smoke、44 次性能库 tails）。四组完整 8K 模型在采数版与性能版均通过；前序结果与已验证基线逐位一致，right_raw 通过 FP64 检查，性能库保留结果与采数版逐位一致。Host、格式、源码身份与完整三 lane 模拟器检查通过，三个 work 共 192 次 MMAD。

同期 Stage 0～9 → Stage 0～10：定长 7530.495 → 8498.790 μs，变长 7643.640 → 8604.525 μs；最大流水为 Vector 57.45% / 57.70%，未达到 80%。新增计算没有完全隐藏在前序流水中，前缀耗时增量约 0.96 ms，不能视为独立 Stage 10 耗时。

实现、精度边界、原始 CSV、模型哈希和复现脚本见 [Stage 10 记录](perf/stage10/README.md)。仍缺 Stage 11～12，不代表完整 Finalize、ATK 或 A3 验收。


## Stage 11/12：最终梯度与参数归约

Stage 11 接入 dk/dBeta、dg 反向扫描、gate 链式求导和 chunk 局部和，输入/输出分别双缓冲；完整 8 行使用三步并行扫描。Stage 12 使用独立 AIV kernel 和 16 chunk 输入双缓冲，同 stream launch 保证全部 partial 就绪。保留 FP32 beta 有效值、稳定 gate 导数及可选 Q/K norm。

448 次小规模检查及四组完整 8K 定长/变长 × norm 采数/性能版全部通过。新增阶段对照 FP64；前序采数、dq/dv 与同输入 Stage 10 截断执行逐位一致，性能版七个最终输出及 partial 与采数版逐位一致。Host、格式、源码身份、主 kernel 三 lane 与独立归约 AIV 模拟器检查通过。

设备 2 同卡 msprof：完整 Stage 0～12 定长 12.243 ms、变长 12.434 ms，并行扫描较首版分别加速 1.94% / 2.30%；最大流水均为 Vector 62.9%，仍未达到 80%。完整结果计入两个 kernel，Stage 12 归约约 17～18 μs。

实现、精度范围和原始证据见 [Stage 11/12 记录](perf/stage12/README.md)。当前为完整 Stage 的私有开发入口，公开 A2 接入、真实训练 dump 双标杆、ATK/A3 验收尚未完成。

## 2026-10-13：原生输出布局与 ATK 开发验证

新增 dq/dk/dv 的 BNSD/BSND、NTD/TND 原生写回。12 组布局对照的七个输出逐位一致；
默认布局兼容和非法布局检查通过，dG/dBeta 仍固定 BSND/TND。

独立 CPU FP64 golden 与 FP32 对照的第一轮 200 case 已完成：192 通过，8 个完整
8K case 的 dq 小值计数项未达标，其余六个输出全部通过。当前 FP32 对照未模拟
Stage 8/10 的 BF16 三项残差。用户已确认纯 FP32 局部矩阵乘为精度目标；不能用
之前的 Stage 开发门槛通过记录代替本轮双标杆通过。另两轮在最后一例期间中断，
重连后确认进程已退出、最终报告未生成，待精度修复后重新完整执行。

用例、校准、复现与限制见 [ATK README](../../../../../../tests/atk/chunk_kda_bwd_finalize/README.md)。

### dq FP32 修复验证（进行中）

Stage 8 仅 dq 局部矩阵乘升级 FP32，关闭 HF32；Stage 7 保留原 FP32 操作数，
Stage 10 重载 BF16 dAqk 平面。L1/UB/L0 容量不变，workspace 每核增至 1472 KiB。
原 10 个失败点全部修复；完整 case 192 的七个输出通过 CT 双标杆，小值超限点归零。
最终构建通过 Host tiling、Stage 8 定长/变长单核双次复用和原生布局回归。
三轮完整 ATK 正在运行，不能用局部通过代替 600 次最终报告。

修正 golden 的 FP64→BF16 二次舍入后重新跑三轮，前 192 个边界 case 全部通过，
8K 场景仍有失败。round 2 / case 192 新增的 3 个 dq 点来自 Stage 7 指数舍入。
整数/小数指数分解实验已修复这 3 个点；13 个冻结 chunk 的 dq 通过原 CT L1。
尾块、布局及完整模型待验。运行中的三轮报告属于 FP32 v2，不属于该指数实验库。

## 2026-10-14 分项性能优化

用户指定门禁：每项先测性能，有提升后跑 20 个精度 case，通过才保留。
沿用私有 A2 fast-launch 构建/测量入口，与此前 600/600 的 exp2-v3 基线一致。
本轮不修改数学公式、CPU 标杆、输入范围、精度阈值和公开接口。

- 固定基线库 SHA256：`92d9e6a8e60014f04ea325f3d0931a0db919a6da8ab1405d913f97ef52ceaf6a`。
- R1：删除 Stage 9 FP32 dq workspace 写回。两轮 A/B 中 norm 开启提升约 1%–1.6%，关闭回退约 0.5%–0.7%，因此不采用全分支删除。
- R1b：仅在 Q_NORM 模板删除该写回，仍出现非 norm 约 1.5%–1.9% 回退；补测原基线后确认，放弃。R1/R1b 均未进入精度门禁，Stage 9 恢复原代码。
- R2：Stage 7 独立 FP32 输出双槽；两轮交替 A/B 的八项提升 2.24%–4.24%，性能门禁通过。候选库 SHA256 `b52edf29d075dc4b93587e477ce0a98f00db4a49034a93d36ffc3c0776fae8b0`；20/20 case、每例七个输出全部通过原 CT L1 双标杆，已保留。本轮未重跑全部 600 次验证。
- 性能覆盖 B=1,H=96,K=V=128、T=8192/16384、norm/beta sigmoid 四种组合；每组预热 20 次、统计 20 次。
- 20 case 使用原矩阵中的 0,4,7,23,39,47,55,63,71,87,95,103,119,135,143,159,175,191,192,199；seed 不变，七个输出按原 CT L1 双标杆比较。FP32/FP64 标杆源文件固定哈希缓存，不能用 NPU 输出充当标杆。
- 记录目录：本地 `validation/perf-opt-20261014`，服务器 `/data/wys/kda-finalize-a2-stage9-20261012/perf-opt-20261014`。

## 2026-10-14 第二轮分项优化
基线为已通过20 case的 R2。候选一：按 validRows==64 分派 Stage 7/9/11 完整块模板，固定 active/count，尾块保持原路径；算术、UB/L1/workspace 和同步协议不变。候选二：Stage 11 的 dt_bias partial 改为 8 行树形归约；复用已结束使用的 product，保留 gateOut，尾块保持原顺序。两项分别先跑八配置两轮交替 A/B，性能提升才跑原20 case；失败撤销，不修改标杆或阈值。记录在 validation/perf-opt-next-20261014。

完整块候选：两轮交替 A/B 八项耗时降低1.86%～2.07%；20/20 case、每例七个输出全部通过原 CT L1，保留。库 SHA256：`93d415fbb21d4ee42e3d7c784f3ff7956896dd0d6f9deb163f20898bed9a11e7`。第二项归约仅在此版本上实验，以 full.so 为独立基线。

归约候选：两轮交替 A/B 八项耗时降低0.47%～1.34%；20/20 case、每例七个输出全部通过原 CT L1，保留。最终库 SHA256：`80ff0813a874d3ca5e886163da1dd718fc6ec5794de58e69a626259e7fda6ac5`。最终源码哈希与服务器构建输入一致。本轮共两次20 case回归，不代表重跑全部200 case/600次；未改变CPU标杆或精度阈值。Vector 最大占比约68.6%，未达到80%。

## dAqk 前移实验（2026-10-14，已撤销）
基线为 reduce.so（80ff0813a874d3ca5e886163da1dd718fc6ec5794de58e69a626259e7fda6ac5）。发布 KE_READY 后预处理当前 work 的 dAqk，前段算术顺序与原 Stage 7 一致；Stage 7 只处理 dAkk。原 BF16 lower 槽与 zV 重叠，迁到每代新增 [368,384) KiB；对角线保存在空隙 [204,204.25) KiB，FP32仍在原 [320,336) KiB。每核四代head槽合计增64 KiB，20核增1.25 MiB；输入输出双槽，完成搬出后才允许 UB 复用。Cube 两种 lower 改为独立基址读入。workspace仍使用uint64地址，2代复用顺序不变。先八配置两轮A/B，收益后20case，失败回退。已获准删除旧golden20缓存，精度报告保留。

实验结果：KE_READY 后前移版本在8K norm场景出现dg/dA_log/dt_bias非有限值，两个小规模定位用例通过但不能代表整体正确；跨work生命周期问题尚未完整定位，未采用。将同一预处理移到Stage5后、S6_READY等待前，8K四开关有限值检查通过，但两轮交替A/B八项耗时回退0.58%～1.21%，性能门禁失败，未执行20case验收。四个源码文件均恢复实验前快照；保留已验证reduce.so。此轮没有新增优化被保留。

## Stage 11 等待重叠实验（2026-10-14）
第一项：将 Stage 10 的 BAND_LOADED 等待移到 Stage 11 的独立输入预取之后，仅 right 仍在等待后加载。算术、UB 容量、事件数量和 workspace 不变。基线 reduce.so；八配置两轮交替 A/B 耗时降低 2.73%～3.48%，随后原20 case、七个输出全部通过 CT L1 双标杆，采用。未修改标杆或精度阈值，未重跑完整200 case/600次。Vector最大占比约70.6%，未达到80%。证据位于 validation/perf-stage11-20261014；服务器同名目录。

第二项：将 Stage 11 常驻 q/k Cast 移到 inputReady/outputFree 等待之前。以已验证预取版为基线，两轮 A/B 八项回退0.40%～1.62%，未通过性能门禁，未执行20case验收，已撤销。最终源码和远端两个开发加载入口恢复预取版；最终库SHA256为 c1c46f02f11f878390b8d6503ecb46c25d5021c12bac7503450011403b980b73，源码SHA256为 02c668dc66bff816e0d8f686fb4d93ea9a535d0dc8403ac726be3d1bbb6d0b20。

## 2026-10-15～16 顺序优化实验（已完成）
A2 SSH恢复后完成八项顺序实验，每项单独修改、八配置两轮交替A/B。Stage7/11提前释放输入、gate链交织、Stage3提前通知、dAqk独立L1、bandExp清零裁剪及Stage1先kE后rH均无稳定全面收益，全部撤销，未进入精度验收。

采用两项：
- Stage11完整块sums清零裁剪：初次平均收益约0.47%，低于自动0.5%筛选线；追加独立两轮A/B仍八项为正，无norm约0.69%～0.76%，norm约0.07%～0.21%。按用户“稳定收益”要求进入20case，七输出全部通过后作为小收益优化采用；原gate=false记录保留。
- Stage8/10 L0残差高低位复用：相对清零裁剪版，八项两轮平均耗时降低1.92%～2.80%，每轮均为正。原20case、七输出全部通过，采用。

最终8K约12.53～13.03ms，16K约25.02～26.06ms；主kernel Vector最高72.5%，未达到80%。统计包含Stage01Kernel和Stage12Kernel的Task Duration，两轮各20采样中位数取平均。最终库SHA256：117b9c419bbb2fb6118906d4b2afe589baa67e517388cba9cfe9285dc4bd8797。远端build与ascend_ops两个开发入口恢复此库；本地与远端采用同一源码快照。

本轮共两次原20case七输出精度回归，未重跑全部200case/600次；FP64 golden、纯FP32局部矩阵乘标杆、种子和CT L1阈值均未改变。精确重编译基线的库哈希相同、性能波动小于0.1%。证据及作废采集说明见工作区validation/perf-sequential-20261015/README.md；服务器同名目录。无commit或push。

## 继续推进80%流水占比（2026-10-16，已完成本轮实验）
以l0reuse.so为基线，恢复连接后完成候选验证：
- dq前32行裁剪零K项：八配置两轮A/B为-0.15%～+0.23%，无稳定收益，撤销。
- Stage7独立输入提前搬入：八项回退0.37%～0.59%，撤销。
- Stage11限制循环展开：编译库SHA256与l0reuse一致，无代码生成变化，撤销；早期无kernel记录的采集作废。
- Stage8取消未使用的BF16 dAqk搬运：八项耗时降低0.56%～1.40%，两轮均为正；原20case、每例七输出全部通过CT L1，采用。
- Stage11禁止内联：以skipunused为基线，八项回退0.32%～0.66%，撤销。
- dBeta按16行双槽搬出：以skipunused为基线，八项回退0.10%～1.96%，撤销。

每次仅改一项，性能无收益的候选不进入20case精度门禁。最终8K约12.33～12.96ms，16K约24.70～25.88ms；主kernel Vector最高72.8%，尚未达到80%。统计为Stage01Kernel和Stage12Kernel的Task Duration之和，两轮各20采样中位数取平均。

最终库SHA256：79ba610becf4a45a1065bd2a666af0d1fb69c376c197487b03c7696e15221267。本地与远端源码匹配已验收快照，远端build和ascend_ops两个开发入口均恢复此库。本轮一次原20case七输出精度验收，未重跑完整200case/600次；纯FP32局部矩阵乘标杆、FP64 golden、种子和CT L1阈值均未改变。

经用户授权仅删除perf-sequential-20261015/golden20缓存。因/data可用空间异常，msprof原始采集暂存/dev/shm/wys；有效op_summary、日志及精度报告归档到/data/wys/kda-finalize-a2-stage9-20261012/perf-bound-next-20261016。详见工作区validation/perf-bound-next-20261016/README.md。无commit或push。
## 按优先级单项优化（2026-10-16，已完成）

以 skipunused 为初始基线，九个候选逐项执行八配置两轮交替 A/B；仅性能通过者进入原 20 case、七输出 CT L1 双标杆验收。保留两项：

- Stage 1/3/5 完整块特化：无 norm 四项提速 0.41%～0.69%，norm 四项 -0.071%～+0.029%。原自动 gate=false 保留；四项每轮收益超过 0.3%、所有项每轮回退小于 0.1%，按稳定小收益单独审核。20/20 case 七输出全部通过后采用。
- Stage 10 按 head 提前通知：以上述版本为基线，八项提速 0.41%～1.73%，两轮各项均为正，自动性能门禁通过。20/20 case 七输出全部通过后采用。

事件分配外提、第二 head 等待延后、仅无 norm 的完整块特化、outputLayout 特化、beta 广播复用、head 参数缓存均无稳定收益；Stage 9/11 融合回退 4.42%～8.52%。这七个候选全部撤销，未进入精度验收。

最终已验收版本八配置的 8K 耗时为 12.171/12.695/12.181/12.703 ms，16K 为 24.334/25.515/24.325/25.418 ms，顺序为无附加反向、仅 norm、仅 beta sigmoid、两者均开启。统计口径仍为 Stage01Kernel 与 Stage12Kernel 的 Task Duration 之和，两轮各 20 样本中位数取平均。主 kernel 最高 Vector 占比 74.2%，未达到 80%。

本轮两次原 20 case 七输出验收；未重跑完整 200 case/600 次。FP64 golden、纯 FP32 局部矩阵乘标杆、种子与阈值不变。最终库为 rightready.so，SHA256 bcdedd9df4a6bceffae3ed5bfaebc3aa4945259ae6b10625c9a9fcb630a6ccd1。本地与远端使用同一源码快照，远端 build/ascend_ops 两个开发加载入口恢复此库；没有 commit/push。

完整模拟器基线采集已正常结束，覆盖六个 work 和三条 AIC/AIV lane，仅用于依赖分析，不当作硬件耗时。实验脚本、源码快照、op_summary、精度报告和结果汇总见工作区 validation/perf-priority-20261016；服务器位于 /data/wys/kda-finalize-a2-stage9-20261012/perf-priority-20261016。

## Stage9 双 AIV 协作实验（2026-10-17）

原全部改动保存在本地检查点 `d8e4624f` 和工作区完整快照中。候选在 A2 的 `/dev/shm/wys/kda-finalize-coop-20261017` 隔离构建，原远端源码、build/ascend_ops 加载库未修改。

按 A→B→A→B 完成八配置采集，每配置20预热+20采样，主kernel+归约kernel总时增加9.47%～16.15%，两轮所有配置均回退。主kernel最高Vector：单轮配置中位数基线74.00%、候选64.70%；两轮平均口径73.90%/64.65%，未达80%。性能门禁失败，候选撤销，源码恢复检查点中的已验收 tile16。

10个同步边界用例七输出与基线逐位一致，覆盖协作路径、奇数head、尾块、变长空序列/完整块/尾块混合、bank回绕、多核组、布局、norm、beta_mode=2和极值gate。性能未通过，因此未进入原20case双标杆验收，也未跑完整200case；没有改动FP32标杆、FP64 golden或CT L1标准。归档见工作区 `validation/perf-coop-20261017/README.md`。

## 四项优化实验（2026-10-17）

| 候选 | 双轮八配置耗时降幅 | 七输出精度 | 结论 |
| --- | --- | --- | --- |
| 末带 dG 驻留 | -0.877%～-0.479% | 未进入精度验收 | 撤销 |
| 首 tile gate 提前计算（全分支） | -0.134%～+0.850% | 未进入精度验收 | 撤销 |
| 首 tile gate 提前计算（仅 norm） | +0.121%～+0.791% | 20/20 case 全通过 | 保留 |
| Stage10 bkPos 预取 | -1.221%～+0.060% | 未进入精度验收 | 撤销 |
| Stage11 末带16行 | +0.278%～+1.528% | 20/20 case 全通过 | 保留 |

gateearly-norm原自动性能gate=false，按已说明的稳定小收益复核后进入20case验收；tile16自动性能gate通过。每项相对当时已验收基线独立测量，原20case七输出全过才保留。两次精度验收共40次case执行，未重跑全部200case；纯FP32局部矩阵乘标杆、FP64 golden和CT L1标准不变。

| 配置 | 8K(ms) | 16K(ms) |
| --- | ---: | ---: |
| 无附加反向 | 12.004 | 24.112 |
| 仅 l2norm | 12.549 | 25.046 |
| 仅 beta sigmoid | 12.017 | 24.082 |
| 两者开启 | 12.555 | 25.063 |

固定B=1,H=96,K=V=128,BF16，BNSD定长；表中包含Stage01Kernel和Stage12Kernel的Task Duration，两轮各20样本中位数平均。最终Vector最高73.575%，未达80%。16行版本降低了Vector工作时间和总耗时，占比略降，不宣称流水目标达成。

最终库tile16.so，SHA256 75c165084f40a2edbc4cb0f83872a7e408750015a1e178b6d79e2234bebd4748。本地与远端源码和两个开发加载入口均恢复此验收版本，无commit/push。证据位于validation/perf-next-20261017，服务器同名目录；raw profiling保留在/dev/shm/wys，有效CSV及精度报告另行持久归档。

## 局部流水优化表逐项实验（2026-10-17）

以tile16为基线，完成六项及三个分支复核。每版独立做八配置A→B→A→B，未叠加未通过候选。下列范围为八配置双轮平均总耗时降幅，正值表示提速：

| 候选 | 降幅范围 | 结论 |
| --- | ---: | --- |
| 删除最终FP32 dq私有写回 | -0.460%～+0.665% | 撤销 |
| 仅无norm删除写回 | -0.105%～+0.416% | 撤销 |
| Stage7高低位双block合并搬运 | -1.005%～-0.037% | 撤销 |
| Stage7第三BF16输出槽 | -0.297%～+0.920% | 撤销 |
| 仅无norm开第三槽 | -0.108%～+0.442% | 撤销 |
| Stage8 dAqk仅搬32行 | -0.754%～+0.436% | 撤销 |
| Stage10 dAqk仅搬32列 | -3.237%～-2.632% | 撤销 |
| 整条band零区裁剪 | -1.230%～+1.318% | 撤销 |
| 仅无norm裁剪零区 | -1.140%～+0.329% | 撤销 |

九版各10个边界用例、七输出均与基线逐位一致，合计90次。性能门禁全部失败，未进入原20case双标杆验收，未重跑200case；未修改FP32标杆、FP64 golden、种子或CT L1阈值。不能以边界逐位一致替代正式精度验收。

本轮候选双轮最高Vector为73.4%；最后一组原版为73.3%，未达80%。原版最后两轮8K总耗时12.123/12.592/12.045/12.585ms，16K为24.064/25.189/24.090/25.131ms，顺序为无附加反向、仅norm、仅beta sigmoid、两者开启。统计含主kernel和归约kernel。

正式源码和build/ascend_ops两处加载库未改动；隔离实验目录也已恢复tile16，库SHA256仍为75c165084f40a2edbc4cb0f83872a7e408750015a1e178b6d79e2234bebd4748。无新代码提交或push。完整脚本、九版源码/库、op_summary、边界结果与哈希清单见工作区validation/perf-local-20261017，A2的/data/wys/kda-finalize-a2-stage9-20261012下同名目录。


## 局部重叠四项实验（2026-10-17）

同源码重编译的四个AIC/AIV指令段完全一致，双轮profiler差异为-0.194%～+0.100%。各候选独立从基线构建，完成八配置A→B→A→B。下表正值表示提速。

| 候选 | 总耗时降幅 | 最高Vector | 结论 |
| --- | ---: | ---: | --- |
| Stage11复用UB中的db_base | -0.312%～+0.044% | 73.625% | 撤销 |
| Stage1最后一块rH归约时预取k/gk | -2.299%～-1.043% | 73.000% | 撤销 |
| Stage9→11首tile独立输入预取 | -3.592%～+0.251% | 71.300% | 撤销 |
| dBeta整chunk合并写回 | -1.772%～+0.091% | 72.500% | 撤销 |

四版各10个边界case、七输出与冻结基线逐位一致，共40次；重编译控制另10次。四项无稳定收益，未进入原20case双标杆验收，未重跑200case，未放宽精度标准。第三项首次构建的循环作用域错误已修正，表中为修正后完整设备测试结果。

原验收源码及两处加载库未改动，隔离目录恢复tile16，库SHA256仍为75c165084f40a2edbc4cb0f83872a7e408750015a1e178b6d79e2234bebd4748。本轮无代码提交或push。原版8K为12.000/12.565/12.069/12.556ms，16K为23.988/25.101/23.960/25.086ms，顺序为无附加反向、仅norm、仅beta sigmoid、两者开启；最高Vector约73.5%，未达80%。

证据：工作区validation/perf-overlap-20261017/report.md与evidence.zip，A2的/data/wys/kda-finalize-a2-stage9-20261012下同名目录。包含五组有效op_summary、逐项源码/库、性能门禁及边界报告；原始profiler目录在/dev/shm/wys/kda-finalize-overlap-20261017。

## Hybrid H14 验收（2026-10-21）

H14仅调整Stage 11的dG对齐跨行写回；数据布局、双buffer、事件及算术不变。已通过9个边界case和原20case、每case七输出的冻结FP32/FP64双标杆CT L1；另测H=4096/4097、T=2，stride上限两侧七输出与H10逐位一致。参考hash保持`1a5e5d498ce1cc5474f628c4279d871e305a3b9cb4b31b2bcba48e7109f6c706`。

八配置两轮交替A/B全部改善，目标B1/H96/T8192/K=V128/BF16/BNSD、无附加反向，主核+归约核从同轮H10的10.8495降至10.7074 ms（1.31%）。目标Vector约66.5%，最高配置约70.7%，未达到8 ms或80%。库hash `421e5f480fd173a212a11a67e42dc03e63f0c7fe3893bf0cbaec5d9539d5b3e0`，性能、精度及归档清单一致。完整200case三seed和公开入口验收未重跑。

H12/H13精度边界通过但性能门禁失败，未保留，未进入20case。设计、生命周期、失败记录及H14八配置表见[hybrid-plan](hybrid-plan.md)；精度、性能、指令段及完整模拟器摘要见`hybrid-evidence/h14-*.json`。H15仅为下一项候选，尚未验收。
