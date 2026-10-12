# A2 Stage 0～8 验证与性能

2026-10-12，`a2-server`，Ascend910B3，device 1，20 AIC / 40 AIV，CANN 9.1.0。私有分阶段入口，尚不是完整 Finalize。

## 采用版本

采用 **r1**：Stage 8 按 32 行带计算 `dq_local_raw`、`left_raw`，各用 `Ah@Bh + Ah@Bl + Al@Bh`，FP32 L0C 累加后写 GM。L0A/B/C 两槽交替，保留既有 UB 双缓冲和下一 work 状态预取；直接展开 Stage 公式，仅扩展已有装载、乘加辅助函数。

Stage 7 先生成完整 kNeg，立即通知 Cube，再生成 qPos/bkPos。Cube 搬入两 head 后才允许下一带覆盖 GM。L1 `[384,448)` / `[448,512)` 分别保存两 head 的下三角高低位及 kNeg，不覆盖下一代 akk。两个下三角矩阵每 chunk 搬一次，kNeg 每带搬一次；复用 keReady/keFree，不增加同方向硬事件。workspace 每 head 两代，各 320 KiB。

方法参考 [cannbot-skills PR #1069](https://gitcode.com/cann/cannbot-skills/pull/1069) 的双缓冲、MTE2 preload 和延后等待；固定参考提交 `4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca`，本轮从本地参考仓读取。未照搬教程的 tile 或收益数字。

## 精度与流水

- **316 次 CPU 标杆检查通过**：首版 Stage 8 三种 gate 尾块 66 次、Stage 0～7 回归 176 次、完整模型采数/性能库 8 次；最终 r1 三种 gate 尾块 66 次。覆盖奇数 head、单核槽位回绕、定长/变长、空序列及 gate=-5。
- r0/r1 的四组完整模型（累计/平坦 gate × 定长/变长）在采数版和性能版均逐位一致；检查全部 capture、trace、dv 及最后两代有效 workspace。见 [accuracy.json](accuracy.json)、[equivalence-final.json](equivalence-final.json)。
- Stage 8 局部门槛使用实际 BF16 三项输入的 FP64 GEMM；分带公式另以原始 gk 做 FP64 检查。最初漏传了 Stage 7 的绝对误差预算，经定位补齐 `sum(abs(A))*2e-38`；kernel 和前序门槛未改。失败定位及修正依据见 [development-notes.json](development-notes.json)。21 项 CPU 校准覆盖正确结果及遗漏残差、错误符号、转置、串带、全零，见 [calibration.json](calibration.json)。
- Host tiling、clang-format、Python 语法及本地/服务器源码哈希检查通过。
- 完整模拟器：H=2/T=192/单核，1 AIC + 2 AIV，共 120 次生产 MMAD，其中 Stage 8 为 72 次。Stage 8 的 L1 搬运与 MTE1 重叠约 2.33 μs，MMAD 与 MTE1 重叠约 2.45 μs；与两 AIV Vector 的重叠从 r0 的 0.99/0.98 μs 增至 r1 的 2.44/2.49 μs。Vector 指令数不增加。见 [r0](timeline-r0.json)、[r1](timeline-r1.json)。模拟器时间不替代硬件耗时。

## msprof

B=1/H=96/T=8192/K=V=128；变长长度 `[0,1,31,64,65,4096,3935,0]`。同一输入、scale、设备及软件环境；gk 沿用基准脚本的 `[-32,0]` 随机值，合法累计 gate 的精度单独验证。关闭 capture/trace 和额外验证 GEMM；每组 20 次预热，取随后 10 次目标 kernel 的 `Task Duration(us)` 中位数。

| 版本 | 定长耗时 μs | 最大流水占比 | 变长耗时 μs | 最大流水占比 |
| --- | ---: | ---: | ---: | ---: |
| Stage 0～7，`7d30217e` | 6375.497 | Vector 74.40% | 6527.870 | Vector 74.20% |
| Stage 0～8，r0：整带完成后通知 | 7599.582 | Vector 62.80% | 7704.344 | Vector 62.70% |
| **r1：提前发布 kNeg** | **6463.000** | **Vector 73.55%** | **6593.072** | **Vector 73.20%** |

r1 比同工作量 r0 快 **14.96% / 14.42%**，收益明显超过采样波动。加入 Stage 8 后相对 Stage 7 耗时增加 **1.37% / 1.00%**，新增 Cube 计算大部分被重叠；两个 Stage 范围不同，不作为等工作量加速比。

**最大流水占比仍略低于 Stage 7，90% 目标未达。** 当前占比最高的仍是 Vector；单纯减少 Stage 8 MMAD 的收益空间不足以填补这一差距，后续应结合 Stage 9/10 的数据生命周期继续优化整体流水。

首版定长的一次自动导出异常退出，已用退出 0 的复采替换。r1 变长两次在 benchmark 完成后的自动导出阶段退出 -9；离线恢复的两份 CSV 各包含完整 30 条目标记录，本次离线导出退出 0。两次中位耗时为 6613.522 / 6593.072 μs，占比 72.90% / 73.20%，差异约 0.31%。表中采用复采数据；原始记录及异常退出码均保留，不能把它们写成正常退出。

完整 min/median/mean/max、各流水占比、退出码及 CSV 对应关系见 [performance.json](performance.json)。命令：`msprof --output=<目录> --ai-core=on --aic-metrics=PipeUtilization --task-time=on --ascendcl=off python .../bench_stage01.py --library stage8/r1-perf.so --stage 8`；变长追加 `--varlen`。

源码与库 SHA256 见 [provenance.json](provenance.json)。完整源码、脚本、日志、CSV 与 trace 归档于服务器 `/data/Ensley0304/kda-finalize-a2-stage8-20261012/stage8/report.tar.gz`，本地为工作区 `validation/stage8-20261012/report.tar.gz`。本轮不代表完整七输出、ATK 双标杆或 A3 验收；Stage 9/10 接入时须延长带内指数、qPos/bkPos 的消费者保护。
