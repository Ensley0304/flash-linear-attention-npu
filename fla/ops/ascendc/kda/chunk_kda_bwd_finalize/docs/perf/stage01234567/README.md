# A2 Stage 0～7 验证与性能

2026-10-12，`a2-server`，Ascend910B3，device 1，20 AIC / 40 AIV，CANN 9.1.0。当前为私有分阶段入口，尚不是完整 Finalize。

## 采用版本

采用 **r5**：16 行输入/输出双缓冲、32 行逆序分带、BF16 高位与残差；qPos/bkPos 复用正向指数。对角线、center、带内指数留 UB，仅采数版导出。Stage 6 使用独立 FIX 通知；Stage 7 进入、退出均保护 UB 复用，末带写完才释放 GM 槽。

当前只生成 Stage 7 操作数；接入 Stage 8/10 时必须插入带间消费和 L1 搬运，不能直接沿用“先生成两带”的测试调度。

## 精度与流水

- r3 完成 **228 次 CPU 标杆检查**，包括 Stage 0～6 回归和 Stage 7 定长、变长、尾块、奇数 head、槽位回绕及 gate=-5 极值。
- 最终 r5 完成 **70 次新增标杆检查**；四组模型 case 的全部 capture、trace、dv 与已验证 r3 **逐位一致**。性能库另用 CPU 标杆检查全部 dv 和各核最后两代有效 workspace。原有容差未放宽。
- 高低位门禁经过正确值及缺失残差、残差符号、转置、缩放错误校准，见 [校准记录](precision-calibration.json)。
- 完整模拟器 trace：H=2/T=192/单核，1 AIC + 2 AIV，48 次生产 MMAD；每 AIV 的 Exp 从 r3 的 87 条降为 69 条，GM 写指令从 270 条降为 252 条。首个 Stage 7 的 Vector 与 MTE2 重叠约 0.34～0.36 μs，与 MTE3 重叠约 3.75～3.78 μs，见 [流水记录](timeline-r5.json)。
- Host tiling 检查、Python 语法和 clang-format 检查通过。

## msprof

B=1/H=96/T=8192/K=V=128；变长长度 `[0,1,31,64,65,4096,3935,0]`。使用相同随机输入和 scale，gk 为基准脚本沿用的 `[-32,0]` 随机值；合法累积 gate 的精度另行验证。每组预热 20 次，取随后 10 次目标 kernel 的中位数；关闭 capture、trace 和 L1 验证 GEMM。

| 版本 | 定长耗时 μs | 最大流水占比 | 变长耗时 μs | 最大流水占比 |
| --- | ---: | ---: | ---: | ---: |
| Stage 0～6，`18653f63` | 3046.871 | Vector 74.70% | 3118.612 | Vector 74.60% |
| Stage 0～7，r3 | 6580.122 | Vector 74.90% | 6662.653 | Vector 75.15% |
| r4：复用正向指数 | 6403.308 | Vector 74.20% | 6502.750 | Vector 74.25% |
| **r5：移除 UB 元数据验证写回** | **6414.639** | **Vector 74.00%** | **6534.861** | **Vector 74.00%** |

最终版比 Stage 7 初版快 **2.51% / 1.92%**；移除验证写回这一步本身没有测得加速。最大流水占比没有提高，**90% 目标未达**，不能宣称已满足该 bound 要求。Stage 6 与 Stage 7 工作量不同，二者总耗时不作为等工作量加速比。

八组正式采集均正常退出，每组 CSV 含 30 条目标记录。完整 min/median/mean/max、各流水占比和原始 CSV 对应关系见 [performance.json](performance.json)。模块命名冲突误触发的早期采集已隔离，未纳入结果。

采集命令：`msprof --output=<目录> --ai-core=on --aic-metrics=PipeUtilization --task-time=on --ascendcl=off python .../bench_stage01.py --library stage7/r5-perf.so --stage 7`；变长追加 `--varlen`。

源码和动态库 SHA256 见 [provenance.json](provenance.json)，精度计数见 [accuracy.json](accuracy.json)。完整源码快照、脚本、日志和 trace 归档于服务器 `/data/Ensley0304/kda-finalize-a2-stage7-20261012/stage7/report.tar.gz`，本地副本位于工作区 `validation/stage7-20261012/report.tar.gz`。早期失败及修复见 [development-notes.json](development-notes.json)。
