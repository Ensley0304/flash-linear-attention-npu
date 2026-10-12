# Stage 0～5：A2 双缓冲流水

范围：私有六阶段入口，A2 Ascend910B3、20 AIC / 40 AIV、CANN 9.1；B=1/H=96/T=8192/K=V=128。变长长度 `[0,1,31,64,65,4096,3935,0]`。**精度检查通过，最大流水占比超过 90% 的目标未达到。** 不代表完整 Finalize、ATK 或 A3 已验收。

## 实现

Stage 5 计算 FP32 dq_base/dk_base/dg_base/db_base，直接对原始 gk 作差再求指数，最后有效 token 加入 gate_state。Stage 1 按代保存 rH，Stage 3 提前发 ACK 允许 Cube Stage 4 与 Stage 5 重叠；独立 S5_DONE 保护两代 GM 槽。

每个 AIV 使用两份 40 KiB 输入和两份 24 KiB 输出，16 行 tile。ratio、q 和归约临时量按生命周期复用，exponent/product 错开 256/512 B。生产 workspace 和 UB 总量不变；dq/dk 覆盖已消费的 raw 区，dg/db 使用空闲区。退出前排空输出，后续 Stage 必须按这些生命周期继续接入。

## 性能

性能库关闭 capture/trace 和测试专用 L1 乘法。每例预热 20 次，取 `Stage01Kernel<5>` 最后 10 次 `op_summary` 中位数。以下为 Stage 0～5 合计；流水占比不是 HBM 峰值带宽利用率。

| case | 首版六阶段 / μs | 当前版本 / μs | 最大流水线 | 最后 10 次占比范围 |
| --- | ---: | ---: | ---: | ---: |
| 定长 8K | 3095.18 | 3003.52 | AIV Vector 75.55% | 75.5%～75.6% |
| 变长 8K | 3155.92 | 3095.20 | AIV Vector 75.05% | 74.8%～75.2% |

四组独立采集均正常退出，各有 30 条目标任务。当前版本耗时分别降低约 3.0%/1.9%，但两例均未达到 90%。数据见 [最终采集](final-performance.json)、[全部候选](performance.csv) 和 [原始 CSV](raw/final-profile-fixed.csv)。较少阶段的 Stage 0～4 不能作为同工作量加速基线。

32 行复用、减少核数、循环展开和 normal-mask repeat 方案均回退；分块行归约没有稳定收益。UB 地址调整使定长 bank group 冲突占比从 13.1% 降至 10.6%，bank 冲突从 7.7% 降至 7.2%。未采用增加无用计算或减核降速来抬高占比。

后续三个候选各通过 22 次 tails 检查，未替换上述完成完整回归的版本。单轮小幅收益尚未证明稳定，不能作为新的验收结果：

| 候选 | 定长 / μs | 变长 / μs | 最大流水占比（定长 / 变长） |
| --- | ---: | ---: | ---: |
| FP32 MulAddDst 合并 beta 乘减 | 2989.14 | 3060.71 | Vector 74.35% / 74.15% |
| 满 chunk 编译期分支 | 3123.84 | 3216.08 | Vector 72.45% / 72.00% |
| 事件数组改为独立标量成员 | 2988.57 | 3063.09 | Vector 75.80% / 75.60% |

满 chunk 分支降低了 Scalar 占比，却增加总耗时；其余两项单轮收益约 0.5%～1.1%，仍未接近 90%。当前证据表明，仅调整这些局部指令和分支不足以达标，下一步需要定位稳态跨阶段等待和指令发射空隙。数据及源码/动态库哈希见 [后续实验](experiments.json)，原始 CSV、源码和日志保存在工作区 `validation/stage5-20261012/followup-report.tar.gz`。前两个候选的变长采集退出 137，原始数据重新导出成功（退出 0）；每组均有 30 条目标任务，取最后 10 条统计。事件成员候选的两组采集正常退出。

历史采集中的退出 137 和 `/data` 满盘空输出均保留日志，未作为最终成功采集。工作副本已复制到 `/root/ensley-kda-finalize-a2-stage5-20261012`，原目录保留。

## 验证

[166 次设备检查](accuracy.json)全部通过：Stage 0～4 单核 tails 各 22 次；Stage 5 单核累计 gate / 多核 flat gate tails 各 22 次；两种 gate 的完整 8K 定长/变长 capture 共 4 次、性能库共 4 次；scale=0/-0.5 各 2 次。Host tiling 和 clang-format 检查通过。

capture 检查全部 head/chunk/元素、局部 FP64 标杆及独立 CPU 全路径，原有精度门槛不变。性能库检查全部 dv 和各核最后两代 GM 结果，未直接导出已覆盖的中间代。门槛校准拒绝全零、减半、串 head/代、错误 scale、遗漏 beta 和错误 gate 补偿位置，见 [校准记录](precision-calibration.json)。

H=2/T=192、单 AIC 的完整模拟器记录含一个 Cube 和两个 Vector lane，共 42 次生产 MMAD。两个 AIV 均确认 Stage 5 计算与 MTE2、MTE3 搬运及 Cube Stage 4 重叠，见 [时间线摘要](timeline.json)。模拟器只证明依赖和重叠，不代替硬件性能。

源码及动态库身份见 [provenance.json](provenance.json)。完整日志、源码快照和 trace 归档在工作区 `validation/stage5-20261012/baseline/report`；服务器保留 `stage5/report.tar.gz`。

复现使用 `check_stage01.py --stage 5 --suite model --cores 0 --repeat 1`；性能库精度增加 `--workspace-only`，flat gate 增加 `--gate-mode flat`。性能使用 `bench_stage01.py --stage 5`，变长增加 `--varlen`。
