# Stage 0～3：A2 双缓冲流水

范围：私有四阶段开发入口，A2 Ascend910B3、20 AIC / 40 AIV、CANN 9.1。模型为 B=1/H=96/T=8192/K=V=128；变长长度 `[0,1,31,64,65,4096,3935,0]`。不代表完整 Finalize 或 A3 已验收。

## 实现

- Stage 3 在 FP32 中计算 Zb、dv、db_v；Zb/dv 以 RNE 转 BF16，beta/db_v 驻留 UB。
- 32 行输入/输出双缓冲。下三角按列乘 beta；dv 按行乘 beta，db_v 不乘 beta。有效行以外不读取 v/beta，不写 dv。
- 每 head 原有 512 KiB GM 划分为两个 256 KiB 槽。先发送当前 kE，再消费上一 work 的 Stage 3；FIX ready 和按槽 ACK 分别保证可读及可复用。
- 切换 zV/zW 与 dv_b/v 输入布局时等待两个旧槽；直接消费空槽许可，删除重复的清空、重新置位和 Scalar 整体等待。
- q 尚未使用的 UB 区暂存两个 BF16 输出槽，避免下一 Stage 1 的输入覆盖未完成的 MTE3。capture 的 trace 仅在自身记录复用时等待，计算沿用异步调度。

UB 仍为 192 KiB/AIV，L1/L0 双缓冲和 workspace 总量不变。后续 Stage 4/5 必须延长 akk、exp2_gk/rH、db_v 的生命周期；当前两代调度不能原样扩展到完整算子。

## 测量方法

性能库关闭 capture/trace。每例预热 20 次，取 `Stage01Kernel<3>` 最后 10 次 `op_summary` 记录的中位数；只统计该 kernel。MTE2 ratio 是流水线忙碌占比，不是 HBM 峰值带宽利用率。

首个正确基线为 r0b；早期 r0 存在 UB 布局切换竞态，排除在性能对照之外。候选顺序为：r1 两代 GM 槽，r2 提前消费上一 Stage 3，r3 32 行 tile，r4 删除整轮 Scalar 等待，r5 精简输入事件，r6 仅清零 beta 尾块并整理代码。

最终独立重采结果（Stage 0～3 合计）：

| case | 正确基线 / μs | 最终 / μs | 耗时降低 | 最大流水线占比 | 10 次 MTE2 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 定长 8K | 2848.09 | 2005.63 | 29.58% | AIC MTE2 92.35% | 92.2%～92.4% |
| 变长 8K | 2913.97 | 2070.59 | 28.94% | AIC MTE2 92.50% | 92.3%～92.5% |

两个 case 的最终 msprof 均退出 0，各有完整 30 条目标任务；最后 10 次均超过 90%。相同源码此前 r6 采集为 2012.46/2065.29 μs、MTE2 92.2%/92.5%，两次也均正常退出。

最终结果和逐流水线占比见 [performance.csv](performance.csv)。原始测量中部分 msprof 在导出后退出 137，退出状态及日志单独归档，不作为工具成功记录；完整 CSV 仍保留。最终采集另行核验退出状态和全部 30 条目标任务。

## 精度与时间线

CPU FP64/FP32 独立标杆覆盖全部 head/chunk/元素；Stage 0～2 容差不变。Stage 3 同时检查独立全路径和实际捕获操作数的局部结果，规则见[测试说明](../../../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。性能库另检查全部 dv 及两个 GM 槽的最后有效代。

最终 118 次设备检查全部通过：前三阶段单核 tails 共 66 次，Stage 3 单核/多核 tails 共 44 次，两种 gate 的完整 8K 定长/变长 capture 共 4 次，性能库相同四组共 4 次。另通过 Host tiling 和 clang-format 检查，清单见 [accuracy.json](accuracy.json)。完整七输出、ATK 双标杆及 A3 上板不在本轮范围内。

[门槛校准](precision-calibration.json)拒绝全零、结果减半、beta 乘错轴、漏掉三角掩码、错误 head 和 db_v 多乘 beta。

H=2/T=192、单 AIC 的模拟器完整采集一个 Cube 和两个 Vector lane。基线与最终候选均为 36 次 MMAD；最终两个 AIV 的 Stage 3 Vector 指令都与下一 work 的 MMAD 重叠，基线没有该重叠。具体区间及指令计数见 [timeline.json](timeline.json)；模拟器时间只用于证明依赖和重叠，不代替硬件耗时。

源码及动态库 SHA256 见 [provenance.json](provenance.json)。复现入口为 `check_stage01.py --stage 3` 和 `bench_stage01.py --stage 3`；加入 `--varlen` 采集变长性能。
