# Stage 0/1 跨 work 预取

2026-10-11，A2 910B3/device 1，CANN 9.1。目标：B=1/H=96/T=8192/K=V=128 的定长、变长中，至少一条主流水线 busy 达到 90%，同时缩短实际耗时。基线为双缓冲版本 `9899448a`，两者都关闭 capture/trace，保留同样的两阶段开发入口收尾握手。

## 同条件 A/B

每阶段预热 20 次，最后 10 次取中位数。计时为目标 `Stage01Kernel` 的 `Task Duration(us)`；MTE2 为 `op_summary` 的 `aic_mte2_ratio`，不是整卡 HBM 带宽利用率。

| Stage 0+1 case | 基线 / us | 优化 / us | 基线 MTE2 | 优化 MTE2 |
| --- | ---: | ---: | ---: | ---: |
| 定长，T=8192 | 1849.70 | 1574.73 | 66.4% | **90.55%** |
| 变长，总 T=8192 | 1911.00 | 1615.70 | 66.4% | **90.7%** |

定长 MTE2 十次范围 90.5%～90.6%，变长十次均为 90.7%。Stage 0 单独耗时分别为 1689.02→1273.57 us、1728.73→1313.06 us；单独 Stage 0 的 MTE2 为 87.6%/87.7%，不将其表述为超过 90%。完整统计见 [performance.csv](performance.csv)。

变长长度 `[0,1,31,64,65,4096,3935,0]`，定长 128 个 chunk、变长 131 个 chunk，均使用 20 AIC/40 AIV。主输入 BF16、gk FP32，公式、容差、工作量及 workspace 不变。

## 改动和对照实验

参考 [cannbot-skills PR #1069](https://gitcode.com/cann/cannbot-skills/pull/1069) 的 MTE2 preload：在当前 head 的最后一次 L1→L0 读取后，通过原有 free 事件预取下一 work 的 h/dh/v_new/dv_scan/v，共 112 KiB/head。它与另一 head 的 Cube 计算及当前 Vector 工作重叠。akk/dW 保持当前 work 所有权，下一 work 再补搬 akk、发布 input-ready；预取深度为每 owner 一代，末 work 不预取。

- `r1`：仅定长元数据寄存器计算，1791.77 us，未见明显收益。
- `r2`：加入跨 work 预取，初测 1590.59 us、MTE2 90.45%；采用并做上述最终 A/B。
- `r3`：仅用于诊断，关闭开发入口跨核收尾握手，1635.41 us、MTE2 92.6%。占比更高但耗时更长，未采用。

Stage 0/1 的 L1 仍分配 384 KiB。后续 Stage 7 不能沿用覆盖预取输入的旧布局，qPos/bkPos 已在设计中移到 [384,512)，完整 L1 活跃量为 480 KiB，详见 [L1 内存](../../a2a3/L1内存.md)。Stage 7 尚未实现；本报告不代表完整 Finalize 性能。

模拟器用 H=2/T=128、1 AIC+2 AIV、一次 launch 覆盖连续两个 work，两版均完整结束。基线下一 work 首条输入搬运在 10.293 us 发射，晚于当前 work 最后 MMAD 的 10.011 us；优化后提前到 7.954 us，当前 work 最后 MMAD 到 9.936 us 才结束。两版均为 24 条输入 DMA、16 条 MMAD，未增加工作量。见 [时间线证据](cross-work-timeline.json)；时间戳只用于证明调度顺序，不用于推算硬件耗时。

## 精度与产物

原 CPU 标杆及容差不变，70 次检查通过：Stage 0 单核 tails 22 次、Stage 0+1 单核 tails 22 次、多核 tails 22 次、完整 8K 定长/变长各 1 次、关闭 capture 后的完整 8K workspace 定长/变长各 1 次。覆盖奇数 head、空序列、尾块、同核多代输入槽复用及全部八个中间量。

格式化后的最终源码重新构建为 `release-perf.so`，另通过定长/变长单核 workspace smoke 共 4 次，总计 74 次精度检查。该库独立变长复测为 1636.16 us、MTE2 90.6%，仍达到目标；此项不是上表成对 A/B 的替代值。

- 实际精度库 `bound90/r2-capture.so`，性能库 `bound90/r2-perf.so`；源码和库身份见 [provenance.json](provenance.json)。格式化前后代码仅空白变化，并逐文件核对。
- 原始数据：`a2-server:/data/Ensley0304/kda-finalize-a2-stage01-20261011/optimization/bound90`。
- `accuracy.status` 为精度独立结果；`resume.status` 为后续采集结果。首个采集脚本在定长 CSV 已完整导出、进入 query 阶段后被终止，应用已输出 DONE，全部 10 次样本均存在；变长和小 case 随后单独采集完成。
- 最终库复测的 msprof 主进程也在导出后被终止，单独 export 重试仍返回 137，原因未确认。已独立核验应用 DONE、30 条目标 task（20 次预热+10 次采样）及完整 CSV；`release-accuracy.status` 为 PASS，原始失败状态和日志保留，不将工具退出异常写成采集进程成功。
- 同时归档 Memory/L2Cache 组；不把这些聚合计数或 MTE2 busy 等同于 HBM 达峰。
- 构建、精度及 profiling 命令沿用 [测试入口](../../../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。
