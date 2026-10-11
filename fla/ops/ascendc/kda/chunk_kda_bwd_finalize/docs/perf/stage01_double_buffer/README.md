# Stage 0/1 双缓冲性能记录

2026-10-11，A2 910B3/device 1，CANN 9.1。基线为 `ecb0768b` 的关闭采数版本；优化版使用同一输入脚本、设备和采数配置。两者均保留两阶段开发入口的收尾握手，不代表完整 Finalize。

每阶段预热 20 次、采样 10 次，仅统计目标 `Stage01Kernel`。下表为 `Task Duration(us)` 中位数，全部 min/median/mean/max 及 AIC 时间见 [performance.csv](performance.csv)。

| Stage 0+1 case | 基线 / us | 优化 / us | 耗时减少 |
| --- | ---: | ---: | ---: |
| B=1/H=96/T=8192，定长 | 2678.80 | 1798.35 | 32.9% |
| H=96/总 T=8192，变长 | 2747.55 | 1859.79 | 32.3% |
| B=1/H=3/T=65，定长 | 8.09 | 5.64 | 30.3% |

K=V=128；主输入 BF16、gk FP32；变长长度 `[0,1,31,64,65,4096,3935,0]`。大 case 使用 20 AIC/40 AIV，小 case 使用 4 AIC/8 AIV。定长大 case 的 Stage 0 单独耗时 1681.48 → 1684.21 us，基本持平。

最终定长 case 的 AIC MTE2/Cube 占比分别 64.75%/18.25%；AIV MTE2/Vector/Scalar 分别 49.8%/42.6%/54.6%。没有单条流水线达到本轮采用的 70%–80% 高占用筛查阈值，也不据此断言已达到整卡峰值。UB bankgroup/bank 冲突占比分别 3.9%/4.6%。

## 双缓冲重叠证据

`msprof op simulator --soc-version=Ascend910B3`，B=1/H=2/T=64，1 AIC+2 AIV，仅一次 Stage 0+1 launch。基线、最终版均完整结束，三个核的指令 lane 均存在。

| 实际计算/搬运指令的重叠 | 基线 / us | 最终版 / us |
| --- | ---: | ---: |
| AIV0 MTE2 + Vector | 0 | 2.014 |
| AIV1 MTE2 + Vector | 0 | 2.161 |
| AIV0 MTE3 + Vector | 0 | 0.166 |
| AIV1 MTE3 + Vector | 0 | 0.069 |
| AIC MTE2 + Cube | 0.282 | 0.464 |

按真实搬运、算术指令区间合并后求交，排除 flag/barrier/寄存器配置指令；详细数据见 [trace-overlap.json](trace-overlap.json)。模拟器仅用于验证调度重叠，不用于推算硬件延时。

## 复现与产物

- [测试入口与命令](../../../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。
- [provenance.json](provenance.json)：参考 PR head、源码及实际加载库 SHA256。本地六个实现/测试文件与服务器源码逐一核对一致。
- 原始目录：`a2-server:/data/Ensley0304/kda-finalize-a2-stage01-20261011/optimization`；最终 profile 为 `verified-{baseline,optimized}-{fixed,varlen,small}`，精度库 `verified-capture.so`，性能库 `verified-perf.so`。
- 本地 CSV、日志、基线与最终版的 `visualize_data.bin/trace.json`：`C:/代码/kda_v2_finalize/validation/stage01-double-buffer-20261011`。

精度覆盖和未采用的实验见 [validation.md](../../validation.md)。
