# A2 Stage 0～2：双缓冲与跨 work 流水

2026-10-11，Ascend910B3/device 1，CANN 9.1，20 AIC/40 AIV。实现范围仅为三阶段开发入口，公开 A5 调用不变。

## 最终结果

B=1/H=96/T=8192/K=V=128，每例预热 20 次、采样 10 次，统计目标 Kernel 的 `Task Duration(us)` 中位数。两版都执行每 head 六次 MMAD，关闭 capture/trace。

| case | 首版 Stage 0～2 / μs | 最终 / μs | 首版 MTE2 | 最终 MTE2 |
| --- | ---: | ---: | ---: | ---: |
| 定长 | 2042.94 | **1632.68** | 66.25% | **90.6%** |
| 变长 | 2114.02 | **1686.16** | 66.8% | **90.7%** |
| H=3/T=65 定长 | 7.21 | **6.61** | 22.45% | 24.05% |

两个目标 8K case 均超过 90%，耗时分别降低约 20.1%/20.2%。定长十次 MTE2 范围 90.5%～90.6%，变长 90.6%～90.7%。小 case 不要求达到 90%。这里的 MTE2 指 `aic_mte2_ratio`，不是 HBM 带宽达峰比例。

变长长度为 `[0,1,31,64,65,4096,3935,0]`，共 131 个 chunk；定长为 128 个 chunk。统计见 [performance.csv](performance.csv)，库和源码身份见 [provenance.json](provenance.json)。首版基线是加入 Stage 2 后的 r0，不与少做两次 MMAD 的旧 Stage 0/1 混比。

## 调度与候选

- Stage 2 复用 BF16 NZ dW；kE 按两个 head 分别搬入 L1，ready/free 独立。L0A/B/C 和 Stage 1 输入/输出均保留双缓冲，L1/UB/生产 workspace 不扩容。
- 保留 Stage 0 的五输入跨 work 预取，将 kE 搬入放在最后一批预取之前。
- kE 搬入 L1 后释放 GM 槽，使下一 work 的 Stage 1 与当前 Stage 2 重叠。akk 使用独立 MTE1→MTE2 free 事件，最终 Fixpipe 写回在 End 中排空。
- 此提前推进适用于当前三阶段入口。加入 Stage 3～11 时，必须保留仍需消费的 exp2_gk/rH/akk，延后释放或增加双代存储；不能将本报告性能当作完整 Finalize 性能。

参考 [cannbot-skills PR #1069](https://gitcode.com/cann/cannbot-skills/pull/1069) 的预取、双缓冲及延后消费等待原则。固定同一输入和设备，候选定长结果如下；未采用的改动不在最终代码中。

| 候选 | μs | MTE2 | 结论 |
| --- | ---: | ---: | --- |
| r0：首版三阶段 | 2042.94 | 66.25% | 基线 |
| r1：首个 head 后等待 kE | 2293.22 | 55.1% | 阻塞另一个 head，回退 |
| r2：kE 插到最后一批预取前 | 2025.56 | 70.65% | 保留 |
| r3：四个 32 KiB L0C 槽 | 2211.56 | 62.7% | 无收益，回退 |
| r4：收尾从 FIX 移至 MTE1 | 1798.57 | 83.0% | 减少无关等待 |
| r5：akk 双槽预取 | 1798.98 | 81.9% | 无收益，回退 |
| r6：提前单独算 dKgb_raw | 1793.06 | 82.9% | 收益小，未增加拆分 |
| r7：kE 搬入后推进下一 Stage 1 | 1632.09 | 90.6% | 采用；格式化后复测为上表最终值 |

H=2/T=128、单核、两 work 的模拟器完整结束，三个 MIX lane 均存在。基线下一 work 的 AIV 输入在 11.924 μs 才发射，晚于当前 work 最后 MMAD 的 11.382 μs；优化后提前到 10.166 μs，当前 work 最后 MMAD 至 12.778 μs 才结束。两版同为 24 次 MMAD、28 次 AIC 输入 DMA、每 AIV 24 次输入 DMA，没有增加工作量。见 [timeline.json](timeline.json)；模拟器时间只证明重叠关系。

## 精度与复现

最终 capture 库通过 92 次设备检查，性能库再通过 4 次完整模型 shape 的各核最后代 workspace 检查，总计 **96 次**。覆盖 Stage 0/1 回归、Stage 2 单核复用与多核调度、奇数 head、空序列、所有尾长、8K 定长/变长及累计/flat gate。Host tiling 正常与异常分支检查通过。

Stage 0/1 的八个中间量保留原容差。Stage 2 既检查实际 BF16 操作数的严格 FP64 局部 GEMM，也检查独立 CPU 全路径的 BF16 误差传播预算。预算可拒绝全零、半结果及错误 head，见 [precision-calibration.json](precision-calibration.json)。具体公式、门槛和运行命令见[测试入口](../../../../../../../../examples/fast_kernel_launch_example/tests/kda_finalize_stage01/README.md)。这不是完整七输出或 ATK 验收结论。

最终变长首轮 msprof 在导出后退出 137，完整 30 条 task 给出 1686.21 μs、MTE2 90.7%。随后同库独立重采正常退出 0，得到上表的 1686.16 μs、90.7%，对应 `final-varlen-confirm`。保留两轮数据和原始退出码。全部采集退出码见 provenance，原始日志、parsed simulator 和源码快照归档在本地 `validation/stage2-20261011/report`。
