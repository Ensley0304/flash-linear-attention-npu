# Stage 0～4：A2 双缓冲流水

范围：私有五阶段入口，A2 Ascend910B3、20 AIC / 40 AIV、CANN 9.1；B=1/H=96/T=8192/K=V=128。变长长度 `[0,1,31,64,65,4096,3935,0]`。不代表完整 Finalize、ATK 或 A3 已验收。

## 实现

- Stage 4 计算 `Tza=Zb@akk.T`：BF16 输入、FP32 累加，F322BF16 直接写 NZ L1。
- 每 head 两代 akk 位于 `[0,8)` / `[56,64)` KiB；两代 Tza 位于 `[48,56)` / `[176,184)` KiB。L1 总量仍为 384 KiB，生产 workspace 不增加。
- 当前 work 完成 Stage 2 后，消费上一代 Stage 3 的 ACK，预取两个 head 的 Zb，再用两个 L0 槽计算 Stage 4。末轮排空两代；akk 在最后一次 L1 读取后才能覆盖。
- Zb 复用已释放的 kE ready/free 事件；MTE1→MTE2 共 8 个 ID。Stage 5/6 接入时仍须延长 exp2_gk/rH、db_v、akk/Tza 生命周期。

代码沿用现有 Stage 方法和搬运原语，仅增加 `LoadZb/Stage4`，使用仓库 clang-format 风格。

## 性能

性能库关闭 capture/trace 和测试专用 L1 验证乘法。每例预热 20 次，取 `Stage01Kernel<4>` 最后 10 次 `op_summary` 的中位数。MTE2 是流水线忙碌占比，不是 HBM 峰值带宽利用率；下述耗时为 Stage 0～4 合计。

首个正确版本即满足目标：定长 2020.83 μs、MTE2 91.5%；变长 2080.54 μs、MTE2 91.6%。两组最后 10 次均超过 90%，msprof 均退出 0。最终独立重采见 [performance.csv](performance.csv)，不将较少阶段的旧结果当成同工作量优化基线。

同一源码、同一性能库的最终独立复测：

| case | Stage 0～4 / μs | 最大流水线占比 | 最后 10 次范围 |
| --- | ---: | ---: | ---: |
| 定长 8K | 2036.11 | AIC MTE2 91.2% | 91.2%～91.4% |
| 变长 8K | 2102.43 | AIC MTE2 91.4% | 91.4%～91.5% |

两组 msprof 均退出 0，各有完整 30 条目标任务。两个轮次均满足目标；本轮没有继续增加无收益的调参候选。

## 验证与复现

精度版逐元素检查独立 CPU 全路径和使用实际 Zb 的局部标杆，沿用既定 BF16/FP32 容差。测试专用 `Tza_L1@akk.T` 直接消费实际 L1 数据，核对 NZ 布局和同步；性能版不执行此乘法。性能库另检查全部 dv 和两个 GM 槽的保留代，未直接导出 Tza。

检查清单见 [accuracy.json](accuracy.json)。门槛校准拒绝全零、减半、错误转置、错误 head 和错误代次，见 [precision-calibration.json](precision-calibration.json)。

140 次设备检查全部通过：Stage 0～3 单核 tails 各 22 次；Stage 4 单核累计 gate / 多核 flat gate tails 各 22 次；两种 gate 的 8K 定长/变长全量 capture 共 4 次，性能库相同四组共 4 次。另通过 Host tiling 与 clang-format 检查。

H=2/T=192、单 AIC 的完整模拟器记录包含一个 Cube、两个 Vector lane，共 42 次 MMAD，恰好为 3 work × 2 head × 7 次生产矩阵乘。六次 Tza 写入地址按两代轮换，均直接写 L1；两个 AIV 仍与下一 work 的 MMAD 重叠，见 [timeline.json](timeline.json)。模拟器只验证指令与依赖，不代替硬件耗时。

源码、动态库身份及采集退出状态见 [provenance.json](provenance.json)。复现使用 `check_stage01.py --stage 4` 和 `bench_stage01.py --stage 4`；完整 8K 精度加 `--suite model --cores 0 --repeat 1`，性能变长加 `--varlen`。
