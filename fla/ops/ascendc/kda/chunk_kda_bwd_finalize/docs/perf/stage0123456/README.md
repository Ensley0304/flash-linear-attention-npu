# Stage 0～6：A2 开发验证

Stage 6 已接入，184 次设备检查通过。**最大流水占比未提升，仍未达到 90%。** 范围是私有七阶段入口，不代表完整 Finalize、ATK 或 A3 验收。

## 实现与同步

`dAkk_raw=akk.T@Tza` 使用 BF16×BF16、FP32 累加。akk/Tza 保持两代 NZ L1 驻留；Stage 4 完成后通过 FIX_MTE1 发布 Tza，Stage 6 搬完操作数才释放 akk。双 head 沿用两个 L0A/B/C 槽，Stage 6 与 Vector Stage 5 并行。新增 FP32 ND 结果写入每代 workspace `[216,232) KiB`，不增加 L1、UB 或 workspace 总量。

生产路径每 head 增加一次 64³ GEMM 和 16 KiB GM 写回，不新增 GM 输入。开发 capture 扩至 384 KiB/head/chunk，其中 `[360,376) KiB` 保存 dAkk_raw；前序偏移不变。

## 精度

[184 次设备检查](accuracy.json)全部通过：Stage 0～5 单核 tails 各 22 次，Stage 6 单核累计 gate / 多核 flat gate tails 各 22 次，以及两种 gate 的 8K 定长/变长 capture 与性能库各 4 次。Host tiling、clang-format 和源码身份核对通过。

Stage 6 的局部 FP64 GEMM 使用设备捕获的 BF16 Tza，门槛为 `rtol=3e-4, atol=2e-6`；独立 CPU 全路径传播既定 Stage 4 BF16 误差预算，没有放宽旧阶段门槛。检查全部 capture 元素、有效尾行/列、有限值、保护区以及各核最后两代真实 workspace。性能库直接检查其保留的两代结果，未声称导出全部中间代。

[CPU 门槛校准](precision-calibration.json)在累计/flat gate、capture/workspace 两种模式下接受正确结果，拒绝全零、减半、两种错误转置、串 head 和串代。

## 性能

A2 Ascend910B3/device 1，20 AIC / 40 AIV，CANN 9.1，torch 2.7.1+cpu / torch_npu 2.7.1.post5。B=1/H=96/T=8192/K=V=128；变长长度 `[0,1,31,64,65,4096,3935,0]`。每组预热 20 次，取 `Stage01Kernel` 最后 10 条 `Task Duration(us)` 的中位数。性能库关闭 capture、trace 和测试专用 L1 验证乘法。

| case | 已验证旧库 Stage 0～5 / μs | 新库 Stage 0～6 / μs | 新库最后 10 次范围 / μs | 最大流水占比：旧 → 新 |
| --- | ---: | ---: | ---: | ---: |
| 定长 8K | 3009.25 | 3042.35 | 3025.58～3054.20 | Vector 75.40% → 74.55% |
| 变长 8K | 3084.79 | 3116.44 | 3113.24～3120.54 | Vector 75.15% → 74.40% |

本轮中位数约增加 1.1%/1.0%，不是 Stage 6 的独立耗时。Stage 6 增加的是 Cube/Fixpipe 工作，Vector 指令数未变；定长 Cube 占比约从 16.3% 增至 17.3%，最大流水仍是 Vector。新增工作大部分与 Stage 5 重叠，但未提高 Vector 的流水占比。

同库截断至 Stage 5 的辅助测量为 3105.07/3176.51 μs，全部保存在 [采集汇总](performance.json) 中；不把七阶段比这个截断版本快解释为消除了新增计算。最终比较使用已验证旧库作为基线，数据见 [选定记录](final-performance.json) 和 [原始 CSV](raw/final6-profile-fixed.csv)。

采集异常：开启 AscendCL 的首轮出现进程被杀及初始化等待，未采用不完整数据。重采统一关闭 AscendCL；旧库两组和七阶段变长采集正常退出。七阶段定长在分析/导出时收到 SIGKILL，最终离线重新导出成功（退出 0）；两次完整采集均有 30 条目标任务，中位数分别为 3042.70/3042.35 μs。异常退出没有记作正常采集，原因尚未定位；所有日志保留。

## 流水证据与复现

H=2/T=192、单 AIC 的完整模拟器记录含 48 次生产 MMAD，一个 Cube 和两个 Vector lane。两个 AIV 均确认 Stage 5 与 MTE2、MTE3、Stage 4 和 Stage 6 MMAD 重叠，见 [时间线摘要](timeline.json)。模拟器只用于检查依赖和重叠，不代替硬件时间。

源码/库身份见 [provenance.json](provenance.json)。完整源码、日志、原始 CSV 和 trace 保存在工作区 `validation/stage6-20261012/report.tar.gz`；服务器目录为 `/data/Ensley0304/kda-finalize-a2-stage6-20261012`。

精度：`check_stage01.py --stage 6 --suite model --cores 0 --repeat 1 --library <capture.so>`；性能库增加 `--workspace-only`，flat gate 增加 `--gate-mode flat`。性能：

```bash
msprof --output=<output> --ai-core=on --aic-metrics=PipeUtilization \
  --task-time=on --ascendcl=off \
  python examples/fast_kernel_launch_example/tests/kda_finalize_stage01/bench_stage01.py \
  --library stage6/r0-perf.so --stage 6
```

变长增加 `--varlen`。下一阶段读取 dAkk_raw 时须新增 Fixpipe ready，不能借用 S5_DONE。
