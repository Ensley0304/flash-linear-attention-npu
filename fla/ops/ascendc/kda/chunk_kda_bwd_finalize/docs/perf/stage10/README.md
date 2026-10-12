# A2 Stage 0～10 开发验证

基线为 `b688ff97`（Stage 0～9 r11）。本轮只加入 Stage 10，不改变公开 A5 ABI；Stage 11～12 尚未实现。

## 实现

- `right_raw = dAqk_lower[:,s:e].T @ qPos + dAkk_lower[:,s:e].T @ bkPos`。
- 两项各保留 `Ah@Bh + Ah@Bl + Al@Bh`，六次 MMAD 在 FP32 L0C 累加后写回。两个 head 使用独立 L0A/B/C 槽，保留双缓冲。
- L1 保留下三角矩阵；qPos、bkPos 依次复用已消费的 kNeg 区。L1 512 KiB、每 AIV UB 192 KiB、每 AIC workspace 1280 KiB 均不增加。
- 两轮带间握手保护 GM：第一轮传 kNeg，第二轮传 qPos/bkPos 并等待 right_raw 写完。Stage 9 与 Stage 10 无结果依赖，可以重叠；下一带不能提前覆盖。
- right_raw 复用 workspace 96 KiB 起的旧 kNeg 槽，开发入口保留最后 band 0。capture 单独保存两带，每 head/chunk 为 816 KiB。

## 精度

采数/性能库分别构建，前序门槛不变。已通过 8 次 Stage 10 smoke、132 次三种 gate×Q norm tails、40 次 Stage 0～9 smoke、44 次性能库 tails。覆盖奇数 head、空序列、尾行、单核多 work 回绕及实际 dq/dv 输出。

Stage 10 对照实际六项 BF16 乘积与独立 FP64 居中公式，检查全部元素、零填充、未计算带保护区和保留 workspace。局部门槛为 `3e-6 * Σ(abs(Ai) @ abs(Bi)) + 2e-38`，分带沿用 Stage 8 的操作数传播预算；零输出、漏分支、漏残差、符号、缩放和转置错误均被校准用例拒绝。

完整 8K 定长/变长×Q norm 开关共四组模型在采数版、性能版均通过。全部 head/chunk 的 right_raw 通过 FP64 检查；前序 capture（前 784 KiB）、dq/dv 与已验证 Stage 9 基线逐位一致。trace 仅将阶段标签 10 改回 9 后比较，其余字段逐位一致。性能库的实际 dq/dv、保留 dg/right_raw 与采数版逐位一致。

Host tiling、格式及本地/远端源码 SHA256 检查通过。模拟器 H=2/T=192/单核三个 work 正常结束，完整包含一个 AIC、两个 AIV；192 次生产 MMAD 与设计一致（Stage 10 新增 72 次）。原始时间线保留在服务器，不以模拟器时长代替硬件性能。

## 性能

A2 Ascend910B3，device 3；PyTorch 2.7.1+cpu / torch_npu 2.7.1.post5，CANN 使用 `/data/wys/Ascend/cann_9.1.0_910b_0605/cann`。B=1、H=96、T=8192、K=V=128，Q norm 关闭；变长切分 `[0,1,31,64,65,4096,3935,0]`。关闭 capture/trace，20 次预热、10 次采样，统计 `Stage01Kernel` 的 Task Duration 中位数。

| case | Stage 0～9 | Stage 0～10 | 增量 | 最大流水占比（9 → 10） |
| --- | ---: | ---: | ---: | --- |
| 定长 | 7530.495 μs | 8498.790 μs | +968.296 μs（12.86%） | Vector 65.30% → 57.45% |
| 变长 | 7643.640 μs | 8604.525 μs | +960.885 μs（12.57%） | Vector 65.20% → 57.70% |

四次 msprof 均正常结束，各有 30 条目标记录。Stage 10 增加了计算与搬运，总耗时上升；没有提高最大流水占比，也未达到 80% bound。前缀耗时差不能当作独立 Stage 10 耗时。

## 复现与边界

在 `/data/wys/kda-finalize-a2-stage9-20261012` 下执行归档的构建、精度、模型和 profiling 脚本。源码与动态库 [SHA256](evidence/sha256.json)、[精度记录](evidence/r1-check.json)、[模型结果](evidence/model-capture.json)、[性能结果](evidence/r1-profile.json)、[基线结果](evidence/baseline-profile.json)、原始 op_summary 及日志在 [evidence](evidence/)；大体积动态库和模拟器原始数据留在服务器 `stage10/`。

这是分阶段开发验证，不代表完整 Finalize、ATK、A3 或正式合入验收。
