# ChunkKdaBwdIntra 性能分析

## 当前实现

当前稳定 fallback 是无 workspace 的单 launch AIV 路径。核间 task 仍为 `(chunk, HV, 16-token block)`。
除下文严格限定的 rowBlock3 Cube 实验 eligibility 外，`safe_gate=true` 默认进入 block-wise tiling key 5/7：核内按 `BK=32` 搬入整个 chunk 的
q/k/g feature tile，并按 source token 同时更新 16 个目标行。`safe_gate=false` 继续使用
已通过精度回归的逐行 tiling key 0/2；legacy safe key 1/3 保留为回退。

已实施的静态优化：

- q/k/g 每个 `(task, BK)` 仅二维搬入一次并驻留 UB，不再被 16 个输出行重复读取；
- `dq_local` 与 `dk_left_pre` 复用一次 `k*gate`；
- unsafe 分支仅在当前 16×16 对角块直接计算 gate，跨块仍按上游首/末参考点分解；safe 分支的对角 left/right 分别使用末/首方向端点，避免 feature-side gate 放大大 BF16 输入；
- safe 分支的首/尾参考向量直接引用 g feature cache；每个 source/reference gate 只计算一次；
- future `dAkk*beta` 先做 FP32 标量合并；
- `db` 在同一 task 内跨 K tile 归约，避免全局 partial tensor 和第二次 launch；
- 每个 16-token task 只下发四次二维 `DataCopyPad`，一次性搬入 dAqk/dAkk 的
  `[R,BT]` 行 slab 和 `[curT,16]` 列 slab；所有 blockLen 均为 32B 的整数倍，
  不再为每个输出行重复提取 dA 行/列；
- varlen 每个 task 只连续搬运一条 32B packed chunk metadata，不携带大 tiling 数组、不做 device 二分；
- task 写区间互斥，无 atomic、跨核同步或随序列增长的 workspace。
- dA 行 slab 在 task 开头一次整理成 `[source,16]`，列 slab 同步完成 causal 清零和
  `dAkk*beta`；单个 `S_V` 后，source 热循环通过 `Brcb + stride Mul/Add` 消费连续系数，
  不再逐 row 读取标量或触发 `S_V`。
- 输出累积按 `[rowCount,32]` 二维批量搬入/写回，`db` 仍按 row 独立 FP32 ReduceSum。

## 已知 profiling 基线

目标 shape 为 `B=1,T=8192,H=HV=32,K=128,BT=64,BF16,safe_gate=true`：

| 实现 | kernel duration | AIV time | 关键指标 |
|---|---:|---:|---|
| Triton 兄弟仓 | 19.272 ms | 19.272 ms | 28 block，AIV scalar 59.7%，MTE2 49.7% |
| legacy AscendC | 477.937 ms | 455.734 ms | 40 block，AIV scalar 44.9%，MTE2 42.7% |
| AIV block-wise `75535cd` | 48.660 ms | 48.660 ms | clean wheel，原 22 项精度通过 |

外围 layout/cast 总计约 2.45 ms，不是主要差距。方向端点修正不增加 Vector API 调用或搬运，
但其性能仍须重新上板确认，不能直接沿用 `75535cd` 的实测值。

## 理论模型

设有效 chunk 长度为 `L`，单 task 行数为 `R<=16`，则主要工作量近似为：

```text
source vector loads: O(L * K / BK)
Exp vectors:         O((L + R) * K / BK)
FP32 vector FLOPs:   O(R * L * K)
dA GM elements:      2 * (R * BT + L * 16)
```

满 16 行 task 下，相比 legacy，q/k/g source feature 搬运和公共 gate Exp 理论上最多减少
约 16 倍；FP32 乘加总元素量不变，但从大量 16/32 元素短调用改为 `[16,32]` repeat。
这里是静态调用/访问量估算，不替代上板 profiling。

核间并行 task 数为 `chunks * HV * ceil(BT/16)`。长序列和常见多头形状可以覆盖全部 AIV；极小单 chunk/单头 case 会受可并行 task 数限制。

## 待上板确认

当前机器没有 CANN/NPU，以下均是需要 profiling 验证的假设，不是实测结论：

1. block-wise 后 Scalar 绝对耗时是否下降至少 80%；
2. source cache 后 MTE2 绝对耗时是否下降至少 70%；
3. `BT=128` 下单 task 变长是否造成核间尾部不均衡；
4. FP32 Exp 吞吐是否高于 source 搬运压力；
5. BSND/TND Python layout materialization在端到端时间中的占比。
6. dA 一次性 Scalar 重排的成本，以及是否值得再替换为专用 transpose/Gather。

建议采集 kernel duration、AIV utilization、MTE2 bandwidth、Vector utilization、各流水 stall 和 task tail。基准至少覆盖 `(BT,K)=(64,128),(128,128)`、FP16/BF16、dense/varlen、`HV/H=1/2/4` 和 safe/unsafe。

## 下一阶段候选

最新 profiling 仍显示目标 shape 的 Cube utilization 为 0，因此先用 rowBlock3 off-left 建立最小
Cube 路径，不直接恢复曾发生设备超时的 stage 内 AIC/AIV 握手。该切片只替换
`row=[48,64), source=[0,48)` 的两路 left contraction：

```text
concat(dAqk_row3_left, dAkk_row3_left) [32,48]
    @ (K_left * exp2(g_48 - g_left))   [48,128]
    -> (dq_row3_left, dk_row3_left)    [32,128]
```

它覆盖满 `BT=64` chunk 约 18.5% 的 outer-contraction 工作量，但仍保留 diagonal、right 和前三个
rowBlock 的 AIV 计算，所以第一轮目标是证明 Cube 能稳定带来净收益，不预设一次降至 10 ms。

实验 eligibility 固定为 `safe_gate=true`、BF16、dense、`B=1`、`H=HV`、`BT=64`、`K=128`
且三份 scratch 合计不超过 256 MiB
和满 chunk；其他 case 继续使用 key7/legacy。执行由 aclnn executor 在同一 stream 串行排入：

```text
AIV prep -> Cube stage -> AIV consume
```

Cube stage 使用 `KERNEL_TYPE_AIC_ONLY`，host 侧仅启动实际使用的 AIC；HF32 关闭，stage 之间
不使用任何 CrossCore flag。这样用额外 launch 和 GM scratch
换取确定的生命周期，避免复杂 ready/done 事件再次造成死锁。稳定 key7 不删除，也不改变非实验
shape 的调度。

scratch 按 `(chunk,HV)` 分配 46 KiB（A 6 KiB、B 24 KiB、C 16 KiB），目标 shape
`B=1,T=8192,H=HV=32` 共 4096 slot，约 184 MiB，另加平台 libapi workspace。第一版不启用
双 buffer、persistent MMAD、slot overlap 或 workspace alias；这些只在 NPU 精度和 profiling
门禁通过后逐项评估。

完整方向仍可将其余 16-token 子块重写为 FP32 Cube GEMM：

```text
dAqk_left  @ (K * exp2(g_ref_left - g))
dAkk_left  @ (K * exp2(g_ref_left - g))
dAqk_right^T @ (Q * exp2(g - g_ref_right))
dAkk_right^T @ (beta*K * exp2(g - g_ref_right))
```

AIV 负责 gate 预处理和行缩放，AIC 负责 GEMM。后续必须按一个 contraction 切片逐步扩展，不能
在 rowBlock3 canary 之前引入 diagonal、right、double buffer 或同 launch 深融合。当前实验路径
尚未完成 NPU 编译、精度或性能验证；性能结论必须按 prep、Cube、consume 三个 kernel duration
之和统计，并与 48.660 ms 稳定基线比较。
