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
| key12 MIX BK32 | 49.168 ms | 48.990 ms | 单 kernel、28 项通过；Vector 53.3%，Scalar 37.9%，MTE2 3.2% |
| key12 MIX BK64 | 32.477 ms | 32.372 ms | 单 MIX kernel；Vector 14.272 ms / 44.1%，Scalar 16.471 ms / 50.9%，MTE2 1.104 ms / 3.4% |
| key13 MIX BK64 | 31.034 ms | 30.955 ms | source gate batch；较 key12 -4.44%，Vector 12.516 ms / 40.4%，Scalar 17.353 ms / 56.1% |
| key14 MIX BK64 | 31.036 ms | 30.956 ms | row post-scale batch；与 key13 基本持平，未形成净收益 |

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

key12 的 Cube MAC 仅 0.036 ms；`cube_utilization` 不能替代绝对 MAC 时间判断，AIC 的其余墙钟
主要是等待 AIV。与 key7 相比，rowBlock3 Cube 令 Vector 下降约 2.184 ms，但 Scalar、同步和
MIX 调度抵消了收益。当前瓶颈是 AIV Vector + Scalar，不是 MTE，因此不优先做 double buffer。

## BK64、key13 实测与 key14 row post-scale 批处理

BK64 已把 key12 从 49.168 ms 降到 32.477 ms，说明减少 feature-loop 和小 Vector API 发射有效；当前主要瓶颈仍是 AIV Scalar + Vector，而不是 MTE2。key13 在不改变 Cube、workspace、跨核事件、输出累加顺序和 `db` 四段 BK32 归约顺序的前提下，只批处理 source 维 gate：

1. 每组最多 16 个连续 source，共用一次 repeat 形式的 `Sub -> Muls -> Exp`；
2. 每个输出累加器仍按原 source 顺序逐项执行 `OuterAccumulate`，不引入 FMA 或重排同一累加器的加法；
3. 单个 BK64 tile 的 source gate 发射从 272 组降到 17 组；包括未批处理的 row post-scale gate 后，总 gate 组数从 496 降到 241；
4. gate UB 从 256 B 增加到 4 KiB，AIV0 估算总 UB 约 143.9 KiB，仍低于 192 KiB；
5. key12 保留为原 BK64 MIX 回退，host 只需把 stage4 从 key13 切回 key12 即可撤销实验。

key13 同卡 profiling 的 6 次 kernel duration 为 `31.030～31.044 ms`，中位数 31.034 ms，
相较 key12 减少 1.443 ms（4.44%）。Vector 下降 1.756 ms（12.31%），但 Scalar 增加
0.882 ms（5.36%），抵消了接近一半收益；MTE2/MTE3 基本不变。因此下一步不做 double buffer，
而是减少仍由逐行循环下发的 post-scale gate 和 `Mul/Add`。

key14 保持 source-loop、Cube、workspace、跨核事件和 `db` 归约树不变，只批处理 row post-scale：

1. 四个 row block 共 14 个独立 gate family，分别用最多 16 次 repeat 的
   `Sub -> Muls -> Exp` 构造 `[rowCount,64]` gate；
2. 对应 `dqAcc/dkLeft`、`dqDiag/dkLeftDiag`、`dkRight` 和 `dkRightFuture` 使用相同 row repeat
   完成 `Mul/Add`，不引入 FMA；
3. 单个 BK64 tile 的 row post-scale gate 组数从 224 降到 14，总 gate 组数从 241 降到 31；
4. 复用 key13 的 4 KiB gate UB，AIV0 总 UB 仍约 143.9 KiB；
5. 每个输出元素的算术顺序不变，source 累加顺序和 `db` 四段 BK32 归约顺序不变；
6. key13/key12 保留为立即回退，host 只需切换 stage4 tiling key。

key14 同卡实测为 31.036 ms，与 key13 的 31.034 ms 基本持平，因此保留为已编译的无收益实验，
不再继续围绕逐行 Vector API 做小步调整。

## key15 full-Cube

key15 将 previous-left、diagonal-left、diagonal-right 和 future-right 四类 contraction 全部改写为
六次块对角 FP32 `BlockMmadTla`。AIV 只负责 A/B 打包、safe gate 内外因子、`db/dg` 和输出累加。
每个逻辑 AIC 复用一份 600 KiB workspace，20 核约 12 MiB；两个 AIV lane 按 `{0,3}` 与 `{1,2}`
分配 row block，并继续使用一代 ready/done 反转 flag。该版本不启用 double buffer。

key13 仍是立即回退。key15 的首轮验收门槛是完整精度和 100 次 repeated launch 均通过，且目标 shape
kernel duration 低于 22 ms；未取得干净 wheel 的同卡 msprof 前，不声明已经获得性能收益。

建议采集 kernel duration、AIV utilization、MTE2 bandwidth、Vector utilization、各流水 stall 和 task tail。基准至少覆盖 `(BT,K)=(64,128),(128,128)`、FP16/BF16、dense/varlen、`HV/H=1/2/4` 和 safe/unsafe。

## 下一阶段候选

稳定 key7 profiling 显示目标 shape 的 Cube utilization 为 0，因此先用 rowBlock3 off-left 建立最小
Cube 路径。该切片只替换
`row=[48,64), source=[0,48)` 的两路 left contraction：

```text
concat(dAqk_row3_left, dAkk_row3_left) [32,48]
    @ (K_left * exp2(g_48 - g_left))   [48,128]
    -> (dq_row3_left, dk_row3_left)    [32,128]
```

它覆盖满 `BT=64` chunk 约 18.5% 的 outer-contraction 工作量，但仍保留 diagonal、right 和前三个
rowBlock 的 AIV 计算，所以第一轮目标是证明 Cube 能稳定带来净收益，不预设一次降至 10 ms。

实验 eligibility 固定为 `safe_gate=true`、BF16、dense、`B=1`、`H=HV`、`BT=64`、`K=128`
且 workspace 不超过 256 MiB 和满 chunk；其他 case 继续使用 key7/legacy。已验证的三 launch
canary 仅用于确认 FP32 MMAD 数值，其总 kernel duration 约 46.011 ms，净收益有限且不满足交付
要求。当前实现改为一次 MIX launch：

```text
AIV0 pack A/B + rowBlock0  ┐
AIV1 rowBlock1/2           ├─ 与 AIC FP32 Cube 重叠 -> AIV0 rowBlock3
AIC  wait ready -> Cube    ┘
```

key12/key13/key14 使用 `KERNEL_TYPE_MIX_AIC_1_2`，host `blockDim=min(slot_count,AIC count)` 并设置 MIX
schedule mode。两个 AIV 子核共同提交 ready，AIC 完成后广播 done；双方按相同的逻辑 core 和
slot 步长推进。HF32 关闭，稳定 key7 不删除，也不改变非实验 shape 的调度。

scratch 按 `(chunk,HV)` 分配 46 KiB（A 6 KiB、B 24 KiB、C 16 KiB），目标 shape
`B=1,T=8192,H=HV=32` 共 4096 slot，约 184 MiB，另加平台 libapi workspace。第一版不启用
双 buffer、persistent MMAD、slot ring 或 workspace alias；这些只在单 kernel NPU 精度和
profiling 门禁通过后逐项评估。

BK64 key12 已证明缩短 feature-loop 能形成净收益，key13 又证明 source-gate 批处理能稳定减少
Vector 时间。当前唯一实验变量是 key14 的 row post-scale gate/Mul/Add 批处理：每个 gate family
对最多 16 个输出行做 repeat，随后仍按原阶段顺序更新独立行。key7、key12、key13、独立三阶段
诊断、Cube MMAD、scratch 和 flag 协议均不变；`db` 在每个 BK64 tile 内仍分两次 BK32
ReduceSum，维持四段累加顺序。

完整方向仍可将其余 16-token 子块重写为 FP32 Cube GEMM：

```text
dAqk_left  @ (K * exp2(g_ref_left - g))
dAkk_left  @ (K * exp2(g_ref_left - g))
dAqk_right^T @ (Q * exp2(g - g_ref_right))
dAkk_right^T @ (beta*K * exp2(g - g_ref_right))
```

AIV 负责 gate 预处理和行缩放，AIC 负责 GEMM。后续必须按一个 contraction 切片逐步扩展，不能
在 key14 编译、完整精度和 profiling 通过前引入更大的 Cube 切片、double buffer 或 workspace
ring。profiling 必须仍只出现一条 KDA MIX kernel 记录，并与
48.660 ms key7、49.168 ms BK32 key12、32.477 ms BK64 key12 以及 46.011 ms 三 launch canary 比较。
