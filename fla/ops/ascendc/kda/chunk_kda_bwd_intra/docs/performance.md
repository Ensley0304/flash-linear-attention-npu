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
六次块对角 FP32 单 tile `TileMmadTla`。六个 contraction 的最大 M/N/K 为 `128/128/96`，
统一使用一个 `128x128x128` 容量的 L1/L0 buffer，不进入 BlockMmad 多层循环。MMAD 与
Fixpipe copyout 均显式使用 `unitFlag=0b11`，并以 `M_FIX/FIX_M` 闭合每次 L0C 生命周期。

key17 的首版复用 key15 `DirectMmad` 后在 A2 触发 L0B 同址读写冲突；改为
`BlockMmadTla<MmadPingpong>` 后，96/128 行的大矩阵又进入了 key10 未覆盖的
多 M/N/K tile 事件循环并在 endpoint 门禁超时。当前 key17 按 PR190 的显式
TileMmad 完成协议，将 M/N 分解为 `64x64` 单 tile，K=96/64 保持完整。
计算映射、FP32 输入输出、点积归约顺序与 key13 回退路径不变。
AIV 只负责 A/B 打包、safe gate 内外因子、`db/dg` 和输出累加。
每个逻辑 AIC 复用一份 600 KiB workspace，20 核约 12 MiB；两个 AIV lane 按 `{0,3}` 与 `{1,2}`
分配 row block，并继续使用一代 ready/done 反转 flag。该版本不启用 double buffer。

clean isolated wheel 的 endpoint guard 已在 key15 上超时，因此 host 已恢复分派 key13，key15
只保留为隔离实验。重新启用的门槛是 endpoint、完整精度和 100 次 repeated launch 均通过，且目标
shape kernel duration 低于 22 ms；未取得干净 wheel 的同卡 msprof 前，不声明已经获得性能收益。

建议采集 kernel duration、AIV utilization、MTE2 bandwidth、Vector utilization、各流水 stall 和 task tail。基准至少覆盖 `(BT,K)=(64,128),(128,128)`、FP16/BF16、dense/varlen、`HV/H=1/2/4` 和 safe/unsafe。

## key19 PR190-style MIX full-Cube

key16/17/18 的三 launch split 路径未解决总体性能目标，且手写 TileMmad 状态机在 endpoint
门禁上仍有风险，因此只保留为诊断实例。public 目标 shape 改用独立 key19，不修改已验证 key13。

key19 直接采用 PR190 已运行的完整外层结构，而不是只复制事件 API：

- 单次 `MIX_AIC_1_2` launch，host `blockDim=min(totalChunks,AIC count)`；
- AIC/AIV 使用同一逻辑 core 映射并按 chunk 步进；
- HV 两个 head 为一个窗口，四个 per-core slot 按 `0/1`、`2/3` 双窗口复用；
- AIV 先完成窗口内两个 head 的 A/B，再按顺序等待并消费 C；
- 使用 PR190 的普通 CrossCore flag id 2/4 和 `0x2` 双 AIV 汇合/广播；
- Cube 采用仓内 `chunk_kda_fwd` 的 IEEE-FP32
  `BlockMmadTla<MmadPingpong<..., true, false>>` 配置。

每 slot 600 KiB，四槽、20 个逻辑 AIC 共约 46.9 MiB。该结构同时去除原 split 路径约
184 MiB 的 task-sized scratch 和三次 launch。当前只声明源码实现完成；在 clean wheel 的
endpoint、repeated-launch、完整精度和 msprof 通过前，不声明性能收益。

## 下一阶段候选

先验证 key19 的结构迁移，不再同时引入新的数学重排或更复杂的流水：

1. clean 单算子编译，确认 key19 MIX 实例和设备对象进入 wheel；
2. endpoint guard 单测必须在 45 秒内结束；
3. 连续 launch 与完整 37 项精度回归通过，容差不放宽；
4. msprof 确认一次 Python 调用只有一条 KDA MIX kernel，并同时具有 AIC/AIV 时间；
5. 与 key13 的 31.034 ms kernel 基线做同卡、同 shape A/B。

只有上述门禁全部通过，才继续根据 profiling 决定下一步。如果 AIV 仍为主瓶颈，优先把
`PackBGroup` 的重复 q/k/g 搬运改为 PR190 式 resident 输入；如果 AIC 出现明显空泡，再复用
PR190 的 L1/L0 ping-pong，而不是另写一套事件协议。
