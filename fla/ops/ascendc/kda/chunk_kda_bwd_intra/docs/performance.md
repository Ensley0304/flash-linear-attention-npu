# ChunkKdaBwdIntra 性能分析

> 2026-07-22 correction: the source-default A2 path remains the two-slot GM A/B bridge. The 2.813329 GB A/B-on-chip figure is only an architectural lower-bound model; it is not attainable by calling AscendC `UB -> TSCM` on DAV_2201 because that route is software-emulated through GM and Matmul KFC. The supported target-shape model is therefore still 5.078254 GB of issued GM traffic (1269.563 GB/s at 4 ms). A realistic AIC stage-local L1 cache would remove only repeated A reads: 0.083886 GB, or 1.65% of the current target traffic; every B image has one consumer and C must still return to AIV. Reaching 4 ms must come from fewer staged operands/results, more reuse inside each AIV or AIC, coarser fusion, and better overlap—not from labeling the emulated TSCM route as on-chip. The target remains unverified until clean-wheel msprof.

## 当前实现

当前保留两层实现。通用 shape 使用单 launch AIV block-wise key 5/7，核间 task 为
`(chunk, HV, 16-token block)`；`safe_gate=false` 继续使用已通过精度回归的逐行 key 0/2，
legacy safe key 1/3 保留为回退。DAV_2201（A2/A3）目标域
`BF16 + safe + BT64 + K128 + dense full chunk + H=HV` 选择 grouped key 23：每个
`(chunk, HV)` 由 1 AIC + 2 AIV 协作，按四个 late-block stage 流水两个固定 workspace
slot。当前设备验证目标是 A2，A3 仍需独立构建和上板验证。pair-wise mixed key 15 仅保留为
源码/重编级回退，不能在已构建产物中运行时切回。
grouped 源码当前以 `KDA_GROUPED_OVERLAP_STAGE_EPILOGUE=false` 为默认，即 32-row
whole-tail；`true` 是尚待设备 A/B 的 8-row stage-local overlap 候选。
`KDA_GROUPED_OVERLAP_TASK_STORE=false` 同样保留串行 task 写回；`true` 是依赖
`tail-blocks=batch` 与 whole-tail epilogue 的跨 task 输出重叠候选。

已实施的静态优化：

- q/k/g 每个 `(task, BK)` 仅二维搬入一次并驻留 UB，不再被 16 个输出行重复读取；
- `dq_local` 与 `dk_left_pre` 复用一次 `k*gate`；
- unsafe 分支仅在当前 16×16 对角块直接计算 gate，跨块仍按上游首/末参考点分解；safe
  分支进一步对对角块使用中点 reference；
- safe 分支的首/中/尾参考向量直接引用 g feature cache，每个 source/reference gate 只计算一次；
- grouped safe 路径按 late block 合并 off-left，把 10 次 ready/done 交接和 20 次 GEMM
  调用降为 4 次交接、17 次 GEMM；off-right 按 early block 独立执行，以保留各自末 token
  reference；对角 midpoint reference 独立保留；
- 每个 stage 只搬入一次 dA row slab，并分别写出 compact `Aoff` 与 `Adiag`。off-right 将
  `Aoff` 的相同地址解释为带父 stride 的 ColumnMajor 转置子视图，不执行第二次 GM 读取或
  UB Scalar 转置；
- `beta*k` 在 task feature load 后以 FP32 预计算并驻留 UB，四个 stage 复用该 cache；
- grouped AIV 的 `(batch, chunk, head)` 坐标只在每核入口做一次 mixed-radix div/mod；固定
  `usedCoreNum` 步长在热循环内改用预计算的 head/chunk/batch 进位递推。目标 shape 从每个
  logical task 重复解析，降为每个物理 AIV 解析一次，且不改变 task 或浮点运算顺序；实际
  Scalar cycle 收益仍需 key 23 profile 证明；
- 六类 AIV `HardEvent` 各在 kernel 生命周期内分配一次并跨 task 复用，替代每个短同步辅助函数
  中的 `AllocEventID/ReleaseEventID`；所有 `SetFlag/WaitFlag` 数量、顺序和 dA/gate overlap
  保持不变。pair-scratch ping-pong 打开时 `MTE3_V` 也只同时占用三个 ID；事件管理的 Scalar
  收益和 repeated-launch 安全性仍需 key 23 clean build/profile 证明；
- stage 0 ready 发布后再构造 stage 1 首次使用的持久 right-outer、pair bridge 并清零三个
  accumulator，使这些 AIV 指令与 stage-0 Cube 计算重叠；该调度由编译期常量控制且不改变
  浮点运算顺序；
- 编译期 `KDA_GROUPED_OVERLAP_STAGE_EPILOGUE=true` 候选在每次 `ConsumeStage` 后完成当前
  8-row block 的 `db` K-reduce 和 `dk_left_pre*beta`，使 stage 0--2 的 epilogue 有机会与
  后续 AIC stage 重叠；源码默认 `false` 继续在 `StoreTaskOutputs` 执行原 32-row whole-tail，
  两条实现都保留，候选收益尚未上板证明；
- M16/M32 与 K16/K32 的 FP32 BlockMmad 使用 `MmadPingpongTlaMulti` 管理共享 L1/L0；
  同一 stage 的 off-right 与 diagonal-right 共用一个 pre-set/final-wait 包络，使 17 个 GEMM 只需
  11 个 MMAD 事件包络；结果完成性由随后同一 `PIPE_FIX` 上的 done flag 保证；
- 源码默认 `KDA_GROUPED_PERSISTENT_MMAD_ENGINES=true`，把 left32/right16 分别映射到互斥的
  L1/L0/event 分区，并把 11 个每-task 包络变为每个 AIC `Process` 的两个常驻包络；`false`
  保留 scoped 回退，两条路径的 GEMM、workspace、slot 和 `PIPE_FIX` done 顺序一致；
- stage 3 off-left 使用静态 K32 加 K16 tail，保留 `2^24 + (-2^24) + 1` 的 FP32
  cancellation 顺序；三个 off-right 使用独立 M16 与 stride-48 ColumnMajor 子视图，避免
  公共 reference 导致 FTZ，也避免 FIX 跨 pair 写入；
- `q/k/g/beta` 和首/中/末 reference 分别使用 7 次二维 `DataCopyPad` 搬入；q/k 以一次连续 Cast
  提升到 FP32，避免四个 block 分别启动 DMA/Vector；
- 十条实际消费的 reference 以 first/middle color 8、right color 12 驻留；right-outer、left-outer
  与 diagonal gate 使用 mask-64、repeat-8 的 `Sub` 直接广播 reference，保持原 `Sub` 的左右
  操作数顺序，同时避开 color-4 `gCache` 和 pair-bridge 两路 reference 的同 bank-group 双读；
  不再生成 8x128 reference 副本；
- 不再构造无消费者的 stage-0 left-outer 与 block-3 right-outer gate；连续 gate bank 合并为
  6 次 `Exp` API；六个 off-diagonal pair 仅各自预计算一条 128 元素 bridge，再用
  `src0RepStride=0` 的 mask-64 `Mul` 直接跨 8 行读取 bridge 并与持久 right/left outer gate
  相乘，每个 AIV 的指数元素从 28×1024 降为 15,104；该直广播避免为左右 inner gate 各复制
  一份 8x128 bridge；
- 每个 off-right pair 的三路 gated feature 先全部留在 scratch，再由一个 V/MTE3 事件包络批量
  写入 workspace，避免三次独立的 Vector→GM 交接；
- AIV 消费同一 stage 的 off-right C 时，`KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME=true`
  把等距的 1/2/3 个 GM row tile 用一次 `DataCopyExtParams` 搬到连续的三块 scratch，并对连续的
  `dkRightAcc`、C 和 `RightOuterGate` 发射一次 FP32 `MulAddDst`。三个非零 stage 的 MTE2 与
  MulAddDst API 数均从每个 AIV 的 `1+2+3=6` 降为 3；目标 shape 两个 AIV 合计各减少 24,576
  次 API 启动。搬运字节、每个元素的操作数、FP32 指令和 destination 均不变，因此这是发射合并，
  不是 reference 重关联或 GM 流量缩减；
- 生产每个 off-right/diagonal-right B 时，`KDA_GROUPED_COALESCE_RIGHT_B_WRITES=true` 利用
  UB 中相邻的 `q*gate`、`beta*k*gate` 两块 8x128 scratch，以一次带 GM destination byte-gap
  的 CopyOut 写入 32x128 右操作数。10 个右 B、两个 AIV 在目标 shape 的 MTE3 API 数从
  163,840 降为 81,920，0.671089 GB 写入字节保持不变；
- off-left 与 diagonal-left 的 32x128 C 也按每个物理 AIV 的两段 8 行窗口合并读取：7 个左 C
  在目标 shape 的 MTE2 API 数从 114,688 降为 57,344，0.469762 GB 读取字节不变；dq 与
  dk-left 的两条 `MulAddDst` 仍独立执行，不改变 accumulator 地址或运算次序；
- `db` 在同一 task 内跨 K 归约，使用 64+64 两级 `WholeReduceSum` 生成 32 个紧凑结果，避免
  全局 partial tensor、第二次 launch、逐行 Scalar compact 和逐行 `ReduceSum` 临时区；
- 每个 16-token AIV fallback task 只下发四次二维 `DataCopyPad`，一次性搬入 dAqk/dAkk 的
  `[R,BT]` 行 slab 和 `[curT,16]` 列 slab，不再为每个输出行重复提取 dA 行/列；
- varlen 每个 task 只连续搬运一条 32B packed chunk metadata，不携带大 tiling 数组、不做
  device 二分；
- AIV fallback 的 task 写区间互斥，无 atomic、跨核同步或随序列增长的 workspace；
- dA 行 slab 在 task 开头一次整理成 `[source,16]`，列 slab 同步完成 causal 清零和
  `dAkk*beta`；单个 `S_V` 后，source 热循环通过 `Brcb + stride Mul/Add` 消费连续系数；
- AIV fallback 的输出累积按 `[rowCount,32]` 二维批量搬入/写回，`db` 仍按 row 独立 FP32
  ReduceSum。

## 已知 profiling 基线

目标 shape 为 `B=1,T=8192,H=HV=32,K=128,BT=64,BF16,safe_gate=true`：

| 实现 | kernel duration | AIV time | 关键指标 |
|---|---:|---:|---|
| Triton 兄弟仓 | 19.272 ms | 19.272 ms | 28 block，AIV scalar 59.7%，MTE2 49.7% |
| legacy AscendC | 477.937 ms | 455.734 ms | 40 block，AIV scalar 44.9%，MTE2 42.7% |
| AIV phase1 | 48.660 ms | 47.508 ms | Vector 59.6%，Scalar 37.2%，MTE2 2.9%，Cube 0% |

phase1 的 Python 端到端中位数为 51.107 ms，外围 layout/cast 约 2.45 ms。MTE2 已不是主瓶颈，
继续只优化 GM 搬运无法逼近 4 ms。grouped key 23 尚未上板，本文不把静态估算写成实测收益；
在取得 key23-only clean build、精度回归与 msprof 证据前，可引用的性能结论仍只有上述 phase1
数据。

## 理论模型

设有效 chunk 长度为 `L`，单个 AIV fallback task 行数为 `R<=16`，则主要工作量近似为：

```text
source vector loads: O(L * K / BK)
Exp vectors:         O((L + R) * K / BK)
FP32 vector FLOPs:   O(R * L * K)
dA GM elements:      2 * (R * BT + L * 16)
```

满 16 行 task 下，相比 legacy，q/k/g source feature 搬运和公共 gate Exp 理论上最多减少约
16 倍；FP32 乘加总元素量不变，但从大量 16/32 元素短调用改为 `[16,32]` repeat。这里是
静态调用/访问量估算，不替代上板 profiling。AIV fallback 的核间并行 task 数为
`chunks * HV * ceil(BT/16)`；极小单 chunk/单头 case 会受可并行 task 数限制。

key 15/23 都把 64×64 causal 区域保持为 10 个 16×16 block pair 的数学分区。key 23 将
同一 late block 的 off-left 项合入大矩阵，而 off-right 按 early block 保持独立 M16，名义
计算量仍为 5.369G MAC（10.737 GFLOP）；stage 3 off-left 的 K32+K16 tail 可能产生额外补齐
周期，必须以 Cube profile 为准。4 ms 对应约 1.342 TMAC/s。四类有效 causal 计算为
4.362G MAC；pair-bridge 因式分解后的目标 shape gate 预处理约为：

```text
15,104 Exp elements/AIV * 2 AIV/logical-task * 4096 logical tasks
= 123,731,968 FP32 Exp elements（约 123.7M）
```

每个 AIV 的 15,104 个元素由 3,072 个持久 right-outer、11,264 个 stage left/diagonal gate
和 768 个 pair bridge 组成。相对 phase1 的约 285.2M 个 Exp 元素减少约 56.6%，相对 bridge
优化前 grouped 方案的 218.1M 减少约 43.3%；实际 `Exp` API 启动数为每 AIV 6 次。这里仍是
静态计数，新增 bridge 乘法的代价和 FP32 舍入/FTZ 必须上板验证。理论唯一 GM 读写下限约
1,155 MiB，4 ms 对应约
302.8 GB/s，因此算量、AIV Exp/标量发射和跨核同步仍需分别确认。

每个 grouped workspace slot 为 26,624 个 FP32，即 106,496 B（104 KiB），双槽每逻辑 AIC
为 208 KiB；workspace 总量只随实际使用的 AIC 核数增长，不随 `T` 增长。两个 AIV 在
slot n+1 准备数据时，AIC 计算 slot n，AIV 随后消费结果并完成外层 gate、`db` reduce 和
四个输出累加。32 行 beta 在 feature load 时一次 `Brcb` 后常驻 `REDUCE_UB`，同时服务
`k*beta` cache 和最终 `dk_left_pre*beta`，默认 whole-tail 不再重复广播；目标 shape 静态减少
8,192 次 `Brcb` 及其 Vector barrier。源码默认把 `db` reduce 与最终缩放放在全部 stage 之后；
候选把它们按 8 行前移到各 stage consume 之后，只有前三个 block 可能被后续 AIC 覆盖。候选
另外复用 reference cache 前后的着色空隙，不增加 179,936 B UB 上限。该机制是
GM 双槽手工 ping-pong。
源码另保留 `KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH` 局部候选：它只为三块 4 KiB pair scratch
增加 alternate set，使 UB 增至 192,256 B，并把上一组 gated-feature 的 MTE3 drain 与下一组
Vector 计算重叠；默认 `false` 仍是 179,936 B 单 scratch。目标 shape 每 task 有 6 个 pair、两个
物理 AIV，候选覆盖的 pair MTE3 数据为 147,456 B/task，即全 shape 的 603,979,776 B；同 scratch
复用前的 pair completion wait 从 6 次/AIV/task 降到 3 次，目标总数从 49,152 降到 24,576。
这些字节只是“可与后续 Vector 重叠”的上限，GM 总字节不变，实际隐藏比例必须由同卡 msprof
证明。两种模式都不是完整 TQue 双缓冲。
Cube 侧另有默认关闭的 HF32 A/B。IEEE 基线在 MMAD 中保持两个 FP32 输入不变；HF32 候选仅在
MMAD 前舍入输入，可能提升吞吐但会损失尾数精度。因此它不能与调度候选混合归因，必须固定其余
维度，先做 37/37 精度门禁，再比较 `HF32 Eligible`、AIC 时间和总 kernel 中位数。
`KDA_GROUPED_BATCH_TAIL_BLOCKS=true` 现为源码默认：它不增加 UB，也不改变单个元素的算术次序，
只把四个 8x128 尾块的 gather、六个 Vector 表达式和 scatter 批量执行。静态上整 task 的
Vector API 约从 204 降到 186，`PIPE_V` barrier 从 93 降到 84，MTE2/MTE3 API 分别从
55/60 降到 43/48。目标 shape 上，仅尾部就把 MTE2 API 从 131,072 降到 32,768，Vector
表达式 API 从 196,608 降到 49,152，MTE3 API 从 131,072 降到 32,768，807,403,520 B
搬运量保持不变。`false` 保留逐块 scalar 回退；这些仍是源码计数，实际收益取决于指令展开、
stride DMA 和 AIV critical path，必须用两份 clean wheel 验证。
此外，物理 AIV 的八行 `rowStart` 已从每个 task/stage 重算改为 kernel 入口一次计算，三个
persistent right-outer gate block 也从动态小循环改为常量 offset 显式展开。目标 shape 的源码模型
分别消除 114,688 个行起点表达式、32,768 次 consume 内 `GetSubBlockIdx` 读取和 24,576 次
三块循环迭代；它不改变任何 FP32、Vector、MTE 或 GM 字节，是否真正减少 Scalar spill/issue
必须由新 key 23 的 `aiv_scalar_time`、指令数及反汇编共同确认。
`KDA_GROUPED_OVERLAP_TASK_STORE` 在 batch tail 上进一步改变跨 task 调度，但不减少上述静态
API 数。完成的 `dq/dk/dg` 原位留在三个 accumulator bank，`db` 留在 128-B prefix；下一 task
先完成 feature load 和 stage 0/1 workspace 发布，随后才提交旧 task 的四个 MTE3 输出。AIC
因此可以在旧输出写回期间计算下一 task 的前两个 Cube stage，AIV 在清零 accumulator 前通过
`MTE3_V` 收口。候选不增加 179,936-B UB 上限；收益只可能来自缩短 task-boundary AIC 空洞，
必须比较相同 batch-tail 实现的 `store-serial`/`store-overlap`，不能与 scalar-tail 基线混比。
`KDA_GROUPED_PERSISTENT_MMAD_ENGINES` 则只改变 AIC 本地资源生命周期。left32/right16 分别占用
40/36 KiB L1、8/4 KiB L0A、32/32 KiB L0B、16/8 KiB L0C，并使用 event 0..3/4..7；
L0B 恰好使用 64 KiB。源码用容量、512 B 对齐和 event 区间断言约束该布局。目标 shape 静态上
69,632 次逻辑 GEMM 不变，MMAD envelope 从 45,056 个降为 20 个 AIC `Process` 各两个、合计
40 个，但实际 AIC Scalar/event stall 收益必须
通过 clean build 和 msprof 证明。

`KDA_GROUPED_REUSE_VECTOR_MASK=true` 是不改变数值语义的 Scalar 发射候选。key 23 的 gate、
pair、stage 累加和输出尾部均为连续 FP32，优化包络内的元素数都是 64 的整数倍且 repeat 不超过
255。实现对每个局部包络只执行一次 normal 64-lane `SetVectorMask`，随后使用相同 Sub/Mul/Muls/
Exp/Add/MulAddDst 的低阶 repeat API，结束时 `ResetMask`；操作数、指令次序和 FP32 repeat 分组
不变。`false` 保留原 Level-2 count API 逐调用配置 mask/counter 的回退，收益必须用两份 clean
wheel 的 AIV Scalar 比例和 kernel 中位数确认。

`KDA_GROUPED_COALESCE_DB_REDUCE=true` 进一步针对 Scalar/API 发射瓶颈：
K=128 的两个 64 元素 half 仍分别归约，但把所有独立行合并进
`WholeReduceSum.repeatTime`。whole-tail 第一层从 32 次 API 调用降为
2 次，stage-local 路径每 stage 从 8 次降为 2 次；partial 地址、第二层
mask-2 归约和 FP32 树均保持不变。`false` 是逐行回退，实际收益以
`coalesced`/`per-row` 两份 clean wheel 的 AIV Scalar 与 kernel 中位数为准。

## A/B 性能归因方法

当前实现保留十二个独立编译期开关，并由验证 runner 在 clean archive 中生成可追溯变体。
`--stage-epilogue` 接受 `source|overlap|tail`：`source` 沿用提交中的常量，`overlap` 强制
stage-local 候选，`tail` 强制原 whole-tail；`--pair-scratch` 接受 `source|pingpong|single`。
`--tail-blocks` 接受 `source|batch|scalar`，`--mmad-engines` 接受
`source|persistent|scoped`，`--vector-mask` 接受 `source|reuse|per-call`，
`--db-reduce` 接受 `source|coalesced|per-row`，`--task-store` 接受
`source|overlap|serial`，`--stage-a` 接受 `source|packed|split`，`--cube-mode` 接受
`source|ieee|hf32`，`--stage-io` 接受 `source|gm|tscm`（A2 只允许 `gm`）。当前源码默认分别解析为
`factor`、`overlap`、`tail`、`single`、`batch`、`persistent`、`reuse`、`coalesced`、`serial`、
`split`、`ieee` 和 `gm`。runner 使用解析后的十二维身份
`pair-<factor|direct>_setup-<overlap|prologue>_epilogue-<overlap|tail>_scratch-<pingpong|single>_tail-<batch|scalar>_mmad-<persistent|scoped>_vmask-<reuse|per-call>_dbr-<coalesced|per-row>_store-<overlap|serial>_stagea-<packed|split>_cube-<ieee|hf32>_io-<gm|tscm>` 绑定 wheel、
精度与 profile 证据：

| 对比 | pair gates | shared setup | stage epilogue | pair scratch | tail blocks | MMAD engines | Vector mask | db reduce | task store | stage A | Cube mode | 拟隔离变量 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 源码默认基线 | factor | overlap | tail | single | batch | persistent | reuse | coalesced | serial | split | ieee | 当前源码默认方案 |
| scoped MMAD 回退 | factor | overlap | tail | single | batch | scoped | reuse | coalesced | serial | split | ieee | AIC 每-use 事件包络对照 |
| packed-A 候选 | factor | overlap | tail | single | batch | persistent | reuse | coalesced | serial | packed | ieee | 两次冗余 stage-A MTE3 row-copy 提交的影响 |
| HF32 候选 | factor | overlap | tail | single | batch | persistent | reuse | coalesced | serial | split | hf32 | Cube 输入舍入的精度与吞吐影响 |
| tail-scalar 回退 | factor | overlap | tail | single | scalar | persistent | reuse | coalesced | serial | split | ieee | 尾部逐块 API、DMA 启动与同步对照 |
| task-store 基线 | factor | overlap | tail | single | batch | persistent | reuse | coalesced | serial | split | ieee | 与 overlap 使用相同 batch-tail 指令 |
| task-store 候选 | factor | overlap | tail | single | batch | persistent | reuse | coalesced | overlap | split | ieee | 跨 task 输出 MTE3 与下一 task Cube 重叠 |
| scratch 候选 | factor | overlap | tail | pingpong | batch | persistent | reuse | coalesced | serial | split | ieee | 局部 3-bank MTE3/Vector 重叠的影响 |
| epilogue 候选 | factor | overlap | overlap | single | batch | persistent | reuse | coalesced | serial | split | ieee | 8-row 前移与后续 AIC 重叠的影响 |
| bridge 对照版 | direct | overlap | tail | single | batch | persistent | reuse | coalesced | serial | split | ieee | pair bridge 减少 Exp 与新增 Mul 的影响 |
| setup 对照版 | factor | prologue | tail | single | batch | persistent | reuse | coalesced | serial | split | ieee | stage-0 与共享初始化重叠的影响 |
| mask 对照版 | factor | overlap | tail | single | batch | persistent | per-call | coalesced | serial | split | ieee | 外部 mask 复用减少 Scalar 控制的影响 |
| db 归约对照版 | factor | overlap | tail | single | batch | persistent | reuse | per-row | serial | split | ieee | 跨行合并减少 Reduce API/Scalar 发射的影响 |

比较时必须固定 commit、物理卡、CANN、目标 shape、warmup/repeat 和 profiler 指标，并要求每个
变体自己的 clean wheel 完整通过 37/37。性能数字取同一 msprof 进程内丢弃 3 次 warmup 后的
10 次 kernel duration 中位数；manifest 中的 `build_variant`、`pair_gates`、`shared_setup`、
`stage_epilogue`、`pair_scratch`、`tail_blocks`、`mmad_engines`、`vector_mask`、
`db_reduce`、`task_store`、`stage_a`、`cube_mode` 和 `stage_io` 必须与 state 一致。不能使用一个变体的精度结果
为另一个变体背书，也不能用
单次样本或不同 CANN 环境的差值判断收益。runner 为每轮生成唯一 profile group，并用
`profile_evidence.pass` 绑定 commit、wheel、CANN、37/37 与 CSV/manifest 哈希；比较 A/B 时应
连同该凭证保存原始 CSV。
runner 还会生成并纳入哈希清单的 `kda_profile_summary.json`。仓内
`compare_chunk_kda_bwd_intra_profiles.py` 只接受两份完整 evidence，强制 commit、物理卡、CANN、
CATLASS、shape、其余十一维变体身份完全一致，仅允许 `pair_scratch=single/pingpong` 及其派生产物
不同；默认中位数至少改善 1% 才选择 pingpong，并单独报告是否达到 4 ms。它同时比较 AIC/AIV、
Cube/Vector/Scalar、MTE1/2/3 与 Fixpipe 中位时间，避免把无关环境差异误判为局部双缓冲收益。

最小 epilogue A/B 应只改变第三维：默认基线使用
`--pair-gates factor --shared-setup overlap --stage-epilogue tail --pair-scratch single --tail-blocks batch --task-store serial --mmad-engines persistent --vector-mask reuse --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm`，候选使用
`--pair-gates factor --shared-setup overlap --stage-epilogue overlap --pair-scratch single --tail-blocks batch --task-store serial --mmad-engines persistent --vector-mask reuse --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm`。最小 scratch
A/B 只把 `pair-scratch` 由 `single` 改为 `pingpong`；最小 tail 回退 A/B 只把 `tail-blocks` 从
`batch` 改为 `scalar`；最小 MMAD A/B 只把 `mmad-engines` 从 `persistent` 回退为 `scoped`；最小 mask
A/B 只把 `vector-mask` 从 `reuse` 改为 `per-call`；最小 db 归约 A/B 只把
`db-reduce` 从 `coalesced` 改为 `per-row`。最小 task-store A/B 必须把其他八维固定为
`factor + overlap + tail + single + batch + persistent + reuse + coalesced`，只把 `task-store` 从 `serial` 改为
`overlap`；`overlap` 与 scalar tail 或 stage-local epilogue 是非法组合。最小 stage-A A/B 只把
`stage-a` 从 `split` 改为 `packed`；HF32 最小 A/B 只把 `cube-mode` 从 `ieee` 改为 `hf32`，
并保持其余十一维不变。也可用十二个 `source` 复现
当前提交默认值；不得把 `source` 标签直接写入变体身份，identity 记录的必须是解析后的具体值。

先运行默认 persistent 版取得 grouped key 23 的真实 AIC/AIV bound，并用 scoped clean wheel
确认 45,056→40 个源码级 MMAD 包络是否转化为实测收益。随后固定 batch tail 对
`store-serial`/`store-overlap` 做最小 A/B，再分别测试局部 scratch ping-pong 与 packed stage-A；
前者只延后三个立即 MTE3 wait，后者只减少每个非零 stage 的两次 AIV MTE3 row-copy 提交，均不能预设为 4 ms
的主收益。当前 phase1 的 MTE3
占比约 0.7%，而 phase1 并非 key 23，因此只能用于排序实验风险，不能代替 grouped profile。
完整 UB 双缓冲需要把约 175.7 KiB 的单任务数据通路复制为约 351.4 KiB，超出 192 KiB UB
上限，不是可行方向。

## 待上板确认

当前机器没有 CANN/NPU，以下均是需要 profiling 验证的假设，不是实测结论：

1. key 23 是否生成 mixed task、AIC time 是否非零，以及 profiler 的 HF32 标记是否与变体一致
   （默认 IEEE 为 `NO`，HF32 候选为 `YES`）；
2. 17 个 FP32 GEMM 的 Cube utilization、M16 子视图、K16 tail copy 和 FIXPIPE 时间；
3. ready/done 双槽是否真正覆盖 AIV gate 构造与 AIC MMAD，是否存在跨核 wait 长尾；
4. 剩余 FP32 Exp、对角 causal mask 和 `db` reduce 的绝对耗时，以及 stage-local epilogue
   是否真正与后续 AIC 重叠、pair scratch ping-pong 是否减少 AIV MTE3/Vector 气泡、四尾块
   batch 是否减少 Vector/Scalar issue 与小 DMA 启动、persistent MMAD 是否减少 AIC
   Scalar/event stall、跨 task store 是否覆盖 task-boundary AIC 空洞；
5. 目标 shape kernel 是否达到 4 ms，以及外围约 2.45 ms layout/cast 是否需要后续融合；
6. FP16/unsafe/GVA/varlen/tail 等未命中 key 23 的 shape 是否保持已有回退性能与精度；
7. DAV_2201 分发在 A2 与 A3 上是否分别完成 clean build、精度和事件验证。

建议采集 kernel duration、AIV/AIC utilization、MTE2 bandwidth、Vector/Cube utilization、
各流水 stall 和 task tail。基准至少覆盖 `(BT,K)=(64,128),(128,128)`、FP16/BF16、
dense/varlen、`HV/H=1/2/4` 和 safe/unsafe。

## pair-wise key 15 重编回退

key 15 对每个因果 block pair 选择共享 gate reference，并将四个逻辑 GEMM 打包为两个：

```text
[dAqk_left; dAkk_left] @ (K * exp2(g_ref - g_early))
[dAqk_right^T, dAkk_right^T]
    @ [Q * exp2(g_late - g_ref); beta*K * exp2(g_late - g_ref)]
```

AIV 负责 gate、因果 mask、矩阵拼接和行缩放，AIC 负责 IEEE-FP32 GEMM。key 15 保持完整
编译，但 host 在 grouped 开关打开时不会选择它。若 key 23 的 clean build、精度或事件验证
失败，必须关闭 `ENABLE_GROUPED_SAFE` 后重新编译才能回到 key 15，不存在已构建 wheel 内的
运行时回退。该切换不影响通用 key 0/1/2/3/5/7。

## grouped-by-late 深融合 key 23（源码实验实现）

mixed key 15 按 10 个 causal block pair 分别下发左右两个 GEMM，共 20 次物理 GEMM。key 23
按 late block `l=0..3` 保持四个 stage：AIV 为 stage `l+1` 准备矩阵时，AIC 计算 stage `l`；
每个 stage 只做一次 ready/done 交接，不再按 block pair 交接。严格 safe-gate 路径的调用数为：

| stage | off-left | off-right | diagonal | 合计 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 | 2 |
| 1 | 1 | 1 | 2 | 4 |
| 2 | 1 | 2 | 2 | 5 |
| 3 | 1 | 3 | 2 | 6 |
| 总计 | 3 | 6 | 8 | 17 |

off-left 在每个存在 early block 的 late stage 合并一次；off-right 则对每个 early block 各执行
一次 M16 GEMM。输入 `g` 已经是本算子使用的 base-2 累计 gate；只有从上游原始 natural-log
gate 推导时才需乘 `log2(e)`。以下直接令 `G=g`，令 `F_l` 为 late block 首 token 的 K 维
`G`，令 `R_e` 为 early block `e` 末 token 的 K 维 `G`：

```text
Poff     = 16*l
A_left   = [Dq(l,0:l); Dk(l,0:l)]                       # [32,Poff]
B_left   = vcat_e(K_e * exp2(F_l-G_e))                  # [Poff,128]
C_left   = A_left @ B_left                              # [32,128]
           再按 late token 乘 exp2(G_l-F_l)

A_right_e = [Dq(l,e)^T,Dk(l,e)^T]                       # [16,32]
B_right_e = [Q_l * exp2(G_l-R_e);
             beta_l*K_l * exp2(G_l-R_e)]                # [32,128]
C_right_e = A_right_e @ B_right_e                       # [16,128]
            再按 early token 乘 exp2(R_e-G_e)
```

设备端通过 `Exp((G_i-ref)*ln(2))` 实现 `exp2(G_i-ref)`。off-left 的所有 early block 可以安全
共享 late-first reference `F_l`；每个 off-right 必须使用它自己的 early-last reference `R_e`。
对角部分仍以当前 block midpoint 为 reference 并单独执行，因果 mask 与 key 15 保持一致。

对于 stage `l`，每个 AIV 将同一份 4 KiB raw dA slab 写成 `Aoff` RowMajor
`[32,Poff]` 和 `Adiag` RowMajor `[32,16]`。off-left 直接读取 `Aoff`；off-right 把相同 compact
地址解释为完整 ColumnMajor `[Poff,32]` 转置视图，再以 `early*16` 为行偏移取得
`[16,32]` 子视图。CATLASS 子视图保留父矩阵 stride `Poff`，所以 stage 3 的三个 M16 分别
使用 stride 48，无需重新打包、第二次 GM 读取或 UB Scalar 转置。

grouped UB 不具备复制整条数据通路的空间。实现使用一个 179,936 B slab，低于当前 A2 目标的
192 KiB 上限并保留 16,672 B。reference cache 从 82,176 B 的 color-8 地址开始驻留
F1..F3/M0..M3，R0..R2 经 128 B padding 移到 color 12；相比“先 Copy reference，再连续 Sub”，repeat-stride `Sub`
在目标 shape 静态移除 163,840 个 mask-64 Copy API 和 117,440,512 个 FP32 中间元素复制，新增
131,072 个 mask-64 Sub API，净减少 32,768 个 Vector API 启动与 40,960 个依赖 barrier。Copy
本身不做算术，新路径保持每条 `Sub` 的 `src0/src1` 顺序，所以不改变 FP32 数值语义。三个持久 gate
bank 保存会被后续 stage 消费的 early-block right-outer gate，六个 stage 奇偶 bank 保存
left-outer、diagonal late 和 diagonal early gate；UB 尾部从 176,768 B 的 color-4 地址保存六个
K-wide pair bridge，
再用 96 B 常驻每个物理 AIV 固定的 causal mask/零向量。bridge
`exp2(F_s-R_e)` 作为 `Mul` 的 `src0`，以 repeat stride 0 在两个 mask-64 半行上重复读取，直接
与持久 `exp2(R_e-G_e)` 和 stage `exp2(G_s-F_s)` 相乘，得到该 pair 的两个 inner gate。相对先
复制 bridge 再相乘的版本，目标 shape 静态减少 98,304 次逻辑 8x128 Copy pass（底层
196,608 个 mask-64 Copy API）、49,152 个依赖 barrier 和 100,663,296 个 FP32 中间元素复制；
这是 UB 内部流量估算，不改变 GM 理论下限。
单调不增的累计 gate 使三个因子都不大于 1，避免公共参考点溢出；但该乘法
重结合仍可能改变直接 `Exp` 的 FP32 舍入/FTZ 边界，所以直接构造路径作为编译期回退保留，
接受优化前必须跑极值 gate 与完整精度回归。raw dA 在 MTE3 完成后复用为 scratch；这是显式
生命周期复用，不是完整 UB 双缓冲。`dk_left`/`dk_right` accumulator 分别使用 color 12/color 4，
消除最终差值的同 bank-group 双读；pair bridge 的 color-4 destination 读取 color-8 first 和 color-12 right reference，
也消除六次 bridge `Sub` 的同组双读。两项仅改变 UB 地址，不改变 FP32 指令顺序。
`KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH=true` 时仅追加
三块 `8x128` FP32 scratch：alternate 起点为 179,968 B，总 UB 为 192,256 B，剩余 4,352 B；
stage 2/3 用两个持久 `MTE3_V` event 延迟 scratch reuse wait，最终 diagonal drain 复用已消费的
event0。默认 `false` 保留单 scratch 与原事件闭环。源码通过
`static_assert(KDA_GROUPED_UB_BYTES <= 192 * 1024)` 约束两种模式上限。

源码默认 `KDA_GROUPED_OVERLAP_STAGE_EPILOGUE=false`。无论该开关取值，一次 beta 广播都会在
feature load 后驻留 `REDUCE_UB`。候选设为 `true` 时，输入 `db` 的 32 行驻留 reference
cache 前的 128-B prefix；每个 stage consume 完成后，用 reference storage 后的着色 suffix 保存该 8-row
block 的 64+8 个归约中间值，随后
更新 `db` 并原位缩放对应 `dk_left_pre`。stage 0--2 可与下一 AIC stage 并行，stage 3 仍落在
task tail。`false` 分支保留原有整块 product、32-row 两级归约和一次整体 beta 缩放；两者都尚未
完成 key23 clean build、37 项 NPU 精度和同提交 msprof A/B，因此本文不预设候选更快。

`KDA_GROUPED_OVERLAP_TASK_STORE=true` 进一步利用 whole-tail 完成后的生命周期：三个 accumulator
已经分别保存最终 `dq/dk/dg`，reference cache 前缀保存最终 `db`，其余 feature/gate/scratch
区域均已退役。AIV 在保留这四个输出的同时搬入下一 task，准备并发布 stage 0/1，然后提交旧
task 的四次 strided MTE3；AIC 可先消费两个 ready，AIV 仅在清零 accumulator 前等待输出完成。
完整 4096-FP32 差值 scratch 复用 `KDA_GROUPED_BATCH_ROW_TMP_UB=[163072,179456)`，没有越过
179840-B causal-state 起点。该候选依赖 batch tail 与 whole-tail epilogue，且不会改变任一输出
内部的 FP32 操作次序；仍需两个 clean wheel 证明设备编译、37 项精度和实际跨 task 重叠。

stage 3 off-left 使用静态 K32 BlockMmad 加 K16 tail，保留 K48 的归约/抵消次序；off-right
使用三个独立 M16，不再把三个 early block 拼成一次大 M 调用。不同 BlockMmad 串行共享一个
CATLASS Resource；off-left/diagonal-left 独立闭环，同一 stage 的 off-right 与 diagonal-right
共享一个 MMAD 事件包络。这些约束已经写入源码，但仍需 clean build、stride-48
M16/slot canary 和重复 launch 在设备上证明。

当前 key 23 每槽固定为 26,624 个 FP32（104 KiB），以下区间均以 FP32 元素计，右边界不包含：

```text
prepare input:
  BoffL   [    0,  6144)    Aoff    [ 6144,  7680)
  BoffR0  [ 7680, 11776)    BoffR1  [11776, 15872)
  BoffR2  [15872, 19968)    BdiagR  [19968, 24064)
  Adiag   [24064, 24576)    BdiagL  [24576, 26624)

consume output:
  CoffL   [    0,  4096)    CdiagR  [ 4096,  6144)
  CoffR0  [ 7680,  9728)    CoffR1  [11776, 13824)
  CoffR2  [15872, 17920)    CdiagL  [19968, 24064)
```

`KDA_GROUPED_PACK_STAGE_A=true` 使用同尺寸 alternate layout；packed A 的有效宽度随 stage
为 16/32/48/64，以下列出最大预留区间：

```text
prepare input:
  BoffL   [    0,  6144)    Apacked [ 6144,  8192)
  BoffR0  [ 8192, 12288)    BoffR1  [12288, 16384)
  BoffR2  [16384, 20480)    BdiagR  [20480, 24576)
  BdiagL  [24576, 26624)

consume output:
  CoffL   [    0,  4096)    CdiagR  [ 4096,  6144)
  CoffR0  [ 8192, 10240)    CoffR1  [12288, 14336)
  CoffR2  [16384, 18432)    CdiagL  [20480, 24576)
```

输出只覆盖已退役的输入区间。每次 CATLASS 调用仍必须在 FIXPIPE 写 C 前完整读完与 C 重叠的
A/B，并保证 M16/M32 实际 store 不越界；该条件必须用 clean build、slot canary 和 NPU one-hot
验证，不能只凭静态地址计算认定安全。host 与 kernel 的 `GROUPED_SLOT_ELEMENTS` 均应为
26,624；key 15 自己的 workspace 布局保持不变，两个路径不能混用 workspace 尺寸。

### 为什么否决 14-GEMM 和 8-GEMM

旧 14-GEMM 草案把同一 late stage 的所有 off-right 合成一个矩阵，并让所有 early block 使用
公共 late-first reference。该变换在无限精度实数代数上成立，但改变了 safe-gate 的 FP32
指数求值次序：某个 early block 的 inner/outer 因子可能先 FTZ 为 0 或先 overflow 为 Inf，
后续缩放无法恢复。一个针对性模型是 gate 每步 `-5/ln(2)`、`q[32]=1`、
`dAqk[32,0]=2^127`；使用该 early block 末 token reference 的路径保留约 `2^-103.8` 的非零
`dk[0]`，公共 late reference 路径却会因外层指数先 FTZ 得到 0。因此 14-GEMM 不是严格
safe-gate 实现，不能通过放宽阈值接受；当前 17-GEMM 路径以 3 次额外 M16 换取原参考点次序。

更激进的 8-GEMM 会进一步把 diagonal 与 off-diagonal 统一到公共 reference，同样会扩大中间
指数范围并改变 FP32 非结合归约/抵消次序。公共 intra 接口也没有携带足以在运行时证明指数与
乘积安全范围的 gate provenance，因此当前不启用这种带 guard 的实验分支。

### 保留 reference 的零填充块对角候选

零填充块对角与上述被否决的代数重关联不同：每个逻辑 GEMM 仍使用原来的 first/middle/right
reference、原来的 gate 值和原来的有效 K 顺序，只把矩阵放进互不相交的 M/K 区间，其余位置
写精确 `+0.0f`。它会插入额外的零乘积，因此只在非 padding 操作为有限值时保持普通数值结果，
不宣称 NaN/Inf 或有符号零逐 bit 等价；任何实现仍必须通过现有极值 gate、cancellation、one-hot
和完整 NPU 精度门禁。

`scripts/model_chunk_kda_bwd_intra_blockdiag.py` 对目标 `BT=64,K=128` 做了结构证明和 A2 容量
预算。把每个 stage 全部压成一个 GEMM 虽可将调用数从 17 降到 4，但 FMA 增至 5.45 倍，四个
stage 的保守 workspace 往返约为 5.587 GB/目标 shape，不能因为调用数少就直接实现。当前更
可落地的 11-GEMM 候选不混合 A 的 RowMajor/ColumnMajor 方向：stage 1..3 将 off-left 与
diag-left 横向合成一个 `M32,N256` GEMM，将全部 off-right 合成一个 ColumnMajor 块对角 GEMM，
diag-right 保持独立；stage 0 仍为两个 diagonal GEMM。该候选 FMA 为原 17-GEMM 的 1.85 倍，
保守 workspace 往返约 4.698 GB/目标 shape。最大 `N256` tile 使用 73,728 B L1、8,192 B
L0A、65,536 B L0B、32,768 B L0C；最大 right `M48` tile 使用 45,056/12,288/32,768/
24,576 B，均在 A2 容量内，但 L0B 已无余量。

这些数字是设计模型，不是实测性能。应先取得当前 17-GEMM key 23 的 clean-wheel msprof，确认
小 GEMM 调度/FIXPIPE 确实是主瓶颈，再决定是否实现 11-GEMM；否则 1.85 倍 Cube 计算和额外
零填充搬运可能抵消调用数收益。模型已纳入 validation source manifest，clean build 会校验
17→4 与 17→11 两组固定结构、reference 不重关联、有效 K 顺序和 A2 tile 上限。

17-GEMM 已作为独立 key 23 写入 kernel/tiling，并保留 key 15 的重编回退。本地静态检查不等于
设备验证：key 23 尚未完成 key23-only CANN clean build、预期 37 项 NPU 精度回归或 msprof。
因此当前可引用的实测结论仍只有 AIV phase1 的 48.660 ms，不能宣称 key 23 已取得性能收益或
达到 4 ms。
