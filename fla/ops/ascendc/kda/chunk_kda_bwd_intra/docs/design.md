# ChunkKdaBwdIntra AscendC 设计

> 2026-07-22 correction: the A2/910B delivery default is the two-slot GM A/B bridge (`KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER=false`). DAV_2201 has no physical AIV-UB-to-AIC-L1 path: AscendC software-emulates `UB -> TSCM` through GM and a registered Matmul KFC client. This direct CATLASS kernel has no KFC client, so enabling the retained TSCM experiment on A2 would be unsupported and would not remove the GM round trip. The main 179,936-byte UB slab remains single-buffered; CATLASS L1/L0 and the GM stage bridge provide the current local and cross-core ping-pong respectively.

## 1. 目标与范围

本算子迁移 `flash-linear-attention` 中的 `chunk_kda_bwd_kernel_intra`，计算 KDA 单个 chunk 内部对 `q`、`k`、`beta` 和逐特征 gate 的梯度贡献。主交付路径为 `safe_gate=true`，同时保留独立的 `safe_gate=false` 编译分支。

首版支持 Ascend A2/A3/A5、`float16`/`bfloat16` 的 `q/k`、FP32 的 L0 gate/其余输入和四个输出，`K` 为 16 的倍数且位于 `[16, 256]`，`chunk_size` 为 64 或 128。kernel 以 `BK=32` 分块，并用实际 `curK=16` 处理尾块。公开 Python 接口允许 BF16 `g/beta` 并在进入 L0 前提升到 FP32。L0 内部布局统一为 BNSD；Python 接口负责 BSND/BNSD/TND/NTD 的无歧义转换。

## 2. 数学语义

输入 `g` 已由上游乘一次 `RCP_LN2=log2(e)` 并完成 chunk-local cumulative sum，因此它已经处于累计 log2 空间，本算子不得再次乘 `log2(e)`。上游增量有限且不大于零，所以每个 chunk 内逐特征 `g` 单调不增；grouped pair-bridge 依赖该输入契约，`safe_gate=true` 本身不负责校验任意 standalone tensor 的单调性。对 chunk 内 token `i` 和特征 `d`，记 `E(i,j,d)=exp2(g[i,d]-g[j,d])`；AscendC 通过 `Exp((g_i-g_j)*ln(2))` 实现该值。仅保留 causal 的 `j<=i` 项：

```text
dq_local[i,d] = sum(j<=i) dAqk[i,j] * k[j,d] * E(i,j,d)
dk_left_pre[i,d] = sum(j<=i) dAkk[i,j] * k[j,d] * E(i,j,d)
dk_right[i,d] = sum(j>=i) (
    dAqk[j,i] * q[j,d] + dAkk[j,i] * beta[j] * k[j,d]
) * E(j,i,d)

dq_out = dq + dq_local
dk_out = dk + beta * dk_left_pre + dk_right
db_out = db + reduce_sum_d(dk_left_pre * k)
dg_out = dg + q * dq_local + (beta * dk_left_pre - dk_right) * k
```

该式与上游 Triton kernel 的左侧 causal 累加、右侧转置累加及最终 `dg` 组合完全一致；`dAqk` 已在前序反向阶段包含 scale，本算子不再乘 scale。

## 3. 数据依赖与执行图

通用 AIV 路径中，每个 `(chunk, value_head, 16-token row block)` 是一个独立任务。任务内部再按 32 个 K 特征切片：

```text
q/k/g/beta,dAqk,dAkk,dq/dk/db/dg (GM)
                 |
                 v
  dA [source,16] coefficients + q/k/g feature cache (UB)
                 |
                 v
 source-token loop, 16-row x 32-feature vector math
                 |
                 v
       dq/dk/db/dg row block (GM)
```

不同任务写入不重叠的 token 行，因此无需原子操作。`db` 在同一任务内遍历全部 K tile 后一次写回，避免上游 `NK` 维临时张量和第二次 reduce launch。

目标 A2 grouped 路径将每个 `(chunk, value_head)` 作为一个逻辑任务，由 1 个 AIC 和 2 个 AIV 协作。两个 AIV 各负责每个 16-token block 的 8 行，写区间互斥；四个 stage 按 late block 合并此前全部 off-diagonal source block：

```text
AIV: q/k/g 驻留 UB，构造 stage l / slot n 的 gate、dA 和 gated feature
                         |
                         v ready
AIC: stage 0 做 2 个 diagonal GEMM；stage l=1..3 分别做
     1 个 grouped off-left、l 个独立 off-right 和 2 个 diagonal GEMM
                         |
                         v done
AIV: 消费 stage l / slot n 的结果并累积，同时准备下一 stage
```

消费 stage `l>0` 的 off-right 结果时，pair C 在 GM 中以固定 16x128 双行槽间隔排列，而每个
物理 AIV 读取的 8x128 行、三块 scratch、`dkRightAcc[0:l]` 和 `RightOuterGate[0:l]` 都连续。
默认开启的 `KDA_GROUPED_COALESCE_OFF_RIGHT_CONSUME` 因此用一个带 GM byte-gap 的
`DataCopyExtParams` 代替 `l` 次连续搬运，并用一次长度 `l*8*128` 的 FP32 `MulAddDst` 代替
`l` 次独立调用。该变换不跨 pair 累加，也不改变任何 gate reference 或元素运算。

对应的右 B 生产端也保持同样的几何不变式：每个物理 AIV 的 `q*gate` 与
`beta*k*gate` 是相邻的两块 8x128 UB scratch，在目标 32x128 GM B 矩阵中间隔另一 AIV 的
8 行。`KDA_GROUPED_COALESCE_RIGHT_B_WRITES` 用一次 `blockCount=2`、UB `srcStride=0`、
GM `dstStride=4096 B` 的 CopyOut 替代两次独立写入；TSCM 实验分支仍使用原来的两次 zN 行搬运。

off-left/diagonal-left 的 32x128 C 具有相同的双 8 行几何。默认
`KDA_GROUPED_COALESCE_LEFT_C_READS=true` 用一次 `blockCount=2`、GM `srcStride=4096 B`、
UB `dstStride=0` 的 CopyIn 把 dq 与 dk-left 的两段 C 搬到相邻 scratch；后续两条 FP32
`MulAddDst` 仍分别写入原 dq/dk-left accumulator。

每个逻辑核组使用两个 104 KiB、512B 对齐的 workspace slot，形成 producer/consumer
ping-pong；实际稳态为 `C0||P1`、`C1||(consume0+P2)`、`C2||(consume1+P3)`、
`C3||consume2`，task 尾的 `consume3+store` 仍未被 AIC 覆盖。stage 0 发布 ready 后，AIV
才构造 stage 1 首次使用的持久 right-outer、pair bridge 并清零 accumulator，使这段共享准备
工作与 `C0` 重叠；编译期开关可恢复原 task-prologue 顺序。该流水不会生成随序列长度增长的
中间张量。不同 M/K 形状和 A-layout 的 BlockMmad 通过
`MmadPingpongTlaMulti` 串行复用同一份 L1/L0 resource。默认 scoped 路径的每个逻辑 GEMM 都执行
`preSetFlags -> operator -> finalWaitFlags`，并关闭 UnitFlag；公共 non-UnitFlag 分支在 `operator` 内建立
完整的 `M_FIX/FIX_M` L0C 生命周期，`finalWaitFlags` 再排空该调用的全部事件。每个 stage 的最后一个
FIX 完成后，AIC 再在 `PIPE_FIX` 发布 done，保证本 stage 的全部 FP32 C 矩阵写回后 AIV 才能读取。

AIV 的六类流水依赖各在 `Init` 分配一个 `TEventID`，在 `ProcessAiv` 的全部 task 完成后统一释放；
每次同步仍执行原来的 `SetFlag -> WaitFlag`，但不再在热路径反复分配和释放事件。局部
pair-scratch ping-pong 打开时另外占用两个 `MTE3_V` ID，因此该类型最多同时保留三个 ID，低于
A2 的八个 event 上限。dA 搬入与 gate 构造的重叠继续复用同一个已保留的 `MTE2_V` ID，并在
下一次同类同步前完成 wait，不改变依赖关系或浮点顺序。

`KDA_GROUPED_OVERLAP_STAGE_EPILOGUE` 控制 grouped AIV 尾处理的调度位置。源码默认值为
`false`，runner 将该变体标记为 `tail`：四个 stage 全部消费完后，`StoreTaskOutputs` 一次性对
32 个选中行执行 `dk_left_pre*k` 的 K 维两级归约、累加 `db`，再完成
`dk_left_pre*beta`。两条路径都在 feature load 时只做一次 32 行 beta 广播并常驻
`REDUCE_UB`，供 `k*beta` cache 和最终缩放共同复用。候选值 `true` 标记为 `overlap`：任务开始时
另外把 32 个输入 `db` 行保留在 UB；每个 `ConsumeStage<stage>` 完成当前 block 的左右/对角累加后，立即对该
8-row block 执行相同的 K 维归约、`db` 累加和 `dk_left_pre*beta`。这样 stage 0--2 的 AIV
epilogue 可以与随后已经发布的 AIC stage 1--3 重叠；stage 3 没有后续 AIC，仍位于 task 尾。
原 32-row whole-tail 实现保留在编译期 `else` 分支中，作为精度、编译或性能不满足时的回退。
该候选只改变调度粒度，不改变每行的数学公式；设备端舍入、事件时序和实际重叠仍需 clean
build、37 项精度回归与 profiling 证明。

## 4. Tiling

### 4.1 核间切分

- dense：任务数为 `B * ceil(T/BT) * HV * ceil(BT/16)`；
- varlen：任务数为 `total_chunks * HV * ceil(BT/16)`；
- blockDim 为 `min(task_count, AIV core count)`，每核步进处理后续 task；
- varlen chunk 在 L0 按 canonical sequence-major 顺序打包为 `[seq,start,end,0]`。kernel 每个 task 按 `flatChunk` 连续搬运 32B 元数据到 UB，不在 device 侧二分序列，也不把序列级数组塞入 tiling data。
- grouped key 23 的任务数为 `B * (T/64) * HV`，blockDim 为 `min(task_count, AIC core count)`；runtime 按 `KERNEL_TYPE_MIX_AIC_1_2` 为每个逻辑任务组调度 1 个 AIC 和 2 个 AIV。pair-wise key 15 保持编译，关闭 grouped 开关即可回退。
- grouped AIV 每核仅在进入 `ProcessAiv` 时用除法解析一次初始 `(batch, chunk, head)`；后续仍按 `task += usedCoreNum` 的原顺序执行，但把固定步长预分解为 head/chunk/batch 三层进位，用加法和至多两次条件减法更新地址。该变换不改变 task、GM 地址或任何浮点指令顺序，避免在目标 shape 每核约 205 个 task 的热循环中重复两组 64 位 div/mod。
- 物理 AIV 的 `GetSubBlockIdx()*8` 行起点在整个 kernel 生命周期不变，入口只计算一次 `rowStart`，随后贯穿 feature load、四个 stage 的 A/B 准备与消费以及输出尾部。默认 serial 路径因此在目标 shape 的 8,192 个 task-AIV 对上消除 114,688 个源码级行起点表达式和 32,768 次 consume 内 `GetSubBlockIdx` 读取。三个 persistent right-outer block 也显式展开，消除 24,576 次小循环迭代并使 UB offset 保持编译期常量；浮点、Vector/MTE 指令和地址均不变，实际 Scalar 收益仍以 grouped profile 为准。

### 4.2 核内切分

- `BC=16`：对应上游稳定 gate 子块，也是 dA 行/列驻留 UB 的粒度；
- `BK=32`：q/k 半精度搬运 64B、gate/梯度 FP32 搬运 128B，均满足 32B 对齐；
- safe 主路径同时保留 16 个输出 token 的 `[16,32]` FP32 累加块及三个跨/块内 partial，避免
  `BT*BT*K` 中间量；legacy safe/unsafe 路径仍按单输出 token 累加；
- 每个 task 一次搬入当前 16-token 块的 `[R,BT]` 行 slab，以及全部有效 source token
  对应的 `[curT,16]` 列 slab；行/列块均使用二维 DataCopyPad，kernel 随后只在 UB 中索引，
  禁止逐 GM 标量读，也不为同一 task 的 16 个输出行重复搬运 dA。
- safe 主路径将 dA 行 slab 一次整理为 `[curT,16]`，并在每个 `BK=32` feature tile 上将
  `q/k/g[curT,32]` 一次搬入 UB、整块升为 FP32；同一 source 的 gate 和向量数据供 16 行复用。
- grouped 路径仍保留 10 个因果 16×16 block pair 的数学分区。每个 late stage 将全部
  off-left pair 合为一次 K16/K32(+K16 tail) GEMM；off-right 为保留每个 early block 自己的
  末 token reference，仍逐 pair 执行 M16 GEMM。连同每 stage 的两个 diagonal GEMM，调用数为
  `2 + 4 + 5 + 6 = 17`：

```text
[dAqk; dAkk] @ (k * gate_early)                    -> [dq; dk_left]
[dAqk^T, dAkk^T] @ [q * gate_late; beta*k*gate_late] -> dk_right
```

  对角项使用 block 中点 reference 并在 AIV 侧清零非 causal 元素；off-left 使用 late block
  首 token reference，off-right 使用各 early block 的末 token reference。每个 stage 只搬入一次
  dA row slab，并分别写为 compact `Aoff=[32,Poff]` 与 `Adiag=[32,16]`；每个 off-right 从同一
  ColumnMajor `[Poff,32]` 转置别名取得一个保留父 stride 的 `[16,32]` 子视图，避免第二次 GM
  读取或 UB 逐元素转置。stage 3 off-left 故意采用静态 K32 加 K16 tail，保留前三个 early
  block 的 cancellation 累加顺序；三个 off-right 均使用精确 M16，避免跨 pair 写入。

调用压缩只考虑不改变 reference 的零填充布局。设计模型表明逐 stage 全块对角的 4-GEMM
方案会引入 5.45 倍 FMA，不作为实现方向；避免混合 A 矩阵方向的 11-GEMM 候选约为 1.85 倍
FMA，且最大 `N256` tile 的 L0B 恰好占满 64 KiB。该候选尚未写入 kernel，必须先用当前
17-GEMM key 23 的 clean-wheel profile 证明小 GEMM 调度是主瓶颈，再进入实现和精度 A/B。
模型只保证原 reference 与有效乘积 K 顺序，不对非有限输入或有符号零作逐 bit 等价承诺。

### 4.3 UB 预算

legacy 固定 buffer 约 38.6 KiB。block-wise safe 路径额外使用两份转置 dA 行 slab、完整
`q/k` typed cache、`q/k/g` FP32 cache、六份 `[16,32]` 累加/partial 和一份 block scratch；
按 `BT=128/BK=32` 估算约 128.9 KiB，低于 A2/A3 的 192 KiB UB，并保留约 63 KiB 余量。
第一阶段使用单 buffer，避免在未 profiling 前为 double buffer 再增加约 64 KiB。

pair-wise key 15 的静态 UB 预算为 175,296 B。grouped key 23 使用一个手工着色的
179,936 B UB slab：feature cache、9 个有效 gate bank、三个 accumulator、result/raw
scratch 分配到不同 128B 相位；实际消费的 F1..F3/M0..M3/R0..R2 十条 reference 从
82,176 B 开始驻留，first/middle 使用 color 8，right 经 128 B 类间 padding 后使用 color 12；它们和 color-4 `gCache`、color-0 gate destination 分离，因而 reference/gate
差值可用 repeat-stride-0 `Sub` 直接跨 8 行计算，不再先复制 reference。typed q/k 在 Cast 后复用为
三个持久 right-outer gate，另外六个 gate bank 按 stage 奇偶复用；UB 尾部 color-4 区域保存六个
K-wide pair bridge，随后 96 B 常驻每个
物理 AIV 固定的 causal Select mask 和零向量。该设计没有足够空间再复制整条 UB 数据通路，因此
真正的双缓冲位于 GM workspace，而不是完整的 UB TQue；后续若 profiling 证明有收益，只能
为热点 scratch 增加局部双槽。
`dk_left` accumulator 保持 color 12，`dk_right` 经 256 B padding 移到 color 4，使最终
`dk_left-dk_right` 的双源读取不落在同一 DAV_2201 bank group；padding 随三个 accumulator 一次性清零，
不增加 Vector API。连续清零总计 12,352 个 FP32 元素，低于三参数 `Duplicate` count 接口的
`64*255=16,320` 元素上限，并由编译期断言保护。stage-epilogue `overlap` 候选不扩大 179,936 B slab：32 个常驻 `db` 行放在 reference cache 前的
128 B prefix，单个 8-row K-reduce 的 64+8 个中间值放在 reference storage 后经着色 padding 保留的 suffix。32 行
beta 广播在 `tail`/`overlap` 两条路径中都常驻原 `REDUCE_UB` 区间，避免 whole-tail 再次执行
`Brcb`；`tail` 默认路径继续在所有 gate/reference 退役后复用其余区域完成一次 32-row 归约。
两种路径共享相同 UB 上限，也都不是完整 UB 双缓冲。
`RAW_DA_UB` 在 consume 阶段复用为第三个 scratch，stage 2/3 再由 MTE2 覆写前显式执行
`V_MTE2`，防止双槽流水下上一 stage 的 Vector 尾读与下一 stage 的 dA 搬入竞争。
跨 task 输出重叠候选也不增加 UB：完成的 `dq/dk/dg` 分别原位保存在三个 16-KiB accumulator
bank，32 个 `db` 行保存在 reference cache 前的 128-B prefix。`dq/dk` 输入使用已退役的 typed
与 `k*beta` bank，`dg` 在旧 `dk_right` 被 `dk` 加法和差值同时消费后直接搬入该 accumulator，
完整 4096 元素的 `dk_left-dk_right` 则复用已经由四尾块 batch 断言保护的
`KDA_GROUPED_BATCH_ROW_TMP_UB`，其右边界为 179,456 B，低于 179,840-B causal-state 起点。
每个 grouped slot 为 26,624 个 FP32（106,496 B，即 104 KiB），双槽为 208 KiB/逻辑 AIC；
user workspace 总量为 `used_aic * 2 * 106,496 B`。

`KDA_GROUPED_PACK_STAGE_A=true` 复用同一 slot 的另一种静态布局：AIV 将已经完成 causal mask
的 `dAqk/dAkk` 完整 prefix 各写一次，组成 RowMajor `[32,prefix]` 的 packed A；off-left 与
diag-left 从不同列区间取得原数学矩阵，off-right 与 diag-right 读取同一存储的转置视图。
这不是 UB 双缓冲，也不改变 safe-gate reference、17 个逻辑 GEMM、MMAD/FP32 累加顺序或
106,496-B slot 大小；它只把每个非零 stage 原先分开的 Aoff/Adiag 四次 row-copy 提交合并为
两次连续提交。默认 `false` 保留 split-A 回退，必须通过 `--stage-a split|packed` 两个 clean
wheel 的 37 项精度、slot canary、repeated-launch 与同卡 msprof 才能判断收益。

### 4.4 局部 pair scratch 双缓冲候选

完整复制 grouped AIV 的 UB 数据通路需要约 351.4 KiB，超过 DAV_2201 的 192 KiB UB，因而不采用完整 `TQue` 双缓冲。源码保留独立编译期开关 `KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH`：默认 `false` 继续使用 179,936 B 单 scratch slab；候选 `true` 只在 `KDA_GROUPED_TAIL_UB + 128`（179,968 B）处增加三块 `8x128` FP32 scratch，共 12,288 B，总 UB 为 192,256 B，仍低于 196,608 B 上限，剩余 4,352 B。常驻 mask/zero 在 alternate scratch 前结束，alternate 起点模 512 为 256，维持原 scratch 的 color-8 相位。

候选仅改变 pair gated-feature 的暂存与 MTE3 等待位置，不改变 gate 构造、FP32 Vector 指令顺序、17 个 GEMM、workspace 地址或归约顺序。stage 1 使用 `pair0(primary) -> diagonal(alternate)`；stage 2 使用 `pair0(primary,event0) -> pair1(alternate) -> wait(event0) -> diagonal(primary)`；stage 3 使用 `pair0(primary,event0) -> pair1(alternate,event1) -> wait(event0) -> pair2(primary) -> wait(event1) -> diagonal(alternate)`。最后 diagonal 的 MTE3 drain 复用已消费的 event0，不额外长期占用第三个 `MTE3_V` event。两个 event 在 AIV `Process` 外分配、结束后释放，避免跨 task 泄漏。该方案必须以 `single`/`pingpong` 两个 clean wheel 分别完成编译、37 项精度、repeated-launch 与同卡 msprof；静态时序证明不能替代设备验证。

### 4.5 四尾块批处理默认路径

`KDA_GROUPED_BATCH_TAIL_BLOCKS` 默认 `true`，在所有 stage 结束后复用已退休的四段 UB：
typed/mask 区保存 `dq`，
reference/gate 区保存 `dk`，`k*beta` cache 保存 `dg`，result/raw 区保存 `dk_left-dk_right`。
四段各能容纳连续 4096 个 FP32 元素，因此不改变 179,936 B 单 scratch 或 192,256 B
pair-scratch ping-pong 的 UB 上限。`false` 保留原先逐个 8x128 block 搬入、计算和写回的
clean-wheel rollback。

四个 GM block 之间相隔 16 行，而每个 AIV 只负责其中连续 8 行。GM->UB 使用
`DataCopyExtParams{4, 4096, 4096, 0, 0}`：`blockLen=4096 B`，GM 侧
`srcStride=4096 B`，UB 侧连续落盘；UB->GM 使用 `{4, 4096, 0, 4096, 0}`，其中 UB
`srcStride=0`，GM `dstStride=4096 B`。`db` 对应参数把 4096 B 换成 32 B。这里的 stride
表示前一 block 尾部到下一 block 头部的间隔；GM 侧单位为 Byte，UB 侧单位为 32-Byte
DataBlock。源码按 `(BC-ROWS_PER_AIV)*rowBytes` 计算该 gap，并断言两个 AIV 恰好覆盖整个
16-row block，避免几何常量变化后静默错址。

Gather 完成后仍严格执行原来的逐元素次序：`dq += dqAcc`；`dk += dkLeft` 后再
`dk += dkRight`；`dg += q*dqAcc` 后构造 `dkLeft-dkRight`，再执行
`dg += (dkLeft-dkRight)*k`。只把同一 API 的元素范围从 1024 扩展为 4096，不改变任何
元素的结合顺序、FP32 精度、gate、GEMM 或归约。尾部静态指令数由 24 个 Vector API、
16 次 MTE2、16 次 MTE3 降为 6、4、4，并保留完整的
`V_MTE2 -> MTE2_V -> V_MTE3 -> MTE3_MTE2/MTE3_V` 生命周期。目标 shape 的 4096 个
逻辑 task、每 task 两个物理 AIV 合计把尾部 MTE2 API 从 131,072 降为 32,768，Vector
表达式 API 从 196,608 降为 49,152，MTE3 API 从 131,072 降为 32,768；搬运字节仍为
807,403,520 B。该计数只是源码模型预期；默认 `batch` 仍必须完成 clean build、37 项精度、
repeated-launch 与同卡 msprof，`scalar` wheel 用于结果和性能回退对照。

### 4.6 AIC 持久双引擎候选

每个 CATLASS `MmadPingpongTlaMulti` 内部已经对 L1A/L1B/L0A/L0B 使用两级 ping-pong。源码默认的
`scoped` 调度把 left16、left32 和 right16 实例映射到相同的 L0 地址与 local event 0..3。
A2 分段结果显示 handshake、stage0-right、stage0-left 和 stage0-both 均可退出，但 stage 1
切为逐调用 `preSetFlags/finalWaitFlags` 后仍然超时。随后在 UnitFlag 分支外手工增加
`SetFlag/WaitFlag<HardEvent::FIX_M>` 的 clean-wheel 预检也在 120 秒后退出，证明不能把普通
事件协议叠加到 UnitFlag 的细粒度 MMAD/Fixpipe 协议上。默认路径因此改为
`KDA_GROUPED_ENABLE_UNIT_FLAG=false`，直接复用公共实现已经配对的 `M_FIX/FIX_M` 事务；该候选仍待
clean-wheel A2 验证，但不改变任何 GEMM、gate 或 FP32 累加次序。
`KDA_GROUPED_PERSISTENT_MMAD_ENGINES=true` 仍保留为编译期实验：它通过算子私有派生 wrapper 重绑
protected L1/L0 tensor 和 event 数组，不修改公共 CATLASS；left32 同时处理实际 K16/K32 以及
stage 3 的 K32+K16 tail，right16 保持原 M16/K32 路径。

同一 dispatch policy 还保留正交的 `KDA_GROUPED_USE_HF32_CUBE` 开关。源码默认 `false`，
即所有 17 个 GEMM 使用 IEEE-FP32 Cube 输入；候选 `true` 只让 MMAD 前的两个 FP32 输入按
HF32 舍入，workspace、FP32 accumulator、GEMM 次序和 Vector 路径均不改变。该候选可能显著
提高 A2 Cube 吞吐，但属于精度语义变化，必须先通过 cancellation、FTZ/overflow、极端 gate 和
完整 37 项 clean-wheel 回归，失败时不能通过放宽阈值接纳。

| 引擎 | L1 | L0A | L0B | L0C | local event |
|---|---:|---:|---:|---:|---|
| left32 | 40 KiB | 8 KiB | 32 KiB | 16 KiB | 0..3 |
| right16 | 36 KiB | 4 KiB | 32 KiB | 8 KiB | 4..7 |
| 合计 | 76 KiB | 12 KiB | 64 KiB | 24 KiB | 0..7 |

L0B 恰好用满 DAV_2201 的 64 KiB，所有起点保持 512 B 对齐，并由编译期容量、互斥区间和 event
上限断言保护。wrapper 显式重绑 non-UnitFlag 路径使用的 `l0CEventList`；scoped 回退类型也经
同一 wrapper 映射回零地址/event0，使普通 `M_FIX/FIX_M` 事务与实际 L0C 分区保持一致。
候选额外断言 L1A/L1B/L0A/L0B 都保持两级、L0C 保持一级且 `ENABLE_L1_RESIDENT=false`：两个
workspace slot 会被 AIV 反复覆写，即使 GM 地址与 tile 坐标相同，也必须重新搬入，不能让 CATLASS
按地址命中旧的 L1 resident 内容。left/right 的 L1/L0 K tile 也固定为 32，因此实际 K16 一轮完成，
K32 一轮完成，K48 严格分成 K32+K16 两轮；每轮末尾归还当前 L1/L0 free token，索引只在两槽间
轮换，不要求跨调用重置。

持久化候选在整个 `ProcessAic` 外只为 left/right 各建立和排空一次 event envelope，task/stage 内继续
按 `off-left -> off-right -> diag-right -> diag-left` 调用，不改变 workspace 覆盖、17 个逻辑
GEMM、约 18 次物理 MMAD 或 K48 cancellation 次序。AIC 的 done flag 仍在 `PIPE_FIX` 上，AIV
仍先等待 done 再读 slot。目标模型的 4,096 个 task、69,632 次逻辑 GEMM 保持不变；保守 scoped
路径使用 69,632 个包含普通 `M_FIX/FIX_M` 的完整 envelope，理论持久化路径为 20 个 AIC Process
各两个、合计 40 个。当前 UnitFlag 与持久化协议都有 A2 timeout 反证，不能进入性能 A/B；必须先让
non-UnitFlag scoped 候选通过 clean wheel、37 项、repeated-launch、slot canary，再用同卡 msprof
量化关闭 UnitFlag 的成本。

### 4.7 FP32 Vector mask 复用候选

grouped key 23 的 gate、pair product、stage accumulator 和输出尾部都以 FP32 处理，候选包络中的
连续长度均为 64 的整数倍，单次 repeat 也不超过 `uint8_t` 的 255 上限。源码默认
`KDA_GROUPED_REUSE_VECTOR_MASK=true`：每个局部 Vector 包络先设置一次 64-lane normal mask，随后
对原有 `Sub/Mul/Muls/Exp/Add/MulAddDst` 使用 `isSetMask=false` 的低层 repeat 重载，包络结束后
执行 `ResetMask`；低层 API 的 mask 形参按接口契约固定传 `MASK_PLACEHOLDER`。跨八行的
reference/bridge 广播仍保留原来的两个 64 元素 half、repeat-stride 0
和操作数方向；连续 API 仍保留原来的 64 元素 repeat 分组、指令顺序与 `PipeBarrier`。

该候选只减少每条 Level-2 API 重复设置 mask/counter 的 Scalar 发射工作，不近似 `Exp`，不合并
FP32 运算，也不改变 workspace、UB 地址、17 个逻辑 GEMM 或归约顺序。`false` 完整保留逐调用
Level-2 count API 回退。由于低层 API 重载可用性与 CANN 版本相关，`reuse`/`per-call` 必须分别
clean build，并各自通过 37 项精度、repeated-launch、slot canary 和同卡 msprof；源码静态检查
不能代替设备编译与运行证据。

### 4.8 db K=128 跨行合并归约候选

grouped key 23 的每行 `db` 增量都严格按两个 64 元素 half 做第一层
`WholeReduceSum`，再对两个 partial 做第二层求和。源码默认
`KDA_GROUPED_COALESCE_DB_REDUCE=true`：第一条指令以 `dstRepStride=8`、
`srcRepStride=16` 同时处理所有行的前 64 个元素，第二条以相同 stride
处理后 64 个元素；输出仍落在原来的 `row*8+0/1`，第二层 mask-2
归约不变。因而每个 repeat 的输入集合和 FP32 二叉归约树都没有改变，
只把独立行的发射顺序从 row-major 改为 half-major。

whole-tail 的第一层 API 数从 32 降为 2，8-row stage epilogue 从每个
stage 8 次降为 2 次。`false` 分支完整保留原逐行循环，runner 用
`--db-reduce coalesced|per-row` 构建两份 clean wheel。该候选必须分别
通过 37 项、极端 gate、repeated-launch、slot canary 和同卡 msprof；
静态等价映射不能替代设备上的编译、精度和性能证据。

### 4.9 跨 task 输出写回重叠候选

`KDA_GROUPED_OVERLAP_TASK_STORE` 默认 `false`，由 runner 记为 `store-serial`。候选
`true` 只允许与 `KDA_GROUPED_BATCH_TAIL_BLOCKS=true`、
`KDA_GROUPED_OVERLAP_STAGE_EPILOGUE=false` 组合：先按 whole-tail 的原顺序完成
`db += reduce(dk_left_pre*k)` 与 `dk_left_pre *= beta`，再把最终 `dq/dk/dg` 原位留在三个
accumulator bank。该 task 的浮点结果完成后，AIV 先推进 mixed-radix task 坐标，搬入下一 task
的 feature，并发布下一 task 的 stage 0 和 stage 1；这些区域都不与旧 accumulator/`db` prefix
重叠。随后才提交旧 task 的四个 strided MTE3 输出，让 AIC 可以在输出写回期间计算下一 task 的
前两个 Cube stage。AIV 仅在清零 accumulator 前等待 `MTE3_V`，形成稳态：

```text
task n:   consume3 -> finalize outputs in accumulators
task n+1: feature load -> prepare/publish C0 -> shared gates -> prepare/publish C1
task n:                                               -> MTE3(dq,dk,dg,db)
AIC n+1:                         C0 ----------------------> C1
AIV n+1:                                              wait MTE3 -> zero -> consume C0
```

这里不在旧输出之后额外插入 `MTE3_MTE2`：`MTE3_V` 保证旧输出已经停止读取 accumulator，
随后 Vector 清零 accumulator，而 `ConsumeStageAiv<0>` 在第一次读取新 workspace 前无条件执行
`V_MTE2`。因此形成 `old MTE3 -> V clear -> new MTE2` 的传递依赖；源码契约同时禁止在该热路径
重新加入会串行化流水的直接 `MTE3_MTE2` 等待。该静态时序仍需 repeated-launch 与 slot canary
在设备上验证。

候选没有改变任何单个输出内部的 FP32 顺序：`dk` 仍为 `(dk_in+dk_left)+dk_right`；`dg` 仍先
执行 `dg_in+q*dq_acc`，再加 `(dk_left-dk_right)*k`；`dq` 输入只在前一项读取完 `dq_acc` 后
原位累加。独立输出之间的发射顺序可以调整，但操作数、API、repeat 分组、归约树、17 个 GEMM
和 gate 计算均保持不变。源码以 UB 边界断言、事件顺序契约和非法组合 `static_assert` 保护该
分支；这些只能证明静态生命周期，仍必须以 `store-serial`/`store-overlap` 两个 clean wheel 的
37 项精度、repeated-launch、slot canary 和同卡 msprof 判断正确性与收益。

## 5. safe_gate 分支

- `safe_gate=true` 是主验证和主优化分支，并复现兄弟仓 Triton kernel 的 16-token 参考点分解。对当前子块之前的 left 项使用子块首 token 的 gate，对子块内 left/right 项使用中间 token 的 gate，对后续 right 项使用子块末 token 的 gate；内层累加完成后再乘参考点到目标 token 的外层因子。它不截断指数差，数学结果仍是 `exp2(g_i-g_j)`，但避免把长距离 gate 差一次送入 `Exp`，保留原实现的稳定计算顺序。
- `safe_gate=false` 精确复现上游普通分支的计算顺序：当前 16×16 对角块直接计算 FP32 `exp((g_i-g_j)*ln(2))`，此前/此后的跨块项仍分别围绕块首/块末 gate 因式分解。
- block-wise safe 路径按 source token 计算一次公共 gate，再通过 `Brcb` 和带 stride 的 FP32
  `Mul/Add` 广播到 16 个目标行；没有融合乘加，保持 legacy 的 `Mul -> Mul -> Add` 舍入顺序。
- host 默认使用 tiling key 5/7 实例化 FP16/BF16 block-wise safe 路径；0/2 保持 unsafe，
  1/3 保留 legacy safe 实例作为回退。热循环内没有 safe/unsafe runtime 分支。
- DAV_2201（A2/A3）上仅当 `BF16 + safe + BT64 + K128 + dense + T%64==0 + H==HV` 时选择 grouped key 23。FP16、unsafe、GVA、varlen、tail、BT128、其他 K 和 A5 均继续走已有 key；pair-wise key 15 作为同一目标域的源码级重新编包回退保留。当前实机验证目标仍是 A2，A3 需要独立编译与精度验证。

## 6. 精度策略

- q/k 在 UB 转 FP32；gate、dA 和输入累积梯度保持 FP32；key 15/23 Cube 的 A/B/C 也全部为 FP32，并显式关闭 HF32；
- 所有乘加、指数、K 维 reduce 均为 FP32；
- grouped key 23 对六个 off-diagonal pair 预计算 `bridge(s,e)=exp2(F_s-R_e)`；bridge 从
  color-4 UB 读取，通过 `src0RepStride=0` 的两条 mask-64 `Mul` 直接广播到 8 行，并与已驻留的
  `exp2(R_e-G_e)` / `exp2(G_s-F_s)` 相乘得到左右 inner gate。该写法保持 bridge 为 `Mul` 的
  `src0`、outer 为 `src1`，不改变 FP32 操作数顺序，同时省去每个 pair 的两次 8x128 中间复制
  和一次依赖 barrier。KDA 累积 gate 单调不增时三个
  因子均不大于 1，不会引入“大公共参考点”溢出；但 FP32 乘法会改变直接 `Exp` 的舍入/FTZ
  边界，因此源码保留直接 inner-gate 编译期开关，且必须通过极值 gate、FTZ、overflow、
  cancellation 和完整 NPU 回归后才能接受该优化；
- 工程关闭自动同步。block-wise safe 路径在 dA/beta 完成 `MTE2_S` 后，由 Scalar 一次性完成
  行 slab 重排、causal 清零和 `dAkk*beta` 系数合并，再用单个 `S_V` 将整个 coefficient block
  交给 Vector；source 热循环不再做逐 row UB 标量读取或 `S_V`。legacy 路径维持 EVENT4 已验证
  的逐 source 同步语义。grouped AIC/AIV 路径把四个 8-row beta tile 连续搬入 UB，经一次 `Brcb`
  形成 32 个广播块，再用两条带 stride 的 FP32 `Mul` 覆盖完整 32×128 cache；causal mask 和
  零向量由每个物理 AIV 在 task 循环前用 Scalar 构造一次并常驻，随后所有 task/stage 通过两条
  `Select` 清零无效 dA。六类同步事件在 AIV 生命周期开始时申请、每个同步点仍配对
  `SetFlag/WaitFlag`、全部 task 完成后统一释放；
- tail token 和 causal mask 通过循环边界实现，不读取 chunk/sequence 外数据；
- 基准重点覆盖大负累积 gate、非整 chunk、GVA (`HV/H>1`)、dense/varlen、FP16/BF16，以及 safe/unsafe 两个分支。

## 7. 性能路线

已验证的 AIV phase1 在目标 shape 上从 legacy 477.94 ms 降到 48.66 ms，但 profiling 显示 MTE2 仅约 2.9%，Vector/Scalar 计算与发射已成为主瓶颈。因此第二阶段引入上述 grouped 深融合和 AIC/AIV 双槽流水。key 23 已写入 host/kernel 并完成本地结构检查，但尚不能声明 NPU 精度或 4 ms 性能达标；上板顺序为默认 scoped 路径 clean wheel、单次 grouped 定向、完整 37 项、目标 shape msprof。只有默认路径稳定后，才按 AIC/AIV stall 数据选择跨 task store、stage epilogue 或局部 scratch 候选；persistent MMAD 必须先重建设备事件协议，不能直接参加性能 A/B。

## 8. 接口约束

内部输入为：`q,k,g,beta,dAqk,dAkk,dq,dk,db,dg`，可选 `cu_seqlens/chunk_indices`，属性 `chunk_size/safe_gate/total_chunks`；输出为 `dq_out,dk_out,db_out,dg_out`。所有输出均为新 tensor，不原地修改输入累积梯度。
