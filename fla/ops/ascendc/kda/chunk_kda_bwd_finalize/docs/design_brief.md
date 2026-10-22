# chunk_kda_bwd_finalize Atlas A2系列产品设计方案

## 1. 目标

主模型 case：`B=1、H=32、T=11264、K=V=128、chunk=64`。
性能目标：`BasicInfo Task Duration < 4 ms`。

当前无竞态实现主 case 30 次正式样本的中位数为 `3720.244 us`，平均值为
`3724.532 us`，范围为 `3668.913~3774.015 us`。完整样本与验证结果见
[validation.md](validation.md)。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
| --- | --- | --- |
| API 输入 | `q/k/v/akk/vNew/h/dh/dvScan/beta` | BF16 |
| API 输入 | `gk/rawG/aLog/dtBias/dAqk/dqRaw` | FP32 |
| 可选 metadata | `cuSeqlens/chunkIndices` | INT64；主模型 case 不使用 |
| API 输出 | `dq/dk/dv/dBeta` | BF16 |
| API 输出 | `dG/dALog/dDtBias` | FP32 |
| Vector 计算 | 指数、缩放、逐元素运算和归约 | FP32 |
| Cube 输入与累加 | L0A/L0B 输入；L0C 累加 | BF16；FP32；Intra 输入由 Vector FP32 结果转换为 BF16 high-only，不计算 low residual 项 |
| FP32 relay | `DkState/DVb/DKgb/Zv/Zw/DQ/DK/Right` | FP32 |
| BF16 relay | `KE/DW/Zb/Tza/dAkkRaw/A/Bmat/Kneg/Qpos/BKpos` | BF16 |

## 2. 变量与任务映射

| 符号 | 含义 |
| --- | --- |
| `CG` | 每个 logical core 同时推进的 head 数；本方案固定 `CG=1` |
| `Nc` | 本次 launch 使用的 logical core 数；每 core 为 `1 AIC + 2 AIV` |
| `logicalCore` | 当前 logical core 编号，`0<=logicalCore<Nc` |
| `Nchunk` | dense 时 `B*ceil(T/64)`；varlen 时为有效 chunk metadata 数 |
| `H` | head 数 |
| `head` | 当前 head 编号，`0<=head<H` |
| `a` | AIV 编号，`a=0/1`，负责 token 行 `[32*a,32*a+32)` |
| `R` | 当前 chunk 有效行数，`0<R<=64` |
| `rows_a` | `max(0,min(32,R-32*a))` |
| `D` | 状态维，当前为 `128` |
| `b/localChunk` | 当前 batch 编号和 batch 内 chunk 编号 |
| `tokenStart` | dense 为 `localChunk*64`；varlen 为 `seqBegin+localChunk*64` |
| `stateIndex` | dense 为 `localChunk`；varlen 为当前有效 chunk metadata 编号 |
| `tokenOffset(W)` | API token tensor 的元素偏移；dense 为 `((b*H+head)*T+tokenStart)*W`，varlen 为 `(head*T+tokenStart)*W` |
| `laneTokenOffset(W,a)` | 当前 AIV 的 API token tensor 元素偏移：`tokenOffset(W)+32*a*W` |
| `stateOffset` | API state tensor 的元素偏移；dense 为 `((b*ceil(T/64)+stateIndex)*H+head)*D*D`，varlen 为 `(stateIndex*H+head)*D*D` |
| `laneStateOffset(a)` | 当前 AIV 的 API state tensor 元素偏移：`stateOffset+64*a*D` |
| `workOrdinal` | 当前 logical core 的有效任务序号，从 0 开始，按 head、再按本 core 的 chunk 顺序递增 |
| `s` | relay 槽号，`s=workOrdinal mod 2`；双槽用于相邻 work 的 ping-pong，不表示两个 head 同时常驻 |
| `slotBase` | 当前 logical core 中 AIC 与两个 AIV 共享的 GM relay 槽基址：`(logicalCore*2+s)*180 KiB` |
| `Nphys` | 物理 AIV 数，`Nphys=2*Nc` |
| `physicalAIV` | 物理 AIV 编号，`physicalAIV=2*logicalCore+a` |
| `hasDtBias` | 是否存在 `dtBias` 输入，存在时为 1，否则为 0 |
| `Align512(x)` | 将 `x` 向上对齐到 `512 B` |
| `gateScratchBase` | 全部 relay 槽之后的 Gate 跨 AIV 临时区基址，公式见第 5 章 |
| `PA_base/PB_base` | PA/PB 持久记录区基址，公式见第 5 章 |

每个 head 都按“连续切段，再轮转 core”的方式分配 chunk：

```text
q = Nchunk div Nc
r = Nchunk mod Nc

for head = 0 .. H-1:
    chunkBegin = 0
    for rank = 0 .. Nc-1:
        logicalCore = (head + rank) mod Nc
        chunkCount  = q + (rank < r ? 1 : 0)
        logicalCore 处理当前 head 的 [chunkBegin, chunkBegin + chunkCount)
        chunkBegin += chunkCount
```

例如 `Nchunk=10、Nc=4`，先切为长度 `3、3、2、2` 的四个连续段：

| head | chunk `[0,3)` | chunk `[3,6)` | chunk `[6,8)` | chunk `[8,10)` |
| ---: | --- | --- | --- | --- |
| 0 | core 0 | core 1 | core 2 | core 3 |
| 1 | core 1 | core 2 | core 3 | core 0 |

因此每个 core 在同一 head 内只处理一段连续 chunk；下一个 head 将起始 core 后移一位，
避免总由固定 core 承担较长段。主 case `Nchunk=176、Nc=20` 时切为 16 段 9 个 chunk
和 4 段 8 个 chunk。

## 3. Stage 总览

| Stage | 执行单元 | 主要动作 | 结束条件 |
| --- | --- | --- | --- |
| S0 | Vector | 准备 `KE`、`E/Q/K` 和早期 Base | `KE_READY` |
| S1 | Cube | WY 链，分 S1a/S1b | `BASE0_READY`、`BASE1_READY` |
| S2 | Vector | `BasePrologue`、`Zb`、Base 梯度和 Gate 早期记录 | `ZB_READY` |
| S3 | Cube | `Tza`、`dAkkRaw` | `DAKK_READY` |
| S4 | Vector | 先准备 DQ 输入，再与 Base 尾部交错准备 DK/Right 输入 | 三个独立 input ready |
| S5 | Cube | `DQ` | `DQ_READY` |
| S6 | Vector | 立即消费 `DQ` | DQ relay 可复用 |
| S7 | Cube | `DK` | `DK_READY` |
| S8 | Vector | 立即消费 `DK` | DK relay 可复用 |
| S9 | Cube | `Right` | `RIGHT_READY` |
| S10 | Vector | 消费 Right、最终输出、Gate，并准备下一 work 的 `KE` 与 BaseEarly | `WORK_DONE` |
| S11 | Vector | 跨物理 AIV 归约 PA/PB | 写出 `dALog/dDtBias` |

## 4. Stage 详细设计

### S0 · Vector：source 与早期 Base

```text
E         = exp2(gk)
KE        = BF16(K * E)
dqBase    = scale * dqRaw * E
F         = exp2(gk_last - gk)
centerExp = exp2(0.5 * (gk_first + gk_last))
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `E/Q/K` | UB | `[0,16) / [16,24) / [24,40) KiB` | 不涉及 | shape=`[32,128]`；`E/K` 为 FP32，`Q` 为 BF16，驻留至当前 work 的最终消费 |
| `dqBase/F/gk` | UB | `[40,56) / [56,72) / [72,88) KiB` | 不涉及 | shape=`[32,128]` FP32；Base 早期结果与原始指数输入 |
| `centerExp` | UB | `[186.25,186.75) KiB` | 不涉及 | shape=`[128]` FP32；S4/S10 缩放向量 |
| `KE` | GM | slot `[112,128) KiB` | 使能 | 两 AIV 各写 32 行，供 S1b；每 work 写一次、S1b 读一次，跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `Q/K/gk` | GM（API 输入） | `q_base/k_base + laneTokenOffset(D,a)*2 B`，各最大 `8 KiB`；`gk_base + laneTokenOffset(D,a)*4 B`，最大 `16 KiB` | 使能 | 每 work 各搬入一次，在 UB 驻留并由后续多个 Stage 消费；无效行先置安全值 |
| `dqRaw` | GM（API 输入） | `dqRaw_base + laneTokenOffset(D,a)*4 B`；当前 AIV 最大 `16 KiB` | 不使能 | 每 work 单次流式读取且无 GM 复用；采用依据见 `validation.md` |
| 当前 relay slot | GM workspace | `slotBase=(logicalCore*2+s)*180 KiB`；槽内 `[0,180) KiB` | 使能 | 单槽 `180 KiB`；每 logical core 两槽共 `360 KiB`；`s=workOrdinal mod 2` 选择当前槽，`WORK_DONE` 前不得被后续同槽 work 覆盖 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| bank-group padding | UB | `[88,88.25) KiB` | 不涉及 | `256 B` 空洞，使工作面相对常驻输入错开 8 个 bank group |
| 六个工作面 | UB | `[88.25,184.25) KiB` | 不涉及 | 六个 16 KiB 区域按 Stage 分时改义 |
| 小记录与固定量 | UB | `[184.25,192) KiB` | 不涉及 | Base/Gate 小向量、PA/PB 累计、bias、`exp(aLog)` 与常量 1 |

### S1 · Cube：WY 链

```text
S1a: DkState = Vnew @ dH_state^T
     DVb     = Akk^T @ dVscan
S1b: Zv      = dVscan @ V^T
     DW      = dVscan @ H_state^T
     Zw      = DW @ KE^T
     DKgb    = Akk^T @ DW
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `DW` | L1 | `[240,256) KiB` | 使能 | 从 GM relay 回读，作为 `Zw/DKgb` 的输入 |
| `DkState/DVb/DKgb` | GM | slot `[0,96) KiB` | 使能 | FP32 Base 输入；每 work 各写一次、两个 AIV 各读自己的 32 行，跨 Stage relay |
| `Zv` | GM | slot `[96,112) KiB` | 使能 | FP32 写出，供 S2 直接读取；每 work 写一次、两个 AIV 分片读取 |
| `Zw` | GM | slot `[112,128) KiB` | 使能 | `KE` 搬入 L1 后覆盖原地址，FP32 写出供 S2 读取；跨 Stage relay |
| `DW` | GM | slot `[128,144) KiB` | 使能 | BF16 写出并回读 L1，只在 Cube 链内部复用；写后重读 |
| 下一 work 的 WY 结果 | GM | 另一 slot `[0,144) KiB` | 使能 | 当前 Right 完成后执行下一 work 的 S1a/S1b；`BASE0_READY` 立即发布，`BASE1_READY` 等当前 `WORK_DONE` 后发布；跨 work 预取 relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `KE` | L1 | `[256,272) KiB` | 使能 | 从 S0 GM relay 搬入，作为 `Zw` 右输入 |
| `Akk` | L1 | `[272,280) KiB` | 使能 | S1b 从 API GM 搬入，作为 `DKgb` 左输入 |
| `KE` | GM | slot `[112,128) KiB` | 使能 | S0 发布 `KE_READY` 后读取一次并驻留 L1；跨 Stage relay |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| 当前 GEMM 的 A/B 工作面 | L1 | `[0,64) / [64,128) KiB` | 使能 | 从 API GM 或 GM relay 按当前 GEMM 搬入，末次 MMAD 后覆盖 |
| WY A tile | L0A | `L0A[0,16) KiB` | 不涉及 | 单缓冲，shape 最大为 `[64,128]` BF16，`1份 × 16 KiB = 16 KiB`；六个 GEMM 顺序复用，末次 MMAD 后释放 |
| WY B tile | L0B | `L0B[0,32) KiB` | 不涉及 | 单缓冲，shape 最大为 `[128,128]` BF16，`1份 × 32 KiB = 32 KiB`；六个 GEMM 顺序复用，末次 MMAD 后释放 |
| WY 累加 | L0C | `L0C[0,32) KiB` | 不涉及 | 单缓冲，shape 最大为 `[64,128]` FP32 NZ，`1份 × 32 KiB = 32 KiB`；每个 GEMM 的 FIX 完成后复用 |

S1b 只搬入一次 `dVscan`，先以 `V` 为另一输入计算 `Zv`，再切换为 `H_state` 计算
`DW`。`DW` 以 BF16 写入 slot `[128,144) KiB`，再回读到固定 L1 `[240,256) KiB`；
随后结合固定 L1 中的 `KE/Akk` 计算 `Zw/DKgb`。

### S2 · Vector：Zb 与 Base 梯度

```text
rh      = rowsum(H * dH)
dbBase  = rowsum(V * DVb)
dv      = beta * DVb
Zb      = BF16(tril-1((Zw - Zv) * beta))
Ds      = DkState * F
U       = DKgb * E
dkBase  = Ds - beta * U
gBase   = Q*dqBase - K*Ds - K*(beta*U)
dbBase -= rowsum(K * U)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `dkBase/gBase` | UB | `[168.25,184.25) / [72,88) KiB` | 不涉及 | shape=`[32,128]` FP32；保留到 S10 最终合并 |
| `dv` | GM（API 输出） | `dv_base + laneTokenOffset(D,a)*2 B`；写 `rows_a*D*2 B`，最大 `8 KiB` | 不使能 | 当前 chunk 的 `dv`；由本 Stage UB 工作面写出，每个有效元素只写一次且当前 kernel 不再读取，采用依据见 `validation.md` |
| `Zb` | GM | slot `[148,156) KiB` | 使能 | BF16 严格下三角矩阵，供 S3；避开仍被另一 AIV 读取的 FP32 `Zw`；跨 Stage relay |
| `gateCorrection/rh` | GM | slot `[144,145) / [145,146) KiB` | 使能 | 各 shape=`[2,128]` FP32，供 S8/S10 最终修正；`expG` 保留在 UB，不写 relay；多消费者记录 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `E/Q/K/dqBase` | UB | `[0,56) KiB` | 不涉及 | S0 常驻数据；按各自最后消费者保护 |
| WY 结果 | GM | slot `[0,128) KiB` | 使能 | `DkState/DVb/DKgb/Zv/Zw`；等待 S1a/S1b 对应 ready 后读取；跨 Stage relay |
| `H/dH` | GM（API 输入） | `h_base/dh_base + laneStateOffset(a)*2 B`；各读取 `64*D*2 B = 16 KiB` | 使能 | state 归约输入，同一 work 内复用 |
| `beta` | GM（API 输入） | `beta_base + laneTokenOffset(1,a)*2 B`；读取 `rows_a*2 B`，最大 `64 B` | 使能 | Base 与后续 Gate 计算复用 |
| `V` | GM（API 输入） | `v_base + laneTokenOffset(D,a)*2 B`；读取 `rows_a*D*2 B`，最大 `8 KiB` | 使能 | Base 计算输入，同一 work 内复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| Base 工作面 | UB | `[88.25,184.25) KiB` | 不涉及 | 六个连续工作面各 `16 KiB`，`6份 × 16 KiB = 96 KiB`；每个 AIV 将 64 行 state 分成两个 32 行 tile，按计算顺序与 DkState/DKgb MTE2、逐元素计算及 `dkBase` 驻留分时复用 |
| 小归约区 | UB | `[184.25,192) KiB` | 不涉及 | `rh/dbBase`、mask 与 Base/Gate 固定小记录分时复用 |

`BASE0_READY` 前只执行 S0 的 `dqBase/F/centerExp`。收到 `BASE0_READY` 后计算
`rh/dv/dbBase`；收到 `BASE1_READY` 后生成并发布 `Zb`，随后先转入 S4 生成
`A/Kneg`，再返回完成 `Ds/U/dkBase/gBase`。

### S3 · Cube：dAkk 链

```text
Tza     = Zb @ Akk^T
dAkkRaw = Akk^T @ Tza
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `Tza` relay | GM | slot `[48,56) KiB` | 使能 | 避开同时存活的 Base 数据；同一 Cube 链写后重读 |
| `dAkkRaw` | GM | slot `[112,120) KiB` | 使能 | 覆盖已消费的 `Zw`，供 S4；跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `Zb` | GM | slot `[148,156) KiB` | 使能 | S2 `ZB_READY` 后作为第一个 GEMM 输入；跨 Stage relay |
| `Akk` | GM（API 输入） | `akk_base + tokenOffset(64)*2 B`；当前块 shape=`[64,64]` BF16，大小 `8 KiB` | 使能 | 两个 GEMM 分别按转置视图读取，存在复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| S3 A/B 工作面 | L1 | `[0,64) / [64,128) KiB` | 使能 | 每个 GEMM 从 GM 搬入操作数，末次 MMAD 后覆盖 |
| S3 A tile | L0A | `L0A[0,8) KiB` | 不涉及 | 单缓冲，shape=`[64,64]` BF16，`1份 × 8 KiB = 8 KiB`；两个 GEMM 顺序复用，末次 MMAD 后释放 |
| S3 B tile | L0B | `L0B[0,8) KiB` | 不涉及 | 单缓冲，shape=`[64,64]` BF16，`1份 × 8 KiB = 8 KiB`；两个 GEMM 顺序复用，末次 MMAD 后释放 |
| S3 累加 | L0C | `L0C[0,16) KiB` | 不涉及 | 单缓冲，shape=`[64,64]` FP32 NZ，`1份 × 16 KiB = 16 KiB`；每个 GEMM 的 FIX 完成后复用 |

### S4 · Vector：分批准备 Intra 输入

```text
A     = BF16(tril0(dAqk))
Bmat  = BF16(tril-1(dAkkRaw))
center = exp2(0.5*(gk_first + gk_last))
neg    = center / E
pos    = E / center
Kneg   = BF16(K * neg)
Qpos   = BF16(Q * pos)
BKpos  = BF16(beta * K * pos)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `A` | GM | slot `[128,136) KiB`，大小 `8 KiB` | 使能 | shape=`[64,64]` BF16；复用已消费的 DW relay，供 DQ Cube 读取 |
| `Kneg[0]` | GM | slot `[136,144) KiB`，大小 `8 KiB` | 使能 | AIV0 写 shape=`[32,128]` BF16；供 DQ/DK Cube 读取 |
| `Kneg[1]` | GM | slot `[156,164) KiB`，大小 `8 KiB` | 使能 | AIV1 写 shape=`[32,128]` BF16；供 DQ/DK Cube 读取；与 `Kneg[0]` 合成 shape=`[64,128]` |
| `Bmat` | GM | slot `[112,120) KiB` | 使能 | `DAKK_READY` 后生成，再发布 DK input ready；后续 DK/Right 复用 |
| `Qpos/BKpos` | GM | slot `[64,80) / [96,112) KiB` | 使能 | 最后发布 Right input ready；跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `E/Q/K/centerExp` | UB | `[0,40) / [186.25,186.75) KiB` | 不涉及 | S0 驻留数据 |
| `dAkkRaw` | GM | slot `[112,120) KiB` | 使能 | S3 完成后原位改写为 `Bmat`；跨 Stage relay |
| `dAqk` | GM（API 输入） | `dAqk_base + tokenOffset(64)*4 B`；当前块 shape=`[64,64]` FP32，大小 `16 KiB` | 不使能 | `A` 的输入；每 work 单次流式读取且无 GM 复用，采用依据见 `validation.md` |
| `beta` | GM（API 输入） | `beta_base + laneTokenOffset(1,a)*2 B`；读取 `rows_a*2 B`，最大 `64 B` | 使能 | `BKpos` 的输入，并在 Base 与 Gate 计算中复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| cast/scale 工作面 | UB | `[88.25,168.25) KiB` | 不涉及 | 三组输入分批生成，避免五组同时占 UB |
| mask/搬运区 | UB | `[152.25,168.25) / [184.25,188.25) KiB` | 不涉及 | 上下三角 mask、cast 输出与小量搬运分时复用 |

`Zb` 发布后立即生成 `A/Kneg` 并发布 DQ input ready，使 S3 与 DQ 输入准备并行。
两个 AIV 的 `Kneg` 分片分别写入 `[136,144)` 和 `[156,164) KiB`，避免覆盖另一 AIV
尚未读完的 `DVb [32,64) KiB`；AIC 将两片搬入连续 L1 后执行 DQ/DK。
随后 S2 完成 Base 梯度。收到 `DAKK_READY` 后原位生成 `Bmat` 并发布 DK input ready，
最后生成 `Qpos/BKpos` 并发布 Right input ready。

### S5 · Cube：DQ

```text
DQ = A @ Kneg
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `DQ` | GM | slot `[148,180) KiB` | 使能 | `[64,128]` FP32，发布 `DQ_READY`；跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `A/Kneg` | L1 | `[288,296) / [320,336) KiB` | 使能 | S4 input ready 后搬入；`Kneg` 由两个 GM 分片拼成连续 shape=`[64,128]`，供 DQ/DK 复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| DQ A tile | L0A | `L0A[0,8) KiB` | 不涉及 | 单缓冲，shape=`[64,64]` BF16，`1份 × 8 KiB = 8 KiB`；MMAD 后释放 |
| DQ B tile | L0B | `L0B[0,16) KiB` | 不涉及 | 单缓冲，shape=`[64,128]` BF16，`1份 × 16 KiB = 16 KiB`；MMAD 后释放 |
| `DQ` 累加 | L0C | `L0C[0,32) KiB` | 不涉及 | 单缓冲，shape=`[64,128]` FP32 NZ，`1份 × 32 KiB = 32 KiB`；FIX 写完后释放 |

### S6 · Vector：消费 DQ

```text
dqDelta = DQ * pos
dq      = BF16(dqBase + dqDelta)
gBase  += Q * dqDelta
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `gBase` | UB | `[72,88) KiB` | 不涉及 | 累加 DQ 对 gate 梯度的贡献 |
| `dq` | GM（API 输出） | `dq_base + laneTokenOffset(D,a)*2 B`；写 `rows_a*D*2 B`，最大 `8 KiB` | 不使能 | 只写 `rows_a` 有效行；每个有效元素只写一次且当前 kernel 不再读取，采用依据见 `validation.md` |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `dqBase/Q/E/centerExp` | UB | `[40,56) / [16,24) / [0,16) / [186.25,186.75) KiB` | 不涉及 | 计算 `pos` 与最终 dq |
| `DQ` | GM | slot `[148,180) KiB` | 使能 | S5 `DQ_READY` 后两个 AIV 各读取自己的 32 行，末次读取后可复用 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| DQ/scale 工作面 | UB | `[88.25,168.25) KiB` | 不涉及 | MTE2、逐元素乘加和输出 cast 分时复用 |

### S7 · Cube：DK

```text
DK = Bmat @ Kneg
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `DK` | GM | slot `[32,64) KiB` | 使能 | `[64,128]` FP32；覆盖已消费的 `DkState/DVb` relay；跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `Bmat/Kneg` | L1 | `[304,312) / [320,336) KiB` | 使能 | S4 发布 DK input ready 后搬入；`Kneg` 复用 S5 已拼接的连续 L1 数据 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| DK A tile | L0A | `L0A[0,8) KiB` | 不涉及 | 单缓冲，shape=`[64,64]` BF16，`1份 × 8 KiB = 8 KiB`；S5 结束后复用原址 |
| DK B tile | L0B | `L0B[0,16) KiB` | 不涉及 | 单缓冲，shape=`[64,128]` BF16，`1份 × 16 KiB = 16 KiB`；S5 结束后复用原址 |
| `DK` 累加 | L0C | `L0C[0,32) KiB` | 不涉及 | 单缓冲，shape=`[64,128]` FP32 NZ，`1份 × 32 KiB = 32 KiB`；FIX 写完后释放 |

### S8 · Vector：消费 DK

```text
dkLeft = beta * (DK * pos)
dBeta  = dbBase + rowsum(K * (DK * pos))
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `dkLeft` | UB | `[104.25,120.25) KiB` | 不涉及 | shape=`[32,128]` FP32；保存至 S10 与 `dkRight` 合并 |
| `dBeta` | GM（API 输出） | `dBeta_base + laneTokenOffset(1,a)*2 B`；写 `rows_a*2 B`，最大 `64 B` | 不使能 | 写当前有效 token 的结果；每个有效元素只写一次且当前 kernel 不再读取，采用依据见 `validation.md` |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `K/E/centerExp/dbBase` | UB | `[24,40) / [0,16) / [186.25,186.75) / [184.75,184.875) KiB`，有效共 `32.625 KiB` | 不涉及 | 计算 `pos`、逐行归约 |
| `DK` | GM | slot `[32,64) KiB` | 使能 | S7 `DK_READY` 后两个 AIV 分片读取；跨 Stage relay |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| DK/归约工作面 | UB | `[88.25,168.25) KiB` | 不涉及 | `DK` 搬入、缩放和行归约分时复用 |

### S9 · Cube：Right

```text
Right = A^T @ Qpos + Bmat^T @ BKpos
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `Right` | GM | slot `[0,32) KiB` | 使能 | L0C FP32 累加后以 `[64,128]` FP32 写出；跨 Stage relay |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `A/Bmat` 转置面 | L1 | `[416,424) / [432,440) KiB` | 使能 | 两项 GEMM 的左输入；从 GM relay 搬入 |
| `Qpos/BKpos` | L1 | `[352,368) / [384,400) KiB` | 使能 | S4 Right input ready 后从 GM relay 搬入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| Right A 输入槽 | L0A | `L0A[0,8) KiB` / `L0A[8,16) KiB` | 不涉及 | 两份顺序槽，各 shape=`[64,64]` BF16，`2份 × 8 KiB = 16 KiB`；分别承载 `A^T/Bmat^T` |
| Right B 输入槽 | L0B | `L0B[0,16) KiB` / `L0B[16,32) KiB` | 不涉及 | 两份顺序槽，各 shape=`[64,128]` BF16，`2份 × 16 KiB = 32 KiB`；分别承载 `Qpos/BKpos` |
| `Right` 累加 | L0C | `L0C[0,32) KiB` | 不涉及 | 单缓冲，shape=`[64,128]` FP32 NZ，`1份 × 32 KiB = 32 KiB`；连续累加两项 GEMM 后以 FP32 写出 |

### S10 · Vector：最终合并、Gate 与下一 work 预取

```text
dkRight = Right * neg
dk      = BF16(dkBase + dkLeft + dkRight)
gbar    = gBase + K*(dkLeft - dkRight) + last_row_correction
Rgate   = reverse_suffix_scan(gbar, cross_AIV_carry)
p       = sigmoid(exp(aLog) * (rawG + dtBias))
dG      = Rgate * lowerBound * exp(aLog) * p * (1-p)
PA     += sum(dG * (rawG + dtBias))
PB     += rowsum(dG)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| 下一 work `E/Q/K/gk` | UB | `[0,40) / [72,88) KiB` | 不涉及 | 当前 work 的最终 DK 和 Gate scan 不再读取旧 source 后，由 `BuildKe` 换入下一 work source |
| 下一 work `dqBase/F/centerExp` | UB | `[40,72) / [186.25,186.75) KiB` | 不涉及 | 当前 Gate 的 V 读完成后生成；与当前 `dG` 的 MTE3 写回重叠，下轮 S0 直接复用 |
| `PA/PB` 累计 | UB | `[188.75,188.78125) / [189.25,189.75) KiB` | 不涉及 | shape=`[8] / [128]` FP32；当前物理 AIV、当前 head 在线累计 |
| `dk` | GM（API 输出） | `dk_base + laneTokenOffset(D,a)*2 B`；写 `rows_a*D*2 B`，最大 `8 KiB` | 不使能 | 只写当前有效 token，每个有效元素写一次且当前 kernel 不再读取；采用依据见 `validation.md` |
| `dG` | GM（API 输出） | `dG_base + laneTokenOffset(D,a)*4 B`；写 `rows_a*D*4 B`，最大 `16 KiB` | 使能 | 只写当前有效 token；不使能 L2 Cache 会导致非有限值，回退依据见 `validation.md` |
| `cross_AIV_carry` | GM workspace | `gateScratchBase+logicalCore*8 KiB`，有效区 `[0,512) B` | 使能 | shape=`[128]` FP32；AIV1 写、AIV0 读，当前 work 消费后供下一 work 覆盖；跨 AIV 读改写记录 |
| 下一 work `KE` | GM | 另一 slot `[112,128) KiB` | 使能 | Gate 前预取并发布下一代 `KE_READY`；跨 work/Stage relay |
| PA record | GM workspace | `PA_base+(physicalAIV*H+head)*32 B`；单 record `32 B` | 使能 | head 结束时写一次，供 S11；跨 kernel 持久记录 |
| PB record | GM workspace | `PB_base+(physicalAIV*H+head)*512 B`；单 record `512 B` | 使能 | 有 bias 时 head 结束写一次，供 S11；跨 kernel 持久记录 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `E/K/dkBase/gBase` | UB | `[0,16) / [24,40) / [168.25,184.25) / [72,88) KiB` | 不涉及 | 直到最终输出和 Gate 所需读操作完成才改变语义 |
| `dkLeft` | UB | `[104.25,120.25) KiB` | 不涉及 | S8 结果 |
| `Right` | GM | slot `[0,32) KiB` | 使能 | S9 `RIGHT_READY` 后两个 AIV 分片读取 FP32；跨 Stage relay |
| `rawG` | GM（API 输入） | `rawG_base + laneTokenOffset(D,a)*4 B`；读取 `rows_a*D*4 B`，最大 `16 KiB` | 不使能 | 每 work 单次流式读取且无 GM 复用；采用依据见 `validation.md` |
| `aLog/dtBias` | GM（API 输入） | `aLog_base + head*4 B`，读取 `4 B`；`dtBias_base + head*D*4 B`，读取 `512 B` | 使能 | Gate 参数输入，在当前 head 的计算中复用 |
| Gate 修正记录 | GM | slot `[144,146) KiB` | 使能 | S2 写入，S8/S10 读取；跨 Stage、多消费者记录 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| Right/scan/Gate 工作面 | UB | `[40,56) / [88.25,184.25) KiB` | 不涉及 | `Right` 搬入、每 AIV 32 行的五级 FP32 逆向扫描、sigmoid 导数和归约分时复用 |
| Gate 小记录与常量 | UB | `[184.25,192) KiB` | 不涉及 | 修正量、PA/PB、bias、`exp(aLog)` 与固定驻留且仅初始化一次的常量 1 |

消费 `DK` 后、等待 `RIGHT_READY` 期间先计算 `p`；`Right` 的 MTE2 搬入与 `1-p`、
`p*(1-p)` 计算重叠。当前 Gate 完成 V 读取后先建立
`V->MTE2` 依赖，再生成下一 work BaseEarly；此时当前 `dG` 可继续通过 MTE3 写回。
`WORK_DONE` 必须晚于当前输出、PA/PB 累计、`dG` 写回和下一 work BaseEarly；只有该事件
完成后，当前 slot 才允许在下一个同奇偶 work 中回绕。

### S11 · Gate 参数最终归约

```text
dALog[h]      = sum_over_physical_AIV(PA[physicalAIV,h])
dDtBias[h,:]  = sum_over_physical_AIV(PB[physicalAIV,h,:])
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `dALog/dDtBias` | GM（API 输出） | `dALog_base + head*4 B`，每 head 写 `4 B`；`dDtBias_base + head*D*4 B`，每 head 写 `512 B` | 使能 | 全部 head 合计分别为 `128 B/call` 和 `16 KiB/call` |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| PA records | GM | `PA_base+(physicalAIV*H+h)*32B` | 使能 | 每 record 8 个 FP32，仅有效 lane 参与归约；跨 kernel 持久记录 |
| PB records | GM | `PB_base+(physicalAIV*H+h)*512B` | 使能 | shape=`[128]` FP32；无 bias 时不分配；跨 kernel 持久记录 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| 当前 record/累加 | UB | `[0,24) KiB` | 不涉及 | 逐物理 AIV 读取并归约，完成一个 head 后复用 |

## 5. 用户额外 workspace

只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统
workspace。每 logical core 使用两个 `180 KiB` relay 槽；每个槽是当前 work 的 GM 临时
地址池，由 AIC 与两个 AIV 共享，没有统一 tensor shape。槽内对象和复用条件见对应 Stage。

```text
relayBytes = Nc * 2 * 180 KiB
gateScratchBytes = Nc * 8 KiB
paBytes          = Nphys * H * 32 B
pbBytes          = hasDtBias ? Nphys * H * 512 B : 0

relayBase       = 0
gateScratchBase = relayBase + relayBytes
paBase          = Align512(gateScratchBase + gateScratchBytes)
pbBase          = Align512(paBase + paBytes)
workspace       = Align512(pbBase + pbBytes)
```

| 区域 | 数量与单份大小 | 主 case 大小 |
| --- | --- | ---: |
| relay | `Nc*2` 个槽，每槽 `180 KiB` | `7.37280 MB` |
| Gate scratch | 每 logical core 一个 `8 KiB` 记录；当前仅使用前 `512 B` 保存跨 AIV carry | `0.16384 MB` |
| PA | `Nphys*H` 个记录，每个 `32 B` | `0.04096 MB` |
| PB | `Nphys*H` 个记录，每个 `512 B`；无 `dtBias` 时不分配 | `0.65536 MB` |
| 用户额外 workspace | 上述区域按 `512 B` 对齐 | **`8.23296 MB`** |

主 case 为 `Nc=20、Nphys=40、H=32、hasDtBias=true`。该容量由并发 logical core 数和
head 数决定，不随 `T` 或总 chunk 数线性增长；relay 槽在 `WORK_DONE` 后按 work 轮转复用。
relay、Gate scratch 和 PA/PB 均存在跨 Stage、跨 AIV 或跨 kernel 的生产消费关系，空间表中的
L2Cache 统一为“使能”；它们不属于单次流式搬运的 bypass 候选。

## 6. 跨 Stage 同步契约

模式 `0x2` 是 `1C2V` 集体同步；即使某个 AIV 的 `rows_a=0`，也必须参与同代事件。
表中的“集合通知”已经包含两个 AIV 的汇合，不再额外叠加同语义的 AIV 间屏障。

| 事件 ID | 第一次语义 | 复用语义 | 被保护的数据 | 发布要求 |
| ---: | --- | --- | --- | --- |
| 0 | `KE_READY` | `RIGHT_INPUT_READY` | `KE`；后续 `Qpos/BKpos` | `KE` 写出完成后集合通知；Right 输入发布前保留 AIV 间屏障 |
| 1 | `BASE0_READY` | `DK_READY` | `DkState/DVb`；后续 `DK` | Cube 结果完成后通知 |
| 2 | `BASE1_READY` | `RIGHT_READY` | `DKgb/Zv/Zw/DW`；后续 `Right` | Cube 结果完成后通知；下一代 `BASE1_READY` 等当前 `WORK_DONE` |
| 3 | `ZB_READY` | `DK_INPUT_READY` | `Zb`；后续 `Bmat` | `Zb` 写出完成后集合通知；DK 输入发布前保留 AIV 间屏障 |
| 4 | `DAKK_READY` | 无 | `dAkkRaw` | Cube 结果完成后通知 |
| 5 | `DQ_INPUT_READY` | 无 | `A/Kneg` | 两个 AIV 直接集合通知；`Kneg` 分片不与 `DVb` 重叠，不增加额外 AIV 间屏障 |
| 6 | `DQ_READY` | 无 | `DQ` | Cube 结果完成后通知 |
| 7 | `WORK_DONE` | 无 | 当前 slot 全部输出、读者和 Gate 累计 | 输出与读者闭合后集合通知，不重复增加 AIV 间屏障 |

执行顺序：

```text
Vector: KE -> BaseEarly -> wait S1a -> BasePrologue
        -> wait S1b -> Zb -> DQ输入(A/Kneg) || S3
        -> BaseGradients -> wait dAkk -> DK输入(Bmat) -> Right输入(Qpos/BKpos)
        -> wait/消费 DQ -> wait/消费 DK -> 计算p
        -> wait Right -> Right MTE2 || 计算(1-p)和p*(1-p)
        -> 最终dk/gbar -> Gate scan -> 下一work KE -> 当前Gate
        -> V到MTE2依赖 -> 下一work BaseEarly || 当前dG写回
        -> WORK_DONE

Cube:   S1a -> wait KE -> S1b -> wait Zb -> S3
        -> wait DQ输入 -> DQ -> wait DK输入 -> DK
        -> wait Right输入 -> Right -> 下一work S1a -> wait 下一work KE
        -> 下一work S1b -> wait 当前WORK_DONE -> 发布下一work BASE1_READY
```
