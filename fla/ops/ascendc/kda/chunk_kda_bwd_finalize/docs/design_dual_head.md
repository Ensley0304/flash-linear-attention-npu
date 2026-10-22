# chunk_kda_bwd_finalize Atlas A2系列产品双 work 流水设计方案

方案设计规则版本：`V2`

## 1. 目标

主模型 case：`B=1、H=32、T=11264、K=V=128、chunk=64`。
性能目标：`BasicInfo Task Duration < 4 ms`。

> 实测状态：本方案已淘汰。主 case Gate 输出未通过回归，Python 整调用约 `7.60 ms`；
> 8 行分片和重复 GM 搬入的开销大于双 work 隐藏的等待。本文保留为实验记录，不作为
> 当前实现依据；当前稳定实现见 [design_brief.md](design_brief.md)。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
| --- | --- | --- |
| API 输入 | `q/k/v/akk/vNew/h/dh/dvScan/beta` | BF16 |
| API 输入 | `gk/rawG/aLog/dtBias/dAqk/dqRaw` | FP32 |
| 可选 metadata | `cuSeqlens/chunkIndices` | INT64；主模型 case 不使用 |
| API 输出 | `dq/dk/dv/dBeta` | BF16 |
| API 输出 | `dG/dALog/dDtBias` | FP32 |
| Vector 计算 | 指数、缩放、逐元素运算和归约 | FP32 |
| Cube 输入与累加 | L0A/L0B 输入；L0C 累加 | BF16；FP32 |
| FP32 relay | `DkState/DVb/DKgb/Zv/Zw/DQ/DK/Right` | FP32 |
| BF16 relay | `KE/DW/Zb/Tza/dAkkRaw/A/Bmat/Kneg/Qpos/BKpos` | BF16 |

## 2. 变量与任务映射

| 符号 | 含义 |
| --- | --- |
| `Nc` | 本次 launch 使用的 logical core 数；每 core 为 `1 AIC + 2 AIV` |
| `logicalCore` | 当前 logical core 编号，`0<=logicalCore<Nc` |
| `Nchunk` | dense 时 `B*ceil(T/64)`；varlen 时为有效 chunk metadata 数 |
| `head` | 当前 head 编号，`0<=head<H` |
| `a` | AIV 编号，`a=0/1`，负责 token 行 `[32*a,32*a+32)` |
| `R` | 当前 chunk 有效行数，`0<R<=64` |
| `rows_a` | `max(0,min(32,R-32*a))` |
| `Tv` | Vector 行 tile，固定为 `8`；每个 AIV 最多处理 4 个 tile |
| `D` | 状态维，当前为 `128` |
| `Wn` | 当前 logical core 的第 `n` 个有效 work，按 head、再按本 core 的 chunk 顺序排列 |
| `s` | `Wn` 使用的本地 lane 和 relay 槽，`s=n mod 2` |
| `lane(s)` | 当前 AIV 的 UB work 上下文基址：`64*s KiB` |
| `meta(s)` | 当前 AIV 的 UB 小记录基址：`(184+1.5*s) KiB` |
| `slotBase(s)` | `(logicalCore*2+s)*180 KiB`；AIC 与两个 AIV 共享 |
| `l1Stream(s)` | `120*s KiB`；两个 work 的 Cube source stream 共 `240 KiB` |
| `Nphys` | 物理 AIV 数，`2*Nc` |
| `hasDtBias` | 存在 `dtBias` 时为 1，否则为 0 |
| `Align512(x)` | 将 `x` 向上对齐到 `512 B` |

任务归属与单 work 版本相同：每个 head 先把 chunk 切成 `Nc` 个连续段，再把起始 core
按 head 轮转。双 work 只改变同一 core 内的执行顺序，不改变 head 或 chunk 的归属。

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

## 3. Stage 总览与流水

### 3.1 当前阻塞与 4 ms 预算

主 case 有 `Nwork=176*32=5632` 个 work，20 个 logical core 平均处理 `281.6` 个 work。

| 项目 | 总时间 | 平均每 work |
| --- | ---: | ---: |
| 当前正式中位数 | `4530.30 us` | `16.09 us` |
| 跳过 Gate/参数归约诊断版 | `4110.28 us` | `14.60 us` |
| 4 ms 目标上限 | `4000 us` | `14.20 us` |

当前代码只有下一 work 的 `KE` 和 Cube `S1a` 提前执行，UB 中的 `E/Q/K/dqBase/dkBase/gBase`
仍只有一份。AIC 完成下一 work 的 `S1b` 后，必须等待 AIV 完成上一 work 的 S10/Gate，AIV
才能生成下一 work 的 `Zb`。因此稳态主阻塞为：

```text
AIC: S1a(n+1) -> S1b(n+1) -> 等待 Zb(n+1)
AIV:                     S10/Gate(n) -> S2(n+1) -> Zb(n+1)
```

跳过 Gate 后仍为 `4110.28 us`，说明 Gate 的可见关键路径约为 `1.49 us/work`，非 Gate
流水仍比目标多 `0.40 us/work`。新方案必须同时隐藏 S10/Gate 和至少一段 Vector/Cube
交接等待，目标按 `<=3.85 ms` 设计，给正式验收保留 `0.15 ms` 波动余量。

```text
Tnew = 4530.30 us - DeltaOverlap + DeltaMTE
```

其中 `DeltaOverlap` 是双 work 消除的 AIC/AIV 空泡，`DeltaMTE` 是重读 `dqRaw/DK` 的可见
代价。达到 `3.85 ms` 要求 `DeltaOverlap-DeltaMTE >= 680.30 us`；若 `DeltaMTE` 达到允许
上限 `150 us`，则双 work 至少要隐藏 `830.30 us`。这两个量必须通过同口径 profiling
验收，不能仅凭代码执行顺序认定达标。

### 3.2 双 work 执行规则

| Stage | 执行单元 | 主要动作 | 结束条件 |
| --- | --- | --- | --- |
| S0 | Vector | 准备 lane 的 `E/Q/K/F/KE/centerExp` | `V2C(s,0)` |
| S1 | Cube | WY 链 | `C2V(s,0)` |
| S2 | Vector | 先发布 `Zb`，再计算 `dkBase/gBase/dv` 与 Gate 早期记录 | `Zb ready` 后本 Stage 尾部与 S3 并行 |
| S3 | Cube | `Tza/dAkkRaw` | `C2V(s,1)` |
| S4 | Vector | 准备 `A/Kneg` | `V2C(s,2)` |
| S5 | Cube | `DQ` | `C2V(s,2)` |
| S6 | Vector | 消费 `DQ`，准备 `Bmat` | `V2C(s,3)` |
| S7 | Cube | `DK` | `C2V(s,3)` |
| S8 | Vector | 消费 `DK`，准备 `Qpos/BKpos` | `V2C(s,4)` |
| S9 | Cube | `Right` | `C2V(s,4)` |
| S10 | Vector | 最终 `dq/dk/gbar/Gate`，本地释放 lane | 完成后直接在同 lane 执行下一代 S0 |
| S11 | Vector | 归约 PA/PB records | 写出 `dALog/dDtBias` |

相邻两个 work 构成流水窗口，但不构成新的任务单位。下表表示依赖拓扑，不设置全局 phase
屏障；AIC/AIV 只等待当前 lane 的直接输入，其余时间继续推进另一 lane。

| phase | Cube | Vector |
| ---: | --- | --- |
| P0 | - | `S0(W0)` |
| P1 | `S1(W0)` | `S0(W1)` |
| P2 | `S1(W1)` | `S2-Zb(W0)` 后继续 `S2-Base(W0)` |
| P3 | `S3(W0)` | `S2-Zb(W1)` 后继续 `S2-Base(W1)` |
| P4 | `S3(W1)` | `S4(W0)` |
| P5 | `S5(W0)` | `S4(W1)` |
| P6 | `S5(W1)` | `S6(W0)` |
| P7 | `S7(W0)` | `S6(W1)` |
| P8 | `S7(W1)` | `S8(W0)` |
| P9 | `S9(W0)` | `S8(W1)` |
| P10 | `S9(W1)` | `S10(W0)` 后立即 `S0(W2)` |
| P11 | `S1(W2)` | `S10(W1)` 后立即 `S0(W3)` |
| P12 | `S1(W3)` | `S2(W2)` |

P12 以后按 P3-P11 的模式继续。窗口不跨 head：当前 head 的最后一对 work 排空并写完
PA/PB record 后，再开始下一 head。最后不足两个 work 时只执行有效 lane；无有效行的 AIV
仍参加同代同步。S10 与下一代 S0 在同一 Vector 时段顺序执行，只有 S10 完成最终读取、
写回和 Gate 累计后，才允许 S0 改写该 lane。

### 3.3 UB 与 L1 固定布局

每个 AIV 的两份上下文都保留在 UB；下表是整个 kernel 的固定地址。

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| lane 0 `E/Q/K` | UB | `[0,16) / [16,24) / [24,32) KiB` | shape=`[32,128]` FP32 / BF16 / BF16 |
| lane 0 `F→dkBase/gBase` | UB | `[32,48) / [48,64) KiB` | 两个 shape=`[32,128]` FP32；S2 后前者改义 |
| lane 1 `E/Q/K` | UB | `[64,80) / [80,88) / [88,96) KiB` | shape=`[32,128]` FP32 / BF16 / BF16 |
| lane 1 `F→dkBase/gBase` | UB | `[96,112) / [112,128) KiB` | 两个 shape=`[32,128]` FP32；S2 后前者改义 |
| Stage 工作面 | UB | `[128,184) KiB` | 14 个 4 KiB 面，按 8 行 tile 分时复用 |
| lane 0/1 小记录 | UB | `[184,185.5) / [185.5,187) KiB` | 各 1.5 KiB；`centerExp/dbBase/scanCarry/control` |
| 当前 head PA/PB 累计 | UB | `[187,188) KiB` | 共 1 份 1 KiB；S10 严格按 work 顺序更新 |
| 固定控制区 | UB | `[188,192) KiB` | 事件、mask、标量和组件固定预算 |

两份 lane context 和小记录共占 `132 KiB`，只剩 `60 KiB`；扣除 PA/PB 与固定控制后，
Stage 工作面可用 `56 KiB`。8 行 FP32 面为 `4 KiB`，可提供 14 个面并给输入/输出各两份
ping/pong；16 行 FP32 面为 `8 KiB`，只能提供 7 个面，不能同时容纳 S2/S10 的输入、计算
中间量和输出双缓冲。因此双 work 版本固定使用 8 行 tile。

每个 AIC 的 L1 使用两份 source stream 和一份共享 Cube 工作区。AIC 同一时刻只执行一个
Cube Stage，因此共享工作区不按 lane 加倍。

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| lane 0 source stream | L1 | `[0,120) KiB` | `H/dH/Vnew/dVscan/V/Akk` |
| lane 1 source stream | L1 | `[120,240) KiB` | 与 lane 0 相同 |
| `DW/KE` | L1 | `[240,256) / [256,272) KiB` | S1 共享工作区 |
| S3 工作区 | L1 | `[272,304) KiB` | `Zb/Tza/Akk` 两个读取视图 |
| Intra 工作区 | L1 | `[288,448) KiB` | S3 完成后改义，供 S5/S7/S9 |
| 预留 | L1 | `[448,512) KiB` | 组件固定开销与地址余量 |

`dqBase` 不跨 Stage 保留：S2 与 S6 各计算一次。`dkLeft` 不跨 Stage 保留：S8 为
`dBeta` 读取一次 `DK`，S10 为最终合并再读取一次。主 case 因此比单 work 版多读取
`dqRaw` 和 `DK` 各一次，共 `369.098752 MB`；换取第二份完整 UB 上下文，不新增 Base
的 GM 写回/读回，也不新增 `exp2`。这些读取使用 8 行 input ping/pong，在 AIC 执行另一
lane 的 Cube Stage 时搬入；若实测新增 MTE2 暴露时间超过 `0.15 ms`，本方案不能判定达标。

## 4. Stage 详细设计

### S0 · Vector：准备 lane 上下文

```text
E         = exp2(gk)
KE        = BF16(K * E)
F         = exp2(gk_last - gk)
centerExp = exp2(0.5 * (gk_first + gk_last))
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/Q/K` | UB | `lane(s)+[0,16) / [16,24) / [24,32) KiB` | 当前 AIV 的 source，上下文释放前保留 |
| `F` | UB | `lane(s)+[32,48) KiB` | S2 使用后原址改为 `dkBase` |
| `centerExp` | UB | `meta(s)+[0,0.5) KiB` | shape=`[128]` FP32，供 S4/S6/S8/S10 |
| `KE` | GM | slot `[112,128) KiB` | shape=`[64,128]` BF16，供 S1 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| API 输入 | GM | API tensor 地址 | 读取 `Q/K/gk`；无效行先置安全值 |
| 同 lane 上一代 | UB/GM | `lane(s)` 与 `slotBase(s)` | 仅在上一代 S10 本地释放后改写 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `gk/E/F` 工作面 | UB | `[128,136) KiB` | 两个 8 行 FP32 bank，搬运与计算交替 |
| `K/KE` 输出 bank | UB | `[136,140) KiB` | 两个 8 行 BF16 bank，MTE3 完成后复用 |
| mask/标量 | UB | `[180,184) KiB` | 尾行、首末行和控制记录 |

### S1 · Cube：WY 链

```text
DkState = Vnew @ dH_state^T
DVb     = Akk^T @ dVscan
Zv      = dVscan @ V^T
DW      = dVscan @ H_state^T
Zw      = DW @ KE^T
DKgb    = Akk^T @ DW
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DkState/DVb/DKgb` | GM | slot `[0,32) / [32,64) / [64,96) KiB` | 各 shape=`[64,128]` FP32，供 S2 |
| `Zv/Zw` | GM | slot `[96,112) / [112,128) KiB` | 各 shape=`[64,64]` FP32；`Zw` 覆盖已搬入 L1 的 `KE` |
| `DW` | GM | slot `[128,144) KiB` | shape=`[64,128]` BF16，Cube 链内部回读 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `H/dH` | L1 | `l1Stream(s)+[0,32) / [32,64) KiB` | 各 shape=`[128,128]` BF16 |
| `Vnew/dVscan/V` | L1 | `l1Stream(s)+[64,80) / [80,96) / [96,112) KiB` | 各 shape=`[64,128]` BF16 |
| `Akk` | L1 | `l1Stream(s)+[112,120) KiB` | shape=`[64,64]` BF16；保留至当前 lane 的 S3 |
| `KE/DW` | L1 | `[256,272) / [240,256) KiB` | 各 shape=`[64,128]` BF16；共享区分时复用 |
| API 矩阵与 `KE` | GM | API 地址与 slot `[112,128) KiB` | `V2C(s,0)` 后搬入当前 lane cache |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 共享工作区 | L1 | `[240,272) KiB` | 当前 S1 的 `DW/KE`，不按 lane 加倍 |
| WY A tile | L0A | `[0,4) / [4,8) KiB` | 双 bank，最大 shape=`[32,64]` BF16 |
| WY B tile | L0B | `[0,16) / [16,32) KiB` | 双 bank，最大 shape=`[64,128]` BF16 |
| WY 累加 | L0C | `[0,16) / [16,32) KiB` | 双 bank，最大 shape=`[32,128]` FP32 NZ |

### S2 · Vector：Zb 与 Base 梯度

```text
dqBase  = scale * dqRaw * E
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

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `dkBase/gBase` | UB | `lane(s)+[32,48) / [48,64) KiB` | 覆盖 `F` 并保留至 S10 |
| `dbBase` | UB | `meta(s)+[0.5,0.75) KiB` | shape=`[32]` FP32，供 S8 |
| `dv` | GM | API 输出地址 | BF16，只写有效行 |
| `Zb` | GM | slot `[148,156) KiB` | shape=`[64,64]` BF16，供 S3 |
| Gate 早期记录 | GM | slot `[144,146.5) KiB` | `gateCorrection/rh/expG` FP32，供 S10 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/Q/K/F` | UB | `lane(s)+[0,48) KiB` | S0 结果；S2 完成前不得覆盖 |
| WY 结果 | GM | slot `[0,128) KiB` | `C2V(s,0)` 后按 8 行 tile 读取 |
| `dqRaw/H/dH/V/beta` | GM | API tensor 地址 | Base 与归约输入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| relay/API 双 bank | UB | `[128,144) KiB` | 两组 8 行 FP32 输入 bank，按对象分时复用 |
| `dqBase/Ds/U/betaTerm` | UB | `[144,160) KiB` | 四个 8 行 FP32 面，按依赖覆盖 |
| `K/Q` cast 与归约 | UB | `[160,176) KiB` | 四个 8 行 FP32 面，按依赖覆盖 |
| `dv/Zb` staging | UB | `[176,180) KiB` | 两个 BF16 输出 bank |
| mask/小归约 | UB | `[180,184) KiB` | 三角 mask、`rh/dbBase` 临时 |

### S3 · Cube：dAkk 链

```text
Tza     = Zb @ Akk^T
dAkkRaw = Akk^T @ Tza
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `Tza` | GM | slot `[48,56) KiB` | shape=`[64,64]` BF16，两次 GEMM 间交接 |
| `dAkkRaw` | GM | slot `[112,120) KiB` | shape=`[64,64]` BF16，供 S6 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `Zb/Tza` | L1 | `[272,280) / [280,288) KiB` | 当前两次 GEMM 的 BF16 输入/中间量 |
| `Akk` | L1 | `l1Stream(s)+[112,120) KiB` | 当前 lane source stream 保留的 BF16 输入 |
| `Akk` 两个读取视图 | L1 | `[288,296) / [296,304) KiB` | 两次 GEMM 分别使用的布局 |
| `Zb/Akk` | GM | slot `[148,156) KiB` 与 API 地址 | `V2C(s,1)` 后读取 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| S3 工作区 | L1 | `[272,304) KiB` | AIC 串行使用；S3 完成后 `[288,304)` 可改作 Intra 输入 |
| S3 tile | L0A/L0B/L0C | 与 S1 相同地址分时复用 | BF16 输入、FP32 累加 |

### S4 · Vector：准备 DQ 输入

```text
A    = BF16(tril0(dAqk))
neg  = centerExp / E
Kneg = BF16(K * neg)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `A/Kneg` | GM | slot `[24,32) / [32,48) KiB` | shape=`[64,64] / [64,128]` BF16，供 S5 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/K/centerExp` | UB | lane source 与 `meta(s)` | S0 结果，继续保留 |
| `dAqk` | GM | API tensor 地址 | 当前 8 行矩阵输入 |
| `dAkkRaw` ready | GM | slot `[112,120) KiB` | S4 末尾消费 C2V 事件；数据保留到 S6 再搬入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `dAqk/E/neg` 双 bank | UB | `[128,152) KiB` | 8 行 FP32 搬运和计算 |
| `K/A/Kneg` staging | UB | `[152,164) KiB` | BF16 输入与双输出 bank |
| mask | UB | `[180,184) KiB` | 下三角和尾行控制 |

### S5 · Cube：DQ

```text
DQ = A @ Kneg
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DQ` | GM | slot `[148,180) KiB` | shape=`[64,128]` FP32，供 S6 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `A/Kneg` | L1 | `[288,296) / [320,336) KiB` | `V2C(s,2)` 后从 relay 搬入 |
| `A/Kneg` | GM | slot `[24,32) / [32,48) KiB` | 末次 MTE1 后允许改义 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| DQ tile | L0A/L0B/L0C | `[0,16) / [0,32) / [0,32) KiB` | 双 bank BF16 输入、FP32 累加 |

### S6 · Vector：消费 DQ 并准备 DK 输入

```text
pos     = E / centerExp
dqBase  = scale * dqRaw * E
dqDelta = DQ * pos
dq      = BF16(dqBase + dqDelta)
gBase  += Q * dqDelta
Bmat    = BF16(tril-1(dAkkRaw))
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `gBase` | UB | `lane(s)+[48,64) KiB` | 原址加入 DQ 对 Gate 的贡献 |
| `dq` | GM | API 输出地址 | BF16，只写有效行 |
| `Bmat` | GM | slot `[112,120) KiB` | 原位覆盖 `dAkkRaw`，供 S7 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/Q/gBase/centerExp` | UB | lane context 与 `meta(s)` | S0/S2 结果 |
| `DQ/dAkkRaw` | GM | slot `[148,180) / [112,120) KiB` | `C2V(s,2)` 后读取 |
| `dqRaw` | GM | API tensor 地址 | 重新计算 `dqBase`，不跨 Stage 保存 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DQ/dqRaw` 双 bank | UB | `[128,144) KiB` | 两组 8 行 FP32 输入 bank |
| `pos/dqBase/dqDelta/Qfp` | UB | `[144,160) KiB` | 四个 8 行 FP32 面，按依赖覆盖 |
| `dq` 输出 bank | UB | `[160,164) KiB` | 两个 8 行 BF16 bank |
| `dAkkRaw/Bmat` | UB | `[164,172) KiB` | FP32 mask/cast 与 BF16 staging |
| mask | UB | `[180,184) KiB` | 严格下三角控制 |

### S7 · Cube：DK

```text
DK = Bmat @ Kneg
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DK` | GM | slot `[32,64) KiB` | shape=`[64,128]` FP32，供 S8 与 S10 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `Bmat/Kneg` | L1 | `[304,312) / [320,336) KiB` | `V2C(s,3)` 后从 relay 搬入 |
| `Bmat/Kneg` | GM | slot `[112,120) / [32,48) KiB` | 末次 MTE1 后允许改义 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| DK tile | L0A/L0B/L0C | 与 S5 相同地址 | BF16 输入、FP32 累加 |

### S8 · Vector：消费 DK 并准备 Right 输入

```text
pos    = E / centerExp
DKpos  = DK * pos
dBeta  = dbBase + rowsum(K * DKpos)
Qpos   = BF16(Q * pos)
BKpos  = BF16(beta * K * pos)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `dBeta` | GM | API 输出地址 | BF16，只写有效行 |
| `Qpos/BKpos` | GM | slot `[64,80) / [96,112) KiB` | 各 shape=`[64,128]` BF16，供 S9 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/Q/K/centerExp/dbBase` | UB | lane context 与 `meta(s)` | S0/S2 结果 |
| `DK` | GM | slot `[32,64) KiB` | `C2V(s,3)` 后读取；保留至 S10 再读一次 |
| `beta` | GM | API tensor 地址 | `BKpos` 与 `dBeta` 输入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DK/beta` 双 bank | UB | `[128,144) KiB` | FP32/BF16 输入 bank |
| `pos/DKpos/Kfp/product` | UB | `[144,160) KiB` | 四个 8 行 FP32 面 |
| `Qpos/BKpos` bank | UB | `[160,168) KiB` | 两组 BF16 双 bank |
| 归约与输出 | UB | `[168,176) KiB` | `dBeta` 行归约和 staging |

### S9 · Cube：Right

```text
Right = A^T @ Qpos + Bmat^T @ BKpos
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `Right` | GM | slot `[0,32) KiB` | shape=`[64,128]` FP32，两项在 L0C 连续累加 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `Qpos/BKpos` | L1 | `[352,368) / [384,400) KiB` | S8 产生的右输入 |
| `A/Bmat` 转置视图 | L1 | `[416,424) / [432,440) KiB` | S4/S6 结果的转置读取布局 |
| 四个输入 | GM | slot `[64,80) / [96,112) / [24,32) / [112,120) KiB` | `V2C(s,4)` 后搬入 |
| `DK` | GM | slot `[32,64) KiB` | S9 不覆盖，保留至 S10 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| S9 工作区 | L1 | `[288,448) KiB` | 当前 Cube Stage 共享，不按 lane 加倍 |
| Right tile | L0A/L0B | `[0,16) / [0,32) KiB` | 两项 BF16 GEMM 分时复用 |
| Right 累加 | L0C | `[0,32) KiB` | FP32 原地累加并写出 |

### S10 · Vector：最终合并、Gate 与 lane 释放

```text
pos     = E / centerExp
neg     = centerExp / E
dkLeft  = beta * (DK * pos)
dkRight = Right * neg
dk      = BF16(dkBase + dkLeft + dkRight)
gbar    = gBase + K*(dkLeft - dkRight) + last_row_correction
Rgate   = reverse_suffix_scan(gbar, cross_AIV_carry)
p       = sigmoid(exp(aLog) * (rawG + dtBias))
dG      = Rgate * lowerBound * exp(aLog) * p * (1-p)
PA     += sum(dG * (rawG + dtBias))
PB     += rowsum(dG)
```

S10 从最后一个 8 行 tile 向前处理；`scanCarry` 保存下一 tile 的列向量和，因此无需保留
完整 `[32,128]` 的扫描临时。S10 完成后立即在同 lane 执行下一代 S0。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `dk/gbar/dG` | GM | API 输出地址 | 只写有效行 |
| `PA/PB` 累计 | UB | `[187,188) KiB` | 当前 head 共一份，S10 按 work 顺序在线累计 |
| PA/PB records | GM | relay 后持久区 | 每物理 AIV、每 head 各写一次，供 S11 |
| 下一代 lane context | UB/GM | 同 lane 与同 slot | 当前 S10 本地释放后由 S0 重新初始化 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `E/K/dkBase/gBase` | UB | lane context | 最终输出和 Gate 完成前不得覆盖 |
| `centerExp` | UB | `meta(s)+[0,0.5) KiB` | `pos/neg` 缩放 |
| `DK/Right` | GM | slot `[32,64) / [0,32) KiB` | `C2V(s,4)` 后读取；`DK` 为第二次读取 |
| Gate 记录 | GM | slot `[144,146.5) KiB` | S2 产生的修正量 |
| Gate 输入 | GM | API tensor 地址 | `rawG/aLog/dtBias` |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `DK/Right/Kfp/scale` | UB | `[128,144) KiB` | 四个 8 行 FP32 面，分批搬入和缩放 |
| `dkLeft/dkRight/gbar` | UB | `[144,156) KiB` | 三个 8 行 FP32 面，合并后改作扫描输入 |
| scan ping/pong | UB | `[128,136) KiB` | merge 完成后复用，逆向扫描双面 |
| `raw/sigmoid/product` | UB | `[136,148) KiB` | Gate 的三个 8 行 FP32 面 |
| BF16 输出与归约 | UB | `[156,168) KiB` | `dk` staging、PA/PB 归约临时 |
| `scanCarry` | UB | `meta(s)+[0.75,1.25) KiB` | shape=`[128]` FP32 |

### S11 · Gate 参数最终归约

```text
dALog[h]     = sum_over_physical_AIV(PA[physicalAIV,h])
dDtBias[h,:] = sum_over_physical_AIV(PB[physicalAIV,h,:])
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| `dALog/dDtBias` | GM | API 输出地址 | 每个 head 的最终参数梯度 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| PA records | GM | `PA_base+(physicalAIV*H+h)*32B` | 每 record 8 个 FP32 |
| PB records | GM | `PB_base+(physicalAIV*H+h)*512B` | shape=`[128]` FP32；无 bias 时不分配 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 当前 record/累加 | UB | Stage 内局部缓冲 | 完成一个 head 后复用 |

## 5. 跨 Stage 同步契约

每个 slot 只使用一条 Vector→Cube 令牌流和一条 Cube→Vector 令牌流。令牌含义由该
slot 的 Stage 计数确定；表中的下一次生产必须晚于同一事件的上一次消费。

| 事件 ID | 方向 | slot | 依次保护的数据 |
| ---: | --- | ---: | --- |
| 0 | Vector→Cube | 0 | `S0:KE` → `S2:Zb` → `S4:A/Kneg` → `S6:Bmat` → `S8:Right输入` → 下一代 `S0:KE` |
| 1 | Cube→Vector | 0 | `S1:WY` → `S3:dAkkRaw`（S4 末尾消费）→ `S5:DQ` → `S7:DK` → `S9:Right` |
| 2 | Vector→Cube | 1 | 与 ID 0 相同，作用于 slot 1 |
| 3 | Cube→Vector | 1 | 与 ID 1 相同，作用于 slot 1 |
| 4 | AIV1→AIV0 | - | S10 的 64 行 reverse scan carry |
| 5 | AIV0↔AIV1 | - | 当前 head PA/PB record 闭合 |

规则：

1. 每次 Vector→Cube 发布前，两个 AIV 先完成 mode `0x1` 汇合；`rows_a=0` 也参加。
2. 每个 slot 的生产者和消费者按表中顺序各推进一个 Stage 计数，不能跳过令牌。
3. S4 发布 `A/Kneg` 后必须消费当前 slot 的 `dAkkRaw ready`，Cube 才能在同一 C2V 事件上
   发布 `DQ ready`；S4 只消费事件，`dAkkRaw` 数据保留到 S6 搬入。
4. 不单独发布 `lane free`：S10 完成后先在本 AIV 内释放 lane，再执行下一代 S0；下一代
   `KE ready` 本身就是该 lane 已安全复用的证明，避免同一 flag 连续发布两次而中间没有消费。
5. S10 与下一代 S0 顺序复用同 lane；S0 完成后才发布下一代第一个令牌。
6. head 边界先等待两个 lane 的 S10 完成，再由两个 AIV 闭合当前 PA/PB record；下一 head
   从空窗口重新执行 P0。

### 5.1 独立推进伪代码

```text
Vector, 每个 head:
    S0(W0), S0(W1)
    S2(W0), S2(W1)
    S4(W0), S4(W1)
    S6(W0), S6(W1)
    S8(W0), S8(W1)
    S10(W0), S0(W2)                         # 复用 lane 0
    S10(W1), S0(W3)                         # 复用 lane 1
    后续窗口重复以上顺序

Cube, 每个 head:
    S1(W0), S1(W1)
    S3(W0), S3(W1)
    S5(W0), S5(W1)
    S7(W0), S7(W1)
    S9(W0), S9(W1)
    收到 W2/W3 的 KE 后进入下一窗口
```

伪代码只表示 lane 所有权和先后关系。实现时 S2 在发布 `Zb` 后继续 Base 尾部，S4 在发布
`A/Kneg` 后继续准备后续输入；AIC/AIV 不等待对方完成整个表格行。

## 6. 用户额外 workspace

只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统
workspace。双 work 使用现有两个 `180 KiB` relay 槽，不增加第三个槽或 Base 持久区。

```text
relayBytes = Nc * 2 * 180 KiB
paBytes    = Nphys * H * 32 B
pbBytes    = hasDtBias ? Nphys * H * 512 B : 0

relayBase  = 0
paBase     = Align512(relayBase + relayBytes)
pbBase     = Align512(paBase + paBytes)
workspace  = Align512(pbBase + pbBytes)
```

| 区域 | 数量与单份大小 | 主 case 大小 |
| --- | --- | ---: |
| relay | `Nc*2` 个槽，每槽 `180 KiB` | `7.37280 MB` |
| PA | `Nphys*H` 个记录，每个 `32 B` | `0.04096 MB` |
| PB | `Nphys*H` 个记录，每个 `512 B` | `0.65536 MB` |
| 用户额外 workspace | 上述区域按 `512 B` 对齐 | **`8.06912 MB`** |

主 case 为 `Nc=20、Nphys=40、H=32、hasDtBias=true`。workspace 由并发 core、双 slot
和最终参数记录决定，不随 `T` 或总 chunk 数线性增长。
