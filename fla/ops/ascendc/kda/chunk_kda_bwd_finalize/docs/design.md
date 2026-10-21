# A2/A3 Finalize Stage 划分

当前hybrid分支以H14为验收基线：Stage 8/10使用关闭HF32的FP32操作数，两个lower矩阵在L1跨带复用；Stage 7生产路径按32行生成band操作数，FP32输出双槽兼作运算临时区。完整生产路径的Stage 8先产生两个head的dq并通知Stage 9，再产生两个left；两个AIV在进入Stage 11前等待left就绪。Stage 11的dG在H≤4096时用对齐多行DataCopy写回，更大H及capture保持DataCopyPad；原双槽和写回事件不变。下文历史设计按日期保留，当前地址、生命周期、通知语义和验收证据见[两版融合优化](hybrid-plan.md)的H3/H4/H10/H14章节。H5b～H9、H11～H13未保留。

2026-10-21性能目标：A2，B=1/H=96/T=8192/K=V=128/BF16，定长BNSD，无l2norm和beta sigmoid反向，Stage01Kernel与Stage12Kernel的Task Duration之和低于8 ms。采用同输入、20次预热及20次采样、两轮交替A/B；算子语义和原精度门禁不变。

2026-10-17 协作实验：Stage9 两 AIV 各处理同一 head 的半带，先 head0 后 head1；Stage7 经 GM 交接原 FP32 指数，Stage11 保留原归属。八配置两轮 A/B 耗时增加9.47%～16.15%，不采用，代码已恢复 tile16。候选及证据在工作区 `validation/perf-coop-20261017`，不改变下述已验收设计。

方案来源：用户确认的 A2/A3 设计稿（2026-10-11）。通用规则参考仓库 V2；任务映射、Vector 分块和数据路径以本方案及 [A2/A3 准则](a2a3/A2A3stage划分准则.md) 为准。

本轮实现范围：Stage 0～12 的 A2 独立开发验证入口，公开接口尚未接入；接口见 [api.md](api.md)，验证记录见 [validation.md](validation.md)。

硬件预算：L1 512 KiB，L0A/L0B 各 64 KiB，L0C 128 KiB，每个 AIV 的 UB 192 KiB。

原生输出布局：Stage 3/9/11 分别写 dv/dq/dk。BNSD/NTD 沿用连续搬运；
BSND/TND 使用 DataCopyPad 多行搬运，每行 128 个 BF16，GM 行间隔为
`(H-1)*128*2` 字节。基址用 uint64_t，Host 检查 DMA stride 范围。
复用既有输出双缓冲及 MTE3 完成事件，不新增 UB、workspace 或转置 kernel。
dG/dBeta 的 BSND/TND 地址规则保持固定。

- **首版**：Stage 0/2/4/6 使用单份 BF16×BF16、FP32 累加，dW/Tza/kE/Zb 均不加残差；Stage 8/10 保留三项残差补算。Vector 中间计算与归约使用 FP32，搬运与同步参考 A2 前向 prepare。
- **提升精度**：逐 Stage 对照高精度标杆定位误差，仅给必要操作数加残差，优先单侧两项补算；残差必须由转换前的 FP32 生成。仍不满足时，再评估对应 GEMM 双输入转 FP32、关闭 HF32 的路径。
- **提升性能**：优先比较 Stage 8/10 的三项、单侧两项和单份 BF16，精度通过后再减项；再评估 Stage 8 沿 M、Stage 10 沿 K 拼接。拼接减少指令与同步次数，不减少乘加量，取舍以 A2 实测为准。

## Stage 0：Cube

```text
dk_state_raw = v_new @ dhᵀ
dv_b         = akkᵀ @ dv_scan
dW           = dv_scan @ hᵀ
zV           = dv_scan @ vᵀ
```

dW：FP32 L0C 经 Fixpipe（F322BF16）转 BF16 NZ 驻留 L1，每 head 16 KiB。

Stage 0 预取：两个 head 的输入先分别搬入既有 L1 lane；每 lane 独立 `MTE2_MTE1/MTE1_MTE2`，h0 就绪后即可计算，同时 MTE2 继续搬 h1。L0A/B 和 L0C 各保留两槽，将操作数搬运、MMAD 发射和结果写回分开排队：预载 dk_state_raw/dv_b，发射两次 MMAD，再在空闲槽预载 dW/zV；`M_FIX` 只在相应 Fixpipe 前等待。输入/结果槽分别按 `MTE1_M/M_MTE1`、`M_FIX/FIX_M` 保护；dW 独立驻留区不参与输入预取覆写。尾 head 不预取、不消费其 input-ready，结束时仅排空初始 free。

每 head 最后一次 Stage 0 L1→L0 读取后，按 `MTE1_MTE2` 释放 h/dh/v_new/dv_scan/v 的 112 KiB 区，立即预取本核下一 work 的同一 owner 输入；下一 work 只补搬 akk，再发布 input-ready。akk/dW 按各自消费事件保护；三阶段入口的提前推进见 Stage 2。首 work 全量装载，末 work 不预取。后续 Stage 7 须采用 [L1 分区](a2a3/L1内存.md) 中避开预取区的布局。

## Stage 1：Vector

```text
rH[k]   = Σv h[k,v] × dh[k,v]
exp2_gk = 2^gk
kE      = k ⊙ exp2_gk
gk_last = gk[valid_rows - 1, :]
```

kE 在 FP32 中计算后，以 RNE 转为单份 BF16，经 UB→GM→L1 供 Stage 2 使用。

Stage 1 双缓冲采用 32 行 tile（Stage 5 之后的 16 行 tile 不变）：W0/W1 与 W2/W3 分别存一份 h/dh BF16 输入，先预取两块；Cast 末次读取后释放输入槽，预取第 i+2 块与第 i 块归约并行。W4/W5、W6/W7、W8/W9 分别为 FP32 h、dh、乘积，W10 为归约临时。rH 完成后 W0～W3 改存两份原始 gk，W4～W7 改存 FP32 k/kE，W8/W9 改存两个 BF16 kE 输出。输入按 `MTE2_V/V_MTE2`、输出按 `V_MTE3/MTE3_V` 各两组事件保护；只在消费或同槽复用时等待，Stage 结束排空输出。k/exp2_gk 仅清零无效尾行，无效 gate 不进入 Exp，kE 尾行仍写零。工作区最高到 UB[176 KiB)，总预留仍为 192 KiB。

## Stage 2：Cube

```text
dKgb_raw = akkᵀ @ dW
zW       = dW @ kEᵀ
```

等待 dW 写入完成后，两次矩阵乘复用同一份 L1 数据；末次 L1→L0 搬运完成后释放。

首版均为 BF16×BF16、FP32 累加并写 GM。两个 head 的 kE 各占一份 L1，独立管理 ready/free；L0A/L0B/L0C 均用双缓冲。dKgb_raw 不等待 kE，仅 zW 消费前等待。保留 Stage 0 的跨 work 输入预取；以 Stage 0～2 全量精度和 msprof 实测验收。

三阶段开发入口中，kE 搬入 L1 后即可释放 GM 槽，让下一 work 的 Stage 1 与当前 Stage 2 重叠；akk 单独等待末次 L1 读取后再覆盖。接入 Stage 3～11 时，必须延后 Vector 提前推进，或双份保存后续仍需消费的 exp2_gk/rH；akk 的释放点改为 Stage 6 末次读取。

## Stage 3：Vector

```text
Zb = tril((zV − zW) ⊙ beta[None, :], -1)

dv   = beta ⊙ dv_b
db_v = Σv(v ⊙ dv_b)
```

Zb 在 FP32 中计算后，以 RNE 转为单份 BF16，经 UB→GM→L1 供 Stage 4 使用。

A2 实现按 32 行处理，输入、输出均双缓冲；先计算 Zb，再复用输入区计算 dv/db_v。dv 转 BF16 写回，FP32 beta/db_v 留在 UB；切换输入布局前等待两个旧槽都读完。

开发入口使用两代 GM 槽，使上一 work 的 Stage 3 与当前 Cube 重叠；Fixpipe 完成后才能读取，消费确认后才能复用。Stage 4 使用两代 akk；后续 Stage 5/6 仍须延长 exp2_gk/rH、db_v 和 akk 的生命周期。

## Stage 4：Cube

```text
Tza = Zb @ akkᵀ
```

Tza：FP32 L0C 经 Fixpipe（F322BF16）转 BF16 NZ 驻留 L1，每 head 8 KiB。

Stage 0～4 开发调度：等待上一代 Stage 3 的 ACK，双 head 预取 Zb 后计算 Tza。akk 使用 L1 两代槽，读完 Stage 4 才允许覆盖；Tza 也保留两槽，L1 总量仍为 384 KiB。后续接入 Stage 6 时，把 akk/Tza 释放点延至其末次读取。

## Stage 5：Vector

```text
dk_state = dk_state_raw ⊙ 2^(gk_last − gk)
dq_base  = scale × dq_raw ⊙ exp2_gk

gate_state = Σtoken(k ⊙ dk_state) + 2^gk_last ⊙ rH

dk_base = dk_state − beta ⊙ exp2_gk ⊙ dKgb_raw

db_base = db_v − Σk(k ⊙ exp2_gk ⊙ dKgb_raw)

dg_base = q ⊙ dq_base
        − k ⊙ dk_state
        − beta ⊙ k ⊙ exp2_gk ⊙ dKgb_raw

dg_base[valid_rows - 1, :] += gate_state
```

Stage 0～5 开发调度按 16 行双缓冲。Stage 1 按代保存 rH（512 B/head），Stage 5 重读原始 k/gk，并直接计算 `gk_last-gk` 后求指数；不依赖已被下一 work 覆盖的 UB。Stage 3 发 ACK 后，Cube Stage 4 与 Vector Stage 5 可并行；Stage 5 另发完成通知，阻止两代 GM 槽提前复用。

Stage 5 的 FP32 dq/dk 复用已消费的 dv_b/dk_state_raw 区，dg/db 放在槽内空闲区，双槽总量仍为 512 KiB/head。后续 Stage 需按新的保留位置消费，不能直接套用完整方案的旧偏移。

UB 使用两份 40 KiB 输入和两份 24 KiB 输出；ratio 写入 dk 后，其 8 KiB 临时区依次复用为 q 和 db 归约。exponent/product 分别错开 256/512 B，减少 bank 冲突，总 UB 仍为 192 KiB。输入末次读取后预取下一 tile，输出按槽等待；退出前排空输出，防止下一 Stage 覆盖。


## Stage 6：Cube

```text
dAkk_raw = akkᵀ @ Tza
```

等待 Tza 写入完成后读取；末次 L1→L0 搬运完成后释放。

A2 七阶段入口先完成两 head 的 Stage 4，再按 head 执行 Stage 6，与 Vector Stage 5 并行。akk/Tza 保持原两代 L1 地址；Stage 4 经 FIX_MTE1 发布 Tza，Stage 6 搬完两个操作数才释放 akk。后续同槽 MMAD/Fixpipe 按既有 L0 双缓冲事件排序，保护 Tza 的末次读取。使用 BF16×BF16、FP32 累加，dAkk_raw 以 FP32 ND 写入每代 GM 槽 `[216,232) KiB`，不与 Stage 5 写回重叠。总 L1/UB/workspace 不增加；测试 capture 新增 `[360,376) KiB`，每条记录扩至 384 KiB。

性能假设：每 head 新增一次 64³ GEMM（0.524 MFLOPs）和 16 KiB 写回；目标定长 case 合计新增约 6.44 GFLOPs、192 MiB GM 写回，没有新增 GM 输入。当前 Stage 5 以 Vector 为主，Stage 6 有机会隐藏在其计算中；最大流水占比能否提高以实测为准，不以新增工作量直接宣称加速。

后续 Stage 7 读取 dAkk_raw 时须等待独立的 Fixpipe 完成通知，不能将 S5_DONE 当作 Stage 6 ready。

## Stage 7：Vector

Stage 7～11 按 32 行输出带逆序执行；每带从原始 gk 计算中心，保留跨带因果依赖。

```text
// 每个 chunk 生成一次
dAqk_strict_lower =  2^16 × tril(dAqk, -1)
dAkk_strict_lower = −2^16 × tril(dAkk_raw, -1)
dAqk_diagonal     = diagonal(dAqk)

// s = 32 × floor((valid_rows − 1) / 32), ..., 0
e = min(s + 32, valid_rows)
center[k] = (gk[s,k] + gk[e−1,k]) / 2
exp2_gk_band = 2^(gk[s:e, :] − center)

kNeg[t, :]  = k[t, :] ⊙ 2^(center − gk[t, :])          (0 ≤ t < e)
qPos[t, :]  = q[t, :] ⊙ 2^(gk[t, :] − center)          (s ≤ t < valid_rows)
bkPos[t, :] = beta[t] × k[t, :] ⊙ 2^(gk[t, :] − center) (s ≤ t < valid_rows)
```

范围外直接写零，不计算指数。不得用已下溢的 exp2_gk 重建居中指数。

上述 Cube 操作数采用 BF16 高位＋残差；Stage 8/10 按三项 `Ah@Bh + Ah@Bl + Al@Bh` 计算，FP32 累加。

Stage 7 开发入口：16 行 tile，strict-lower 输入及高低位输出均双缓冲；q/k/原始 gk 每 chunk 搬入一次，qPos/bkPos 共用同 tile 的正向指数。对角线、center、带内指数保留 UB，只有采数版导出。等待 Stage 6 的 `S6_READY[bank]`（FIX，flag 6/7）。Gather 索引初始化一次，驻留 UB 84.75 KiB；进入、退出均经 `V_MTE2` 保护 UB 复用。截断到 Stage 7 时捕获两带，workspace 留 band 0，末带写完即释放；执行到 Stage 8 时增加下述带间消费握手及独立 L1 区。Stage 10 使用下述第二轮带间握手。

Stage 7 输出按实际产出顺序交替使用双槽。带内全零 tile 直接清零高低位两份输出；完整有效 tile 不预先清零临时量，尾 tile 保留清零和原精度转换。

## Stage 8：Cube

```text
dq_local_raw = dAqk_strict_lower[s:e, :] @ kNeg
left_raw     = dAkk_strict_lower[s:e, :] @ kNeg
```

开发入口首版：每带 32 行，两个 `32×64 @ 64×128` GEMM 各用三项 BF16 补算、FP32 累加，最后一次 Fixpipe 写 FP32 ND。L0A/B/C 各两槽，两结果交替装载、计算；不在补算项之间落 GM。

Stage 8 专用 L1 为 `[384,448)`、`[448,512)` KiB，分别属于两个 head；每区前 32 KiB 保存两个 strict-lower 的高低位，后 32 KiB 保存 kNeg 高低位。strict-lower 每 chunk 只搬一次，kNeg 每带搬一次。原 `[0,384)` 的下一代 akk、状态预取保持不动。复用 Stage 2/4 已释放的 `keReady/keFree` 事件，不新增同方向硬事件。

Stage 7 每带先生成完整 kNeg，从 MTE3 发 `BAND_READY=1`，随后生成 qPos/bkPos，与 Stage 8 并行；Cube 搬入两个有效 head 后从 MTE2 发 `BAND_LOADED=2`，Vector 才可覆盖该带 GM。无效奇数 head 仍参与每次握手。最后一带读取且全部 Vector 写回完成后释放该代 workspace；Cube 输出由自身顺序和 L0C free 事件保护。Stage 9/10 接入时还需保护带内指数和 qPos/bkPos，不能直接复用此提前释放条件。

开发 workspace 每 head 两代，每代扩为 320 KiB，新增 `[256,288)` 的 dq_local_raw、`[288,320)` 的 left_raw；全核为 `usedCoreNum×1280 KiB`。采数区扩为 720 KiB，656/688 KiB 起保存两结果；短于 32 行时未计算的第二带保持 guard。性能版不导出采数。采用提前发布 kNeg 的 r1，性能数据见 [Stage 0～8 记录](perf/stage012345678/README.md)。

## Stage 9：Vector

以下逐元素计算及输出均限于当前带 `[s,e)`，base、q/k、beta、对角线和 rstd 取对应切片。

```text
dq_local = dq_local_raw ⊙ exp2_gk_band × 2^-16

dq_pre = dq_base + dq_local + dAqk_diagonal[:, None] ⊙ k

dg = dg_base + q ⊙ dq_local

未启用 Q L2Norm backward：
    dq = dq_pre

启用 Q L2Norm backward：
    dq = (dq_pre − q ⊙ Σk(dq_pre ⊙ q)) ⊙ q_rstd
```

Stage 9 保持 16 行输入、输出双槽：先搬入已就绪的 dq_base/dg_base，再等待同 head 的 Stage 8 raw；q/k 转换和对角线广播提前到输入等待之前。UB/GM 布局、ready 通知次数与带尾排空规则不变。

## Stage 10：Cube

```text
right_raw = dAqk_strict_lower[:, s:e]ᵀ @ qPos
          + dAkk_strict_lower[:, s:e]ᵀ @ bkPos
```

Stage 10 开发入口沿用逆序 32 行带，与 Stage 9 重叠。每项计算 `Ah@Bh + Ah@Bl + Al@Bh`，六次 MMAD 在 FP32 L0C 中累加后一次写回。两个 head 分别使用 L0A/B/C 双槽；L1 保留 strict-lower，高低位 qPos、bkPos 依次复用 kNeg 的 32 KiB/head，仍为 512 KiB。

每带的 `BAND_READY/BAND_LOADED` 使用两轮握手：第一轮传 kNeg；第二轮在第一轮消费完成后发布 qPos/bkPos，Cube 完成 right_raw 写回后释放。Stage 9 不等待第二轮计算，下一带等待。right_raw 复用已消费的 workspace kNeg 槽（96 KiB，FP32 32×128/带），由 Stage 11 在第二轮握手后消费；workspace 仅保留最后一带，采数区另存两带。workspace 大小不变。

## Stage 11：Vector

逐元素计算及输出限于当前带 `[s,e)`；各输入取对应切片。

```text
left  = left_raw ⊙ exp2_gk_band × 2^-16
right = right_raw / exp2_gk_band × 2^-16

db_delta = Σk(left ⊙ k)
dk_delta = beta ⊙ left + right + dAqk_diagonal[:, None] ⊙ q

dg += k ⊙ (beta ⊙ left − right)

dk_pre = dk_base + dk_delta
db_eff = db_base + db_delta

未启用 K L2Norm backward：
    dk = dk_pre

启用 K L2Norm backward：
    dk = (dk_pre − k ⊙ Σk(dk_pre ⊙ k)) ⊙ k_rstd
```

**新增 beta sigmoid backward：**

```text
未启用 beta sigmoid：
    db_out = db_eff

启用 beta sigmoid：
    普通模式：
        db_out = db_eff × beta × (1 − beta)

    allow_neg_eigval=True：
        db_out = db_eff × beta × (1 − beta/2)
```

每带从后向前扫描 dg；后缀和及两个参数部分和在 chunk 开始时清零、跨带累加，全部带完成后写出 partial。

```text
dg_reverse_cumsum[t,k] = suffix[k] + Σ(i=t..e−1) dg[i,k]  (s ≤ t < e)
suffix[k]             = dg_reverse_cumsum[s,k]           // 本带结束后更新
gate_input[t,k]        = raw_g[t,k] + dt_bias[k]
exp_a_log              = exp(a_log)
gate_sigmoid[t,k]      = sigmoid(exp_a_log × gate_input[t,k])

d_g[t,k] = dg_reverse_cumsum[t,k] × lower_bound × exp_a_log × gate_sigmoid[t,k] × (1 − gate_sigmoid[t,k])

partial_d_dt_bias[k] += Σ(t=s..e−1) d_g[t,k]
partial_d_a_log      += Σ(t=s..e−1,k) d_g[t,k] × gate_input[t,k]
```

## Stage 12：Vector

```text
d_a_log[h]     = Σchunk partial_d_a_log[chunk,h]
d_dt_bias[h,k] = Σchunk partial_d_dt_bias[chunk,h,k]
```

Stage 11/12 开发入口：Stage 11 在每带第二轮握手后消费 right_raw，使用逆序 8 行输入、输出双槽。完整 8 行用三步并行反向扫描，复用已失效的 left 区；尾 tile 保留逐行扫描。复用 q/k、beta、对角线及带内指数；suffix 和参数 partial 跨带驻留。gate 导数用 `exp(-abs(x))/(1+exp(-abs(x)))²`，避免 `1-sigmoid(x)` 的消减误差。beta sigmoid 因子在 chunk 入口用 FP32 计算一次。

每 head/chunk 输出 136 个 FP32 partial（128 个 bias、8 个 a_log 分量），不使用原子加。Stage 12 在同一 stream 另启 AIV kernel，双槽预取 partial，按固定 chunk 顺序求和。此 kernel 边界提供全核同步；性能统计同时计入两个 kernel。开发接口的 dk 沿用输入布局，dG/dBeta 固定 BSND/TND；公开 A5 ABI 暂不变。

## 直接依赖

```text
Stage 0：算子输入
Stage 1：算子输入
Stage 2：Stage 0、1
Stage 3：Stage 0、2
Stage 4：Stage 3
Stage 5：Stage 0、1、2、3
Stage 6：Stage 4
Stage 7：原始 gk/q/k/beta/dAqk、Stage 6
Stage 8：Stage 7
Stage 9：Stage 5、7、8（同一带）
Stage 10：Stage 7
Stage 11：Stage 5、7、8、9、10（同一带），及后一带的扫描后缀和
Stage 12：所有相关 chunk 的 Stage 11
```

Stage 0/1 可重叠；Stage 5 可与 Stage 4/6 重叠；同一带的 Stage 9 可与 Stage 10 重叠。Stage 7 若复用 Stage 5 或上一带的工作区，需等待计算及 DMA 消费结束。Stage 12 等待所有 chunk 的全部带完成。

## 备选：复用前向结果简化计算图

以下为待验证候选。令 A=akk、E=k⊙2^gk；A 为下三角，且精确算术下满足：

```text
v_new = A @ diag(beta) @ (v − E @ h)
dv_b  = Aᵀ @ dv_scan
```

可在 dv_b 就绪后直接计算：

```text
// Cube
dKgb_raw = dv_b @ hᵀ
P        = dv_b @ v_newᵀ

// Stage 7 Vector：保留原有符号和缩放
dAkk_strict_lower = −2^16 × tril(P, -1)
```

- 省去 dW/zV/zW/Zb/Tza 及其搬运；Stage 3 保留 dv/db_v，原 Stage 4/6 由上述 Cube 计算替代。P 只用于严格下三角改写，不等同于原完整 dAkk_raw。
- 逻辑 GEMM 从 12 次减到 8 次。对 B=1、H=96、T=8192、K=V=128，保留 Stage 8/10 三项补算且不新增补偿时，理论 Cube 运算量约从 270.6 降至 231.9 GFLOPs（−14.3%），不代表实测加速。
- 启用前用真实前向 dump 对比中间量及最终梯度：BF16 舍入会破坏严格等式，任意独立输入不保证满足该关系。另行确定 dv_b 送入 Cube 的精度、布局和搬运路径；精度与 A2 性能均通过后，再更新 Stage、依赖及内存规划。

## dq 精度修复实验

2026-10-14 性能候选 R1/R1b 已放弃：全分支删除或仅 norm 分支删除 Stage 9
FP32 dq 写回，均出现非 norm 性能回退；恢复原调度。测量记录见 validation.md。

性能优化 R2（已保留）：Stage 7 将 FP32 输出与 BF16 高低位输出分开。
严格下三角阶段用 UB[178,182)/[182,186) KiB 两槽，每槽 FP32 16×64；
band 阶段复用已消费完的输入 UB[88,96)/[96,104) KiB 两槽，每槽 FP32 16×128。
前者在 Stage 7 矩阵 packing 时生产，后者在 kNeg packing 时生产；各自最后消费者为 MTE3。
原 q/k/gk、mask、beta、bandExp 与其余算术槽不变，峰值仍不超过 192 KiB。
额外分配两个 MTE3_V free event，进入 Stage 7 时初始化；写 FP32 槽前等待 free，
用既有 V_MTE3 ready 发布后搬出并重置 free，随即在独立槽计算 BF16 高低位。
复用该 FP32 槽时才等完成；每个 band 在进入 Stage 9/11 前收齐两槽 free 并重新初始化，
避免其 UB 重映射覆盖尚在搬出的数据。Stage 7 退出时收齐并释放两 event。
MTE3_V 共使用四个 event（原两槽加本轮两槽）；GM 地址和跨核协议不变。
两轮交替 A/B 的八项耗时降低 2.24%～4.24%；20 个 case 的七个输出全部通过原 CT L1 双标杆，保留此项。

局部矩阵乘以用户确认的纯 FP32 为目标。先只将 Stage 8 的
`dAqk_lower @ kNeg` 改为 FP32，并关闭 HF32，其余 GEMM 保持原路径。
Stage 7 额外保存 FP32 dAqk_lower 16 KiB、kNeg 32 KiB，每个 workspace bank
由 320 增至 368 KiB。Stage 8 先完成 left，再复用本 head 的 64 KiB L1
计算 dq；Stage 10 重载原 BF16 dAqk 两平面。L0 双槽、UB 双缓冲保留。
仅运行到 Stage 8 时，`BAND_LOADED` 在两种输入都读完后通知，防止下一带提前覆盖；
完整流水仍提前通知，由 Stage 9 的 `S8_READY` 等待保护 FP32 输入的生命周期。
先验证冻结的失败 chunk，再跑完整 200 case。实验通过前保留原库及失败记录。
FP32 搬运参考仓内 A2 前向 Prepare 的 LoadData2D/3Dv2 路径；本算子尺寸为
M=32、N=128、K=64，L0A/B 每槽分别使用 8/32 KiB，均不超过 32 KiB。

后续指数实验：Stage 7 按 `2^x = 2^n × exp((x-n)×ln2)` 计算，
`n=round(x)` 并限制在 [-126,127]，保证 `2^n` 可用 FP32 正常数表示。
范围内 Exp 参数不超过半个 ln2；范围外保留剩余指数，不截断原始 x。
复用 operand/rebuilt 临时区，无新增 UB。用于减少大指数乘 ln2 的舍入误差，
先检查冻结失败点，再检查尾块、布局和完整精度；设备结果未通过前不作采用结论。

## 完整 chunk 路径优化（2026-10-14）
Stage 7 入口按 validRows==64 分派一次，满块模板固定 Stage 7 行数和 Stage 9/11 每 tile 的 active；尾块保留通用路径。算术顺序、双缓冲、片上布局、workspace 和跨核协议不变。相对 R2 八项耗时降低1.86%～2.07%，20 case 的七个输出均通过原 CT L1，采用。

## Stage 11 参数梯度归约优化（2026-10-14）
满 8 行时，dt_bias partial 先按 4/2/1 行做树形归约，再累加一次；复用已结束 dA_log 计算的 product 区，gateOut 保留给 MTE3。尾块仍逐行累加。无需增加 UB 或同步事件，改变 FP32 加法顺序。相对完整块版本八项耗时降低0.47%～1.34%，20 case 七个输出均通过原 CT L1，采用。

## Stage 11 输入预取（2026-10-14）
先搬入前两个 tile 的 left/dk/dg/db/raw gate，再等待 Stage 10 的 BAND_LOADED，最后搬 right 并发布 inputReady。保留 Stage 9 输出完成与 UB 复用保护，不增加 buffer；后续 tile 在同一通知之后正常搬入。两轮 A/B 八项耗时降低 2.73%～3.48%，原 20 case 七个输出全部通过，采用。

## Stage 11 完整块清零裁剪（2026-10-16）
完整 chunk 的两次 WholeReduceSum 覆盖 sums 的全部 16 个元素，省去此前的 Duplicate 和对应 PIPE_V 等待；尾块保留清零。内存、公式和归约顺序不变。两组独立的八配置两轮 A/B 均有小幅收益，原 20 case、七个输出全部通过，采用。

## Stage 8/10 L0 残差复用（2026-10-16）
BF16 残差矩阵乘将 A/B 高低位各搬入一次，在现有 L0 双槽内复用，依次计算 HH、HL、LH。每槽 A 占8 KiB、B占32 KiB，容量不变；三项计算结束才释放操作数。Stage 8 的纯FP32 dq路径不变，Stage 10 两个矩阵的累加顺序不变。相对清零裁剪版，八配置两轮A/B耗时降低1.92%～2.80%，原20case七个输出全部通过，采用。

## Stage 8 删除无效 BF16 dAqk 搬运（2026-10-16）
Stage8首带只搬dAkk高低位平面；dq使用原FP32输入，不再搬随即被覆盖的BF16 dAqk。Stage10仍独立加载BF16 dAqk。内存、公式与同步不变。相对L0复用版八配置两轮A/B耗时降低0.56%～1.40%，原20case七输出全部通过，采用。

## Stage 1/3/5 完整 chunk 特化（2026-10-16）
入口按validRows==64选择FULL_CHUNK模板，固定前段行数；尾块保持原路径，运算、双缓冲和同步不变。无norm四配置两轮均有约0.41%～0.69%收益，norm四配置变化在-0.08%～+0.03%；全配置平均未过原自动0.5%筛选线，按稳定小收益单独复核后跑原20case，七输出全部通过，采用。仅无norm启用的变体实测回退，不采用。
## Stage 10 按 head 提前通知（2026-10-16）

Stage 10 每个 head 保持 HH/HL/LH 的原累加顺序。head 0 完成第二个矩阵并将 right 写回后，通过 S10_FIRST_READY 通知 AIV；head 1 随后完成。两条 AIV 各消费一次新通知，head 0 可提前计算 Stage 11，head 1 仍等原 BAND_LOADED。

head 0 在 Stage 11 结束前也消费 BAND_LOADED，保留整对 workspace 的复用保护。Stage 10 前缀和奇数 head 的空 owner 同样消费通知。双 buffer、片上容量、GM 地址宽度和精度计算不变，不新增功能函数。该调整已通过八配置两轮 A/B 和原 20 case 七输出验收；Stage 9/11 融合实验性能回退，未采用。

## Stage11 首 tile 预计算与末带16行（2026-10-17）

执行到Stage11的完整chunk非采数路径：前一带保留8行tile；启用norm时，将首tile的gate系数放到right等待之前计算，使用原gateTemp/denom，增加一次对应MTE2→V输入完成同步，不新增事件或容量。最后一带start=0改为16行tile，不提前gate计算；gateTemp/denom在q/k末次消费后复用其空间。尾块、Stage0–10前缀和采数路径保留原实现。本轮实际验收的是完整Stage0–12入口。

16行末带UB布局（KiB）：有效q[0,8)、k[16,24)；输入双40KiB槽[24,64)/[88,128)；lower mask[64,80)和小量[80,88)保留；query/key/product[128,152)；dk双4KiB槽[152,160)；指数[160,176)、beta广播[176,178)保留；gate双8KiB槽[8,16)/[184,192)；packedDb双512B槽[178,179)、sums起于179、rowScale起于180、db双64B槽起于181。最高地址192KiB，仅复用最后一带已死的q/k上半区和gk，lower mask不复用。

16行逐元素运算及搬运合批，逆向扫描仍先处理高8行、再处理低8行；dA_log/ddt_bias按同样的两组8行顺序累加，FP32运算和残差精度策略不变。输入/输出双buffer、原跨核通知和最后释放闭环保留，GM地址仍用64位。

末带dG驻留、Stage10 bkPos预取及全分支gate预计算实验均未保留。实测与精度见validation.md及工作区validation/perf-next-20261017。

## 局部流水候选复核（2026-10-17，均未采用）

依次测试最终dq私有写回裁剪、Stage7高低位合并搬运、第三BF16输出槽、Stage8搬32行、Stage10搬32列、整条band零区裁剪，并复核三个仅无norm版本。九版均未通过八配置双轮性能门禁，不改变当前双buffer、数据布局及计算路径。边界七输出逐位一致，但没有候选进入原20case正式精度验收。完整证据见工作区validation/perf-local-20261017/README.md。
