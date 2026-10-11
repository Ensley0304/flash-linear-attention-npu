# A2/A3 Finalize UB

每个 AIV 独占 192 KiB：AIV0 处理 h0，AIV1 处理 h1，两者采用相同地址表。保留当前 chunk 的数据；Stage 7～11 按 32 行输出带逆序执行，带内 Vector tile 最多 16 行。

以下偏移单位为 KiB，区间左闭右开。使用显式 TBuf 切片；下表已包含输入、输出和临时区，不再隐含添加 TQue、双缓冲或高阶 API 工作空间。

## 固定地址

| 区间 | 数据 | dtype / 生命周期 |
| --- | --- | --- |
| [0,16) | q | BF16，Stage 5～11 |
| [16,32) | k | BF16，Stage 1～11 |
| [32,64) | exp2_gk → 原始 gk | FP32；Stage 1～5 存指数，Stage 7 前重新读入原始 gk |
| [64,80) | exp2_gk_band | FP32，32×128；每带 Stage 7～11 |
| [80,80.25) | beta_eff | FP32，Stage 3～11 |
| [80.25,80.5) | db_v → db_base | FP32，Stage 3～11 |
| [80.5,80.75) | dAqk_diagonal | FP32，Stage 7～11 |
| [80.75,81) | q_rstd | FP32，可选 |
| [81,81.25) | k_rstd | FP32，可选 |
| [81.25,81.75) | rH | FP32，Stage 1～5 |
| [81.75,82.25) | gk_last | FP32，Stage 1～5 |
| [82.25,82.75) | gate_state | FP32，Stage 5 跨 tile 累加 |
| [82.75,83.25) | center | FP32，Stage 7 |
| [83.25,83.75) | suffix | FP32，Stage 11 跨 tile、跨带 |
| [83.75,84.25) | partial_d_dt_bias | FP32，整个 chunk |
| [84.25,84.75) | dt_bias | FP32，Stage 11 |
| [84.75,85) | partial_d_a_log、标量参数 | FP32，含对齐 |
| [85,88) | 元数据、对齐预留 | 3 KiB |
| [88,192) | W0～W12 工作槽 | 每槽 8 KiB |

工作槽地址为 `[88+8j,96+8j)`，j=0..12。一个槽容纳 16×128 FP32；半槽容纳相同形状 BF16，或 16×64 FP32。常驻区预留 88 KiB，工作区 104 KiB，合计 192 KiB；非活跃区域不代表可以额外申请缓冲。

## 各 Stage 工作区复用

下表列出同一时刻需要保留的量；箭头表示旧值最后消费后原址更新。API 不支持原址重叠时，使用该行剩余工作槽中转。

| Stage | 工作槽安排 |
| --- | --- |
| 1 | 32 行 tile；W0/W1、W2/W3：两份 h/dh BF16 输入；W4/W5：FP32 h；W6/W7：FP32 dh；W8/W9：乘积；W10：归约临时。rH 完成后 W0～W3 改存 gk ping/pong，W4～W7 改存 FP32 k/kE，W8/W9 改存 BF16 kE 输出 ping/pong。W11/W12 空闲。 |
| 3 | 先处理 zV/zW/Zb，再复用槽处理 dv_b/v。Zb、dv 转 BF16 后写出；db_v 留在常驻区。两部分不同时保留大矩阵。 |
| 5 | W0：dk_state_raw→dk_state；W1：dKgb_raw→exp2_gk×dKgb_raw；W2：dq_raw→dq_base；W3：gk→衰减因子；W4/W5：q/k FP32；W6/W7：dk_base/dg_base；W8～W12：乘积、归约及搬运临时。 |
| 7 | 先分 tile 生成两个 strict_lower 高低位及对角线；再逐个生成 kNeg、qPos、bkPos。W0/W1：q/k FP32；W2/W3：正/负居中指数；W4：当前操作数；W5：重建值/残差；W6：BF16 高低位；W7～W12：mask、转换及临时。 |
| 9 | W0：dq_local_raw→dq_local；W1：dq_base→dq_pre；W2：dg_base→dg；W3/W4：q/k FP32；W5：乘积；W6：dq BF16；W7～W12：L2Norm 归约及临时。 |
| 11，K/beta | W0/W1：left_raw/right_raw→left/right；W2：dk_base→dk_pre；W3：dg；W4/W5：q/k FP32；W6：乘积；W7：dk/db 输出；W8～W12：归约及临时。 |
| 11，gate | 保留 W3 的 dg，等 dk/db 写出后复用其余槽。W0：raw_g；W1：gate_input；W2：sigmoid；W4：逆向扫描结果；W5：d_g；W6～W12：扫描、归约及转换临时。 |

Stage 5 按 token 正序处理，gate_state 跨 tile 累加；最后一个有效 tile 修正 dg_base 末行，再写 GM。dq_base/dk_base/dg_base 均以 FP32 暂存 GM，db_base 留 UB。

Stage 7 高低位由原始 FP32 操作数计算：
`hi=BF16_RNE(x)，lo=BF16_RNE(x−FP32(hi))`。两个 strict_lower 每 chunk 生成一次；三个居中操作数每带重算。先屏蔽无效行和三角位置，再计算指数或缩放，不用 exp2_gk 恢复原始 gk。

Stage 9 的 dg 按当前带写 GM，Stage 11 再读。Stage 11 在带内从后向前处理最多 16 行；每个 tile 都将扫描首行更新到 suffix，下一 tile/下一带沿用。suffix、两个 partial 在 chunk 开始清零，只有最后一带结束后才写 partial。

## 搬运与尾块

- 单个工作槽复用前同时满足 Vector 最后读取、MTE2 写入及 MTE3 读取完成；仅发出写回指令不能覆盖。
- q/k/rstd 等按实际 layout 搬入连续 UB；不足 16 行用 mask 和零填充。beta sigmoid 开启时使用保存的 FP32 beta_eff。
- dG、dBeta 按接口固定 BSND/TND 写出，dq/dk/dv 按 outputLayout 写出；非对齐拷贝按有效长度使用 DataCopyPad，不越界补写。
- Stage 1 为两个 32 行输入 tile 和两个 kE 输出预留独立地址及 ready/free 事件；提前两块搬入，Cast/Muls 末次消费后释放输入，输出同槽复用前等待 MTE3。FP32 计算临时按 Vector 顺序复用。其余 Vector Stage 的双缓冲留待实现时独立核算。
- 归约/扫描 API 的实际 scratch 必须落入表中剩余槽。若目标 CANN 要求更大 scratch，带宽仍保持 32 行，只缩小内层 Vector tile；不得超过 192 KiB。

## Stage 12：独立 kernel

不保留主 kernel 的 UB。dt_bias 任务处理一个 head 的 32 个 K 通道，每批最多 64 个 chunk：输入 8 KiB、累加及对齐 1 KiB、临时 16 KiB，共预留 25 KiB。

a_log 任务处理连续 8 个 head，每批最多 64 个 chunk：读取各 partial 的 32 B 记录共 16 KiB，累加及对齐 1 KiB、临时 16 KiB，共预留 33 KiB。两类任务复用 UB；最终输出只写有效元素。任务与 GM 地址见 [workspace 与同步](Workspace与同步.md)。
