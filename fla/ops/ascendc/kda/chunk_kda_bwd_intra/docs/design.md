# ChunkKdaBwdIntra AscendC 设计

## 1. 目标与范围

本算子迁移 `flash-linear-attention` 中的 `chunk_kda_bwd_kernel_intra`，计算 KDA 单个 chunk 内部对 `q`、`k`、`beta` 和逐特征 gate 的梯度贡献。主交付路径为 `safe_gate=true`，同时保留独立的 `safe_gate=false` 编译分支。

首版支持 Ascend A2/A3/A5、`float16`/`bfloat16` 的 `q/k`、FP32 的 L0 gate/其余输入和四个输出，`K` 为 16 的倍数且位于 `[16, 256]`，`chunk_size` 为 64 或 128。通用 kernel 以 `BK=32` 分块，并用实际 `curK=16` 处理尾块；严格限定的 key12～key15 目标 shape 使用 `BK=64`/全 K128 Cube。公开 Python 接口允许 BF16 `g/beta` 并在进入 L0 前提升到 FP32。L0 内部布局统一为 BNSD；Python 接口负责 BSND/BNSD/TND/NTD 的无歧义转换。

## 2. 数学语义

对 chunk 内 token `i` 和特征 `d`，记 `E(i,j,d)=exp2(g[i,d]-g[j,d])`。仅保留 causal 的 `j<=i` 项：

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

每个 `(chunk, value_head, 16-token row block)` 是一个独立任务。任务内部再按 32 个 K 特征切片：

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

## 4. Tiling

### 4.1 核间切分

- dense：任务数为 `B * ceil(T/BT) * HV * ceil(BT/16)`；
- varlen：任务数为 `total_chunks * HV * ceil(BT/16)`；
- blockDim 为 `min(task_count, AIV core count)`，每核步进处理后续 task；
- varlen chunk 在 L0 按 canonical sequence-major 顺序打包为 `[seq,start,end,0]`。kernel 每个 task 按 `flatChunk` 连续搬运 32B 元数据到 UB，不在 device 侧二分序列，也不把序列级数组塞入 tiling data。

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

### 4.3 UB 预算

legacy 固定 buffer 约 38.6 KiB。block-wise safe 路径额外使用两份转置 dA 行 slab、完整
`q/k` typed cache、`q/k/g` FP32 cache、六份 `[16,32]` 累加/partial 和一份 block scratch；
按 `BT=128/BK=32` 估算约 128.9 KiB，低于 A2/A3 的 192 KiB UB，并保留约 63 KiB 余量。
第一阶段使用单 buffer，避免在未 profiling 前为 double buffer 再增加约 64 KiB。

key12 的 AIV consume 编译为 `BT_CAPACITY=64/BK=64`。其 block-wise buffer 约 118.9 KiB；
AIV0 再加现有 rowBlock3 prep 约 21.3 KiB 后合计约 140.1 KiB，仍低于 192 KiB。BK64
是 FP32 Vector 单次 64-element mask 的上限，16-row repeat 和 8-block stride 均在 `uint8_t`
范围内；不为该实验引入 double buffer 或 UB alias。
key13 将 gate buffer 从一行扩为 16 行，额外增加 3.75 KiB，AIV0 合计约 143.9 KiB；其余
buffer 与 key12 相同。repeatTime 最大为 16，仍远低于 Vector API 的 255 上限。
key14 复用 key13 的同一份 `[16,64]` FP32 gate buffer，不增加新的 UB buffer、别名或
double buffer，因此 AIV0 预算仍约 143.9 KiB。row repeatTime 同样最大为 16，FP32 mask 为 64，
相邻行 repeat stride 为 8 个 32B block。

### 4.4 rowBlock3 off-left Cube 实验路径

第一次引入 Cube 只替换 `BT=64` 中最后一个 16-token 块的跨块 left 累加，即
`rowBlock3=[48,64)`、`source=[0,48)`。不改 16×16 diagonal、right 累加、`db/dg` 组合或其他
rowBlock，避免同时引入 causal mask、转置 GEMM 和复杂的 AIC/AIV 生命周期。AIV prep 将两路 dA
堆叠为 `A[32,48]`，并构造共同的 gated K `B[48,128]`；Cube 只执行一次：

```text
A = concat(dAqk[48:64,0:48], dAkk[48:64,0:48], axis=0)  # FP32 [32,48]
B = k[0:48] * exp2(g[48] - g[0:48])                     # FP32 [48,128]
C = A @ B                                                # FP32 [32,128]
```

`C[0:16]` 初始化 `dqAcc`，`C[16:32]` 初始化未乘 beta 的 `dkLeft`。consume AIV 随后复用原
block-wise 路径的 `exp2(g[48:64]-g[48])` 外层缩放，再继续 diagonal、`db`、beta、`dg` 和输出
累加。这样不会重复计算 `source=[0,48)`，也不会改变 `dkLeftPre` 在 `db` 归约前的语义。

实验 fastpath 只允许 `safe_gate=true`、BF16 q/k、dense、`B=1`、`H=HV`、`BT=64`、`K=128`
且 workspace 不超过 256 MiB、chunk 满 64 token；其余 dtype、BT、K、GVA、varlen、tail 和 unsafe
case 全部走稳定 key7/legacy 回退。host 当前对目标 shape 分派已验证的 key13，key15 仅保留为
未启用的独立实验 key；key13/key12 保留为逐级回退。设备侧实验 MIX 路径均使用
`KERNEL_TYPE_MIX_AIC_1_2`：一个逻辑 AIC 与两个 AIV 子核处理相同的 `(chunk,HV)` slot。

```text
AIV0: pack A/B -> ready -> rowBlock0 ------- wait done -> rowBlock3 consume
AIV1:          -> ready -> rowBlock1/2 ----- wait done
AIC :             wait ready -> FP32 Cube -> done
```

ready 使用 `0x2` 汇合两个 AIV 子核，确保 AIV0 完成 A/B 的 GM 写回后 AIC 才读取；done 由 AIC
广播，两个 AIV 子核都消费同一代反转 flag，避免下一 slot 误读陈旧事件。前三个 rowBlock 与 Cube
重叠，rowBlock3 只在 done 后读取 C。A/B/C 均保持 FP32，使用 IEEE FP32 模式并显式关闭 HF32。
设备源文件受 `TORCH_MODE` 保护地引入 `lib/matmul_intf.h`，仅为 CANN MIX 生成包装器注入的
`matmul::clearWorkspace` 提供声明；实际矩阵收缩使用 CATLASS 单 tile `TileMmadTla`，不启动
Matmul API server。key15 的六个 FP32 contraction 均落入单个 `128x128x128` tile，MMAD 与
Fixpipe copyout 显式使用 `unitFlag=0b11`，并以 `M_FIX/FIX_M` 事件闭合 L0C 生命周期。
每个 `(chunk,HV)` workspace slot 为 46 KiB：A 6 KiB、B 24 KiB、C 16 KiB，全部 512B 对齐；
目标 shape 的 4096 个 slot 共约 184 MiB。当前版本使用唯一 slot，不做复用或双 buffer，先保证
无覆盖、无死锁；通过 NPU 门禁后再评估 ring workspace 和更深的 contraction 覆盖。

单 kernel key12 的 BK64 版本已把目标 profiling 从 49.168 ms 降到 32.477 ms。key13 只把
source-loop gate 构造按 16 行打包为 repeat 指令，实测进一步降到 31.034 ms。key14 在 key13
基础上把 16 个独立输出行的 post-scale gate 以及对应 `Mul/Add` 改为 repeat：单个 BK64 tile
的 row post-scale gate 组数从 224 降到 14，总 gate 组数从 241 降到 31。每个输出元素仍保持
原来的 `Sub -> Muls -> Exp -> Mul -> Add` 链；Cube、workspace、slot 映射、ready/done 和
source 累加顺序均不变。`db` 仍按连续四个 32-feature FP32 子段归约并依次累加，避免 gate
批处理改变归约树。

## 5. safe_gate 分支

- `safe_gate=true` 是主验证和主优化分支，并复现兄弟仓 Triton kernel 的 16-token 参考点分解。对当前子块之前的 left 项使用子块首 token 的 gate；子块内 left/right 均使用第 8 个 token 的中点参考；后续 right 项使用子块末 token。内层累加完成后再乘参考点到目标 token 的外层因子。内外因子乘积仍严格对应 `exp2(g_i-g_j)`，中点参考缩短对角块两侧的指数范围。
- `safe_gate=false` 精确复现上游普通分支的计算顺序：当前 16×16 对角块直接计算 FP32 `exp((g_i-g_j)*ln(2))`，此前/此后的跨块项仍分别围绕块首/块末 gate 因式分解。
- block-wise safe 路径按 source token 计算一次公共 gate，再通过 `Brcb` 和带 stride 的 FP32
  `Mul/Add` 广播到 16 个目标行；没有融合乘加，保持 legacy 的 `Mul -> Mul -> Add` 舍入顺序。
- host 默认使用 tiling key 5/7 实例化 FP16/BF16 block-wise safe 路径；0/2 保持 unsafe，
  1/3 保留 legacy safe 实例作为回退。热循环内没有 safe/unsafe runtime 分支。

## 6. 精度策略

- q/k 在 UB 转 FP32；gate、dA 和输入累积梯度保持 FP32；
- 所有乘加、指数、K 维 reduce 均为 FP32；
- 工程关闭自动同步。block-wise safe 路径在 dA/beta 完成 `MTE2_S` 后，由 Scalar 一次性完成
  行 slab 重排、causal 清零和 `dAkk*beta` 系数合并，再用单个 `S_V` 将整个 coefficient block
  交给 Vector；source 热循环不再做逐 row UB 标量读取或 `S_V`。legacy 路径维持 EVENT4 已验证
  的逐 source 同步语义。所有事件均按同步点申请、等待并释放，不使用跨 task 持久 ID；
- tail token 和 causal mask 通过循环边界实现，不读取 chunk/sequence 外数据；
- 基准重点覆盖大负累积 gate、非整 chunk、GVA (`HV/H>1`)、dense/varlen、FP16/BF16，以及 safe/unsafe 两个分支。

## 7. 性能路线

第一阶段使用无中间大张量、单 launch、全 AIV block-wise safe 路径。已上板的 `75535cd`
基线在目标 shape 上为 48.660 ms kernel、51.107 ms end-to-end；在此之前的 legacy 路径约
477.94 ms。BK32 的单 kernel key12 已通过 28 项回归并测得 49.168 ms；其 AIV Vector/Scalar
分别为 26.121/18.561 ms，表明最小 Cube 切片尚未形成净收益。后续只从该稳定 fallback 小步演进：

1. 上板验证方向端点修正后的原 22 项加 1 项极值回归，并重新确认目标 shape profiling；
2. 若 MTE2 仍高，再评估 source cache double buffer 或多个 rowBlock 合并；
3. 若 Vector/Exp 仍主导，将跨 16-token 子块的 dA×gated-vector 搬到 Cube；
4. 根据实际 `K/BT/HV` 决定 task 是否沿 K 二次切核。

rowBlock3 off-left 是第 3 步的最小实验切片。稳定 key7 始终保留为非实验形状的默认路径和即时回退。
当前 key14 仅进一步减少 target MIX AIV 的 row post-scale gate/Mul/Add API 发射次数，不扩大
Cube contraction、eligibility 或 workspace 生命周期；key13 仍是目标 shape 的立即回退。
key14 的性能结论仍须重新完成编译、完整精度和单行 profiling 后给出。

## 8. 接口约束

内部输入为：`q,k,g,beta,dAqk,dAkk,dq,dk,db,dg`，可选 `cu_seqlens/chunk_indices`，属性 `chunk_size/safe_gate/total_chunks`；输出为 `dq_out,dk_out,db_out,dg_out`。所有输出均为新 tensor，不原地修改输入累积梯度。
