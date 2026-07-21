# ChunkKdaBwdIntra AscendC 设计

## 1. 目标与范围

本算子迁移 `flash-linear-attention` 中的 `chunk_kda_bwd_kernel_intra`，计算 KDA 单个 chunk 内部对 `q`、`k`、`beta` 和逐特征 gate 的梯度贡献。主交付路径为 `safe_gate=true`，同时保留独立的 `safe_gate=false` 编译分支。

首版支持 Ascend A2/A3/A5、`float16`/`bfloat16` 的 `q/k`、FP32 的 L0 gate/其余输入和四个输出，`K` 为 16 的倍数且位于 `[16, 256]`，`chunk_size` 为 64 或 128。kernel 以 `BK=32` 分块，并用实际 `curK=16` 处理尾块。公开 Python 接口允许 BF16 `g/beta` 并在进入 L0 前提升到 FP32。L0 内部布局统一为 BNSD；Python 接口负责 BSND/BNSD/TND/NTD 的无歧义转换。

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

## 5. safe_gate 分支

- `safe_gate=true` 是主验证和主优化分支，并复现兄弟仓 Triton kernel 的 16-token 参考点分解。对当前子块之前的 left 项使用子块首 token 的 gate，对子块内 left/right 项使用中间 token 的 gate，对后续 right 项使用子块末 token 的 gate；内层累加完成后再乘参考点到目标 token 的外层因子。它不截断指数差，数学结果仍是 `exp2(g_i-g_j)`，但避免把长距离 gate 差一次送入 `Exp`，保留原实现的稳定计算顺序。
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

第一阶段使用无中间大张量、单 launch、全 AIV block-wise safe 路径。已采集的 legacy A2
基线在目标 shape 上约 477.94 ms，主要问题是同一 source 被 16 行重复搬运/计算以及热循环
Scalar 同步。后续按以下顺序演进：

1. 上板验证 block-wise 路径的 22 项精度、重复 launch 与目标 shape profiling；
2. 若 MTE2 仍高，再评估 source cache double buffer 或多个 rowBlock 合并；
3. 若 Vector/Exp 仍主导，将跨 16-token 子块的 dA×gated-vector 搬到 Cube；
4. 根据实际 `K/BT/HV` 决定 task 是否沿 K 二次切核。

## 8. 接口约束

内部输入为：`q,k,g,beta,dAqk,dAkk,dq,dk,db,dg`，可选 `cu_seqlens/chunk_indices`，属性 `chunk_size/safe_gate/total_chunks`；输出为 `dq_out,dk_out,db_out,dg_out`。所有输出均为新 tensor，不原地修改输入累积梯度。
