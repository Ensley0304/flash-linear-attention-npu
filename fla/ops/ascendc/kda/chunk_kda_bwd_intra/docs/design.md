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
     16-token row/column metadata (UB)
                 |
                 v
  causal source-token loop, 32-feature vector math
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
- 每次只保留一个输出 token 的 FP32 累加向量及三个跨/块内 partial，避免 `BT*BT*K` 中间量；
- dA 的行使用连续 DataCopy，列使用二维 DataCopyPad 收集，禁止逐 GM 标量读。

### 4.3 UB 预算

固定 buffer 包括两条 dA 行、两条 dA 列、beta、q/k 半精度输入、q/k/g FP32 向量、梯度累加向量、临时向量和 reduce 临时区；safe 分支另有三个 `MAX_K=256` 的参考 gate 向量。按 `BT=128/BK=32` 估算小于 32 KiB，给编译器和 API 临时区保留充足余量。

## 5. safe_gate 分支

- `safe_gate=true` 是主验证和主优化分支，并复现兄弟仓 Triton kernel 的 16-token 参考点分解。对当前子块之前的 left 项使用子块首 token 的 gate，对子块内 left/right 项使用中间 token 的 gate，对后续 right 项使用子块末 token 的 gate；内层累加完成后再乘参考点到目标 token 的外层因子。它不截断指数差，数学结果仍是 `exp2(g_i-g_j)`，但避免把长距离 gate 差一次送入 `Exp`，保留原实现的稳定计算顺序。
- `safe_gate=false` 精确复现上游普通分支的计算顺序：当前 16×16 对角块直接计算 FP32 `exp((g_i-g_j)*ln(2))`，此前/此后的跨块项仍分别围绕块首/块末 gate 因式分解。
- 两条路径使用独立 tiling key 和模板实例；false 分支不计算 safe 路径专用的块中参考点。
- host 使用不同 tiling key 实例化 `ChunkKdaBwdIntraKernel<T, true/false>`，热循环内没有逐元素 runtime 分支。

## 6. 精度策略

- q/k 在 UB 转 FP32；gate、dA 和输入累积梯度保持 FP32；
- 所有乘加、指数、K 维 reduce 均为 FP32；
- tail token 和 causal mask 通过循环边界实现，不读取 chunk/sequence 外数据；
- 基准重点覆盖大负累积 gate、非整 chunk、GVA (`HV/H>1`)、dense/varlen、FP16/BF16，以及 safe/unsafe 两个分支。

## 7. 性能路线

首版优先实现无中间大张量、单 launch、全 AIV 向量化的稳定基线。profiling 后按以下顺序演进：

1. 将跨 16-token 子块的 dA×gated-vector 搬到 Cube，并保留 AIV gate 预处理；
2. 使用双缓冲覆盖 source row 的 MTE2 与 Vector；
3. 根据实际 `K/BT/HV` 调整 task 是否沿 K 二次切核；
4. 仅以端到端 profiling 和精度双标结果决定是否增加专用 tiling key。

## 8. 接口约束

内部输入为：`q,k,g,beta,dAqk,dAkk,dq,dk,db,dg`，可选 `cu_seqlens/chunk_indices`，属性 `chunk_size/safe_gate/total_chunks`；输出为 `dq_out,dk_out,db_out,dg_out`。所有输出均为新 tensor，不原地修改输入累积梯度。
