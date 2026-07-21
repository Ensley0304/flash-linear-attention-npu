# ChunkKdaBwdIntra 性能分析

## 当前实现

当前 kernel 是无 workspace 的单 launch AIV 路径。核间 task 为 `(chunk, HV, 16-token block)`，核内按 `BK=32` 遍历特征，再逐 source token 计算 FP32 gate 和三类梯度累加。

已实施的静态优化：

- 历史 token 只搬运 `k/g`，不搬未使用的 `q`；
- `dq_local` 与 `dk_left_pre` 复用一次 `k*gate`；
- unsafe 分支仅在当前 16×16 对角块直接计算 gate，跨块仍按上游首/末参考点分解；safe 分支进一步对对角块使用中点参考 gate；
- safe 分支的首/中/尾完整 K 参考向量在 task 开头各搬一次并驻留 UB，供同一 16-token block 的所有输出行复用；
- future `dAkk*beta` 先做 FP32 标量合并；
- `db` 在同一 task 内跨 K tile 归约，避免全局 partial tensor 和第二次 launch；
- dA 列提取除末 7 列外按每行 32B 搬运，末 7 列才回退到 4B `DataCopyPad`；
- varlen 每个 task 只连续搬运一条 32B packed chunk metadata，不携带大 tiling 数组、不做 device 二分；
- task 写区间互斥，无 atomic、跨核同步或随序列增长的 workspace。

## 理论模型

设有效 chunk 长度为 `L`，单 task 行数为 `R<=16`，则主要工作量近似为：

```text
source vector loads: O(R * L * K / BK)
Exp vectors:         O(R * (L - 1) * K / BK)
FP32 vector FLOPs:   O(R * L * K)
```

核间并行 task 数为 `chunks * HV * ceil(BT/16)`。长序列和常见多头形状可以覆盖全部 AIV；极小单 chunk/单头 case 会受可并行 task 数限制。

## 待上板确认

当前机器没有 CANN/NPU，以下均是需要 profiling 验证的假设，不是实测结论：

1. `BK=32` 下短 Vector 指令和同步事件是否成为主要开销；
2. MTE2 是否因每个输出行重复读取 source token 而成为主瓶颈；
3. `BT=128` 下单 task 变长是否造成核间尾部不均衡；
4. FP32 Exp 吞吐是否高于 source 搬运压力；
5. BSND/TND Python layout materialization在端到端时间中的占比。

建议采集 kernel duration、AIV utilization、MTE2 bandwidth、Vector utilization、各流水 stall 和 task tail。基准至少覆盖 `(BT,K)=(64,128),(128,128)`、FP16/BF16、dense/varlen、`HV/H=1/2/4` 和 safe/unsafe。

## 下一阶段候选

若 profiling 确认当前路径受重复 source 读取或短向量指令限制，优先实现 16-token 子块的 AIC/AIV 融合路径。对每个目标子块选择 gate reference 后，可将计算重写为四个 FP32 Cube GEMM：

```text
dAqk_left  @ (K * exp2(g_ref_left - g))
dAkk_left  @ (K * exp2(g_ref_left - g))
dAqk_right^T @ (Q * exp2(g - g_ref_right))
dAkk_right^T @ (beta*K * exp2(g - g_ref_right))
```

AIV 负责 gate 预处理和行缩放，AIC 负责 GEMM，每 AIC 固定复用一份 scratch。该方案不产生序列级中间张量，但引入跨核同步和约数 MiB 的固定 workspace，必须在精度回归与 profiling 后决定是否替换当前稳定基线。
