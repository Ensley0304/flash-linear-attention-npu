# ChunkKdaBwdFinalize

## 功能

KDA 反向优化链路的最后阶段，合并 Prepare 和 Dhu 的结果，输出 Q/K/V、
beta、Gate 与参数梯度。可融合 Q/K L2 归一化反向，仅支持 Ascend950。

## 输入

以下为 Finalize 算子自身的接口；Python V2 入口的类型转换与可选参数处理见
[优化接口](../chunk_kda_bwd/docs/api.md)。

| 输入 | 类型 | 含义 |
|---|---|---|
| q、k、v、v_new | BF16 | 前向 Token 张量 |
| akk | BF16 | 块内中间量 |
| h、dh | BF16 | 前向状态与状态梯度 |
| dv_scan | BF16 | Dhu 输出 |
| gk、raw_g | FP32 | 累积 Gate 与原始 Gate |
| d_aqk、dq_raw | FP32 | Prepare 输出 |
| beta、a_log | BF16 / FP32 | 衰减与 Gate 参数 |
| dt_bias | FP32 | Gate 偏置 |
| q_rstd、k_rstd | 可选 FP32 | 成对提供的归一化中间量 |
| cu_seqlens、chunk_indices | 可选 INT64 | 变长序列元数据 |

输入须连续，K=V=128、chunk_size=64、Hq=Hv。支持定长及变长序列，
Token 为 [B,H,T,D] 或 [H,T,D]；h 为 [B,Nc,H,K,V] 或 [Nc,H,K,V]，
dh 保持 [B,H,Nc,K,V] 或 [H,Nc,K,V]。

## 输出与属性

dq/dk/dv 为 BF16，d_beta 与 beta 同类型，d_g/d_a_log/d_dt_bias 为 FP32。

| 属性 | 要求 |
|---|---|
| scale | 必填，与前向一致 |
| lower_bound | 默认 -5，范围 [-5,0) |
| chunk_size | 64 |
| safe_gate、use_gate_in_kernel、use_exp2 | True |
| state_v_first | False |

## 实现

采用 MIX_AIC_1_2，矩阵计算与向量处理协同执行；末阶段同步后归约参数梯度。
Tiling key 1/2 对应定长/变长，3/4 为对应的归一化反向版本。

| 文件/目录 | 职责 |
|---|---|
| op_host/ | 算子注册、形状推导、Tiling 与 ACLNN 接口 |
| op_kernel/ | Ascend950 kernel |
| [优化设计](../chunk_kda_bwd/docs/design.md) | 分带、精度与存储方案 |

构建时设置 `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd`，
会包含 V1/V2 及优化链路依赖。

## A5 内存检测

Ascend950 编译配置包含 `-fno-jump-tables`，用于规避 CANN 9.1.0 /
msSanitizer 26.1.0 动态插桩执行 Finalize 跳转表时复现的 507015。
该选项不改变 tiling、workspace 或流水线；需要源码行号时另加 `-g` 构建诊断版本。

以下验证基于交付版本 `b19bf2a0`，不是 PR 新 head 的全量验收。
2026-09-21 的交付矩阵回归中，新旧 release 的 200 条输出逐张量位级一致；
12 个代表用例的整条 V2 链路设备 kernel 耗时总和变化为 −0.44%～+0.45%。
这组数字不包含 CPU 调用开销，也不代表所有 shape 的性能保证。

同一交付矩阵的前 50 条内存复测：ATK 50/50 Pass，memcheck 原始 ERROR=0，
key 1/2/3/4 均有实际执行记录，无 507015 或检测记录丢失。原始日志同时包含
6204 条共享 GM 写入所有权 WARNING，均位于 Cube 的 Fixpipe→GM 写回位置；
该结果不是零告警验收。附加单例 racecheck 的两条潜在 L1 RAW 保留用于进一步核查。

检测结果须同时保留 ATK 表格和 sanitizer 原始日志。Cube 与 Vector 原位更新共享
workspace 时可能产生跨核所有权 WARNING；ATK Pass 不能替代零告警结论。
核内 racecheck 的潜在 L1 RAW 报告还需结合跨核 ready/return 依赖分析，
不能仅凭确定性通过就忽略，也不应未经核对就向性能流水线添加屏障。
