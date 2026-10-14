# GDN forward 发布分支接口合同

本次移植来源为 main 修复提交 `95004a10a0176113348c50e33ce0a98106eb5b49`，目标基线为
`v26.9.1@36e67d2decea70745554f7aebf9a6a29e9168a91`。公开原型仍以
`op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.h` 与既有 Python wrapper 为准，29 个 ACLNN
参数和 Python 十项返回均不修改。

## 输入、属性与输出

`q/k/v` 为 FP16 或 BF16。四种布局均为四维：BNSD/NTD 使用 `[B,H,T,D]`，
BSND/TND 使用 `[B,T,H,D]`；g/beta 固定为 `[B,T,Hv]`，允许 FP32 或 QKV dtype。
K=128，V=128/256，chunk_size=64/128，Hv 必须是 Hk 的正整数倍。
输出 O 固定为 `[B,T,Hv,V]`，不随输入布局改变。

`cuSeqlensOptional` 与 `chunkIndicesOptional` 同时省略时为 dense；变长时同时提供，
物理 B=1，cu 从 0 单调不降到 T，允许空序列沿用原有处理；索引按 sequence-major 顺序列出每条序列的
`(sequence, local_chunk)`，扁平化为 int64 数组。空序列和非法边界保持原校验行为。

初始状态可选，`stateVFirst=false` 时为 `[N,Hv,K,V]`；最终状态是否输出仍由
`finalStateOutOptional` 是否为空决定。可选中间输出、Q/K hats/rstd、beta_eff、A 和 H
沿用原公共合同。H 使用 NT-first：dense `[B,NT,Hv,K,V]`，packed `[1,total_NT,Hv,K,V]`；
state_v_first 交换状态的最后两维。

## 本次支持差异

DAV_2201 的 A2/A3 在原 Phase6 基础场景增加 BSND/TND 原生 QKV 路径，支持范围与
冻结来源保持一致。原 BNSD/NTD、定长/变长、GVA、初始/最终状态、V128/V256 和 BT64/128
保持支持。公开数学语义、尺度、dtype 和默认属性不改变。

Phase6 条件仍为 use_exp2/use_qk_l2norm/allow_neg_eigval/state_v_first 关闭、
a_log/dt_bias/beta_eff/H 扩展不请求。A5 原 Prepare/H/O 路径及其扩展输入输出保留；
不支持的扩展组合仍由原 host/Prepare 校验返回错误，不静默忽略。

A2/A3 的 g/beta 在 host 转换到 FP32，满足既有融合 kernel 的消费合同；A5 两个门控输入
均为低精度时仍按原 Prepare 合同传递。空输入快捷返回、必选参数检查和输出错误码保持原样。

## 私有 L0 兼容

原输入、输出和前三个属性不改；从来源实现追加以下可选属性。属性只在唯一融合 L0 内部使用，
不增加公共 ACLNN/Python 参数：

| 属性 | 默认值 | 内部含义 |
| --- | --- | --- |
| output_g_cumsum | true | 是否写公开累计门控输出 |
| raw_g_layout | 0 | 0=BHT，1=BTH prepared 输入 |
| qkv_layout | 0 | 0=BHTD，1=BTHD 原生 QKV |
| o_layout | 0 | 0=BHTV，1=BTHV 直接输出 |

旧 L0 调用省略这些属性时保持历史合同；public host 仅在 DAV_2201 设置新布局标志。
旧 arch35 实现继续消费原属性和布局。算子定义改动需要对应 CODEOWNER 检视。

运行/精度支持结论以本分支实际验证记录为准，源码移植和目标编译不替代实机验收。
