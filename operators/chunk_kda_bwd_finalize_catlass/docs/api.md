# KDA backward Finalize：冻结接口

2026-10-19：用户要求以已验收版本为基线，按 CANNBot 重新设计实现。沿用此前确认的接口、七输出和精度要求，本轮不新增数学语义。完整接口依据为仓内 `fla/ops/ascendc/kda/chunk_kda_bwd_finalize/docs/api.md`；本文件冻结本次设备验证支持域，避免把 A5 的预留能力误认为当前基线已支持。

## 输入与输出

K=V=128、chunk=64、Hq=Hk=Hv。连续 ND 输入，定长 token 张量为 `[B,H,T,D]`，变长为 `[H,T,D]`。q/k/v/v_new/dv_scan 为 BF16；gk/raw_g/dq_raw 为 FP32。akk 为 BF16，末维为64；dAqk 同形状、FP32。h/dh 为 BF16，定长 `[B,ceil(T/64),H,128,128]`，变长 `[Nc,H,128,128]`；state_v_first=false。

beta_eff 为 FP32，形状等于 token 张量去掉末维；a_log 为 FP32 `[H]`；dt_bias 为 FP32 `[H,128]`。Q/K rstd 为可选 FP32，形状等于 token 张量去掉末维，必须成对提供。

输出顺序固定为 `dq, dk, dv, db, dg, da_log, ddt_bias`。dq/dk/dv 为 BF16，支持定长 BNSD/BSND、变长 NTD/TND 原生输出。db 为 FP32，固定 `[B,T,H]` / `[T,H]`；dg 为 FP32，固定 `[B,T,H,128]` / `[T,H,128]`。da_log 为 FP32 `[H]`，ddt_bias 为 FP32 `[H,128]`。全部有效输出必须写入且有限。

## 属性与边界

- scale 为有限标量，与前向一致，可为零或负值；lower_bound 范围 `[-5,0)`，默认-5。
- beta_mode=0：对有效 beta 求导；1：乘 beta×(1−beta)；2：乘 beta×(1−beta/2)。beta_eff 不能提前量化为 BF16。
- norm 开关控制 Q/K L2Norm 反向，使用保存的归一化 q/k 及 rstd。
- gk 为以2为底的累计 log gate；gate 反向读取 raw_g/a_log/dt_bias，safe_gate/use_gate_in_kernel/use_exp2 固定开启。
- 变长使用单调非减 cu_seqlens 和配套 chunk 索引，可含空序列，总 token 数非零。最后一个 chunk 的有效行数为1～64。掩码位置不能参与危险指数运算或输出归约。
- H 可为奇数；强制单核和多 work 回绕均须支持；所有 GM 地址乘积在64位中计算。
- 拒绝不匹配的 dtype/shape/layout、单独提供 rstd、越界 beta_mode、非法元数据、空总输入及非128的 K/V。原200case之外另测 host 拒绝行为。

## 标杆与验收

唯一数学源码为 `tests/atk/chunk_kda_bwd_finalize/reference.py`，由旧 `scripts/reference.py` 原字节迁移，旧位置及直调工程只保留转发。输入构造、种子、FP32/FP64公式和四处既定BF16交接全部冻结；不假设独立输入满足额外前向恒等式。

原CT L1双标杆是不可放宽的必要门槛。CANNBot compare_precision 的分区检查另行校准和记录，不能以其通过代替原CT结果。每次性能候选先定向精度，再八配置两轮交替A/B；稳定收益后原20case七输出；最终候选fresh full原200case×三个seed及边界/非法输入验收。

## 直调与性能口径

04/05使用独立C++ ACL host，完成设备分配、tiling、kernel直调和七输出拷回；输入准备不计入设备kernel耗时。必要预处理、主kernel及归约kernel的Task Duration全部计入，不能只报某一个kernel。性能目标：8K无norm、无beta sigmoid反向约9.60ms以内，其余七配置需稳定优于原版；A5仅作历史参考。Vector占比是诊断指标，不是完成条件。

通过直调full后再接入仓库适配和ATK。基线开发入口与A5路径在新实现验收前保持可用。
