# A2 KDA Finalize：S24 快照

本版提供 `arch22` 实现和私有 fast-launch 验证入口，覆盖 Stage0～12。算子代码取自已验收 S24，来源提交 `96945164`；最后归档提交 `5bda1e52`。未包含未验证的 S27，不改现有 A5 kernel、公开 ACLNN/torch 接口或注册。

## 支持范围

- BF16 Q/K/V 与状态输入，K=V=128、chunk=64；定长和变长，包括尾块、空序列与奇数 head。
- FP32 局部矩阵乘，关闭 HF32；保留 dW/kE/Zb/Tza 的 BF16 交接。
- Q/K L2Norm 反向成对启用；beta_mode=0/1/2 对应关闭、sigmoid、allow_neg_eigval。
- dq/dk/dv 按 output_layout 原生输出 BNSD/BSND 或 NTD/TND；dG/dBeta 固定 BSND/TND。
- Stage12 参数归约在同一次 launch 内完成，保持固定 chunk 累加顺序。

## 流水与资源

每个 AIC 处理一对 head，两个 AIV 各处理一个 head；work 为 head-pair × chunk，按核跨步分配。两个 workspace bank 允许相邻 work 重叠，空 owner 同样消费集体通知。

| 阶段 | 执行单元 | 主要工作 |
|---|---|---|
| 0/1 | Cube / Vector | 状态及 WY 反向的首批乘积；gate、kE 准备 |
| 2/3 | Cube / Vector | 后续 WY 乘积及 dv、Zb 合成 |
| 4/5/6 | Cube / Vector / Cube | Tza、各梯度 base 和 dAkk |
| 7 | Vector | 严格下三角及分带居中的 kNeg/qPos/bkPos |
| 8/9 | Cube / Vector | dq/left 乘积，dq 合成及可选 Q norm 反向 |
| 10/11 | Cube / Vector | right 乘积，dk、dBeta、gate 反向扫描及 chunk 参数局部和 |
| 12 | Vector | 全体 AIV 同步后归约 dALog/dDtBias，复用前段 UB |

每 AIV 使用192 KiB UB；每 AIC 使用512 KiB L1，L0A/B各两个32 KiB槽，L0C两个64 KiB槽。workspace 每核组1536 KiB：两个 head × 两代 bank ×384 KiB；GM 基址使用64位。partial 独立保存为 `[H,Nc,136]` FP32。

输入/输出槽分别保留双缓冲及各自 ready/free。每个局部槽在最后读取或写回完成后才可覆盖；跨核通知保留首项、稳态、回绕、末项和空 owner 的完整消费顺序。

S20 将 q/k/raw gk 从 Stage5 留到后段，减少重复读入。S24 对完整 chunk 的 Stage8 start=0 和 Stage10 start=32 使用紧凑32行 NZ 和 K32；Stage7 不生成三个无效半块，省去相应写回、读入和零项乘法。其他带、尾块及 capture 路径保持原表示。带内指数采用整数/小数分解，避免对全 chunk 未居中的指数直接求倒数。

## 已有验证

S24 在 A2/910B3、CANN9.1 上完成九边界和原20case七输出回归；140个输出全部通过原纯FP32局部GEMM标杆、独立FP64 golden和CT L1，未改种子或阈值。报告见 [边界](hybrid-evidence/s24-boundary.json) 和 [20case](hybrid-evidence/s24-precision20.json)。

八配置按 S20→S24→S20→S24 交替采集，每配置20次预热、20次样本，使用 msprof Task Duration 中位数。每个 capture 为320个主核、零独立归约核。

| T | 无附加反向/ms | 仅 norm/ms | 仅 beta/ms | norm+beta/ms |
|---|---:|---:|---:|---:|
| 8192 | 9.902088 | 10.775535 | 9.894592 | 10.802801 |
| 16384 | 19.806321 | 21.531130 | 19.826321 | 21.525995 |

B=1、H=96、K=V=128、BF16。原始8K无附加反向模型同轮10.011780→9.902088 ms，改善约1.10%，Vector64.65%；八配置最高69.50%。结果为两轮中位数的均值；[采样摘要](hybrid-evidence/s24-perf.json)、[双轮门禁](hybrid-evidence/s24-perf-gate.json)。

验收库 SHA256：`ea137ad21a880597ca9afc0b6d44a3d839049add83f4f49f3d3c199d19d3fdcf`。

本 PR 将相同 kernel 快照整理到较新的主分支；没有在该 PR 基线上重新构建。完整200case三seed、公开入口、A3/A5编译与运行、整体Example/ST及正式CI仍待完成。9ms/80%目标尚未达到。历史实验日志保留在开发分支，不将旧报告作为本快照的全量验收。
