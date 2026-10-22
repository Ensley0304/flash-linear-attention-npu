# 附件优化思路吸收结果

原始模型 B=1、H=96、T=8192、K=V=128、BF16、BNSD、无 l2norm/beta sigmoid 反向：本轮起始 R2b **10.473 ms → 最终 S3 9.988 ms，提升 4.63%**。仍未达到 9 ms；目标 Vector 约 63.93%，八配置最高约 68.55%，也未达到 80%。

源码分支 `kda-finalize-a2-hybrid`，起始代码 `3db5e900`，最终代码 `26d551ca`。没有采用附件的低精度矩阵乘；FP32 局部 GEMM、稳定指数、功能开关及布局契约保持原样。

## 逐项结论

每次只改一项，先九个边界 case 七输出，再八配置两轮交替 A/B；有稳定收益后才跑原 20 case 七输出。保留项全部通过，未调整标杆、输入种子或 CT L1。

| 候选 | 改动 | 结果 |
| --- | --- | --- |
| [S1](compact-stage11.md) | 前带 Stage 11 用两个 16 行 tile，复用已消费的输入区 | 保留，无 norm 每轮改善 0.61%～1.16%，norm 指令不变；提交 `4df95a05` |
| [S1b](compact-stage11.md#s1b提前搬入-raw_g) | raw_g 提前搬入，与 dk 和扫描重叠 | 保留，无 norm 每轮改善 2.63%～2.86%，norm 指令不变；提交 `3990bc6a` |
| [S4](compact-dg-handoff.md) | dG 从 Stage 9 原地交给 Stage 11，不经 GM | 无稳定收益，撤回 |
| [S2](stage5-input-reuse.md) | 复用 Stage 5 留下的后 32 行 q/k/gk | 无 norm 回退，撤回 |
| [S3](stage7-lower-batch.md) | Stage 7 lower 从 16 行扩大到 32 行批量生成 | 保留，八配置每轮改善 0.706%～2.694%；提交 `26d551ca` |
| [S5](stage11-scan-batch.md) | 合并两个独立扫描组的同步点 | 原始 case 无稳定收益，撤回 |
| [S6](stage11-gate-scan-overlap.md) | Gate 导数运算与扫描交错，共用 barrier | 无 norm 仅 0.069%～0.219%，未过 0.3% 门槛，撤回 |

这些增量收益来自各自同轮对照，不能直接相加。下表另用最终 S3 与起始 R2b 直接复测。

## 最终直接对照

A2 `a2-server`，910B3，设备 4，CANN 9.1。固定输入、20 预热＋20 采样，R2b→S3→R2b→S3；每次采集均有 320 个主核和 320 个归约核记录，前后设备占用检查通过。统计两轮中位数的均值，含主核＋Stage 12 归约核，单位 ms。

| T | 附加反向 | R2b | S3 | 改善 |
| --- | --- | ---: | ---: | ---: |
| 8192 | 无 | 10.473 | 9.988 | 4.63% |
| 8192 | norm | 11.140 | 10.870 | 2.42% |
| 8192 | beta | 10.469 | 9.996 | 4.52% |
| 8192 | norm+beta | 11.168 | 10.937 | 2.07% |
| 16384 | 无 | 20.936 | 20.022 | 4.37% |
| 16384 | norm | 22.306 | 21.754 | 2.47% |
| 16384 | beta | 20.951 | 20.037 | 4.36% |
| 16384 | norm+beta | 22.293 | 21.798 | 2.22% |

最终两轮八配置均通过 >0.3% 门槛。原始 case 两轮分别为 9.982/9.995 ms，对应改善 4.68%/4.58%。

## 精度与交付范围

S1、S1b、S3 各自通过原 20 case 的 140 输出：dq、dk、dv、db、dg、da_log、ddt_bias。另各通过九边界七输出；固定 case ID、种子、参考 hash、实际加载库 hash 均已复核。这里是同一组 20 case 对三个版本的增量验收，不是新增 60 个独立场景。

最终库 SHA256：`84df9828475c1aa434ac80d79fee68a6070666445db2814ac03077b06228ce22`。独立 FP32/FP64 参考 SHA256：`1a5e5d498ce1cc5474f628c4279d871e305a3b9cb4b31b2bcba48e7109f6c706`。完整 200 case 三 seed、公开入口与 A5 未重跑，不能作为正式全量验收完成。

本地源码已恢复 S3；A2 隔离工程的 build 与 ascend_ops 两处库也恢复 S3，并核验上述 hash。失败候选不在最终代码中。

摘要为 `hybrid-evidence/attachment-optimization-summary.json`、`final-s3-*.json` 及各候选 JSON。原始 CSV、构建快照、脚本和日志在工作区 `validation/compact-stage11-20261022/*-evidence.tar.gz`，均按清单逐文件校验。服务器原件在 `/data/wys/kda-finalize-transfer-20261022`。
