# Stage 0～9 验证记录

分支 `kda-finalize-a2`，基线 `50496bc8`。A2：Ascend910B3、CANN 9.1.0。工作目录现为 `/data/wys/kda-finalize-a2-stage9-20261012`；原目录保留。这里只验证私有分阶段入口，Stage 10/11 尚未接入，不代表完整 Finalize、GPU 双标杆或 ATK 验收。

## 实现

Stage 9 逐带合成 dq、更新 dg，支持可选 Q L2Norm backward；实际 dq 转 BF16 写回，FP32 dq/dg 复用 workspace。

- r0：8 行输入、输出独立双槽，等待两 head 的 Stage 8 完成。
- r1：16 行双槽，读完 base 后原位计算，减少循环和同步。
- r2（最终版）：保持 r1 双缓冲，按 head 发布 Stage 8 ready，使 h0 Stage 9 与 h1 Cube 重叠。两 AIV 按同序消费通知，空 owner 也参与，下一带通知保护复用。

参考 [cannbot-skills PR 1069](https://gitcode.com/cann/cannbot-skills/pull/1069)，本地参考提交 `4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca` 的双缓冲、UB 驻留和生命周期复用方法。

## 性能

B=1/H=96/T=8192/K=V=128，设备 3，Q norm 关闭。变长切分 `[0,1,31,64,65,4096,3935,0]`。每组单份 CSV 均有 30 条目标记录，取 20 次预热后 10 次采样的中位数。

| 版本 | 定长耗时 μs | 定长最大流水 | 变长耗时 μs | 变长最大流水 |
| --- | ---: | ---: | ---: | ---: |
| Stage 8 同期基线 | 6430.220 | Vector 73.80% | 6605.945 | Vector 73.00% |
| Stage 9 r0 | 8144.150 | Vector 65.05% | 8291.911 | Vector 64.70% |
| Stage 9 r1 | 7899.706 | Vector 66.00% | 8048.397 | Vector 65.70% |
| Stage 9 r2 | 7717.991 | Vector 67.70% | 7836.817 | Vector 67.85% |

r2 比 r0 快约 5.23% / 5.49%，比 r1 快约 2.30% / 2.63%。最大占比仍未达到 90%，也低于 Stage 8；Stage 8/9 工作量不同，不能作为同功能加速比。这里只采集 PipeUtilization，未据此宣称带宽饱和或逐核负载均衡。

部分 msprof 自动导出退出码为 -9，定长 r2 已离线恢复完整 30 条记录。早期 r0 目录含重复采集，本表选最后一份完整 CSV，不混合多次运行；因此与早期临时表略有差异。原始 CSV、采样范围和全部流水指标见 [performance.json](verified/performance.json)。

## 精度与同步

- r2 采数版：三种 gate × Q norm 开关，132 次 tails 检查通过。
- Stage 0～8 回归：18 次 smoke 检查通过；r2 性能版：44 次 tails 检查通过。
- 完整 8K 模型：定长/变长 × Q norm 开关四组，Stage 9 全部输出通过局部 FP64 和直接因果指数差公式检查。r2 采数/性能版与已验证 r1 的实际 dq/dv、有效 workspace 逐位一致；采数版 capture/trace 也一致。
- Host tiling、Python 语法、源码 SHA256 与格式差异检查通过。旧 Stage 容差不变；校准能拒绝全零、漏对角线、漏局部项、错误缩放和漏 Q norm。
- r2 模拟器正常退出，三 lane 完整；120 次生产 MMAD，其中 Stage 8 为 72 次，48 次对应 L1 搬入。摘要确认 Cube/Vector 及搬运重叠，但没有单独隔离 Stage 9 指令占比。

完整模型使用 `model.py` 检查新增 Stage 9，并非重新计算所有前序 Stage 的完整模型 FP64 标杆。证据和复现脚本在 [verified](verified/)，原始模拟器及动态库保留在工作目录的 `stage9/`。脚本中的历史绝对路径记录原执行位置，重跑时需改为当前 `/data/wys/` 工作目录。
