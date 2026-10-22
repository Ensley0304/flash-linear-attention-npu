# 附件调度增量实验

基线：`1f1321bf` / `h24.so`。保留原 FP32 局部 GEMM、分带缩放、七输出、norm、beta 反向及原生布局。候选仅在隔离目录构建；稳定性能收益且原 20 case 七输出通过后才能提交实现。

验收：先九个边界 case，再 H96、8K/16K、norm/beta 四组合，两轮交替 A/B，20 预热＋20 采样。统计主核与归约核每次调用 Task Duration 之和；各配置每轮改善超过 0.3% 后进入原 20 case。FP32/FP64 标杆、种子、CT L1 不变。最终 200 case 验收单列，不用 20 case 代替。

## R1：首带 DQ 绕过 dAkk 依赖

生产 Stage 11/12 路径启用；capture 和短前缀沿用基线。每 AIV 仍处理一个 head，work 映射、两代 workspace、尾块规则不变。

原顺序：Cube S4→S6→S8；Vector S5→等 S6→准备两个 lower→kNeg→通知 Cube。

候选：Cube S4→首带 DQ→S6→首带 left→Right；Vector S5→准备 dAqk lower→首带 kNeg→通知→等 S6→准备 dAkk lower→通知→qPos/bkPos→原后处理。后续带使用常驻的两个 lower。DQ 先完成，但 Vector 仍在原 Stage 9 消费，避免同时改变 UB 后处理布局。

| 数据 | 地址与生命周期 |
| --- | --- |
| akk/Tza | 原 L1 两代槽，保留至延后的 S6；与 L1[384,512) KiB 的 S8 区分离 |
| dAqk/kNeg | 原 S7 GM 地址，原 S8 L1 槽；先计算并写回 DQ，才复用 L0 做 S6 |
| dAkk raw | GM[216,232) KiB，S6 写；S7 完整搬入并消费后才能写重叠的 bkPos |
| dAkk lower | GM[368,384) KiB，延后生成一次；Cube 收到独立 ready 后搬到原 L1 lower 槽 |
| B 处理临时区 | 复用 UB[88,112) 和 [178,186) KiB；先排空 kNeg 输出 DMA，再复用。center、bandExp、q/k/gk 和小量不覆盖 |

同步：S6_READY=6+bank 改由首带 DQ 写回后发出；两个 AIV 在首带 kNeg ready 后各消费一次。首带增加一轮 BAND_READY，表示 B lower 已完整写回；Cube 在 left 前消费，原第二轮仍表示 positive 就绪。空 owner 发出相同次数通知并消费 S6_READY；首项、两代回绕、末项都沿原 S5_DONE 回压，不增加 outstanding work。

L1 512 KiB、UB 192 KiB、L0 双槽及每 bank 384 KiB 不增加。算术、矩阵尺寸和舍入不变；潜在收益是 dAkk 准备不再阻挡 DQ，代价是新增握手和暂存区切换。必须测量净收益，失败恢复 h24。

## 后续顺序

1. R2：Base 的局部 UB 驻留，先完成地址预算，避免重复 H20/H21 的简单 dG 驻留。
2. R3：Gate 系数逐 tile 提前准备，保持稳定导数公式；不重复仅首 tile 的全分支方案。
3. R4：同 head 多 chunk 参数部分和累计，保持 chunk 扫描独立，单独检验归约顺序误差。

## R2 资源预算：先保留 dq_base 的最后八行

三个完整 Base 共 96 KiB，不能直接加入当前 UB。先只在完整 chunk 的生产路径保留 dq_base[56:64]，4 KiB，地址 UB[188,192) KiB。Stage 5 的最高临时区结束于 184.5 KiB，Stage 7 的最高生产输出结束于 186 KiB，因此可跨 Stage 5→7 驻留。

Stage 9 首带开始，在任何 product[184,192) 写入之前，将缓存复制到第二个 dq 输入槽的后半段；该槽前八行仍从 GM 搬入。随后缓存死亡，product 恢复原布局。Stage 5 少写最后八行，Stage 9 少读这八行；每完整 chunk/head 少 8 KiB GM，H96/8K 合计 96 MiB。代价是两次 4 KiB UB 复制；不调整 tile、事件数、双缓冲、计算及舍入。尾块、capture、Stage 11 之前的短前缀原样保留。

必须验证 q/k/gk、lower、bandExp 与小量不触及该区；Stage 9 入口原排空与 V→MTE2 桥接保持。仅当原九个边界和八配置性能门禁通过，才运行原 20 case 并决定保留。

## R1 实测：不采用

九个边界 case 七输出全部通过。八配置两轮均回退 6.52%～8.09%；H96/8K 无附加反向，同轮 h24 为 10.622 ms、候选 11.459 ms。代码已恢复 h24 后再开始 R2，没有提交 R1，也未将性能失败者进入原 20 case。拆开依赖本身没有收益：本候选将原本可与 S5 重叠的 S6 延后，并增加首带握手；DQ 虽先完成，Vector 仍在原 Stage 9 消费，因此不能声称形成了完整提前消费流水。

R1 结果限制：仅验证首带 Cube 顺序和输入发布变化；更大范围的 Stage 9 前移还需新的 UB 设计，不在此候选中暗中叠加。

## R3 资源预算：后 32 行带的 Gate 逐 tile 提前

保留已有首 tile（仅 norm）预计算。对完整 chunk 的 8 行 Stage 11 路径，把后续 tile 的原始 gate+bias 和稳定导数计算移到上一 tile 的输出 DMA 排入之后。提前消费下一输入槽的 inputReady，再归还一份相同 ready 供原循环消费；不提前归还 inputFree，避免 raw 被覆写。导数使用原 gateTemp[140,144) KiB、denom[144,148)，下一 tile 的 q/k/product 仅使用[128,140)，不会覆盖导数。

16 行路径的 gateTemp 与 q 共址，不能沿用此预计算，因此保持原路径。尾块、capture 保持原路径；稳定 Exp(-Abs(x))/(1+Exp(-Abs(x)))²、扫描与归约顺序不变。不新增 buffer，不增 GM 流量。代价是每个提前 tile 多一对 ready 消费/重发；只实测净收益，不预设可隐藏整段 Vector 时间。

## R4 资源预算：按 AIV/head 累积参数部分和

保持 work 的 head-pair×chunk 跨步映射，不改负载分配。每个 AIV 对同一 head 负责的 chunk 在 UB 累积原有 136 个 FP32 partial；每 chunk 内扫描和归约完全不变，只改变跨 chunk 合并树。

常驻 bias partial 放 UB[84.25,84.75) KiB，alpha 八项放 UB[83.75,83.78125) KiB。两处分别位于 beta_factor 与 Gather 索引之间、k_rstd 与 beta_factor 之间；不与 Stage 1/3/5/7/9/11 的 scratch 重叠。head 切换时清零，每个 chunk 结束后相加。

同一 head 的本核最后一个 chunk 满足 chunkIndex+usedCoreNum>=chunks。只在这里写 GM，索引为 head×chunks+chunkIndex%usedCoreNum，record stride 保持 136 floats。每个 head 有 min(chunks,usedCoreNum) 个连续有效 record，无空洞，Stage 12 只读取这些 record；不缩小已分配 partial tensor，不增加 workspace，不用 atomic 或全核同步。两个输出片段使用原 outputFree 保护至下一次 UB 复用。

仅完整 Stage 12 非 capture 入口启用；Stage 11 前缀及 capture 保留逐 chunk partial，Stage 12 消费模式与生产者配对。chunk 数少于、等于、大于核数，奇数 head、强制单核、变长空序列均进入边界精度验证。归约顺序改变必须通过原 FP32/FP64 七输出判据，不能放宽 dA_log/dDt_bias 阈值。此项理论收益较小，不承诺能单独推动 8 ms 目标。

状态：R2 全路径候选不采用；R2b 无 norm 特化通过增量验收，保留。R3/R4 资源与依赖设计完成，尚未运行。

R2 第二轮 device 2 采集结束发现外部 ATK 进程，该组作废。同一候选库另名 r2d3，在空闲 device 3 重新完整两轮，不混合不同卡数据。

R2b 预先约定：若 R2 复测仍只有无 norm 稳定改善，单独做编译期无 norm 特化。原全配置门禁失败记录保留。新候选需四个无 norm 配置每轮改善>0.3%，带 norm 的 AIC/AIV 指令段与 h24 哈希一致，且各轮带 norm 耗时回退不超过 0.3%（测量噪声限制）；超出时复核，不直接保留。满足后仍须原 20 case 七输出全过。该规则仅用于不改带 norm 机器码的候选，不放宽精度标准，也不声称八配置都加速。

R2 device 3 完整复测：无 norm 改善 0.76%～1.44%，带 norm 回退 0.29%～0.95%，不满足全配置门禁。九个边界 case 七输出通过，但不进入原 20 case、不提交实现。

R2b 库 SHA256：`1ce88f73f066b3115e873e97af25a56a7de11be7311b5d5ab30f7b894b9fc588`。norm1 AIC/AIV 指令段与 h24 完全相同。九个边界 case 七输出通过；原 20 case 的 140 个输出全部通过，FP32/FP64 标杆和 CT L1 未变。只补了调用处换行，构建时与提交源码的 C++ token 相同。

两轮 A/B：无 norm 四配置每轮改善 1.01%～1.33%；带 norm 各轮变化在回退 0.062% 至改善 0.110%，满足上述专项门禁。原全八配置“均加速”门禁仍记录为失败，不覆盖该结果。

| T | l2norm 反向 | beta 反向 | h24 / ms | R2b / ms |
| --- | --- | --- | --- | --- |
| 8192 | 否 | 否 | 10.589 | 10.480 |
| 8192 | 是 | 否 | 11.288 | 11.289 |
| 8192 | 否 | 是 | 10.617 | 10.483 |
| 8192 | 是 | 是 | 11.189 | 11.183 |
| 16384 | 否 | 否 | 21.177 | 20.946 |
| 16384 | 是 | 否 | 22.299 | 22.297 |
| 16384 | 否 | 是 | 21.197 | 20.959 |
| 16384 | 是 | 是 | 22.363 | 22.365 |

B=1、H=96、K=V=128、BF16；表为两轮各 20 次中位数的平均。910B3 device 3，CANN 9.1.0 0605，主核和归约核 Task Duration 相加。无 norm Vector 占比约 62.8%～62.9%，带 norm 约 67.1%～67.2%，未达到 80%。

证据：A2 `/data/wys/kda-finalize-transfer-20261022/` 下 `r2b-perf.json`、`r2b-specialized-review.json`、`r2b-kernel-text.json`、`r2b-precision20/results.json` 及四次原始 op_summary。工作区 `validation/resident-transfer-20261022/` 保存副本。此为私有 fast-launch 增量验收，不代表新版本已完成 200×3、公开 ACLNN 或 A5 设备验收。
