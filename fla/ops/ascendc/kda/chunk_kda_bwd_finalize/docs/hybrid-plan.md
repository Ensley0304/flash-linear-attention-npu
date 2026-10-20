# A2 两版融合优化

目标：B1 H96 T8192 K=V128 BF16，无附加反向，主核与归约核总耗时先低于 10 ms；原 70% A5 吞吐目标对应 9.604066 ms，继续保留。精度仍以原 FP32 局部矩阵乘标杆、独立 FP64 golden 和七输出门禁判定。

## 主线选择

2026-10-20，设备4，同一缓存输入，20预热+20采样，旧→新→旧→新。下表取两轮 Task Duration 均值，均包含主核与归约核。

| 版本 | 源码/产物 | 耗时 |
|---|---|---:|
| 原版双head | 3d7f1c15，tile16.so，hash 75c16508… | 12.042 ms |
| CANNBot新分支 | 56ca50fd，finalize-p5c，hash 0126dd2a… | 13.644 ms |

原版耗时低约11.7%，选它作为基础。独立分支为 kda-finalize-a2-hybrid；原版与新分支均保留。证据在工作区 validation/hybrid-20261020/compare-summary.json，远端 /data/wys/kda-finalize-a2-hybrid-20261020。

两条入口分别是 torch fast launch 和直调 executable，比较 device kernel 时间；不比较 Python、输入生成或 host 等待时间。mean和median均保留。最终候选还须在相同入口做八配置两轮交替A/B。

## 已核对的差别

原版保留双head、一AIV一head的调度、下一work输入预取、按head提前通知、并行scan、树形参数归约和末带16行tile。新分支是一个head由两AIV分行处理，局部GEMM统一FP32，S7/S9使用16行tile，并提前计算gate。

新分支的树形归约、完整块特化等在原版已有对应实现，不能重复计算收益。原版已有的双AIV协作、Stage9/11直接融合、简单删除dq写回等失败记录继续有效，不原样重试。

新分支rH转Cube候选通过边界验证，但八配置回退0.55%～1.92%，撤销；不迁入原版。

## 实验顺序

| 顺序 | 候选 | 目的与约束 |
|---|---|---|
| 1 | 保留双head调度，逐个局部GEMM试FP32替代三项BF16残差 | 先试Stage8的left，再独立试Stage10的right；关闭HF32。减少残差拆分/重建和MMAD发射，评估增加FP32搬运的净收益。已有FP32 dq不改。实施前补齐L1/UB/workspace生命周期。 |
| 2 | 仅在已减少残差暂存的路径扩大Stage7 tile | 借鉴新分支16行处理，减少广播和循环；不能直接照搬新分支32行AIV内存图。只在容量与双buffer合法时实施。 |
| 3 | 原版后段gate计算与前段空隙重排 | 先对照现有“仅norm首tile提前”与失败的全分支实验；需要新的生命周期或同步位置证据才试，不把旧失败方案换名重跑。 |
| 4 | 跨work交错扩大 | 前述局部实验不足时，再设计workspace代次、credit和片上槽所有权。不得只增加buffer或提前发flag。 |

这些是待验证假设，不承诺收益可以相加。从12.042 ms到10 ms需再降约17.0%，到9.604 ms需约20.2%；几项不足1%的微调不够。

每项独立修改：构建及边界检查 → 八配置两轮交替A/B → 稳定收益后原20case七输出 → 全过才保留。涉及累加顺序变化时边界使用独立标杆，不能要求与旧NPU输出逐位一致替代精度判定。最终继续200case三seed、原功能/layout覆盖和公开入口验收。

## H1：Stage 8 left 改 FP32（撤销）

Stage 7 保存乘过掩码、符号和 65536 缩放的 FP32 dAkk；Stage 8 的 left 和 dq 共用 FP32 kNeg，关闭 HF32。Stage 10 仍用原三项 BF16 残差。生产路径不再拆分/写回 kNeg 的 BF16 高低位；调试 capture 保留原快照。

- L1：每 head 64 KiB，依次放 dAqk 16、dAkk 16、kNeg 32。Stage 8 每带全部加载；Stage 10 随后重载两种 lower 的四个 BF16 平面，不能沿用已被覆盖的 dAkk。
- L0：原双槽不变。left 用槽1，dq 用槽0；每槽 A/B 各32 KiB、C 64 KiB。
- UB：原双 FP32 输出槽复用，lower 的 dAqk/dAkk 串行提交；kNeg 使用原双8 KiB槽。仍为192 KiB。
- GM：每 bank 尾部新增 dAkk FP32 16 KiB，总384 KiB；两 head、两 bank、20核共新增1.25 MiB。不能覆盖原 dAkk_raw，因为它与后续 bkPos 的生命周期重叠。
- 同步：ready/free 计数不变。Stage 8 完成 L1 读取后才归还槽；Stage 10 重载使用同一 keFree/keReady。kNeg 跳过 BF16 输出时归还对应 outputFree，FP32 DMA 由独立 fp32Free 保护。

预期减少 Stage 7 残差指令、Stage 8 MMAD 发射和中途重装 kNeg；代价是新增 FP32 dAkk 写回以及 Stage 10 dAkk 重载。净收益由同入口交替 A/B 决定。

实测：9个边界、七输出通过；八配置两轮均回退0.22%～1.62%，不进入20case保留门禁。目标case约12.015 ms；Vector忙碌时间减少约0.23 ms，但总耗时未下降。补丁和报告保存在 validation/hybrid-20261020/h1-*，源码恢复基线。

## H2：独立将 Stage 10 right 改 FP32（未保留）

从原基线开始，Stage 8 left 保持三项BF16残差。Stage 10 两项乘法改FP32、关闭HF32；相应省去dAqk、qPos、bkPos残差拆分。FP32转置沿用新工作流已验证的CATLASS映射，MMAD设置kDirectionAlign。

- L1：每head仍64 KiB。Stage 10将前32 KiB放FP32 dAqk/dAkk，后32 KiB依次放qPos/bkPos。下一带Stage 8须重载BF16 dAkk。
- L0/UB：容量和双槽不变；Stage 7 qPos/bkPos复用原FP32双输出槽，读写由fp32Free保护。
- GM：新增16 KiB FP32 dAkk；qPos/bkPos原高低位两平面32 KiB改为同大小单FP32矩阵。
- 调试：保留BF16快照，但不写回覆盖供Stage 10读取的FP32 qPos/bkPos。

这项独立验证是否能把省去两组残差处理的收益转为总耗时下降，不叠加已撤销的H1。

实测：9个边界、七输出通过。目标case两轮改善1.02%/1.10%，但norm配置不稳定，16K双开关两轮回退0.13%/0.15%，不满足八配置保留门禁。补丁、产物、报告独立保留，不提交此实现。

## H3：Stage 8/10 统一FP32并共用驻留lower（已保留）

把两者作为同一数据生命周期设计：dAqk/dAkk各16 KiB在Stage 8首带加载，驻留至Stage 10末带；后32 KiB按阶段依次放kNeg、qPos、bkPos。两head仍独立，原跨单元握手不变。这样移除H1/H2单独转换造成的lower格式切换和重载。

Stage 7生产路径不再拆分所有后段BF16残差，FP32输出仍双槽。调试路径保留旧残差快照。L1/L0/UB容量不增加，GM仍为每bank384 KiB。只更新后段GEMM与操作数生命周期，Stage 9/11及其scan、参数归约顺序不改。

以原12ms基线重新跑完整门禁；不能把H1、H2的局部收益相加作为验收结果。

两轮交替A/B均通过；9个边界和原20case、每case七输出均通过原CT L1，参考hash未变。动态库SHA256：`8baaa7a44f94838b9d637d82806b0c9e0dca99395bd4d853a4047a398c143a10`。以下取两轮各20次采样的median均值，包含主核与归约核，单位ms。

| T | norm | beta反向 | 原基线 | H3 | 降幅 |
|---|---|---|---:|---:|---:|
| 8K | 关 | 关 | 11.980 | 11.190 | 6.59% |
| 8K | 开 | 关 | 12.575 | 11.838 | 5.86% |
| 8K | 关 | 开 | 12.016 | 11.244 | 6.43% |
| 8K | 开 | 开 | 12.566 | 11.860 | 5.62% |
| 16K | 关 | 关 | 23.926 | 22.384 | 6.45% |
| 16K | 开 | 关 | 25.211 | 23.743 | 5.82% |
| 16K | 关 | 开 | 23.968 | 22.370 | 6.67% |
| 16K | 开 | 开 | 25.124 | 23.780 | 5.35% |

目标case Vector忙碌时间约8.24→7.41 ms，占比69.3%→66.7%；加速来自减少计算和搬运，并未达到80%占比。尚未达到10 ms或70% A5目标；此版本尚未重跑完整200case三seed及公开入口验收。原始profiling在远端`/data/wys/kda-finalize-a2-hybrid-20261020/h3-*-prof`，摘要与精度报告见`hybrid-evidence/h3-*.json`。

## H4：Stage 7 band改32行并直接生成FP32输出（已保留）

Stage 7的band按32行处理，lower仍按16行。生产路径FP32输出双槽放[88,120) KiB，exponent放[120,136)，power放[136,152)；输出槽在计算期间兼作临时operand，提交MTE3后由独立free事件保护。band exponent保留[160,176)，beta broadcast保留[176,178)，供Stage 9/11按原地址消费。这样无需新增UB，也无需额外一次value到输出槽拷贝。capture继续用原16行路径。

H3的20case门禁通过后才启动H4。对照库使用已验收H3，不能仅与原12ms版本比较后就判定有增量收益。

9个边界、原20case七输出全部通过。两轮八配置均改善1.90%～3.87%。目标case同轮H3为11.320 ms，H4为10.889 ms；四功能配置依次为8K：10.889/11.619/10.856/11.605 ms，16K：21.637/23.210/21.651/23.172 ms。统计口径与H3相同。

动态库SHA256：`fe8ea5f8001da7e928bb18beb1282d548a1a5517a8f73e5df7e040a9c540b2b1`。性能与精度产物hash一致，证据见`hybrid-evidence/h4-*.json`。仍未达到10 ms，完整200case三seed及公开入口验收未重跑。

## H5候选：同一带复用正负指数

先保存本带有效行的`exp2(center-gk)`，qPos用FP32除法求其倒数，再把band exponent替换为正指数供Stage 9/11使用。kNeg的带外行不进入缓存；无效行不做除法。

仅完整finalize且`lowerBound∈[-5,0)`时启用，其余保持直接计算正指数。合法safe-gate累计量在32行带内跨度不超过155（自然对数单位），居中后绝对值不超过77.5；正负指数都在FP32正规数范围内。此证明依赖gk符合前向累计gate契约，不能推广到任意不一致的gk输入。

新增存储为零，复用[160,176) KiB的band exponent，先负后正。舍入会变化，必须重新跑原精度门禁，不能用公式等价代替验证。只有H4精度通过后才开始此实验。
