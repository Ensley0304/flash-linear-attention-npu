# A2 两版融合优化

目标（2026-10-21更新）：B1 H96 T8192 K=V128 BF16，定长BNSD，无l2norm及beta sigmoid反向，主核与归约核总耗时低于8 ms。精度仍以原FP32局部矩阵乘标杆、独立FP64 golden和七输出门禁判定。10 ms及原70% A5吞吐对应的9.604066 ms仅作中间里程碑。

本轮继续采用PR #833的单变量实验、设计先行、定向精度、同条件profiling和最终full验收流程。PR固定到`c53f7811c0330fbf947a7ea14b4785c6c6b401ab`，CANNBot沿用已有`61518bc950fac84c953abe19332e43c24d4c1985`。当前是已验收Ascend C基线的增量优化，不声称已完成CATLASS直调工程或公开入口的全流程验收。

## 8 ms目标与下一轮实验

当前H4目标case为10.889 ms，需要再减少约26.5%。主核AIV时间约10.774 ms，Vector忙碌时间约7.077 ms，Vector占比65.7%。这说明还有待检查的等待与重叠空间，但各pipe时间相互重叠，不能把差值直接当作可消除气泡。

如果Vector工作量不变，8 ms总耗时要求其忙碌时间占总耗时约88.5%；计入归约与启动后，主核需要更高的重叠程度。因此同时评估减少运算与调度重排，不以占比上升代替耗时改善。

| 顺序 | 单项实验 | 关键检查 | 当前状态 |
|---|---|---|---|
| 1 | H5b：仅当前32行带复用正负指数 | 带后qPos继续直接算指数；原失败case 4、长序列和极值gate | 边界通过，性能无收益，不保留 |
| 2 | H6：Stage 7期间提前计算下一work的dW/zV | GM子区最后读取、L1驻留、事件配对、单核回绕和奇数head | 边界通过，性能回退，不保留 |
| 3 | H7：Stage 8先写先算完的left，再写dq | L0C双槽事件不变；让left的Fixpipe与dq的MMAD有机会重叠 | 已启动构建与验证 |
| 4 | 按最新验收库采集稳态指令时间线及逐核周期 | 区分Stage 7/9/11的等待、搬运、Scalar发射和核间不均 | 待前述测试结束，避免采集互扰 |
| 5 | 根据时间线选择更大的跨work窗口或后段分块调整 | 先给出UB/L1峰值、workspace代次及ready/free闭环，再实现一个变量 | 尚未确定，不叠加到前述候选 |

H5b和H6当前均独立对照H4；若先保留其中一项，另一项须重新基于新验收库构建和比较。每项先边界精度，再八配置两轮交替A/B；收益稳定后跑原20case七输出，全过才保留。最终候选还须完整200case三seed和公开入口验收，未完成时不标记工作流complete。

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

## H5：同一带复用正负指数（未保留）

先保存本带有效行的`exp2(center-gk)`，qPos用FP32除法求其倒数，再把band exponent替换为正指数供Stage 9/11使用。kNeg的带外行不进入缓存；无效行不做除法。

仅完整finalize且`lowerBound∈[-5,0)`时启用，其余保持直接计算正指数。合法safe-gate累计量在32行带内跨度不超过155（自然对数单位），居中后绝对值不超过77.5；正负指数都在FP32正规数范围内。此证明依赖gk符合前向累计gate契约，不能推广到任意不一致的gk输入。

新增存储为零，复用[160,176) KiB的band exponent，先负后正。舍入会变化，必须重新跑原精度门禁，不能用公式等价代替验证。只有H4精度通过后才开始此实验。

首版构建成功，边界case 0/2/3通过，case 4出现非有限值，脚本在性能采集前停止，未保留。代码检查发现：qPos还要计算当前带之后的行，这些行没有缓存的负指数，不能走倒数复用。修正版H5b仅在`start <= first < end`时复用，其余行沿用原指数计算；已检查1～64有效行及16/32行tile的缓存索引范围，仍需上板验证，不能据此判定精度通过。

2026-10-21：H5b的9个边界case七输出全部通过。八配置两轮均无收益，回退0.035%～0.776%，不进入20case保留门禁。目标case中，Vector忙碌时间7.076→6.844 ms，但总耗时10.878→10.882 ms；减少的正指数计算未转化为关键路径收益。不保留H5b，继续使用H4。证据见`hybrid-evidence/h5b-*.json`。

## H6：等待Stage 7期间提前算下一work的dW/zV（未保留）

仅完整生产路径启用。当前循环为work i，后段处理i-1；下一项i+1的h/v/dv_scan已由原预取放在L1。发出i-1的S6_READY后，先算i+1的dW、zV，再进入i-1的Stage 8。i+1正式进入Stage 0时只算dk_state_raw、dv_b。

- dW：使用原L1 [8,24) KiB。work i的Stage 2已消费旧dW，直到i+1的Stage 2前没有其它写者。
- zV：使用i+1 bank的GM [64,80) KiB，它与i-1同bank。i-1的VECTOR_ACK已证明Stage 3读完旧zV；H3以后生产Stage 7不再向此区写BF16 lower。因此可单独提前复用这一子区。
- dq/dk/dG及其它仍活跃的bank区域不提前复用；不增加workspace代数，不放宽原S5_DONE/WORK_OPEN信用限制。
- 复用现有inputReady确认未来state预取完成；dW仍由dwReady交给未来Stage 2。L0操作数/结果双槽按原ready/free事件复用。未来state的l1Free必须留到未来Stage 0完成剩余两项读取后才归还。
- 前两次循环和最后排空不提前消费不存在的产品；下一work的奇数head按原有效head判断。capture及短前缀保持原路径。

2026-10-21：9个边界case七输出通过，包含单核多work回绕、奇数head和变长尾块。八配置两轮回退0.029%～2.487%，目标case 10.873→11.110 ms，不保留，也不进入20case门禁。提前的Cube工作没有带来净收益；是否延迟了Stage 8关键路径，须用时间线确认，不能仅凭调度顺序定量归因。候选在工作区`validation/hybrid-20261020/h6-source/`，证据见`hybrid-evidence/h6-*.json`。

## H7候选：Stage 8按计算完成顺序写回

H3之后Stage 8先算slot 1的left，再算slot 0的dq，但Fixpipe循环先等待slot 0。候选仅把写回顺序改为slot 1→0，让已完成的left可在dq计算时写回。公式、累加顺序、GM/L1/L0区间、双槽ready/free及跨核通知数量均不变；S8_READY仍在本head两项结果均写完后发出。

该顺序在源码上允许更多M/FIX重叠，实际能否降低耗时仍须A/B。它不减少总计算和搬运量，不预先声称收益。候选与H5b/H6分别验证，不同时修改三项。

## H8候选：优先产生Stage 9需要的dq

与H7分别比较。Stage 8先算dq(slot 0)，再算left(slot 1)，按0→1写回；head0的dq写完即发S8_READY，提前启动Stage 9。head1仍在dq/left都写完后发S8_SECOND_READY。owner0在Stage 9后本来就等待S8_SECOND_READY，owner1在Stage 9前等待它，因此两侧进入Stage 11读取left时，两个head的left均已写完。奇数head由原空owner分支补发S8_SECOND_READY，位置仍在head0的left写完之后。

不增加flag或buffer，GM/L1/L0区间和每项MMAD累加顺序不变；仅改变两项独立乘法的优先级及head0已有通知的发布位置。需要验证capture/前缀、奇数head、单核回绕及原精度门禁。未上板前不采用，也不把原S8_READY“保证两项结果完成”的含义沿用到新路径。
