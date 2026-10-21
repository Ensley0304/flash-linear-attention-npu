# A2 两版融合优化

目标（2026-10-21更新）：B1 H96 T8192 K=V128 BF16，定长BNSD，无l2norm及beta sigmoid反向，主核与归约核总耗时低于8 ms。精度仍以原FP32局部矩阵乘标杆、独立FP64 golden和七输出门禁判定。10 ms及原70% A5吞吐对应的9.604066 ms仅作中间里程碑。

本轮继续采用PR #833的单变量实验、设计先行、定向精度、同条件profiling和最终full验收流程。PR固定到`c53f7811c0330fbf947a7ea14b4785c6c6b401ab`，CANNBot沿用已有`61518bc950fac84c953abe19332e43c24d4c1985`。当前是已验收Ascend C基线的增量优化，不声称已完成CATLASS直调工程或公开入口的全流程验收。

## 8 ms目标与下一轮实验

当前验收库为H14：两轮目标case均值10.707 ms，需再减少约25.3%才到8 ms。完整H10/H14时间线及H10七组硬件指标已归档；H14目标Vector约66.5%。各pipe时间相互重叠，不能把差值直接当作可消除气泡。

如果Vector工作量不变，8 ms总耗时要求其忙碌时间占总耗时约88.5%；计入归约与启动后，主核需要更高的重叠程度。因此同时评估减少运算与调度重排，不以占比上升代替耗时改善。

| 顺序 | 单项实验 | 关键检查 | 当前状态 |
|---|---|---|---|
| 1 | H5b：仅当前32行带复用正负指数 | 带后qPos继续直接算指数；原失败case 4、长序列和极值gate | 边界通过，性能无收益，不保留 |
| 2 | H6：Stage 7期间提前计算下一work的dW/zV | GM子区最后读取、L1驻留、事件配对、单核回绕和奇数head | 边界通过，性能回退，不保留 |
| 3 | H7：Stage 8先写先算完的left，再写dq | L0C双槽事件不变；让left的Fixpipe与dq的MMAD有机会重叠 | 边界通过，性能基本持平，不保留 |
| 4 | H8：优先产生dq并提前通知head0的Stage 9 | Stage 11仍等两head的left完成；奇数head及回绕验证 | 边界通过，无norm配置回退，不保留 |
| 5 | H4生产库指令时间线，H2/T256/单核组 | 区分Stage 7/9/11的等待、搬运、Scalar发射 | 完整采集并归档，含三条lane |
| 6 | H9：末带Stage 9预取Stage 11独立输入 | 死区复用，dG依赖保持 | 边界通过，八配置回退，不保留 |
| 7 | H10：先产出两个head的dq | 两类结果分开发通知，原双槽及奇数head事件闭环 | 四轮性能复核、20case七输出通过，已保留 |
| 8 | H11：Stage 7指数拆分复用mask | 相同32位长度、repeat≤64、恢复mask | 边界通过，收益不稳定，不保留 |
| 9 | H12：Stage 7 lower直接在独立输出槽计算 | 两矩阵分别用槽0/1，覆盖前等各自MTE3完成 | 边界通过，八配置收益不稳定，不保留 |
| 10 | H13：Stage 10后半带K裁剪 | 保持L1物理stride，只减少全零的L0/MMAD区间 | 边界通过，性能回退，不保留 |
| 11 | H14：dG对齐跨行写回 | 512 B整行，32 B单位stride及16位上限 | 八配置双轮、原20case七输出通过，已保留 |
| 12 | H15：AIV稳态与排空共用后段调用点 | 事件顺序等价，检查实际指令段及Scalar开销 | 边界通过，八配置均回退，不保留 |
| 13 | H16：Stage 7两个算术源错开UB地址 | 256 B对齐、无重叠，输入/输出槽不动 | 边界通过，无norm回退，不保留 |
| 14 | H17：生产路径精简重复outputFree | fp32Free保持所有FP32写回保护，边界通知配对不变 | 边界通过，八配置回退，不保留 |
| 15 | H18：Stage 11完整chunk统一16行tile | q/k重排、q/k/gk恢复，额外80 KiB重读的成本 | 边界通过，无norm回退，不保留 |

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

实测：9个边界case七输出通过；八配置两轮改善范围为-0.119%～0.404%，不满足稳定收益门禁。目标case两轮H4均值10.859 ms，H7为10.860 ms，基本持平；不保留，也不进入20case门禁。证据见`hybrid-evidence/h7-*.json`，库hash为`267a78ed6c53eddd4a0c622efabc3fb585c1688796da156f4e1e9c78ad53cc4d`。

## H8候选：优先产生Stage 9需要的dq

与H7分别比较。Stage 8先算dq(slot 0)，再算left(slot 1)，按0→1写回；head0的dq写完即发S8_READY，提前启动Stage 9。head1仍在dq/left都写完后发S8_SECOND_READY。owner0在Stage 9后本来就等待S8_SECOND_READY，owner1在Stage 9前等待它，因此两侧进入Stage 11读取left时，两个head的left均已写完。奇数head由原空owner分支补发S8_SECOND_READY，位置仍在head0的left写完之后。

不增加flag或buffer，GM/L1/L0区间和每项MMAD累加顺序不变；仅改变两项独立乘法的优先级及head0已有通知的发布位置。需要验证capture/前缀、奇数head、单核回绕及原精度门禁。未上板前不采用，也不把原S8_READY“保证两项结果完成”的含义沿用到新路径。

实测：9个边界case七输出通过。八配置两轮中，无norm回退0.639%～1.731%，有norm改善1.247%～1.693%。目标case两轮H4均值10.847 ms，H8为10.979 ms；未满足保留门禁，不进入20case，也不保留。该结果说明调度收益依赖后续Vector负载；具体原因待时间线确认。证据见`hybrid-evidence/h8-*.json`，库hash为`e82938806602abe06a79a170ad1fd59bccd9c60f991428b47f26fc127fce69ec`。

## H9候选：末带Stage 9预取Stage 11的独立输入

H4生产库的完整模拟器时间线（H2/T256/单核组，无norm/beta）显示，Stage 9→11附近反复出现约2～2.6 μs的Vector空隙，包含输出排空与后继输入搬运。AIV0的S8_READY等待与Vector空隙重合仅0.024 μs，单独提前此通知不是主要机会。仿真时长不等于8K设备耗时，各重合区间也不能相加为收益。

仅完整64行chunk、非capture、需要Stage 11且最后一带start=0时启用。在Stage 9两槽raw输入排入MTE2后，预取Stage 11首个16行tile（行16～31）的left、dk_base、raw_g，分别写UB [24,32)、[40,48)、[56,64) KiB。这些区间原来是已消费的后半k和gk；当前Stage 9/11只再用k的前32行[16,24)，不会覆盖。Stage 9入口原有V→MTE2桥接保证此前Vector读取已结束。

left由H4原S8_READY/S8_SECOND_READY保证本head完成；dk_base来自Stage 5，raw_g来自输入，三者独立于Stage 9。Stage 11的首槽只补搬dG、dBeta和right，第二槽照旧；dG仍在Stage 9原写回排空后读取。MTE2按顺序执行，Stage 11原inputReady覆盖所有预取和补搬，不增加事件或buffer。UB仍192 KiB，GM字节数不变，每work将24 KiB搬运提前。

该候选独立基于H4，不叠加H8。尾块、capture、Stage 9短前缀保持原路径；变长完整chunk可用。先9个边界七输出，再八配置两轮A/B，稳定收益后原20case七输出；未通过前不保留。

实测：9个边界case七输出通过；八配置两轮回退0.497%～1.414%，不保留，也不进入20case。目标case同轮H4为10.855 ms，H9为10.934 ms；Vector忙时基本不变，MTE2忙时3.142→3.218 ms，提前搬运未带来净重叠收益。库hash `53dde753a52c41062223a786435f809c55b4d26e3cfc08b9c8fbe5cb90d29184`，证据见`hybrid-evidence/h9-*.json`。

## H10：Stage 8先生成两个head的dq（已保留）

H4时间线中，AIV1等待S8_SECOND_READY与Vector空隙重合约10.97 μs，高于AIV0的3.12 μs；该通知当前同时保证head1的dq/left完成。候选把两类结果分开发布：先算dq0/dq1、写回并发S8_READY，再算left0/left1、写回并发S8_SECOND_READY。两个AIV都在S8_READY后做Stage 9，在Stage 9后消费S8_SECOND_READY，保证Stage 11读取left安全。奇数head的空owner仍消费每个通知一次。

仅Stage 11/12非capture路径启用；前缀及capture保持原顺序。两相中L0双槽分别对应head0/head1；每相都先发两项MMAD，再按槽写回，下一相复用由原resultFree保护。L1两head操作数保持到各自left读取完才归还keFree，keReady仅在dq相消费一次。GM地址、每项MMAD顺序、HF32关闭、buffer容量和事件数量不变。Stage 10仍等本带Stage 8完成后启动。

代价是head0的dq通知较晚；能否补偿head1等待须实测。独立基于H4，不叠加H8/H9。按原边界→八配置两轮→有收益再20case七输出门禁执行。

首两轮八配置全部正收益，目标case为0.353%/0.288%；因后一轮略低于0.3%，原自动门禁保持false，不直接保留。追加同hash库的两轮交替A/B，预先约定复核条件：四轮每配置均为正收益、每配置四轮平均收益超过0.3%；满足后仍须原20case七输出全过。精度标准不改，两轮追加原始报告及单独复核结论均保留。

四轮复核通过，各配置平均改善0.349%～1.354%；9个边界和原20case七输出全部通过，参考hash不变。库hash `c052ab21475cfbf4843726fb35606874fc24ef7905ee264952980f1a7a4a18b3`。8K四配置为10.807/11.436/10.789/11.547 ms，16K为21.628/22.939/21.573/23.010 ms（无附加反向、仅norm、仅beta、两者开启）；目标Vector约65.96%，最高配置约70.4%。保留此项，仍未达到8 ms。完整200case三seed及公开入口验收尚未重跑。证据见`hybrid-evidence/h10-*.json`，包括原自动门禁false及独立四轮复核结论。

## H11候选：Stage 7指数拆分复用Vector mask

H4的四work时间线中，AIV0有2391条MOVEMASK。指令时长相互重叠，不能把它们的duration总和当作可消除耗时。当前CANN 9.1的dav_c220实现中，count形式的Mul/Exp每次都切换Counter模式、设置mask、计算，再恢复Normal和全mask；低层`isSetMask=false`接口允许显式复用设置。

先只改Stage 7的exp2拆分连续段：Maxs、Mins、两次Cast、Sub、整数Adds/ShiftLeft、Muls、Exp、Mul。所有数据均32位，count=active×128；统一Normal mask=64，repeat=active×2（2～64，capture上限32），连续块stride=1、repeat stride=8。开始设置一次mask，段尾ResetMask；原barrier、舍入模式、中心化和算术顺序保持。尾块按实际active重复，active=0仍跳过。

这是减少mask指令的单项实验，不改变buffer或调度。以届时最新验收库为基线；边界→八配置两轮→有收益再20case七输出。API依据为服务器安装的`asc/impl/basic_api/dav_c220/kernel_operator_vec_{binary,unary}_impl.h`及`asc/include/basic_api`声明，本地副本存于validation/hybrid-20261020/cann-api。

实测：基于H10构建，9个边界case七输出通过。首次msprof因/data满而没有导出CSV，此次无效；重跑时新旧库的输出和临时目录统一使用/dev/shm/wys/kda-finalize-h11-20261021。有效八配置两轮收益-0.309%～1.077%，目标仅0.047%/0.107%，不保留，也不进入20case。目标H10为10.835 ms，H11为10.827 ms，Vector忙时7.071→7.068 ms，未获得稳定的目标收益。

H11库hash `5bd776bf5efe8e770a14cb7bb0520c35faa5a15b6d5ac2ee78490211711b0af7`。四份有效op_summary均含320个主核及320个归约核记录，已下载并按SHA256核对，工作区归档为validation/hybrid-20261020/h11-valid-performance.tar.gz；摘要和精度见`hybrid-evidence/h11-*.json`。当前验收库保持H10。/data仍满，golden20缓存删除须用户单独授权；未删除任何缓存。

后续：用户已授权删除本轮golden20缓存，清理完成；远端源码和动态库已恢复H10并校验。源码、报告及验收库保留。后续精度测试按原标杆和种子重新生成缓存。

## H12候选：Stage 7 lower直接在独立输出槽计算

以H10为基线，重采完整生产库时间线。源码确认lower循环按tile选择FP32输出槽，导致同一tile的dAqk/dAkk先后占用同一槽；第二个矩阵须等第一个矩阵的MTE3。每项还要从公共value区复制4 KiB到输出区。

生产路径改成dAqk使用UB[178,182) KiB、dAkk使用[182,186) KiB。每次先等本槽fp32Free，再直接完成mask乘法和65536缩放，V→MTE3后写原GM地址，MTE3→V归还本槽。下一矩阵可使用另一个槽；下一tile复用同矩阵槽前仍须等待。输入槽仍按tile交替，capture沿用原路径。band开始复用这些区域前继续消费原fp32Free；Stage 9/11、跨核flag和所有GM地址不变。

每head少8次4 KiB的UB复制，目标case共少384 MiB复制量；GM字节数、UB峰值192 KiB、事件数不变。数学运算顺序保持mask后缩放；删除的加零复制可能改变零的符号，不改变有限非零值。性能收益待实测，不能把省去的复制量直接折算为总耗时。

验证：9个边界case七输出→八配置两轮交替A/B（每配置每轮改善>0.3%）→有稳定收益再原20case七输出。任一步失败不保留；FP32标杆、FP64 golden、CT L1及种子均不改。

实测：9个边界case七输出通过。目标case两轮改善0.175%/0.960%，但16K仅beta首轮回退0.595%、8K norm次轮回退0.189%，未通过原门禁，不保留，不进入20case。目标Vector约65.6%～65.7%，未改变主要瓶颈。库hash `d3fe8564ad24a277d9ee4a534355bedd61ac196187e1622d8587d2040f158242`；证据见`hybrid-evidence/h12-*.json`，四份CSV和实验源码归档于工作区`h12-evidence.tar.gz`。保留H10作为下一项的对照。

## H13候选：Stage 10仅计算后半带的有效K区间

H10完整模拟器时间线（H2/T256/单核组）中，S10_FIRST_READY等待与Vector空隙重合为AIV0 4.32 μs、AIV1 2.67 μs；H10的S8_SECOND_READY对应空隙已降至0.19/0.46 μs。具体时间不直接外推到8K硬件，重合区间不相加。

Stage 10计算`lower[:,s:s+32].T @ positive`。s=32时，严格下三角矩阵的前32行全零；因此只需`lower[32:64,32:64].T @ positive[32:64,:]`。本候选仅裁剪Stage 10的L1→L0和MMAD，K=64改32，仍FP32且关闭HF32，两项乘法的累加顺序不变。s=0和capture保持原路径；s=32尾块已有零padding继续有效。

L1保持原64行NZ物理布局和GM搬运量。FP32 NZ行32偏移为32×8个float；A从原列带偏移再加该行偏移，B加同一行偏移，3Dv2的源宽仍64，mExtension改32。L0A每槽4 KiB、L0B每槽16 KiB，原双槽容量不变；L0C及所有事件不变。被省略的乘积为零，不引入近似；设备累加分组是否影响舍入仍由精度门禁验证。

每head少524288 FLOPs，目标case共约6.44 GFLOPs；仅占后段Stage 8/10总计算的12.5%，不视为整算子加速比例。此前BF16版本的整条band零区裁剪失败，本次区别是H3后统一FP32、lower常驻，并只裁剪Stage 10计算，不同时修改Stage 7写回和GM→L1。按原边界→八配置双轮→原20case七输出门禁独立验证；H12完成后才启动，未验收补丁不叠加。

实测：9个边界case七输出通过，目标两轮回退0.087%/0.026%，多数配置也回退，不保留，不跑20case。减少这一段零乘法没有改善整算子时延；当前仍为H10。原始四份CSV及源码、日志归档在工作区`h13-evidence.tar.gz`。

## H14候选：dG使用对齐跨行写回

H10时间线的MTE3与Vector空隙重合约28.1/30.2 μs（两AIV，四work，不可直接换算硬件收益）。dG每行固定128个FP32，512 B，UB地址及GM行首均32 B对齐；当前仍用非对齐搬运指令。仅这一输出尝试普通DataCopy多block路径，dBeta和dq/dk/dv保持原实现。

blockCount=active，blockLen=16个32 B块，srcStride=0，dstStride=(H-1)×16个块。H≤4096时该stride≤65520，可用uint16；更大H或capture回退原DataCopyPad。GM基址/offset仍64位，尾块只写active行，BSND/TND布局不变。原outputReady/outputFree仍保护相同gateOut双槽，字节数、UB/L1/L0和算术全不变。

这是验证两类DMA指令在当前散写路径上的代价，不预设普通DataCopy更快。独立基于H10，原9边界、八配置双轮及有收益后20case七输出门禁不变。

实测通过并保留：9个边界和原20case七输出全部通过，FP32/FP64参考hash不变；另测T=2、H=4096/4097，两条stride路径的七输出均与H10逐位一致。八配置每轮收益0.53%～1.70%，两轮median均值如下（ms）：

| T | 无附加反向 | 仅norm | 仅beta | 两者开启 |
|---|---:|---:|---:|---:|
| 8192 | 10.7074 | 11.4030 | 10.7011 | 11.4004 |
| 16384 | 21.4599 | 22.7818 | 21.3963 | 22.8296 |

目标同轮H10为10.8495 ms，改善1.31%；Vector约65.65%→66.5%，最高配置约70.7%。MTE3忙时变化很小，不能把全部收益归因为带宽提升。库hash `421e5f480fd173a212a11a67e42dc03e63f0c7fe3893bf0cbaec5d9539d5b3e0`，证据见`hybrid-evidence/h14-*.json`，四份CSV及源码/日志完整归档于工作区`h14-evidence.tar.gz`。完整200case三seed及公开入口验收未重跑，目标8 ms未达到。

## H15候选：AIV稳态与末尾排空共用一个后段调用点

H10补采七组硬件指标：目标Vector约65.8%、AIV Scalar约45.9%、AIV I-cache miss rate约10.9%，UB bank-group冲突约11.6%。这些比例不等于可直接消除的耗时；Memory组的main_mem带宽计数偏小且L2带宽为0，不据此判定HBM带宽上限。证据为`hybrid-evidence/h10-hardware-metrics.json`。

AIV入口当前把Stage 3/5/7（含9/11）分别内联在稳态和最后一次排空处。候选用一个retire循环共用该调用点，不新增功能函数、不改变Stage内部实现。普通work只处理前一work；最后一个work依次处理前一work和当前work；首个且非最后work不进入后段。用编译器禁止展开该短循环，避免重新复制大段后处理。

VECTOR_RELEASE的位置保持：稳态在前一work后段之后等待；最后work在两个后段之间等待；只有一个work时先等待再执行当前后段。每个后段仍严格按S02_READY→Stage 3→VECTOR_ACK→Stage 5→S6_READY→Stage 7/9/11→S5_DONE执行，bank由被消费work计算；奇数head空owner的通知/等待仍完整。AIC、GM/UB/L1/L0、事件数量、公式及capture记录位置均不变。

检查首项、单work、连续多work、奇数head、回绕及排空的事件序列等价，再独立编译；比较实际设备指令段大小，确认是否减少复制。收益目标是减少指令取指/Scalar开销，不把I-cache miss rate直接当作加速比例；代价是新增短循环的判断与地址计算。先边界，再八配置双轮，有稳定收益才跑原20case七输出。H14已完成验收，本项重新基于H14，只修改AIV入口调用组织。

实测：9个边界case七输出通过。norm0/norm1的AIV指令段由178984/183880 B降至98096/99808 B，AIC指令段hash不变；但八配置两轮全部回退0.434%～1.650%，目标同轮10.868→11.036 ms。因此不保留、不进入20case。缩小指令段并未转化为性能收益，不能据此断言I-cache是主要瓶颈。库hash `d4d2d03b9ff8478e9ee3d4953dddba290232c2a9a91338a7337edcb8123c8a8d`；原始CSV、源码和日志位于工作区`h15-evidence.tar.gz`，摘要见`hybrid-evidence/h15-*.json`。

## H16候选：Stage 7指数临时区错开bank group

硬件ResourceConflictRatio显示当前地址布局的bank-group冲突约11.6%。Stage 7生产路径的value、exponent、power起点均为整KiB；先只针对这段多输入运算尝试256 B错位，具体冲突下降和总时延收益必须实测，不能把11.6%当成可直接消除的时间。

输出双槽仍UB[88,104)/[104,120) KiB。exponent改为[120 KiB+256 B,136 KiB+256 B)，power改为[136 KiB+512 B,152 KiB+512 B)，均16 KiB、256 B对齐，彼此及bandExp[160,176)不重叠。利用原临时区空隙，峰值仍192 KiB。lower准备、capture地址、Stage 9/11、GM、事件、计算顺序和舍入不变。

H15完成后以届时最新验收版本独立测试，不叠加未通过候选。原边界→八配置双轮→有收益后原20case七输出不变；若性能通过，再同条件补采前后ResourceConflictRatio，核对假设。


H16实测：9个边界case七输出通过。八配置两轮中，带norm改善1.049%～1.273%，无norm回退0.477%～1.031%；目标同轮10.861→10.939 ms。未通过门禁，不保留、不进入20case，也不追加bank冲突指标采集。库hash `a97c1b02a60207a01f8cbdc82abb5f4e8c5a02685f4191a55fd36097b69e88b7`；原始证据为工作区`h16-evidence.tar.gz`，摘要见`hybrid-evidence/h16-*.json`。这次地址改动未改变norm算术，但收益仍因配置而异，不能据此认为bank冲突已降低。

## H17候选：Stage 7生产路径只保留FP32写回槽同步

以届时验收基线为对照，单独精简Stage 7的两处内循环。生产路径不写BF16，但每次仍同时消费/归还outputFree和fp32Free。H14完整时间线里本地hard-event等待与Vector空隙重合约26.8/29.2 μs；不是全部可消除，重复等待须单独实测。

仅CAPTURE=false时，lower和band内循环省掉outputFree的wait/set，FP32输出仍严格由fp32Free保护，V→MTE3 ready和跨核flag完全保留。Stage 7入口先消费旧outputFree、完成布局切换，再各置一次；内循环不动该通知，原band末尾消费/重新置位，Stage 9/11沿用原协议。最后一带结束仍保留各一份outputFree；没有重复set，也没有遗漏wait。capture保持原BF16双槽同步。

lower输出[178,186)及band输出[88,120) KiB复用均受fp32Free保护；Stage 11原入口/退出排空不改。地址、容量、计算顺序和GM字节数不变。完整chunk每head减少20对重复wait/set；收益可能只在指令开销，不能把全部hard-event空隙视为收益。

先9个边界七输出，再八配置两轮交替A/B，每配置每轮>0.3%才进入原20case七输出。任一步失败不保留，标杆/CT L1/种子不变。该候选不叠加未验收H16；若H16通过，先验收并将H17重新基于H16生成。


H17实测：9个边界case七输出通过；八配置两轮全部回退0.825%～2.266%，目标同轮10.826→11.038 ms。不保留、不进入20case。AIC指令段hash保持不变，norm0 AIV指令段反而增加480 B，不能把源码中少了事件等同于最终机器码减少或流水改善。库hash `ad8088c9a35b4f9a25f10a2d66c531aeff479520ed5d8498f221d70c593884c7`，证据见`hybrid-evidence/h17-*.json`及工作区`h17-evidence.tar.gz`。

## H18候选：完整chunk的Stage 11统一16行tile

独立基于H14。当前后32行带的Stage 11用四个8行tile，前32行带用两个16行tile；生产时间线仍有本地hard-event与Vector空隙，Scalar忙时约46%。尝试把后32行带也合成两个16行tile，每chunk的Stage 11共六次迭代降为四次；向量元素数不变，减少每tile控制与搬运发射，收益须扣除以下重读成本。

容量约束：16行双输入需要UB[24,64)/[88,128) KiB，gate双输出用[8,16)/[184,192)，会覆盖原k后32行和gk。第一次Stage 11前，将本带q/k的32行从GM搬到UB[0,8)/[16,24)，Stage 11从first-start读取q/k；其余输入、运算区和输出直接复用已验证的末带16行布局，峰值仍192 KiB。该次结束后重新从GM载入完整q/k/gk到原[0,64)，再进入下一带。每完整chunk/head新增16+64=80 KiB GM→UB，目标case共960 MiB；这是换取更大tile的明确代价，不声称减少搬运。

两次重排均在原Stage 9/11输出排空后执行。消费inputFree[0]后，另发一次V→MTE2桥接保证全部旧Vector读取结束；搬运后MTE2→V确认，再恢复一份inputFree[0]供原入口消费。inputReady[0]无未消费通知。原跨核flag和所有workspace位置不改。Stage 11的扫描仍按两个8行组逆序执行，参数归约按原8行组累计，保持舍入次序。

仅FULL_CHUNK且非capture启用；尾块和capture原路径不变，变长中的完整chunk可命中。先9个边界七输出，再八配置两轮A/B，全部稳定改善>0.3%后原20case七输出。不改标杆、CT L1或种子；若重读成本抵消tile收益则不保留。

H18实测：9个边界case七输出通过。八配置两轮中，带norm改善0.448%～0.608%，无norm回退0.426%～0.990%；目标同轮10.835→10.903 ms。未通过门禁，不保留、不进入20case。库hash `2285d6a0eb82e55a86a175ec3660fac702280ab0387c0af9a4e401ac13183612`，证据见`hybrid-evidence/h18-*.json`及工作区`h18-evidence.tar.gz`。

## H15～H18批次结论

四项均先通过9个边界七输出，再完成八配置两轮交替A/B；没有一项满足性能门禁，因此没有将候选替换正式代码，也未对失败候选重复跑20case。每次采集含320个主核和320个归约核记录，预热/采样及FP32/FP64标杆均保持原样，归档内源码、日志和CSV的SHA256已核对。

正式代码保持H14（`0080465a`），服务器隔离工程也恢复H14源码和两个加载位置的动态库，hash均为`421e5f480fd173a212a11a67e42dc03e63f0c7fe3893bf0cbaec5d9539d5b3e0`。H14原验收目标值10.707 ms；本批次同条件对照为10.826～10.868 ms，应使用各自同轮对照判断收益，不能跨批次相减。

8 ms目标仍未达到。减少指令段、删除重复事件、局部UB错位和扩大tile都不能单独保证关键路径缩短；本批次不再继续枚举这几类参数。下一步应重新审查Stage 7/9/11之间的存活区和跨work排程，先证明能减少真实等待且不会增加同量重读，再进入新实验；H6、H9、H18的已失败方案不能原样重试。完整200case三seed及公开入口验收仍未重跑，不标记工作流完成。
