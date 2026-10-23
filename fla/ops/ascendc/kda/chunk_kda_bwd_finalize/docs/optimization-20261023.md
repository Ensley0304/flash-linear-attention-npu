# 9 ms 目标增量实验

本轮保留 S15：压缩三角 mask，腾出 UB 提前搬入 raw gate。同设备同轮目标 case 从 10.084 降至 9.991 ms，改善 0.92%；原 20 case、140 个输出全部通过。S7～S14 均已撤回。尚未达到 9 ms。

基线 S3 `26d551ca`，原始 B1/H96/T8192/K=V128/BF16/BNSD、无 norm/beta 反向约 9.988 ms。接口、FP32/FP64 标杆、CT L1、种子及七输出不改。目标主核加 Stage12 小于 9 ms；这需要再降低约 9.9%，单个小优化不能保证达到。

按 PR #833 的设计→开发→验证闭环增量执行，保留本分支已确认的 A2 数据路径。每次先九边界，再八配置两轮交替 A/B；稳定收益后原 20 case 的 140 输出全过才提交。完整 200 case 和公开入口验收另计，不将增量门禁写成全工作流完成。

本次重新核验 PR head 仍为 `c53f7811c0330fbf947a7ea14b4785c6c6b401ab`，沿用原 CANNBot `61518bc950fac84c953abe19332e43c24d4c1985` 的开发与验证流程，不升级工具版本或替换当前标杆。当前是既有 Ascend C fast-launch 实现的增量优化，不声称已迁移为 CATLASS 直调工程。

## S7：32 行 lower 直接在双输出槽计算（候选）

旧 H12 在 16 行 lower 上未取得稳定收益。本次只验证 S3 扩大到 32 行后的路径；tail/capture 保持原样，不复用旧性能结论。

输入仍为 UB[88,104)、[104,120) KiB；dAqk 和 dAkk 分别使用输出 [144,152)、[152,160)。先等对应 fp32Free，再直接执行 mask 乘法及 ±65536 缩放，V→MTE3 后写原 GM 地址，MTE3→V 释放槽。每矩阵独立复用，删除四次 8 KiB UB 加零复制；跨核通知和 GM 流量不变。

q/k/gk、lower、常驻小量、UB 总量 192 KiB 和事件数量不变。band 的 power scratch [136,152) 仍由首次 fp32Free[0] 等待保护；槽 1 在 [152,160) 不重叠，第一次复用仍等 fp32Free[1]。算术顺序不变，删除加零可能改变零符号，由原精度门禁检查。

收益假设：减少 lower 复制和同槽写回等待；旧 H12 无稳定收益，因此本项必须按当前 S3 同轮实测，不预估百分比。

S7 实测：九边界七输出全过。原始 case 两轮改善 0.504%/0.692%，但 8K beta-only 为 -0.073%/+0.258%，16K beta-only 为 +0.152%/+0.191%，未通过八配置每轮 >0.3% 门禁。撤回，不运行20case、不提交实现。证据 `hybrid-evidence/s7-*.json` 和工作区 `s7-evidence.tar.gz`，归档已逐文件校验哈希。

## S8：Stage 11 Gate 连续运算共用 normal mask（待实验）

当前 S3 原始 case 的 AIV Vector 约 64%、Scalar 约 44%，两者可重叠，不能相加。Gate 导数六个连续逐元素调用都使用相同 FP32 长度，但 count 接口各自切换 Counter/Normal 模式。尝试在这六项运算间只设置一次 64 元素 normal mask，repeat 为 active×2，末尾恢复 mask；所有原 PIPE_V 依赖保留。

数学仍为 Exp(-Abs(raw×expA))/(1+Exp(-Abs(raw×expA)))²，不改计算顺序或舍入。每行 128 个 FP32，active 为 1～16，repeat≤32，小于 255；尾块也只处理完整的有效行。地址、双槽、跨核协议及 GM 流量全部不改。

只在 S7 完成保留/撤回后独立实施。具体 API 已核对当前 CANN 9.1 basic_api 的 unary/binary 接口，使用 `isSetMask=false` 的 mask/repeat 重载；不将高层 count 重载误当成相同模板。仍按九边界→八配置双轮→原20case执行。

S8 实测：九边界七输出全过。原始 case 两轮仅改善 0.016%/0.102%，norm 多项回退，未通过门禁。已撤回，不跑20case、不提交实现。证据 `hybrid-evidence/s8-*.json`、`s8-evidence.tar.gz`，归档哈希已复核。

## S9：连续任务区间分核（待实验）

附件 `ResidentVector::Process/FindNextWork` 在 head 内连续遍历本核的 chunk 范围。本基线每核跨 usedCoreNum 个 work 访问下一 chunk。候选只在完整生产入口改变逻辑任务映射，prefix/capture 保持原映射；数学计算和最终按 chunk 索引写 partial 的顺序不变。

令 N=ceil(H/2)×chunks，C=usedCoreNum，核 c 的起点为 c×floor(N/C)+min(c,N%C)。仍用原 physical work=c+iC 控制循环、bank、ready/free；实际 head/chunk 使用 logical task=起点+i。每核任务数与旧映射完全相同，所有任务恰好覆盖一次。next/previous/ready 的地址也统一使用 logical task，不能只改当前地址。

原始 N=6144、C=20，各核仍为307/308个任务。逻辑任务保持 head-major 单调递增，奇数 head 的无效 owner 只能出现在该核任务末尾，不会出现从无效 owner 返回有效 owner 却漏预取的情况。每个 chunk 的扫描独立，Stage12 仍按原索引归约，不改变精度运算树。

容量和数据字节数不变，收益只可能来自访存局部性及代码生成；现有 profiling 不能提前证明收益。先做映射覆盖、任务数量和 odd-head 预取不变量检查，再按原设备双门禁判断。

S9 实测：九边界七输出通过，CPU映射覆盖检查通过。原始case两轮 +0.030%/-0.050%，基本持平；norm有0.72%～1.09%收益，但不满足八配置门禁，也未改善主目标。撤回，不跑20case、不提交实现。归档 `s9-evidence.tar.gz` 已校验文件哈希、Vector和调度源码，摘要见 `hybrid-evidence/s9-*.json`。

## S10：最后一次 workspace 读取后提前归还 bank（待实验）

当前完整 chunk 的末带 Stage11 用两个16行槽，入口已搬完 left/right/dk_base/dg_base/db_base。后续没有 workspace 读取：raw_g 是外部输入，dk/dG/dBeta 和 partial 是独立输出。当前 S5_DONE 直到整个 Stage11 及输出排空后才发，阻止下一代 Cube Stage0 复用该 bank。

只对完整非capture路径，将这一通知移到末带两个 right 搬入排队之后，由 PIPE_MTE2 发出，保证全部旧 GM 读取结束。通知编号仍为12+bank；两AIV各发一次，Cube仍在原位置集体等待。尾块、短前缀、capture、奇数head空owner在旧出口通知，不重复发。最后一带的BAND_LOADED仍按原规则消费。

逐区检查：下一 Stage0 写 dk_state[0,32)、dv_b[32,64)、zV[64,80) 和 rH块[256,272) KiB；前者的旧 dk_base 已全部搬入，dq和zV旧消费者已结束，旧 dq_local 已由 Stage9消费。Stage9输出在进入Stage11前排空。其余 lower/left/right 的本 work消费者也已完成，下一代的工作区写入无需等待UB计算或最终输出。

L1/L0/UB/GM容量、数据量、双代窗口、WORK_OPEN/KE_READY背压不改；新增的是通知位置，未增加可同时占用的bank数量。每次归还仍对应一次Cube等待，首项、回绕、尾项和奇数head进入九边界及20case检查。收益假设是下一Stage0与末带Vector重叠，需八配置A/B证明，不从静态等待量直接推算加速。

S10 实测：九边界七输出全过；原始case两轮回退0.344%/0.391%，仅双融合配置约0.1%收益，撤回。未跑20case、未提交实现。归档和摘要均已保存。

## S11：rH 两个64×64对角块（待实验）

沿用已接受的H24将rH搬到Cube的路径，只把四个32×32块改成两个64×64块，减少两次MMAD/Fixpipe，额外计算非对角元素。K维仍128，取对角线后才进入原Stage5，不改变后续数学。

每槽L0A/B各16KiB、L0C16KiB，仍在原双buffer容量内。结果由16增至32KiB，使用GM原WS_DQ_LOCAL[256,288)；Stage5在UB[88,120)一次搬入后Gather，Gather结束才允许输入覆盖。没有增加常驻内存，下一Stage8仍在原时刻覆盖GM。rH索引同步改为64块宽，尾块和head数不影响固定128维状态。

代价是rH块的计算和GM往返翻倍，收益仅可能来自较少的小任务发射；必须用同轮A/B验证。精度门禁不变，失败撤回。

S11实测：九边界七输出通过；原始case两轮回退0.422%/0.381%，其余配置也回退0.38%～2.17%。撤回，保留四个32×32块。没有跑20case或提交实现，归档SHA256和Cube/Vector源码均复核。

## S12：Gate 固定系数合并（待实验）

BeginStage11预先计算exp(a_log)×lowerBound，保存UB[87.75,88) KiB；原expA保留给raw gate。每tile的(dg×expA)×lowerBound改成dg×预计算系数，减少一次全tile乘法和barrier，每chunk多一次64元素系数乘法。

新槽在原192KiB容量内，Stage5的beta广播会覆盖它，但BeginStage11在Stage5之后重建；Stage7/9/11不再覆盖此区。双buffer、GM数据和通知不变。

两种乘法结合顺序在FP32有舍入差异，不能按数学等价宣称精度通过。先检查边界、有限性和原FP32/FP64双标杆，性能达标后必须原20case七输出全过，否则撤回，不放宽误差门槛。仅在S11决策后实施，不叠加未验收补丁。

S12实测：九边界七输出通过；原始case两轮回退1.332%/1.146%，norm改善0.27%～0.65%。未通过性能门禁，撤回，没有跑20case或提交实现。数学操作减少未带来目标case加速，不据此给出收益预估。

## 当前S3时间线及S13：两带交替right产出顺序（待实验）

直接使用已验收S3库，H2/T256/单核组、无norm/beta，模拟器三条MIX lane完整结束。AIV0/1的Vector约85.21/85.24μs，空隙72.21/72.79μs；空隙与MTE2重合38.22/36.73μs、MTE3重合22.54/25.97μs。区间有重叠，不相加，不外推8K硬件时延。

S10_FIRST_READY与Vector空隙重合6.62/4.89μs；BAND_LOADED重合2.51/7.16μs，后者还包含Stage8握手。AIV1有更多等待，但这尚不能证明交换顺序会加速。拟只改Stage10两个独立head的优先级：start=32且两个head都有效时按1→0，start=0按0→1；尾head不足两个及capture仍按0→1。

L1装载、L0装载、MMAD和Fixpipe保持同一head顺序；矩阵0/1在各head内的累加顺序不变。S10_FIRST_READY改为“本带首个head就绪”，另一个head在读right前再等BAND_LOADED；首个head在Stage11结束补等BAND_LOADED。每AIV仍各消费两通知一次，下一带覆盖GM仍受原握手保护。其余Stage、地址、容量、双buffer和输出布局不变。

这项只检验两AIV等待不平衡，不能保证达到9ms。按九边界→八配置双轮→原20case七输出执行，不叠加S7～S12。

S13设备4在首次采集前被其它测试占用，脚本主动停止，未产生性能记录；边界报告保留。改用空闲设备3，对相同hash的S3/S13重新进行完整双轮，旧错误日志保留。

S13有效结果：原始case两轮回退0.810%/0.660%，未通过门禁；九边界七输出通过。撤回，没有跑20case或提交实现。原始时间线中的等待不均不能直接转换成加速，当前交替顺序反而加重关键路径。

## S14：Stage10按head连接L0装载和MMAD（待实验）

S3完整时间线显示S10_FIRST_READY仍与AIV0/1空隙重合6.62/4.89μs。候选不换head顺序，而把Stage10的“两head加载L0”与“两head计算”两个循环合成一个：L1仍提前加载两head，L0按head0装载→发MMAD/Fixpipe→head1装载→发MMAD/Fixpipe。

使用既有operandReady/Free和resultReady/Free，不增加buffer或通知；两head的L0槽独立，当前head的MMAD有机会与另一个head的MTE1重叠。matrix0/1仍按原顺序，FP32累加及HF32关闭不变。代价是第二head的L0装载发射更晚，不能预判收益；独立从验收基线测试，S13未验收时不叠加。

S14实测：九边界七输出通过；原始case两轮仅改善0.139%/0.042%，norm回退1.84%～2.29%。撤回，没有跑20case或提交实现。

## S15：位掩码换出UB，提前搬入compact路径raw gate（已验收）

现有strict-lower FP32 mask占UB[64,80) KiB，但只有0/1。完整生产入口、无norm时将它改为512B位掩码，驻留[84.25,84.75)；零值32B在[84.875,84.90625)，四个mask指针各32B在[85,85.125)。现有gather索引[84.75,84.875)、rH索引[85.25,85.75)及小量不重叠。按launch初始化并用S→V事件确认。

Stage3/7以Select执行严格下三角选择，余下公式和缩放不改。核对当前CANN 9.1 c220实现：普通tensor-mode Select会隐式使用UB184KiB临时区，与本算子Stage3广播冲突。因此使用公开SetCmpMask和已设置CMPMASK的Select模板重载，指针提前放入专用槽，避免隐式scratch；不直接调用底层intrinsic。

腾出的[64,72)、[72,80)用于compact Stage11两个16行raw_g输入。随left/dk/dg一起预取，在S10_FIRST_READY之前发射，原inputReady覆盖全部输入；删除原left死亡后的V→MTE2→V往返。raw_g字节数不变，依然有两个输入槽和输出槽。最后一带仍使用原布局；尾块无compact路径，但使用同一位掩码；norm/capture/短前缀保持FP32 mask原路径。

这是一项有依赖的内存优化：压缩mask本身未必更快，目标是释放空间并消除raw_g晚搬运。选零与乘零可能改变零符号，不放宽精度。事先定义专项门禁，沿用S1口径：无norm四配置两轮各>0.3%；norm两主核指令哈希须与S3相同，双轮耗时变化均在±0.3%；再跑原20case七输出。全八配置原门禁报告仍保存，不覆盖为pass。任何项失败撤回。

设备 5 完成 S3→S15→S3→S15，每轮每配置 20 次预热、20 次采样。下表为两轮中位数的均值，含 Stage12，单位 ms。

| T | 附加反向 | S3 | S15 |
|---|---|---:|---:|
| 8192 | 无 | 10.083656 | 9.991199 |
| 8192 | norm | 10.844167 | 10.847257 |
| 8192 | beta | 10.081281 | 9.998980 |
| 8192 | norm+beta | 10.855787 | 10.865527 |
| 16384 | 无 | 20.251150 | 20.063791 |
| 16384 | norm | 21.704969 | 21.710373 |
| 16384 | beta | 20.189443 | 19.994574 |
| 16384 | norm+beta | 21.757039 | 21.760315 |

无 norm 四配置每轮改善 0.725%～0.976%；norm 的 AIC/AIV 指令段与 S3 完全相同，耗时变化在 ±0.124% 内。专项门禁通过；通用八配置全改善门禁未通过，其原报告保留。设备 3 首次采集被其他进程占用，守卫在采集前退出；上表全部来自设备 5 的完整重测。

九边界及原 20 case 均七输出全过。原 IDs、seed=20261013+ID、FP32/FP64 双标杆及 CT L1 均未改。库 SHA256 为 `87a9739d66f7e98294f5ac697413ee070a12d957f674ef55b82bc1d0c7c5942d`，性能、精度和指令段报告一致。标杆 SHA256 为 `1a5e5d498ce1cc5474f628c4279d871e305a3b9cb4b31b2bcba48e7109f6c706`。本地提交源码与构建快照仅注释和排版不同，代码 token 已核对一致。

证据见 `hybrid-evidence/s15-*.json`；工作区 `s15-evidence.tar.gz` 的 46 个文件逐一校验，归档 SHA256 为 `f41e09a899cc437cfcffcddcaebb19b05a93dcf87723aec2fde3bfef72e4974a`。完整 200 case 三 seed、公开入口和 A5 未重跑。目标 case 的 Vector 占比约 63.9%，9 ms 和 80% 均未达到。
