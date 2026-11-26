# KDA C32/C64 泛化：分阶段 Cube/Vector 流水设计

方案设计规则版本：V2。版本 D3.3，2026-11-26。

接口唯一事实源为 [api.md](api.md)，设备事实与历史实验见 [validation.md](validation.md)。本版整体替换 D2.4 正文；旧稿存入 [历史快照](evidence/design-d2.4-before-redesign.md)。本轮只改文档，不修改、构建或安装算子代码。

**状态：优先重设计 C64 保存反向（disable_recompute=true），详见 §2.12；先恢复高性能分阶段基线，再完成 C32 单 kernel。** C32 已完成的组件开发/实验见 validation §9–10，不能等同完整流水已实现。本文本轮只修改设计与证据，不修改 kernel。C32 保存跨 chunk 方案保留，重算单 kernel 要求保留；本轮不扩展重算调度。下列预算是开发门槛，不是实测承诺。

## 1. 目标

### 1.1 范围与不可妥协条件

遵循仓内 [03-方案设计 V2](../../a2a3-generalize/docs/agents/03-方案设计.md)，之后才是 04/05。只设计 api.md 已确认的 A2/A3 能力，包含 C32/C64、FP16/BF16、六组 K/V、训练状态、packed、gate/norm/beta 及保存/重算；不扩 GVA，不改变 A5。

- C32 前向一次计算 kernel；保存反向、重算反向各一次计算 kernel。核内包含激活、恢复状态、梯度归并和格式转换。禁止用隐藏的 Exp/Cast/Transpose/Fill/Reduce 满足此数量。
- C64 前向 Prepare→FwdH→Finalize；保存反向 BwdPrepare→Dhu→Finalize；重算反向 Recompute→FwdH→BwdPrepare→Dhu→Finalize。允许必要的适配 kernel，但完整统计，优先将转换并入生产者或消费者。
- C64 本身必须恢复到原有高性能结构；C32 必须胜过**性能合格的同规格 C64**。赢过 24 ms 的慢通用 C64 不算达标。
- Python 只路由；ACLNN 检查描述符/属性/offsets。数据相关的安全分支在设备执行，不在 Python 扫描数据。
- 测试使用 fla-org 原脚本及其标杆，保持原阈值；不使用 ATK；A2/A3 任一平台实测可满足本轮平台要求。

### 1.2 源码证据与采用边界

本地实现起点 f2ed09dc9465fcc42e141560407c811e931811c7；上游测试 c55dc553edf44701bf703715ed052d811f8a6376。下列路径相对 `../../a2a3-generalize/fla/ops/ascendc/kda/`。

|证据|核实事实|D3 采用方式|
|---|---|---|
|chunk_kda_fwd_prepare/docs/design.md、op_kernel/chunk_kda_fwd_prepare_policy.h|Prepare 八阶段，四 head；A2/A3 pair wave、独立 RHS 区；C4 先提交全部 head，再推进 C5|保留 stage-major 调度和独立生命周期；不能做完 head0 全链后才做 head1|
|chunk_kda_bwd_recompute/docs/design.md|A2/A3 每核两页、每页四 head；READY/FREE；Vector 可填下一页|借鉴双页信用与预取，不复制固定 BF16/128 数据尺寸|
|chunk_kda_bwd_prepare/docs/design.md|已有四 head 工作项，矩阵乘与后处理分离|BwdPrepare 保留 Cube 主计算，扩 K/V 和 FP32 消费者|
|chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize.h|两 head，两个 GM generation，相邻 work 前后处理重叠，末尾 SyncAll<true>() 参数归并|复用有闭环的调度原则；不是把旧固定128地址直接扩大|
|同目录 chunk_kda_bwd_finalize_cube.h|原生 FP32 Multiply/LoadOperandsFp32，SetHF32Mode(false)，独立 L0 slots|FP32 中间量可以用 Cube；不需要退回逐点 Dot/ReduceSum|
|chunk_kda_fwd_fused/op_kernel/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32_matmul.h|RunPipelineStages、warmup/steady/drain；四个32-token组成 group|借鉴跨 chunk 时间展开；四 chunk 不是四 head，不复制 arch35 SyncCV/内存路径|
|chunk_kda_bwd/op_kernel/arch22/fused32/owner.h、bf.h、cube_service.h|P10 head 串行、AIV1 只握手；B3 仍逐行/逐点；单 outstanding Cube 服务|仅作为数学/正确性对照，停止以此结构增量堆优化|

既有部分 V1 代码按不同 head 分给两个 AIV；D3 新模板遵循 V2 R01：**两个 AIV 都处理每个 head，各负责该操作连续输出维的一半**。旧专用模板保持原行为，不假称已迁移 V2。

### 1.3 性能基线分层

测量配置来自已有脚本 profile_public_c32.py：A3、CANN 9.3.0.P009、device10；B=1，BSND，seed21，scale=0.37，q/k 输入预归一化，核内 norm/beta sigmoid 关闭，beta=0.5；预热2、采样10。A 组 raw safe gate、A_log=0、bias=0、无 h0/dht；B/C 组预激活 gate=-0.1、非零 FP32 h0/dht。两种 C 使用相同输入。不能把这几组叫作用户生产模型全集。

|组|T/H/K/V/dtype|原有优化路径实测：保存 / 重算|用途|
|---|---|---|---|
|A|65/2/128/128/BF16|95.298 / 118.0275 μs，5 / 7 个设备任务|原有 C64 专用路径性能锚点|
|A-long|1024/8/128/128/BF16|需按原始 CSV 固定口径复核|长序列可达性必测，不从短例外推已达标|
|B|65/2/256/256/FP16|无原生同规格旧快路径测量|新增规格，使用阶段模型及本节预算，不能拿慢通用结果作基线|
|C|65/2/64/32/FP16|无完整同规格旧快路径测量|新增规格，同 B|

A 的记录是 P10 包中保留的旧专用路线，不是声称对干净旧 commit 独立重建测过。正式 A/B 必须冻结原快路径源码/产物 hash 并确认实际命中的 TilingKey；P10 证据见 profile-matrix-p10.json 和 p10-evidence.tgz。

从归档 CSV 逐调用、逐任务位置提取的阶段中位数见 [阶段证据](evidence/d3-original-c64-stage-evidence.json)：

|C64 原快路径任务|保存 μs|重算 μs|
|---|---:|---:|
|转置1|12.1995|11.8100|
|Recompute|—|10.5800|
|FwdH|—|12.1900|
|转置2|10.7300|12.3395|
|BwdPrepare|5.5100|4.7100|
|Dhu|19.5800|19.4295|
|Finalize|47.2990|46.8390|

逐任务中位数之和不等于逐调用总和的中位数。上述 CSV 的 Duration(us) 是历史观察值；最终按工作流使用 msprof op_summary 的 Task Duration(us) 复核，核对两种采集口径对应同一任务。

**失败数据隔离：** P10 的 C32 A=13043.58/20940.31 μs；B=24677.27/51729.59 μs；通用 C64 B=24589.88/27558.36 μs。均记录为设计失败，不参与设定优化后性能目标，也不据此声称 C32 加速。不同 dtype、状态及维度不能直接算作同规格回退倍率。

### 1.4 逐阶段预算和可达性

以下是本轮设计提出的**止损上限/开发预算**，不是已测阶段耗时，不改变用户“同规格 C32 快于 C64”的要求。预算给整个公开调用，包含所有 head/chunk 和尾块；禁止将单 tile 时间直接充当全调用时间。

|阶段预算（μs）|A C32|C C32|B C32|C C64|B C64|
|---|---:|---:|---:|---:|---:|
|BwdPrepare / S7–8|8|8|45|10|45|
|Dhu / S9–10|20|20|120|25|100|
|Finalize / S11–14|42|42|280|55|300|
|归并、填充/排空、未隐藏同步及转换|10|10|55|10|55|
|保存合计上限|80|80|500|100|500|
|重算额外：Prepare/Recompute|8|10|90|12|110|
|重算额外：FwdH / 状态 tape 与逆序恢复|12|15|110|18|140|
|重算合计上限|100|105|700|130|750（待重算重设计）|

A 的 C64 门禁为原快路径同条件无显著回退；不能把它放宽到 B 的预算。A 的 C32 80/100 μs 相对95.3/118.0留出约15%以上余量；这是目标，不是“单launch必然节省15%”。B 的保存预算约为 A 的5倍量级，考虑 K/V 翻倍、FP32路径和非零状态，绝非将24ms视为合理。C 仍受短调用固定开销约束，不能只按 FLOPs 比例缩放。

C32 必须同时满足预算和同规格 C64 加速；例如 C64 B 优化到300μs，C32 450μs即使低于500μs也失败。收益必须超过重复测量波动。之后若瓶颈仍有成本合理的改进，继续按工作流一次一个主要变量迭代，预算不是停止优化的理由。

前向也纳入本次重设计。暂定全调用预算如下（μs），均为设计上限，旧前向同规格实测尚缺，必须先补采才可冻结；不能拿反向95μs冒充前向基线。

|前向阶段|A/C C32|A/C C64新FP32链|B C32|B C64新FP32链|
|---|---:|---:|---:|---:|
|S0–S4 Prepare（含求逆）|30|35|160|190|
|S5 FwdH|20|25|110|140|
|S6 Finalize|10|15|70|85|
|未隐藏搬运、填充/排空|10|15|40|55|
|全调用预算|70|90|380|470|

旧C64前向已支持规格仍以冻结旧快路径无回退为准，不以此表放宽。前向预算不能与反向预算重复核销；重算反向只恢复必要量，不重复算前向o。本表是新设计评审输入，实际吞吐尚未测出，不能据数字承诺可达。

**计算量模型。** 对单 head、有效 M、FP32 中间量，普通矩阵乘按2mnk FLOPs。BwdPrepare 为 `4M²V+2MKV`；Dhu 为 `6MKV`；Finalize 中 dW/dkg、dA、dR 和逆矩阵伴随为 `4MKV+4M²(K+V)+4M³`；score 导数的 band GEMM/直接向量分支另计，不隐藏在上述数字中。保存反向普通 GEMM 共 `12MKV+4M²K+8M²V+4M³`。

以 B 为例，C32 的 M=32/32/1，H=2：上述约115.349512 MFLOPs；C64 的 M=64/1：约129.505288 MFLOPs（不含物理padding和score导数）。C32 不存在数量级工作量减少，性能主要依赖流水、搬运和发射；不得仅凭块更小推断胜出。物理计量使用 M/N 向16、FP32 K向8对齐，以及实际分片，不拿有效 FLOPs 当 MMAD 实际工作量。

每 Stage 建模：`lower=max(F/P_effective, B_GM/BW_effective, B_L1/BW_L1)+不可隐藏依赖等待`；每组按 DAG 关键路径计算，全卡取最慢 workgroup 加末尾归并。`P_effective/BW_effective/握手延迟` 必须取目标设备对应 dtype/tile 的测量，不用 BF16 峰值套 FP32，不把多 pipe 耗时简单相加，也不能假定全部重叠。

历史串行 FP32 Cube 协议每矩阵约1.4–2.5μs（含握手）只作为负面约束：若按P10每32×128片反复请求服务，几百次握手就能耗尽预算。D3 按stage/window通知、K归约留L0C、批量装载，必须实测减少的事件数及有效重叠。FP32所有消费者、极值保护的成本尚无完整流水实测，**A/B/C 的可达性当前均为待证，不允许直接进入完整 kernel 大改**。

长例与全范围：沿用已测T1024/H8、原型T8192/H1、H65复用，以及六组维度/两dtype/保存重算建立预算表。新增生产模型case须另记录实际shape，不能用这些探索case替代。长例的时间上限由上面逐Stage模型乘实际wave并加入scan链长生成；在测出有效吞吐与同步前不填写虚假的微秒预测。T8192/H1是最重要的否决例，CG4无法增加其并行度。

### 1.5 方案选择与停止条件

采用“共享矩阵/向量计算原语，分别设计 C32 融合调度和 C64 阶段调度”。拒绝：纯AIV矩阵计算；AIV1闲置；动态GM命令串行Cube服务；每head整链排空；为了过性能降低Aqk/H精度；把所有规格塞进一个无特化循环。

优先迁移 Finalize 全部矩阵乘和 band梯度、FwdH/Dhu矩阵状态更新、Prepare/WU/求逆，再谈小幅指令优化。若 FP32+稳定分支在原型上无法满足预算，返回本章修订结构或预算评审；不能静默改变数值范围、单launch或性能基线。

## 2. Stage 0–15 完整详设

### 2.1 符号、任务域与分核

C∈{32,64}；M是有效chunk长度；S为序列数；Q=Σceil(Ls/C)；H=HK=HV。逻辑状态为[K,V]，物理V-first使用转置寻址。所有新中间缓存FP32；公开原始输入/梯度类型按api。下文G单位可为自然域或log2域，E分别exp/exp2，b∈{1,ln2}（E的导数系数分别1和ln2）；换底系数另记c_b=1或1/ln2，避免把二者混为一个符号。

C32保存反向首版固定CG=1，以单head跨chunk流水为基线；CG2/4多head交错只保留为后续优化，不是首版依赖。C64仍参数化CG∈{1,2,4}，只沿head分组。C64 无依赖阶段 Nbase=Q，scan阶段Nbase=S；C32完整融合Nbase=S。`Nwork=Nbase*ceil(H/CG)`，`blockDim=min(AIC_available,Nwork)`，group按grid-stride取work；同序列的状态不跨block交接。AIV0/1均执行每个有效head的同一Stage，分连续输出列；不再让第二AIV只握手。跨列归约的标量由两侧partial经GM合并，规定AIV0写最终scalar，双方收到ready后继续。

例：B1/T65/H2，C32 Q=3、C64 Q=2。C32 CG1有2个work、CG2仅1个work、CG4也仅1个work；因此首选CG1，不能以“4head”名义进一步减少占核。C64独立阶段CG1有4个work，scan2个work。大H/S场景比较CG4/2/1的波次、容量和流水效率；仅当CG4不会显著损失占核且资源可容纳时列为首选候选。`wave=ceil(Nwork/blockDim)`，尾group只有有效head参与数据访问，协议参与者仍为两AIV。

对packed [0,0,1,66,66,129]、H2：S5、Q(C32)=6，C32有10个序列/head工作项；空序列只写ht/dh0，仍完成所属group协议。取消“同head所有序列必须串行”的旧限制。参数梯度写每work独立partial；所有work结束后使用现有Finalize的AIV-only `SyncAll<true>()` 路径，在同一launch内按head、序列固定顺序归并。此路径必须单独做多wave/空work/末核提前结束测试，不能用跨block自旋counter替代SDK barrier。

### 2.2 C32保存反向：以Dhu为主线的跨chunk流水

**本节是disable_recompute=true的首版正式调度，替换D3.0以head窗口为中心的安排。** 每个work为(sequence,head)，CG=1；跨核处理不同work，核组内部沿同一head的chunk倒序推进。两AIV做真实运算，不能仅其中一个工作。本文静态调度与信用模型已经审查，设备上是否获得足够重叠和速度仍未验证。多head交错放在单head流水完成并测出空泡后评估。

参考现有arch35 C32前向：`chunk_kda_fwd_32_matmul.h:1586`的RunPipelineStages和`:1647`的Process、`chunk_kda_fwd_32_vec.h:4120`的任务循环。源码将同一head当前group的准备与上一group的状态/输出交错；一个task排空后才换head。本反向采用同一思想，方向反转，但不复制其“四chunk一group”常量或A5专属搬运API。初始以一个chunk为流水步，先用测量决定是否值得进一步展开为2/4chunk。

#### 2.2.1 依赖图与三个子链

设一个序列有N个chunk，扫描序号r=0..N−1，物理chunk为c(r)=N−1−r。定义：

- P_r：BwdPrepare，S7/S8。只读前向保存的Aqk/Vnew/H及dO，产生dvLocal、dS、dqRaw；**不依赖Dhu的递推状态**，可以提前计算。
- D_r：Dhu，S9/S10。依赖P_r中的dvLocal和后一个物理chunk传来的Dnext_r，产生dU_r、Dprev_r；Dprev_r是D_(r+1)所需的Dnext。
- F_r：Finalize，S11–S14。依赖P_r的dS/dqRaw及D_r的dU/更新前Dnext，以及保存的H/Akk/G等；产生输入梯度和参数partial；**不是D_(r+1)的前置依赖**。

```text
P_0 ──→ D_0 ──→ D_1 ──→ D_2 ──→ … ──→ dh0
        │        ↑│       ↑│
        ↓        │↓       │↓
        F_0   P_1 F_1  P_2 F_2 …
```

P中的三次GEMM不是全部都必须在Dhu前完成：dvLocal优先，dS/dqRaw只需在F消费前完成。首版为固定调度简单，保证下一步P整体ready；后续可单变量实验进一步延后dS/dqRaw，不在首版加入动态任务队列。

#### 2.2.2 填充、稳态和排空

宏观流水步r安排 `D_r + P_(r+1) + F_(r−1)`。三者独立部分交错发射，绝不是把三条完整子链顺序调用一遍。F_r延迟到下一步，因此D_(r+1)不等待F_r结束。

|阶段|当前Dhu主线|提前准备|滞后Finalize|完成条件|
|---|---|---|---|---|
|填充|初始化Dnext_0=dht或0|P_0，预取P_1原始输入|无|P_0所需结果就绪，输入/结果credit初始化|
|步0|D_0|P_1（存在时）|无|Dprev_0及P_1 ready|
|稳态步r|D_r|P_(r+1)|F_(r−1)|Dprev_r、下一P ready；旧F_(r−1)结束，可回收旧record|
|最后一步N−1|D_(N−1)，写dh0|无|F_(N−2)|最终状态梯度写完，旧record退休|
|排空|无|无|F_(N−1)|全部输出/partial完成，然后S15归并|

N=1：P0→D0→F0，无法制造不存在的跨chunk重叠；N=2：步0重叠D0/P1，步1重叠D1/F0，最后F1。N=0空序列只处理状态和零参数partial，不伪发chunk事件。

以T65为例：物理chunk长度32/32/1，扫描顺序c2(M1)→c1(M32)→c0(M32)，不是从c0正序算Dhu。必须用实际M生成stage有效区，不能把首个扫描chunk误当满32。

#### 2.2.3 每步的Cube/Vector交错次序

下面是有依赖约束的静态提交顺序；GEMM内部按tile展开，相邻提交通过两套L0/operand slot重叠搬运。同一个Cube仍顺序执行MMAD，不声称不同GEMM在同一Cube同时算。

|发射段|Cube工作|两AIV/搬运可同时工作|阻塞点与释放|
|---|---|---|---|
|A|D_r：kg@Dnext|预取P_(r+1)；为F_(r−1)生成Rk/Rv等输入|当前Dnext ready；不等F_r|
|B|P_(r+1)：Aqk.T@dO；随后dO@Vnew.T、dO@H.T按固定tile表提交；无下一P时按静态尾模板跳过|消费A结果形成dU_r，准备D_r的qg/W和dO/dU输入|AIV先消费D的增量并发布dU，再消费P结果；首版Cube完成本段固定P提交后进入C，不使用运行时ready优先队列|
|C|D_r：termQ=qg.T@dO、termW=W.T@dU|P_(r+1)的scale/mask；F_(r−1)已ready的Vector处理|每矩阵完整归约；结果slot信用独立|
|D|F_(r−1)：先dW、dkg，再dA；其余按S12/S13固定依赖顺序|消费termQ/termW生成Dprev_r；完成下一P、准备后续F输入|Dprev ready释放下一步递推依赖；F消费者按自身DAG等待|
|E|继续F_(r−1)中依赖解除的矩阵乘，穿插下一步允许预取的operand|F_(r−1)band导数、gate/norm、cast/写回|退休F_(r−1)，三record回绕安全；进入下一步|

具体允许重叠的区间：①Cube执行P_(r+1)时，AIV消费D_r的kg@Dnext结果、加dvLocal并发布dU；②Cube执行D_r的termQ/termW时，AIV处理P_(r+1)的scale/mask及已完成输入预取；③Cube执行F_(r−1)的dW/dkg时，AIV消费D_r的termQ/termW并发布下一状态；④Cube执行D_r的kg@Dnext时，AIV准备F_(r−1)的Rk/Rv或搬运下一P。以上数据依赖互不成环；跨pipe执行由异步DMA/MMAD和延迟到消费点的Wait支持。只证明存在合法重叠窗口，不证明两段实际时间足够长、编译器一定按预期发射或收益已达预算。

首版在段E完成旧F退休后才跨入下一宏步，因此不承诺F最后的纯Vector尾段被下一Dhu完全隐藏。这一未隐藏尾部计入预算；不能在文档中画成已经跨步重叠。若测量它成为瓶颈，再单独设计跨步提前发射，不让实现自行越过本版退休边界。

保存线路不执行前向S0–S6：qhat/G/betaEff等读保存上下文，Rk/Rv在Finalize局部按需重建。norm的反向归约仍需两AIV交换partial。

A–E是调度顺序约束，不是每段末尾PIPE_ALL。等待只放在实际消费者和slot复用点；AIV运行匹配的确定顺序，禁止先阻塞等尚未发射的F结果，使D_r的dU迟迟无法生产。F的Vector长段与Cube下一项可重叠，但同一AIV的两段Vector算术不能声称并行。

所有GEMM都需Cube，因而P/F会竞争Dhu的Cube时间。设计收益是隐藏Vector/MTE/Fixpipe等待与去除多launch/重复搬运，不是“Dhu把全部P/F耗时免费掩盖”。若Finalize占满Cube，性能模型必须计入其工作量。CG1流水仍失败时先定位瓶颈，再评估多head交错。

逐步预算以 `max(Cube[P_(r+1)+D_r+F_(r−1)], Vector同批总工作, MTE/FIX同批总工作)+未隐藏依赖等待` 为乐观下界，再按真实DAG排程；这个表达式只是资源下界，不能据此保证整段执行达到该值；不得把三个子链的最大值当耗时，因为它们共享Cube。调用总时长包括P0填充、所有稳态步、最后F排空及S15。design 1.4的保存预算保持为A/C 80μs、B 500μs上限；新增record容量本身不等于搬运流量，但切换Vector阶段产生的暂存/回读必须实测计入。重算预算只保留为后续目标，不用于本轮保存线路通过判断。

#### 2.2.4 三个chunk record与两个传输bank

稳态同时存活F_(r−1)、D_r和P_(r+1)，因此**两个完整chunk record不够，首版改为三个record**。`slot(r)=r%3`，初次使用显式初始化，回绕只在该slot的F及所有DMA消费者退休后允许。记录标识为(sequence,head,r)，不等同物理chunk号。

|槽生命周期|必须保留的数据|最后消费者|
|---|---|---|
|PREPARED|dvLocal/dS/dqRaw、保存cache引用|D读取dvLocal；F读取dS/dqRaw|
|SCAN_DONE|更新前Dnext、dU、Hentry、上述P结果|F完成S14；Dnext绝不能改写成Dprev|
|RETIRED|该chunk梯度/partial已提交，所有相关搬运结束|此后才允许r+3覆盖|

D_r的Dprev直接写入已分配slot(r+1)的Dnext区，使用本次FP32后处理结果写一次，不先写本record再复制；最终r=N−1直接写dh0（未请求则不公开写）。本record的Dnext保持只读供F_r。下一record可以同时有P写入，但P区域与Dnext区域不重叠。没有跨record长期别名，也无需因为Dnext引用而阻止旧record退休。

三record只是chunk工作集；input/result传输仍各两个bank，与r%3无对应关系。bank由单调递增的传输批次号%2选择，descriptor含record/阶段/tile，三方按相同静态调度推导；不使用GM动态command。result_free只归还传输槽，record_free必须等F完整结束，两者禁止混用。两AIV各完成本record的最后MTE3并发布RETIRE_READY；AIC集合等待后广播RETIRE_ACK，两AIV都收到ACK才允许下一次复用。ACK不能只由AIV0代替另一侧发出；具体旗语见2.11。

#### 2.2.5 本轮边界

C64保留无依赖阶段按chunk并行、scan阶段按sequence并行的分阶段路线。前向C32只作结构参考，本轮不重写其实现。disable_recompute=false仍要求一次计算kernel，未来在核内接入正序恢复与本节逆序流水；**其跨正逆扫描调度、缓存预算和性能可达性未在本轮完成，不以保存线路完成代替重算线路完成**。先完成保存线路设计/评审，再推进重算，不在实现时临时增加Recompute或FwdH launch。

### 2.3 资源模板：绝对地址、所有权与生命周期

容量依据：arch22现有Finalize用L1 512KiB、L0A/B各64KiB、L0C128KiB、UB192KiB；Host再查目标平台。D3每AIV只使用184KiB，保留末8KiB。以下单位均KiB，区间半开；不是把A5的248KiB UB搬来。空间先固定；下表通用lane规划保留给C64候选，C32保存线路使用紧随其后的专用record映射；超界不能静默选择慢串行模板。

|层级/地址|对象、dtype/物理shape|份数及生命周期|
|---|---|---|
|L1 [0,256)|lane[r]=[64r,64(r+1))；每head64KiB FP32 NZ resident|最多CG4；[0,16) Akk，[16,32) dW/Rk面板，[32,48) dA/middle，[48,64) dRk/Rv面板（均相对lane）；实际面板≤C×64×4；各生产/最后消费见Stage表|
|L1 [256,384)|operand bank0/1，各64KiB|每bank A32KiB+B32KiB；M≤64,N≤64,Kr≤128时均可容纳；下一bank预取与本bank计算并行|
|L1 [384,448)|Cube→Cube reload staging|FP32结果经独立GM relay后回读；不能假设arch22可L0C直写L1|
|L1 [448,512)|空闲|64KiB连续余量；未分配给隐藏buffer|
|L0A/B各[0,32),[32,64)|operand ping/pong FP32 NZ|两个slot；M最后读取后方可LoadData覆盖|
|L0C [0,64),[64,128)|累加结果ping/pong FP32|单tile≤64×64=16KiB，保留64KiB slot；完整Kr归约结束才Fixpipe；Fixpipe完成后归还|
|每AIV UB [0,32),[32,64)|输入bank0/1 FP32或原dtype|各32KiB，批量2D输入，不按标量搬运|
|UB [64,96),[96,128)|结果bank0/1 FP32|各32KiB；每bank按r分8KiB lane，CG4全部结果可独立落地；单次公布切片≤64×64/2 FP32|
|UB [128,160)|当前Stage Vector临时|32KiB，直接差值成对tile/归约/转置；不能同时容纳另一stage临时|
|UB [160,176)|head局部小量与norm partial|16KiB；每head4KiB，beta/rstd/末行G/必要partial，按需换义|
|UB [176,184)|事件元数据及参数partial|8KiB；不用作矩阵输出；末尾归并复用|
|UB [184,192)|保留|不参与计算预算|

**C32保存CG1的专用映射：** L1[0,64)、[64,128)、[128,192)分别属于record0/1/2，内部四个16KiB面板划分同表；[192,256)空闲。这样F_(r−1)的Akk/dW不会被P_(r+1)预取覆盖。L1[256,384)仍为两个operand bank，[384,448)为reload staging，[448,512)空闲。UB小量区[160,164)、[164,168)、[168,172)分别保存三record小量，[172,176)空闲。UB结果两bank按传输批次轮换，当前活跃结果必须在UB复用前消费或批量写入所属record；临时[128,160)只给正在执行的Vector子阶段，阶段切换前将跨段存活数据写回其record。未完成的F局部梯度不能只留在这个临时区后切去P覆盖。L0和UB输入/结果区沿表不变。C32该key不启用head lane索引。

连续空闲L1为两个64KiB区共128KiB、最大连续64KiB；UB保留8KiB；L0两槽全保留。C64K256完整状态256KiB，不强行驻留每head一份；状态按64×64 tile进入operand banks，剩余状态在GM。CG4“4份resident”是四份面板和小量，不是四份完整K256状态。CG1/K≤128可评估利用空闲lane驻留状态，但需独立key地址表与实测收益，不能本版先宣称已做。

GM record（FP32，内部按align256分段）：`A=4C²`，`X=4C*Kp`，`Y=4C*Vp`，`Z=4Kp*Vp`，Kp/Vp向16对齐。每window record顺序为：

|offset|字段|容量|生产/最后消费者|
|---|---|---:|---|
|0|score[6]：Aqk/Akk/dS/dL/dA/middle|6A|S2/S3/S7/S12 → S14|
|6A|key[10]：qhat,khat,G,qg,kg,W,dW,dRk,dqhat,dkhat|10X|S0/S4/S11 → S14；不能覆盖保留中的dW|
|6A+10X|dG、Rk、dkg|3X|S4/S11/S13 → S14|
|6A+13X|value[5]：v,Vnew,dO,dU,dRv|5Y|输入/S5/S9/S12 → S14|
|6A+13X+5Y|Hentry、Dnext、termQ（保存线路）|3Z|H/Dnext到S13；termQ到S10b，不存第二份Dprev|
|其后|betaEff、dBeta、rstdQ/K、metadata|align256(16C+256)|S0/S13 → S14|

`R=align256(6A+13X+5Y+3Z+align256(16C+256))`。前向时value3暂存U，产生Vnew并完成其消费者后才在反向换义为dvLocal/dU；value4先存Rv，到S11 dA完整消费后才由S12覆盖为dRv。key8先存dqRaw，S13转换为dqhat的初始贡献，不先清零丢失dqRaw；key9/dG/dBeta显式初始化。score2由rawS原位scale/mask成dS；dA/middle分别持久化在score4/5，公共result relay仅作传输。Cube结果经AIV搬入UB后批量写回对应record，后继Cube经input_ready再读；最后消费者完成前不得复用score4/5。额外搬运为每个矩阵4C²读和4C²写，计入S11/S12预算。公共保存cache可只读直接引用，首次实现按完整record上界分配，不用假别名低报峰值。C32保存CG1每group三个chunk record：`3*R`；C64旧候选上界已由§2.12.6的独立阶段活跃集取代，不再使用`2*CG*R`。C32重算的record/tape最终组合待专门详设，不能直接继承本式宣称完成。另有operand relay两bank×CG×64KiB、result relay两bank×CG×32KiB，合计`192KiB*CG`；两者与record分离。norm scalar交换每group再留align256(32C*CG)字节（每行8个FP32，含padding）。C32重算hTape为`4QHKpVp`，所有组写独占索引；参数partial为`align256(4*(Kp+1))*S*H`。系统workspace独立按SDK查询，计入总量。

保存线路每组workspace上界为3R+192KiB+align256(32C)，再加全局参数partial和SDK系统区；与T成线性关系的保存cache属于前向已有存活量，不计成新分配但计峰值显存。该上界有意不依赖未证明的槽别名。以C32/KV256/CG1估算record≈1.34MiB，C32保存三代约4.01MiB，加relay约0.19MiB/group；保存模式没有hTape，重算另计；不是此前几十KiB的虚假总峰值。Host用64位checked arithmetic计算；长序列hTape显存单独列入验收。优化record别名只在最后消费者和profile证据闭合后做。

所有Stage沿用以上绝对地址：输入只在UB[0,64)，结果只在[64,128)，临时只在[128,160)，小量只在[160,184)；Cube的lane和operand不重叠。S10b的state_ready是数据就绪条件，不额外占用一个未列出的flag：下一Dhu输入的input_ready在两侧对应状态写回MTE3完成后才允许发布，AIC等待两侧集合后读取完整状态。每个Stage结束只释放其最后消费对象，不释放整个arena。AIV先写GM relay，AIC回读；AIC Fixpipe到GM relay，AIV再批量搬入自己的UB结果半片。R04的结果最终驻留UB，但A2/A3路径需GM中转，不谎称直接Fixpipe UB。

### 2.4 Stage结果总表

|Stage|单元|功能|主要空间/最后消费|
|---|---|---|---|
|S0|Vector|norm、beta、GateEval、chunk cumsum|UB输入/临时，小量；record q/k/G 到S14|
|S1|Vector|安全band准备或直接差值score|UB临时；operand relay 到S2|
|S2|Cube|安全band score GEMM|L1 operand、L0；score relay 到S3|
|S3|Vector|mask、beta、16阶叶子逆|UB临时；Akk叶子到S4a|
|S4a|Cube|三角逆跨叶子块组合|L1 Akk/中间；完整Akk到S4c/S12|
|S4b|Vector|qg/kg/Rk/Rv、scale|key/value record；operand relay到S4c|
|S4c|Cube|W=Akk@Rk，U=Akk@Rv|L1 Akk，W/U relay到S5|
|S5a|Cube|WH=W@Hentry|state tiles与W；relay到S5b|
|S5b|Vector|Vnew=U−WH、decay准备|value record；Vnew到S5c/S6/S7/S11|
|S5c|Cube|state delta=kg.T@Vnew|L0归约；state relay到S5d|
|S5d|Vector|Hnext=decay*Hentry+delta|state record/tape；到下一chunk|
|S6|Cube|o=scale*qg@Hentry+Aqk@Vnew|L0两项完整累加，最终cast写o|
|S7|Cube|BwdPrepare三乘法|record dO/Vnew/H/Aqk；raw结果到S8|
|S8|Vector|scale、因果mask、dqRaw保留|dS/key scratch；到S11–13|
|S9a|Cube|dU增量=kg@Dnext|Dnext只读；relay到S9b|
|S9b|Vector|dU=dvLocal+增量，写dh快照|dU、Dnext；到S10/S11|
|S10a|Cube|termQ=qg.T@dO、termW=W.T@dU|L0完整归约；relay到S10b|
|S10b|Vector|Dprev=decay*Dnext+scale*termQ−termW|Dprev释放逆序scan依赖，Dnext保留到S13|
|S11|Cube|dW、dkg、dA|L1 dW驻留/GM完整面；到S12/S13|
|S12a|Cube|middle、dL、dRk、dRv|Akk驻留；独立结果relay|
|S12b|Vector|dL严格下三角、初始化梯度面|dL到S13；旧scratch不得被+=读取|
|S13a|Vector|band准备/直接差值导数、状态/gate贡献|UB临时/key梯度；到S13b或S14|
|S13b|Cube|安全band矩阵梯度|lane/relay；到S13c|
|S13c|Vector|合并Q/K/G/Beta及末行贡献|record梯度；到S14|
|S14|Vector|逆cumsum、gate/norm/beta导数、输出cast|dq/dk/dv/dg/db、work参数partial|
|S15|Vector|同launch固定顺序参数归并|partial；最终dA_log/dbias|

a/b/c子编号只是同一逻辑阶段的单元边界；不得把Cube和Vector混写成一个类型Stage。重算跳过S1–S4a，直接使用保存Akk；若需Aqk，读取前向保存值，不重新改变舍入。每个Vector Stage对应一段批量向量程序；arch22使用其支持的Vector API，不照抄arch35 VF调用语法。

### 2.5 S0–S4：Prepare / Recompute

S0：qhat=q或q*rstd（FP32），rstd=rsqrt(sum_K(q²)+eps)。norm关不读rstd。两AIV沿K半片做partial，AIV0合并并广播scalar，双方处理自己的片。beta三分支：b、sigmoid(b)、2sigmoid(b)。raw safe：delta=ell*sigmoid(a*x)；raw softplus：delta=−a*softplus(x)；预激活delta=g。a=exp(A_log)或1，x=g+bias或g。G在每chunk按token顺序累计delta*c_b；两个AIV各负责自己的K列，不跨chunk累计。

S1/S2：`R_qk[i,j]=sum_d qhat[i,d]*khat[j,d]*E(G[i,d]−G[j,d])`，`R_kk`以khat替换qhat，先限定j≤i/j<i再计算指数。沿16行band，安全分支用FP32 Qplus/Kplus/Kminus，stack两矩阵提交Cube；FP32累加，输出FP32，不cast half。

**稳定性保护是算法的一部分。** midpoint分解必须检查当前band有效前缀的指数及输入乘积边界：用自然域检查每个factor指数绝对值≤40，且非零输入满足log(abs(input))+exponent≤80。这保证单factor不会因分解溢出，但不证明所有归约精度；下溢/动态范围还须和直接差值control验证。不满足时同一kernel转直接差值向量tile（query4×key4×K64），两AIV分K，各自批量Exp/Mul/Reduce，再合并K partial；无逐(i,j,d)标量GM读取。mask在Exp前，尾部无效值设安全0。

动态分支只在band边界，两个AIV先统一predicate写入协议metadata；AIC据同一predicate跳过GEMM但仍完成本band协议。保留固定band迭代和通知次数，不能只让一个参与者提前continue。安全路径也不得把Exp溢出后的Inf用mask“修复”。极值合法输入可能走保护路径，其性能单独测，不用它掩盖常规规格回退。

S3：Aqk=scale*R_qk（FP32），L=I+strict_lower(betaEff[:,None]*R_kk)。物理尾块为diag(Lvalid,I)，交叉块清0。叶子16×16逆用向量行消元，依赖步16次，不能每元素Scalar Dot往返。S4a按块三角逆公式 `X10=−X11@L10@X00`，C32两叶、C64四叶递归；FP32 Cube中间经GM relay回L1，相关2次中转计预算。Akk完整结果FP32供公开缓存和后续消费者。

S4b：qg=qhat*E(G)，qg_scaled=scale*qg，kg=khat*E(Glast−G)，Rk=betaEff*(khat*E(G))，Rv=betaEff*v；均FP32明确两次乘节点。S4c：W[C,K]=Akk[C,C]@Rk[C,K]，U[C,V]=Akk@Rv；按输出64列tile，K归约在L0C完成，Fixpipe FP32。Akk驻留lane[0,16)，Rk/dW面板复用lane[16,32)需等W消费结束。生成W/U的同一时刻，Vector允许准备下一head或下一chunk的S0/S1。

重算S0不得重复norm/sigmoid：从前向保存qhat/khat/rstd、betaEff读取；norm关读取原q/k；G已保存则不再cumsum/换底，raw重建则执行相同FP32节点。原始q/k/g/beta/v及Akk按api留存，不从qg/kg反除恢复原输入。

### 2.6 S5–S6：状态和前向输出

逻辑状态Hentry[K,V]；S5a `WH[C,V]=W[C,K]@Hentry[K,V]`，FP32累加；S5b Vnew=U−WH；S5c `deltaH[K,V]=kg.T[K,C]@Vnew[C,V]`；S5d Hnext=E(Glast)[:,None]*Hentry+deltaH。物理V-first用对应转置GEMM和地址，禁止外部Transpose。

Hentry、运行Hnext、hTape、返回h均为FP32同一数学值；WH读取保存的入口值，不读取低精度快照。ht最终FP32，h0只读。S6输出 `o=scale*(qg@Hentry)+Aqk@Vnew`，两个完整归约后相加再cast主dtype；Aqk已经含scale，第二项不能再乘scale。L0C结果与下一输入bank独立；S6读取Hentry完成之前不能复用该代state。

重算正序只生成所需hTape和局部量；逆序从tape读入口，仅计算Vnew，不重复更新无消费者的Hnext。空序列ht=h0/0，dh0=dht/0；不访问Glast。C32保存前向写公开cache不代表可以释放仍被S6使用的record。

### 2.7 S7–S10：BwdPrepare与反向状态主链

S7三个FP32矩阵乘：`rawS=dO[C,V]@Vnew.T[V,C]`，`dvLocal=Aqk.T[C,C]@dO[C,V]`，`dqRaw=dO[C,V]@Hentry.T[V,K]`。S8令dS=scale*tril(rawS)，它是**未乘scale的R_qk的梯度**；dvLocal直接使用含scale的Aqk。V-first第三式为dO@Hphysical，不再转置。两AIV沿输出连续列分片，dS mask包含对角。

S9/S10：Dnext已经由上一扫描步写入本record，直接作为更新前dh只读快照，不额外复制；首步由dht/0初始化。再算 `dU=dvLocal+kg@Dnext`；`termQ=qg.T@dO`，`termW=W.T@dU`；完整C归约后S10b执行 `Dprev=E(Glast)*Dnext+scale*termQ−termW`。qg未预乘scale；公开保存/重算两态用相同顺序。禁止先覆盖Dnext再供Finalize使用。termQ完整结果先写本record的第三state区；termW按tile经结果bank进入UB，与同位置termQ及旧Dnext融合，直接写下一record.Dnext。全部tile写完成才公布state_ready；不在公共relay保留跨tile的termQ，不额外分配完整termW状态面。

Dprev写到下一扫描chunk独立record的Dnext区并ready后，该chunk可开始S9；当前chunk S11–S14继续读取当前record的Dnext/Hentry。保存线路按2.2三record轮换；不传跨record状态别名，不用input_free或Dprev ready释放整个record。

### 2.8 S11–S14：Finalize完整消费者与活跃表

S11：`dW=−dU@Hentry.T`，`dkg=Vnew@Dnext.T`，`dA=dW@Rk.T+dU@Rv.T`。dW[C,K]必须完整产生后才能完成dA的K归约；其最后消费者还包括S12的dRk。

S12a：`middle=Akk.T@dA`；`dL=−middle@Akk.T`；`dRk=Akk.T@dW`；`dRv=Akk.T@dU`。均FP32，矩阵形状分别[C,C]、[C,C]、[C,K]、[C,V]。S12b **先把dL对角/上三角清零，再交S13**；不是等最后输出才mask。K256分列处理dRk，但dA不能使用未完成dW。

|子阶段|活跃GM对象|L1驻留/UB使用|首次写与释放|
|---|---|---|---|
|S11 dW/dkg|独立key6(dW)、dkg面|lane[16,32)只驻当前dW64列；UB结果bank|每tile覆盖写；整面dW留到S12 dRk结束|
|S11 dA|dA=record.score4、dW/Rk/dU/Rv|lane[32,48) dA面，L0C归约|第一项初始化L0C，第二项累加；不清掉dW|
|S12 middle/dL|record.score4(dA)/score5(middle)，dL写score3|lane[32,48)按完整消费换义|middle最后乘Akk后可释放；dL mask后写record|
|S12 dRk/dRv|key7、value4；dW仍只读|lane[48,64)当前结果面板|结束后dW槽才可复用|
|S12b 初始化|key8/9/dG/dBeta|UB结果/临时|本chunk首次赋值0或唯一初始贡献，之后才+=|
|S13/S14|dqhat/dkhat/dG/dBeta、dkg/dRk/dRv、dS/dL|UB按K64 tile加载、计算、最后写回|末行G合并后才逆扫；输出完成record退休|

这里明确保留13个key面，不再宣称四面能装所有同时活跃量。后续可以缩减，但须逐对象最后消费者证明。

S13直接贡献：qg项给 `dqhat += scale*dqRaw*E(G)`、`dG += b*scale*dqRaw*qg`；Rk项给 `dkhat += dRk*betaEff*E(G)`、`dG += b*dRk*Rk`；kg项给 `dkhat += dkg*E(Glast−G)`，`dG_i -= b*dkg_i*kg_i`，并向Glast加所有行的相反贡献；状态decay再向Glast加 `b*E(Glast)*sum_v(Hentry*Dnext)`。i=last正负抵消自然发生，不覆盖先前贡献。

score梯度在有效(i,j,d)定义 `e=E(G_i−G_j)`：

- qk：dq_i += dS_ij*k_j*e，dk_j += dS_ij*q_i*e，zq=dS_ij*q_i*k_j*e；dG_i+=b*zq，dG_j−=b*zq。
- kk：z=dL_ij*betaEff_i，仅j<i；dk_i += z*k_j*e，dk_j += z*k_i*e；zk=z*k_i*k_j*e；dG_i+=b*zk，dG_j−=b*zk。
- dBetaEff_i += dot(dRk_i,k_i*E(G_i))+dot(dRv_i,v_i)+sum_{j<i}dL_ij*Rkk_ij。
- dv=betaEff*dRv，再cast输入v dtype。beta=0时dv=0但dBeta可非零；不得用L/beta恢复Rkk。

安全band下S13b用上述分解的矩阵形式同时生成query侧、key侧梯度（dS@Kminus、dS.T@Qplus，以及dL*beta对应两项），FP32 Cube；S13c再乘对应指数与输入累加dG。key侧跨band按固定顺序累加到独占record，不跨group写同一元素。危险band用S1相同直接差值tile，保持相同有效域与FP32消费者；不是逐点GM scatter。Rkk局部重建后消费，不新增公开缓存。

S14：先完成所有Glast贡献，然后在每个chunk内逆向cumsum，乘c_b回到自然log-step梯度。raw safe设z=a*x、t=exp(−abs(z))，sigmoid导数用t/(1+t)^2；dg=dDelta*ell*a*deriv，dA_log=sum(dDelta*ell*deriv*a*x)，dbias按K列sum(dg)。raw softplus用理想sigmoid(x)导数；dA_log=sum(dDelta*delta)，dg=−dDelta*a*sigmoid(x)。softplus数值实现与control固定，不能用分段多项式自身导数冒充理想导数；分界点、±20/±80单独测。

beta反向三支：关闭乘1，普通乘betaEff*(1−betaEff)，双倍乘betaEff*(1−betaEff/2)。norm开用rstd*(dqhat−qhat*sum(qhat*dqhat))，k同理；norm关直接cast，禁止读空rstd。独立Finalize若输入FP32 betaEff则db跟随该独立输入；组合入口在最后按原始beta dtype写回，不能混同。

### 2.9 S15：归并、尾部与最终释放

每work的dA_log/dbias partial先显式初始化再累加，空序列写0。grid-stride工作结束后，所有AIV完成MTE3并排空局部C/V事件，执行已有Finalize模式的SyncAll<true>()；参数输出head分给AIV，按序列id固定顺序读取partial。无原子加，无额外launch。barrier workspace/flag14保留给SDK，flag15按现有MIX初始化保留。

C32这一归并改变旧whole-head owner策略，因此P10的多head测试不能证明新协议；必须重测Nwork>核数、多wave、两种CG尾group、空序列和每个head不同参数。若SDK在该MIX退出顺序下不能正确使用，不能替换为未证明的忙等，也不能暗加Reduce kernel。

### 2.10 搬运汇总、缓存与预算输入

下表按一个有效chunk/head计算逻辑字节，FP32每元素4B；实际DMA按tile/padding另记。读入Cube和Vector各一次属于两条不同消费路径，不伪称一次全芯片读取。超L1的K256面板会重读，明确计入。

|对象|主路径GM读写|驻留/重复成本|
|---|---|---|
|q/k/g/v/beta原输入|S0/S4一次批量读，原dtype字节|反向raw导数需原g，单独计读；保存qhat避免再norm|
|G/qhat/khat|生产各写4MK，Vector梯度按Ktile读|生命周期跨前反向必须保留；重算读/重建策略按api|
|Aqk/Akk|各写4MC，BwdPrepare/Akk消费者各按实际Stage读|Akk在同window L1驻留；不重复求逆|
|Hentry/Dnext|各4KV；W@H、dW、dkg等按输出tile读取|K256无法全resident；上界按每个GEMM面板遍历计，不把4KV当所有Stage总读|
|W/U/Vnew|分别4MK/4MV/4MV生产；各消费者矩阵路径读|L1/UB面板复用；重算正扫与逆扫的重复明确计入|
|普通GEMM relay|每输出矩阵4mn写+4mn读；Cube→Cube同样需中转|分离producer/consumer；K归约不产生每Kr片relay|
|dW/dRk/dkg及三梯度面|每面首次覆盖写；后续每Stage批量读改写|禁止每(i,j)一次GM+=；跨band累加留UB当前Ktile|
|hTape（重算）|写4QHKV，逆序读同量|独立于固定window workspace，计峰值|
|参数partial|每work写4(K+1)，最终归并读同量|空work写0；与原输入/最终输出不别名|

对 tiled GEMM m×k @ k×n，保守GM读取上界为 `4mk*ceil(n/64)+4kn*ceil(m/64)`，输出4mn；若面板L1驻留省读，用实际生命周期从上界扣除。低精度原输入额外cast搬运另计。Stage预算模型逐GEMM相加得到字节数，不允许漏掉GM↔L1↔L0和relay。DataCopyPad使用二维块与有效字节数，K60/100及V32尾片绝不按向上对齐长度写公共GM；padding只发生在私有内部区。

GM→L1采用已验证ND/NZ路径，FP32分形16×8；LoadData转置按操作数形状选择。V-first直接改变GEMM transpose/输出stride。流式原输入可禁L2；同kernel反复写读的workspace按现有协议禁L2并做真实pipe同步，不能用PIPE_ALL替代一致性。缓存策略改变需单变量测量。

### 2.11 保存线路的同步闭环与提交顺序

首版固定CG1。使用已有arch22 `CrossCoreSetFlag<0x2, PIPE_*>` / `CrossCoreWaitFlag<0x2, PIPE_*>` 的两AIV集合语义；源码依据为fused32/cube_service.h已有mode2交接及Finalize末尾SyncAll<true>()。本版选定具体通道，不再保留“另找AIV间同步API”的未决项。ID属于同核组；14/15留给SDK。实际编译分配及DMA一致性仍是设备验收，不由静态模型替代。

|通道|ID|生产者 → 消费者|首次状态与复用|
|---|---|---|---|
|input_ready[p]|0/1|两AIV在PIPE_MTE3 set → AIC在PIPE_MTE2集合wait|初始空；一bank最多一未消费通知|
|input_free[p]|2/3|AIC在PIPE_MTE2 set → 两AIV在PIPE_MTE3 wait|首次本地可用，不预置flag；读取完成后仅当本bank还有下一代才set|
|result_ready[p]|4/5|AIC在PIPE_FIX set → 两AIV在PIPE_MTE2 wait|Fixpipe完成后发布；两侧即使有效半片为空也wait|
|result_free[p]|6/7|两AIV在PIPE_MTE2 set → AIC在PIPE_FIX集合wait|两侧最后一次GM→UB读取后归还；仅有后继复用时set|
|RETIRE_READY|8|两AIV在PIPE_MTE3 set → AIC在PIPE_MTE2集合wait|当前F最后读写及核内事件排空后；每chunk恰一次|
|RETIRE_ACK|9|AIC在PIPE_MTE2 set → 两AIV在PIPE_MTE2 wait|集合收到8才广播9；双方收到后record才free；不积累多chunk信用|
|保留|10–13|不使用|不分配给隐含动态队列|
|全局归并|14|SDK SyncAll<true>()|所有局部交接排空后调用|
|MIX初始化|15|SDK|不使用|

参数partial在F内按扫描顺序加到该work的独占区；RETIRE只保证本record和该work已提交的贡献，不表示整个head跨序列归并结束。最后S15仍先用SDK barrier，再按固定序列顺序归并。

**norm跨两AIV的归约选定现有Cube路径中继，取消未定的AIV互发协议。** 每行两侧分别写q/k点积partial至交换区的独占列，组装逻辑FP32 `[C,4]`（q0/q1/k0/k1），GM物理按[C,8]补零；AIC用固定常量矩阵 `[4,2]` 做两项求和，物理K补8、N补16，pad为0，FP32累加且HF32关闭。结果两AIV各读完整有效 `[C,2]` 小矩阵，之后各写自己K半片的dq/dk。使用普通input/result通道，不增加flag；常量在核内生成，不增加launch。交换区align256(32C)已计入，输出载入两AIV各256B，计入S14/同步预算。norm关闭静态删除该事务，不能等待不存在的结果。

band安全判定同样不得让两AIV独立跳过不同命令：两侧生成0/1危险partial，经固定求和事务广播统一结果，再共同选择fast或direct。AIC若需要分支，只读取两AIV发布完成后的控制metadata；沿已有 `DataCacheCleanAndInvalid<..., SINGLE_CACHE_LINE, CACHELINE_OUT>` 后GetValue方式读这个标量，不读Tensor元素做热循环控制。该控制区独立于数据bank，直到分支两侧完成同一band的结果/退休协议才复用。fast/direct每个band均完成一次同形状控制交接；direct由AIV先写梯度record，AIC跳过MMAD但仍按表完成该事务的确认，禁止一侧删去wait而另一侧仍set。后续带数据的事务数由双方相同predicate和模板显式推导，输入packet/output tile计数分别推进，不沿用fast分支的次数。它不是动态GEMM命令服务，GEMM顺序/shape仍由静态模板决定；元数据可见性需在设备原型覆盖。保护分支的统一选择和额外归约成本必须计入预算。

**输入包与输出tile使用不同计数器。** 输入packet编号i，bank=i%2；每个完整输出tile编号j，result bank=j%2。Kr分片时可有多个输入packet对应一个输出；L0C一直累加，最后一个Kr才发布一次result_ready。最后复用判断分别按输入总数和输出总数，不能用一个“命令数”同时推导两套flag次数。

消费者优先规则：两AIV保持同一输出消费次序。为任何新输入阻塞等待credit之前，先消费已经排在该输入之前的待回收结果；最多允许两个未退休输出tile。若下一输入依赖某个结果，先完成该结果的计算/持久化再发布输入。不允许“先把所有P输入填完，最后才收D结果”。

```text
填充：初始化record0和需要的record1；初始input/result bank均本地free。
      先发布P0；按输出序消费P0，完成D0输入依赖。
每个静态宏步：
  AIV按2.2 A–E配对执行；任何输入提交前：
    drain_due_results()：按序wait result_ready，MTE2搬入私有UB；
    若UB还持有旧活跃结果，先完成后处理或写回所属record；
    MTE2读完后set result_free（仅当有后继复用）；
    需要跨Stage保留的值写入record，MTE3完成后才publish依赖input_ready。
  AIV等待目标input bank的free（非首次），写两侧独占区，再set input_ready。
  AIC wait input_ready；MTE2→L1完成后归还input_free（有下一代时）。
  L1 bank重用前等上一MTE1读取完；L0A/B重用前等M读取完。
  第一个Kr初始化L0C，后续Kr累加；完整归约后取得result_free（非首次）。
  Fixpipe到独立result bank，set result_ready；L0C在FIX完成后才能复用。
  F_(r−1)最后消费、partial写回和DMA结束：
    两AIV set RETIRE_READY；AIC在该静态退休点wait集合后set RETIRE_ACK；
    两AIV wait ACK；slot(r−1)可供下一宏步的P_(r+2)使用。
排空：只执行有效F；最后record完成同样的RETIRE握手。
      无后继bank不发free；所有ready/ACK均有对应wait，最后再SyncAll。
```

两AIV的硬事件在私有pipe内完成：input区V写→MTE3读使用V_MTE3；MTE3结束才能重用UB；结果MTE2写→V读用MTE2_V，V读完才能下一MTE2覆盖（V_MTE2）。AIC使用MTE2_MTE1、MTE1_M、M_FIX与MTE1_MTE2、M_MTE1、FIX_M的双slot配对。每种HardEvent方向分配两ID并成对Release；逐行LoadData共享L0片段时保留已有PIPE_MTE1顺序，不以不同slot名假定实际地址不同。新增矩阵中继也复用此表，不额外申请未记账事件。

等待图审查：input_free只等AIC完成输入搬入，不等Vector后处理；result_free只等两AIV读完结果，不等新input_free；record退休不持有仍被后继输入需要的传输credit。RETIRE只出现在旧F全部必要Cube结果已发布之后，AIC不可提前wait8而尚未提交F必需GEMM。因此宏步和credit均有前向进展路径；若实现改变顺序，必须重新审查。

[静态检查记录](evidence/d3.2-saved-pipeline-static-check.json)：三record扫描长度N=0..129全部通过生命周期模型；两AIV独立推进+AIC的单信用抽象模型，输出数1/2/3/4/5/8/12、窗口1/2共14组，遍历4080状态，无死锁及残留信用；反例“8个输入全部先提交再回收”出现死锁。本模型不覆盖真实指令乱序、GM缓存、不同Kr输入包数量、band分支或设备时延，后者仍必须按验证计划检查，不能称为完整硬件协议证明。

### 2.12 C64 保存模式：三段流水重设计（D3.3）

本节是 C64 保存模板的调度、资源与接线唯一设计；§2.2 的三 record 属于 C32，不搬入 C64。数学、dtype、gate/norm/beta 契约沿用 §2.7–2.9 和 api.md。既有已支持 BF16/128 快路径保持原路由与精度，新 FP32 cache 链使用独立内部 key。不得仅放宽 Host 校验后落到逐行 Dot 的 general kernel。

#### 2.12.1 证据、目标和实施优先级

从本次原始 CSV 按调用内任务位置拆分，证据见 [d33-c64-save-stage-evidence.json](evidence/d33-c64-save-stage-evidence.json)。每项为十次中位数，各项中位数之和不等于总调用中位数。

|任务|BF16 K128 旧快路径 μs|FP16 K256 慢通用路径 μs|处理|
|---|---:|---:|---|
|Prepare|5.920|4941.412|三个 GEMM 改 Cube，缓存/输入转换并入输入阶段|
|Dhu|19.600|2216.796|按 V 面板递推，三个 GEMM 改 Cube；FP32状态不降精度|
|Finalize 主计算|46.719|17404.062|首先替换完整矩阵链与 score 梯度，不能只搬前两阶段|
|Finalize 第二任务|无|3.200|保留其实际功能直至核对实现；归并计划并入 Finalize 尾部|
|其他|两次 Transpose|十五次 Cast/Transpose|优化格式边界，但不能把主要问题归因于 launch|

事实：general Finalize 在 `MatrixGradients/DirectGradients/ScoreGradients` 中仍有多重行循环、Dot/ReduceSum 和单点读写；旧优化 Finalize 则有 Stage0/2/4/6/8、两 generation、独立 Akk/dW 生命周期和下一 work 预取。继承后者的调度原则，不照搬固定 HEAD_DIM=128 的偏移。Prepare 四 head、Dhu 四 head 的旧模板是工程参考，不强制 H=2 时也只启动一个核组。

新链：`BwdPrepare → Dhu → Finalize`，三次计算 launch。内部输入转换在各自 Vector 阶段完成，最终 cast/layout 写回在 Finalize 完成；不增加全量预处理、dL mask 或输出 transpose kernel。若阶段接口接线暂时保留适配 task，必须全数计时，不能宣称已经达到最终三段目标。公开独立入口不改变返回顺序或语义。

B 组保存目标 <=500 μs：Prepare 45、Dhu 100、Finalize 主体300、公共归并/转换及额外固定开销55；按完整调用计，不是每head预算。阶段自身时间包含其内部同步/搬运，末项不重复加总。A 组维持原快路径实测约98 μs且无显著回退。补做同 gate、state、dtype 的 K128/K256 对照，分清算法规模与路由差异。现有 A/B 不是纯维度翻倍实验。

优化施工优先级为 Finalize→Prepare→Dhu→整链去适配；每个阶段独立以保存缓存直调验证后替换。本轮停止继续扩展慢通用实现。C32随后复用这里的计算 Stage，另用其跨chunk调度器。

#### 2.12.2 工作项、tile 与两类流水

C=64，实际有效行 M<=64，Kp/Vp按16补齐；Cube物理 m/n向16、归约维向8补齐。输出 tile <=64×64，归约包<=128，K256分两包在同一 L0C 完整累加，最后一次才 Fixpipe。tail只算必要物理tile，M1按16计算，不固定做满64行；跨chunk数学仍按C64定义。

Prepare/Finalize：`Nwork=Q*ceil(H/CG)`；Dhu：`Nwork=S*ceil(H/CG)`，一个owner覆盖完整序列逆扫。CG仅沿head，两个AIV都处理各head的连续输出半片。参数化CG=1/2/4；H2/S1首先CG1，独立阶段4个work、Dhu2个work。CG4若减少本来不足的核组数，不作为首选。CG2/4用于足够work时的 stage-major 交错，需比较波次和峰值容量。

两条流水分别是：(a) 同一GEMM的 MTE2(n+1)/MTE1(n)/MMAD(n−1)/Fixpipe(n−2)；(b) Cube处理head/work的矩阵阶段时，Vector处理已发布的前一个结果、加载或生成下一阶段操作数。并行执行的是不同pipe；两个Cube阶段本身不声称并行。不能每次矩阵乘后全pipe排空，也不能等整个work的所有Vector尾处理结束才加载下一work。

A2/A3首选已有可核实的 FP32 MMAD + GM中继 + L1回读路径；不假设支持A5/A6的L0C直写另一AIV UB。Cube→Vector经GM中继一次读入UB；Cube→Cube结果在Fixpipe/GM→L1完成后驻留，按R14明确计入中转。`saved_cube.h`已通过的组件只是指令/数值依据；其保守即时Wait版本不能直接作为已经实现上述重叠的证明。

#### 2.12.3 Prepare：P0–P3

|Stage|单元、公式/形状|驻留、输出与最后消费者|
|---|---|---|
|P0|Vector，读取dO的有效[M,V]半片，升FP32；解析token/head/state步长|dO FP32面进入独占转换区，两个AIV ready后Cube可读；同work各GEMM复用，不为每个GEMM重复转换|
|P1|Cube，`rawS=dO[M,V]@Vnew.T[V,M]`|FP32 L0C全V归约后发布score结果；dO面板保留给P2/P3；Vnew本面板末次加载后可回收|
|P2|Cube，`dqRaw=dO[M,V]@Hentry.T[V,K]`；`dvLocal=Aqk.T[M,M]@dO[M,V]`|两路按K/V64列tile；H的V-first用逻辑转置视图；Aqk/dO复用在合法驻留窗口内，不生成全量转置|
|P3|Vector，`dS=scale*tril(rawS)`，发布dqRaw/dvLocal|dS含对角，dvLocal的Aqk已含scale，不再乘scale；三输出FP32|

排布：先发布 head0 的P1结果，Cube继续P2/head1，Vector同步做head0的mask/write；P0(next work)使用另一generation，与当前Cube尾部并行。P3按结果tile消费，不以所有P2完成为前置。输出bank未free时先消费已ready结果，不能只填输入。

#### 2.12.4 Dhu：D0–D4，逆向状态为唯一串行依赖

|Stage|单元与公式|数据依赖/释放|
|---|---|---|
|D0|Vector，首chunk从dht或0初始化FP32 Dnext；生成decay=E(Glast)|输入状态布局在本阶段解释；后续chunk直接读上一步Dprev；无初态读取需求不制造h0转换|
|D1|Cube，`stateV=kg[M,K]@Dnext[K,Vt]`，Vt<=64|沿K完整归约；Dnext保持只读，供Finalize的更新前快照不被覆盖|
|D2|Vector，`dU[:,v]=dvLocal[:,v]+stateV`|dU当前V面板发布后，其W消费者才能读；写公开dU供Finalize|
|D3|Cube，`termQ=qg.T[K,M]@dO[M,Vt]`；`termW=W.T[K,M]@dU[M,Vt]`|termQ可在D1结果交Vector时先计算；termW只等对应dU面板；两个结果按K64×Vt分片|
|D4|Vector，`Dprev=decay[:,None]*Dnext+scale*termQ−termW`|scale在完整归约后乘一次；每片写下一chunk的dh槽，末chunk写dh0或终端scratch；不能写当前dh|

相邻V面板之间无相互归约，形成 `D3(termQ,v+1) / D2(v)` 与 `D1(v+1) / D4(v)` 的合法窗口。每个V片的dU必须完整跨K归约，不能先更新部分Dnext再求剩余dU。首版chunk屏障仅等待本chunk所有V片D4完成，再处理上一物理chunk；下一chunk输入预取可提前。多head时按D1(head0..CG−1)、就绪D3和D2/D4交错，不按head整条序列跑完再换head。

公开dh缓冲承载每chunk更新前状态：逆扫首步初始化末chunk的槽，D4直接写前一chunk槽。空序列dh0=dht或0，不能访问虚构chunk。最后一步没有下一dh槽；若不返回dh0，独立终端scratch按owner分配。Hentry保持前向FP32快照，不能以BF16/FP16运行状态替换。

#### 2.12.5 Finalize：F0–F9，完整矩阵链先行

|Stage|单元、矩阵/向量运算|最后消费者与重叠机会|
|---|---|---|
|F0|Vector，恢复qhat/khat上下文、Rk=betaEff*khat*E(G)、Rv=betaEff*v及安全band输入|原始q/k在norm关时必须保留；Rk/Rv完整有效面写generation；与前work Cube尾部重叠|
|F1|Cube，`dW=−dU@Hentry.T`，`dkg=Vnew@Dnext.T`，`dRv=Akk.T@dU`|K/V按64列；dW面板独立于输入预取区，dkg/dRv交Vector；不能提前释放Akk|
|F2|Cube，`dA=dW@Rk.T+dU@Rv.T`|K/V分包累加同一L0C；dW完整GM面保留到F4，当前面板复用优先；dA完整64×64|
|F3|Cube，`middle=Akk.T@dA`，`dLraw=−middle@Akk.T`|中间完整归约后驻L1；复用dA槽必须等middle的最后读取完成|
|F4|Cube，`dRk=Akk.T@dW`|最后一列完成才释放dW/Akk；可与Vector处理已经发布的dkg/dRv重叠|
|F5|Vector，`dL=strict_lower(dLraw)`，初始化dq/dk/dG/dBeta并累加直接贡献|严格下三角先mask，再允许F6消费；不能把mask推迟到输出；Rv和dRv是不同语义槽|
|F6|Cube，安全band的qk/kk两侧score梯度矩阵乘；并重建所需Rkk|按§2.8公式，dS与dL*beta的左右侧都计算；K64列分片，不能只优化F1–F4而保留原逐行Dot的整个score链|
|F7|Vector，指数还原、score梯度合并、dBeta三路贡献及Glast贡献|Rkk不通过L/beta恢复；beta=0仍产生正确db；危险band执行有效域直接差值向量tile，禁止逐元素GM读写|
|F8|Vector，每chunk逆cumsum、gate导数、norm投影、beta导数、最终cast/layout写回|先完整合并Glast再逆扫；norm跨K partial按固定顺序合并，最后cast到各原输入dtype；norm关不读rstd|
|F9|独立Vector归并launch，全work参数partial按固定顺序归并|主计算launch完成后消费partial；当前实现不声称同launch跨work归并|

F6安全分解与危险band判定沿用§2.5/2.8，不放宽FP16范围承诺。列归约/参数归约不采用非确定atomic；每个(chunk,head)写独占partial，F9按head及chunk固定顺序合并。dBeta/norm所需的两AIV列partial用generation交换区，经AIC集合ready/广播ack后由约定lane归并，两个lane读同一个结果；不能只依赖PIPE_ALL。

F6 当前 band 上限固定为：预激活且 Vp≥64 时64行，其余24行；本轮 stage 重构不继续调这个参数。对角band用中间query行累计G为参考，非对角band用首query行；两AIV检查 `b*(Gi-Gref)` 和 `b*(Gref-Gj)` 最大值不超过32，否则保持直接差值回退。该数值候选已通过完整work90例与上游80例，不构成任意FP32中间值的全域有限性证明。

固定长度公开入口按work数选择CG1/CG2：配对后的work数不足AIC数时选择CG1，其余CG2；奇数尾head只提交有效owner。F1–F4仍按矩阵依赖层推进。满足直接搬运约束（m/n≥16、归约K为8的倍数、连续或转置视图）的FP32操作数由AIC直接ND→NZ搬入L1，不再经AIV打包写GM。其他尾片保留原打包路径。无需后处理的Assign阶段由Fixpipe直接写完整内部结果面，以12号ready在整个阶段完成后发布；负号、累加和严格下三角后处理保留AIV消费者。padding只能写内部补齐面，不能越界写公开输出。

L1实际配置为128KiB传输bank加256KiB的八个32KiB操作数槽。标签包含地址、ND形状、源stride和NZ stride；每个阶段/新band失效workspace范围的标签，公开只读输入可继续复用。当前A操作数槽在装载B时受保护，不能被替换；L1最后MTE1读取完成后才允许覆盖。UB输入与F0准备区合为连续65KiB，F6因子生成借用该区；结果/旧值各16KiB，共97KiB。F5/score/尾段共用84.25KiB arena，总181.25KiB。CG1的dq/dk/dG列面留在该arena中，因子生成和矩阵搬运不得覆盖它。CG1且Vp≥128、M≥16时，在F5发布前一次性生成完整masked dS和dL×beta，分别存主record的Value::Do面前后4096个FP32；该面直到score结束只读，所有band/K面板通过子矩阵视图消费，避免重复掩码、广播、搬运。其他规格保持band内生成，不扩大workspace或公开缓存。

危险band的Cube事务从执行列表剔除，不再执行零矩阵。CG1对角块的两个AIV各读取完整K64面板累计G，计算同一`max(abs((G-Gref)*base))≤32`判定；有符号差值通过UB广播视图保留为各lane的expPlus/expMinus输入，不再次读取G。判定与原正负两向max条件等价，省去两AIV之间的一次ready/ack和控制回读。predicate与操作数一起发布，AIC等待发布后读取两项控制值。非单调非对角块及CG2仍用原两AIV交换协议；已证明单调的非对角矩形不读取控制值。AIC的读取先做MTE2→Scalar排序并失效控制cacheline；每band两个GlobalTensor::GetValue仅用于控制，与通用性能技能禁用建议的明确例外不能扩展到张量逐元素读取。中继输入/结果信用按实际提交计数，下一代复用才消费，退出排空至多两个剩余信用。全回退、交替安全/回退和最后band回退均需验证。

FIX_M在同一个L0C槽下一次使用前等待，退出排空最后两个槽；其余L1/L0硬事件仍保留对应最后消费者。直接写回与中继结果分别计数：直接结果不占中继result_ready/free信用，退出只排空实际中继信用。安全原生F6中，`ds.T@qPlus`与`dlBeta.T@kPlus`在同一L0C连续累加，再一次写到Score::Key；第一次不发布/不释放L0C，第二次不重复等待上一代FIX_M，F7只读取合并后的key梯度。该固定FP32加法顺序通过独立标杆与上游容差验收，不声称逐比特等同于两次独立MMAD再相加。公开F9仍是独立归并launch，跨完整work的双generation尚未接入。

F5退休后，Rk/dW/dkg/dRk/band五个key面复用为qPlus/kPlus/kMinus/expPlus/expMinus；dL固定保留在主record。scalar矩阵放入退休的Rv/Drv区。CG1按K列片外层、query块内层推进：保持24行对角块不变，将同一query块的全部此前key合为一个非对角矩形，M64对应三个对角块和两个非对角矩形。V32下最大scalar占用仍不超过2048个FP32。raw softplus或lowerBound≤0的raw safe gate产生非正log-step，完整非对角矩形以首query为参考时两项指数均≤0，可省去两AIV安全判定交换；对角块及预激活输入仍执行原阈值32的动态判定和回退。此优化依据输入语义，不把任意预激活G视为单调。

CG1将一个K64面板的dq/dk/dG保存在UB，完成全部band后统一写回。安全和直接差值回退采用相同的`Pad16(panelWidth)/2`列归属；回退前写回、回退后重载，避免K100尾片在两AIV间交换归属而读到未完成的结果。F7批量执行整strip计算。F0/F5和band beta使用Brcb广播、WholeReduceSum行归约；F5的末行gate贡献使用固定FP32树归约。norm关的F8以8行批量搬运/cast/gate计算，逆cumsum及参数累加仍按原反向token顺序。仅在完整32列lane保证所有活跃元素先写后读时跳过清零，短尾仍显式初始化。

当前CG1双record用于同work的score流水：Prepare(0)→Cube(0)→Prepare(1)→[Cube(1)与F7(0)]→Prepare(2)→[Cube(2)与F7(1)]，依此推进。两个record轮流保存因子和score结果；dL与梯度基址始终是主record。Prepare(i+2)开始前F7(i)已完成最后读取。原生Assign band的所有Fixpipe写完成后以12号ready发布，非原生band使用实际中继结果信用。最后一个F7和列面写回完成后才发列退休握手，随后进入下一列或Tail。CG2保留stage-major多head队列；不能把本轮已实现的跨band重叠表述为跨完整work或C32流水已完成。

#### 2.12.6 资源、生命周期及带宽账本

本轮C64实际配置及同步以§2.12.5为准；下列完整work跨代资源表属于后续流水目标，不把尚未接入的驻留lane计作已经实现。固定长度BF16/K256、T65/H2、raw-safe无状态的最终三轮验收：保存Finalize 256.4–257.6μs，为同轮原K128专用路径5.46–5.57倍；保存整链493.9–496.6μs。精度96组fla-org原用例函数通过，详见`validation.md`§13.3。该结果不外推为C32或重算整链已经达标。

以下为独立C64新模板上界，取代§2.3中“C64候选2*CG*R”的粗分配；C32资源保持不变。`depth=min(2,4/CG)`，最多4个(head,generation)槽。

|存储|分区（KiB，半开）|所有权与释放|
|---|---|---|
|L1|[0,256)：4×64KiB驻留lane|每lane四个16KiB面：Akk、dW当前列板、dA、middle；依阶段换义须按F2/F3/F4最后消费者释放|
|L1|[256,384)：2×64KiB输入bank|每bank A/B各32KiB，最多64×128 FP32；MTE1最后读完才允许下一代MTE2覆盖|
|L1|[384,448)：中转重载；[448,512)：余量|GM→L1的Cube结果面板回读；不把余量计为额外已驻留状态|
|L0A/B|各2×32KiB|MTE1等待MMAD读完free；不同输入包与输出tile采用独立计数|
|L0C|2×64KiB|一个输出tile跨全部归约包持有，Fixpipe最后读完才free|
|每AIV UB|[0,64)：4×16KiB结果驻留；[64,128)：操作数/向量活跃面；[128,160)：计算临时；[160,176)：scalar/归约；[176,184)：元数据；[184,192)：保留|每结果tile半片最大8KiB，一槽16KiB；FP32 Vector计算只处理K64/V64列板，禁止同时驻留完整K256状态|

这是一张分配上界表，不允许某Stage临时借用仍活跃的另一结果槽。进入04前用每Stage实际LocalTensor清单检验峰值；特别F7直接差值tile和F8 norm归约不能超过操作数/临时区。高风险运算也先mask无效域，padding exponent不做无意义exp。

GM临时每generation/head保守分配：`align256(4*(6*C*C + 8*C*Kp + 3*C*Vp))`，六score面为dA/middle/dL/Rkk/两score梯度临时；八key面为Rk/dW/dRk/dkg/dq/dk/dG/band输入；三value面为Rv/dRv/dO升精度。公开dS/dqRaw/dvLocal/dU、缓存Aqk/Akk/G/H/dh不重复分配。band输入按面板复用，A/B relay另计。参数partial `align256(4*(Kp+1))*Q*H`；Dhu不返回dh0时终端scratch为每owner `4*Kp*Vp`，可与已退休阶段临时复用。系统workspace单列，checked乘加按实际blockDim/CG/depth计算。

Dhu按V面板生成termQ/termW，两块各`4*Kp*64`，不为每chunk额外保存两个完整K×V中间态；公共dh本身是必须返回的快照，不因“状态workspace”再复制一整份。Prepare需要P0的dO升精度面，由本work槽分配并在最后GEMM读取后回收。

搬运审计必须单列下列重复，不能以“驻留”掩盖：

- dW完整面溢出L1后供F2/F4读：至多一次写、两次读，即`12*C*Kp`字节/head/chunk；能驻留的当前面板跳过对应重载。
- Cube→Vector中继：每次有效结果写+读`8*m*n`字节；Cube→Cube中继同样计两程，结果回L1后在该生命周期只读一次。
- FP32状态H/dh每面`4*Kp*Vp`字节；按实际重用列出各GEMM读取次数，禁止标为整链只读一次。K/V翻倍使状态面为4倍，不会自然产生100倍指令量。
- 公开阶段边界按实际输出计量：Prepare `4*(M*M+M*K+M*V)`，Dhu `4*(M*V+K*V)`，Final梯度按原dtype及参数FP32计。输入、输出、GM中转、初始化、partial归并分别统计。

#### 2.12.7 同步与无死锁要求

事件按缓冲槽，不按矩阵语义重复占用。每个新kernel独立命名空间：cross-core 0/1输入ready，2/3输入free，4/5结果ready，6/7结果free，8/9 generation退休集合/广播；10/11保留给两AIV scalar交换集合/广播，12/13未使用；14/15不占用。多head按相同静态顺序通过两个传输bank，flag不按CG倍增；驻留结果由每head槽持有。scalar交换需AIC在双方都到达的静态位置服务，不与等待其输入的GEMM形成环。

- Vector发布输入：V写完→V_MTE3→MTE3写GM→input_ready；AIC等双方ready再MTE2。AIC读完GM才能发input_free。
- Cube发布结果：首次覆盖结果bank前取得result_free；全归约结束M_FIX→Fixpipe→result_ready；两AIV完成最后MTE2读取后才归还result_free。结果bank free与L0C free是两件事。
- Vector UB复用：前次MTE3读完→MTE3_V→下一次Duplicate/计算；不能只等待MTE3_MTE2。该缺口已在组件长循环中复现并修复。
- generation只有全部Cube最后读取、两个AIV最终写回均完成后才退休。Dhu另有状态依赖，输入free不能代替state_ready。
- 每个方向HardEvent使用静态双槽映射；L0/MMAD、L1/MTE1、Fixpipe回读各有对应ready/free。即时Set/Wait只是保守正确版本；改为延后等待前必须覆盖3/7/67次复用及退出，不能重现已有组件的延后回收挂起。
- warmup仅初始化确有后继wait的信用；drain不产生永远没人消费的credit。循环分支、packed空序列、tailhead必须让AIC和两AIV拥有相同事务数量。消费者优先，有界outstanding<=2，禁止先提交全部输入再读结果。

该事件表是拟采用协议，需绑定实际API与分配结果后跑设备检查；不以编号已写出冒称完整流水已验证。

#### 2.12.8 精度、接线及性能门禁

输入按api.md保留，Python只路由；ACLNN检查shape/dtype/属性。保存入口直接把原始layout步长传给阶段；GM接口承诺的FP32 Aqk/Akk/H/dh等不降精度，HF32关闭。Prepare输出FP32 dS/dvLocal/dqRaw→Dhu输出FP32 dU/dh/dh0→Finalize输出原输入dtype梯度和FP32参数partial。独立入口与组合入口分别测试db类型、state_v_first及可选状态，不因组合路径优化破坏独立接口。

原BF16/128合法key先命中旧快路径；新增FP16/非等维/K256/FP32 cache组合命中新分阶段模板。路由不能按“dtype相同”忽略cache dtype/feature。研发中可保留general用于定位，但每份验收必须报告实际key；落入general不得计为性能通过。GDN共享Dhu变更只扩KDA所需新key，原GDN路径回归，避免跨任务覆盖。F8的norm跨K先形成每行partial并固定次序合并，再读回对应dq/dk列板应用投影；这两遍GM读取是容量限制下的显式R14成本，不能称为一次读入完成。

可达性边界：有效三段主GEMM量 `12*M*K*V + 6*M*M*V + 4*M*M*K + 4*M^3`，不含F6 score梯度、padding、Vector和归并。B组H2/M64+1约129.5MFLOPs；物理tail按16计，另加F6全部矩阵tile。组件已测每tile约1.36–1.78μs仅作为含中转吞吐的量级观察，不用于证明500μs达标；实际必须按每work的tile数、并行核组数、依赖最长路径和全部中转计算。Finalize占旧慢链约七成，是首要结构收益来源。

阶段门禁：先独立Finalize完整F0–F9（含安全/危险band、norm/beta/gate、归并），再Prepare与Dhu；每个阶段对上游标杆中间量，随后完整fla-org测试。至少覆盖K64/V32、K256、K60/100尾部、M1/63/64、T65状态快照、exp/exp2、非单位scale、beta0、safe±20、packed空序列与连续不同序列。不得以仅GEMM测试替代整阶段验收。

时间线必须展示至少一个实际的Cube/Vector或MTE/MMAD重叠窗口，区分低H一次work无跨work收益与长例稳态。预算不通过时先定位F6、GM回读、同步/占核，单变量比较CG和深度，不下调精度。验证只用fla-org体系，无ATK；A2/A3任一平台。正式性能仍按§2.13统一口径与三批重复测量。C64保存通过后再进入C32完整保存融合；重算链仍是Recompute→FwdH→三段，新重算详设后续完成。

### 2.13 原型、精度与性能评审顺序

1. **冻结真基线**：原BF16/128专用key及产物hash，沿用§2.12.1的逐阶段复采；补同条件K128/K256控制组。慢generic只作失败诊断。
2. **C64 Finalize先行**：实现F0–F9完整主体，含score梯度、tail、norm/gate/beta和参数归并；验证中间量及300μs主体预算。不能只测F1的矩阵乘。
3. **C64 Prepare/Dhu接通**：验证45/100μs预算、FP32缓存/状态快照和多head/跨面板同步；整链测全部任务，去除适配后仍需保持精度。保存总计<=500μs，旧快路径无显著回退。先证明低H可行，再测T1024/H8、T8192/H1与packed。
4. **C32完整保存融合**：复用通过的阶段，固定CG1验证三record/双bank、两AIV真实运算、N=0/1/2/3及连续回绕；接P_(r+1)/D_r/F_(r−1)、FP32状态、核内参数归并。时间线核实真实重叠，保存单launch必须比优化后的同规格C64更快；仅组件通过不算交付。
5. **重算另行详设**：保留C32重算单kernel要求。先完成正逆扫描调度/缓存与性能评审，再复用已优化阶段接通。不得在保存优化期间暗增launch或把重算记为已完成。

必测定向：M1 q=k=(256,0,…)且v=0、M16强负gate；非单位scale；safe±20非零梯度；beta三分支及beta0；norm关空rstd；T65非零h0/dht两状态布局与更新前dh；packed空序列和跨序列loss可加性；K60/100哨兵越界；workspace非零初值；H大于核数；保存后释放外部q/k引用；exp/exp2缓存单位；Akk保存重算完全一致。

正式性能每个模式预热2、采样10，至少三个交错A/B批次，保留min/median/max及全部样本；主口径为每调用全部Task Duration之和，另报span、task数、每核负载、MTE/Scalar/Fixpipe等待。C32恰一个计算task；C64列出全部转换任务。不放宽原fla-org阈值，不选择性隐藏失败规格。

D3.1保存线路进入04的签核条件：每Stage地址/事件资源检查通过；全部危险值分支正确；保存模式A/B/C及长例预算和原快路径无回退有证据；至少一个目标平台的完整依赖原型证明单launch可行。**当前C32组件验证与旧公开入口复采已完成；C64新流水及C32完整融合均未通过整链性能门禁，不能宣布性能恢复或设计冻结。**
