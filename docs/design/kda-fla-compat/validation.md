# KDA C32/C64 泛化：D3验证与性能追踪

版本D3.2，2026-11-26。与[design.md](design.md)、[api.md](api.md)配套。旧记录完整保留在[历史验证记录](evidence/validation-before-d3.md)，其中AIV_ONLY、whole-head owner、无性能目标、禁止参数partial等旧设计描述已经失效，不适用于D3。

## 1. 已有事实与未执行项

|对象|已有证据|边界|
|---|---|---|
|CPU公式|旧72组合通过|不是D3块求逆、band快速/保护分支及新累加顺序验证|
|P10完整实现|原上游范围内36项+扩展162项通过；安装后2节点复验|不是全部fla-org用例；GVA等范围外未覆盖；不是D3正确性|
|P10 C32|保存/重算设备各一个计算task|head串行、AIV1只握手、B3仍慢；未达性能目标|
|旧C64专用路径|A组95.298/118.0275μs；stage原始CSV已核对|来自P10包保留旧路径，需冻结纯基线源码/产物并补msprof口径|
|慢通用C64|K256约24.59/27.56ms|失败记录，不作为性能达标基线|
|D3前向/反向流水|设计正文已重写|代码未修改，新精度/同步/资源/性能原型均未执行|

本轮未连接设备测试、未编译、未安装、未运行ATK。仅本地读取已有日志/CSV并提取逐任务统计。A2/A3任一平台验收即可，拟继续A3；不扩A5。

## 2. 真性能基线及数据来源

- [P10完整采样摘要](evidence/profile-matrix-p10.json)。原始kernel_details.csv在[p10归档](evidence/p10-evidence.tgz)。
- [逐任务位置统计](evidence/d3-original-c64-stage-evidence.json)：两次Transpose分别统计，不能混在同名集合中；保存/重算分别10次。
- 原快路径保存各任务中位数：12.1995、10.7300、5.5100、19.5800、47.2990μs；重算：11.8100、10.5800、12.1900、12.3395、4.7100、19.4295、46.8390μs。
- 主要统计为每次公开调用全部设备任务Duration之和的中位数；不能把上述各任务中位数相加当同一统计量。
- 最终遵循PR833仓内V2：msprof op_summary Task Duration(us)；历史torch_npu profiler Duration需做同任务对应核对，不能混用span、ACL event或CPU wall time。

新规格没有原生旧同规格快路径时，明确填“无”；使用design中计算/搬运/同步模型和预算，不将错误慢路径当baseline。K256 FP16有状态与K128 BF16无状态不是同规格，不能直接宣称两者倍率为同条件回退。

## 3. 测量配置与命令

固定上游c55dc553edf44701bf703715ed052d811f8a6376，目标A3/CANN9.3.0.P009，同设备、产物、seed21、输入、layout、开关及状态。现有三组输入参数详见design 1.3。前向和反向分别测；反向只把本次调用内任务纳入，不把生成缓存的前向混入。

已有采集脚本命令模板（在隔离A3环境内，先核实hostname/whoami/pwd）：

```bash
ASCEND_RT_VISIBLE_DEVICES=10 FLA_NPU_OVERLAY_VENDOR= python   fla/ops/ascendc/kda/chunk_kda_bwd/tests/upstream/profile_c32_matrix.py   --output <全新输出目录> --samples 10   --only bf16_128_no_state,fp16_k256_v256,fp16_k64_v32
```

该脚本使用torch_npu profiler，是已有诊断采集。正式msprof命令按目标安装版本的`msprof --help`确认后记录原命令，不能在未执行时声称已采集op_summary。前向用profile_public_c32.py去掉--backward-only，明确C32/C64、dtype、T/H/K/V、raw-safe/no-state。每模式预热2、采样10、至少三个交错A/B批次，不复用旧输出目录。

记录实际加载wheel/OPP、源码manifest、kernel hash、TilingKey、CG、blockDim、真实workspace、全部task名称/数量和样本。C32恰好一个计算task；禁止过滤Transpose/Exp等。C64每个转换和内部GateNormalize均计入。

## 4. D3.2保存线路门禁与性能追踪

|模型case|轮次/状态|证据或预计关键路径|结构决策/唯一主要变量|精度|性能|结论|
|---|---|---|---|---|---|---|
|A BF16/K128/T65/H2|基线|Finalize47.3μs、转换约23μs|冻结原专用key|历史范围内通过|95.298/118.0275μs|旧快路径锚点；不允许generic替代|
|B FP16/K256/T65/H2|失败记录|P10逐行/逐点和单服务槽|停止旧owner增量改法|历史通过|C32 24.68/51.73ms|不是合格baseline|
|A/B/C完整Finalize|待实测|S11–14普通GEMM、band梯度和搬运|全矩阵流水，与旧B3比较|待执行|design逐阶段预算|先测完整主体，不只三个GEMM|
|A/B/C Prepare/FwdH/Dhu|待实测|求逆、FP32状态、稳定分支|共享tile原语|待执行|design前反向预算|不降低缓存精度|
|保存CG1跨chunk流水|待实测|P_(r+1)/D_r/F_(r−1)，三record/双传输bank|先固定CG1，验证完整静态调度|待执行|吞吐/等待/波次|时间线上确有重叠才采用|
|多head CG2/4|后续优化|CG1测得剩余空泡|每轮只改CG|待执行|与完成的CG1比较|非首版依赖|
|T1024/H8、T8192/H1|待实测|scan真依赖、重算双扫|完整依赖骨架|待执行|模型预算待设备系数|H1不可用CG4掩饰低并行|
|最终C32对合格C64|待实测|完整公开调用|相同输入仅C不同|fla-org原阈值|逐规格C32更快且过预算|平均收益不可抵消回退|

预算见design 1.4，属于工程止损上限而非实测预测。原型不达预算返回03，不先生产化后再把目标改成胜过慢generic。旧支持范围需同条件无显著回退，新规格两条C路径都要通过预算。

## 5. 精度、协议与资源验收

沿用上游tests/ops/test_kda.py及fla/ops/kda/naive.py。原节点仅换DUT后端，不改reference/阈值/断言；扩维度与接口测试另列node。六组K/V×两dtype×C32/C64×保存重算必覆盖；gate三态、norm两态、beta三态、可选参数、状态布局/packed采用分层矩阵，不冒称全笛卡尔积已测。

逐Stage采集中间值：qhat/rstd/G/betaEff；Aqk/Akk/W/U；h/WH/Vnew；dS/dU/更新前dh；dW/dA/dL/dRk/dRv；合并后的dG及最终参数梯度。比较有效区、tail单位阵和mask，不仅比较最终输出或保存/重算彼此。

必须定向覆盖：

1. M1 score=65536，v=0；M16强负gate；FP32稳定band与直接差值分支边界；不能出现人工Inf×0。
2. norm关且rstd为空；beta关闭/普通/双倍，beta=0时dv=0且db非零；safe±20可表示非零梯度。
3. dL在S12b结束时对角和上三角均0；dW直到dRk结束保持正确；所有梯度面首次写覆盖，非零workspace不污染。
4. T65/K64V32/scale0.37、非零h0/dht，两种状态布局；逐chunk对照Hentry、WH、Vnew、更新前Dnext，末tokenloss与dh0。
5. packed空序列、相邻不同非空序列、独立loss与联合loss可加性，G逆cumsum每chunk重置，exp/exp2换底一次。
6. 两generation连续不同数据回绕、同地址写后读、两个AIV最后消费者、CG尾head、Nwork超过核数、末核提前结束及SDK参数归并。
7. K60/100及V32边界哨兵；物理尾逆矩阵单位阵；源q/k外部引用释放后仍可反向；输入不可写别名。
8. softplus分支边界±ln16与±20/±80，理想导数对照；原始beta dtype与独立FP32 betaEff输出dtype分别检查。

实际UB/L1/L0最高地址、HardEvent数量、cross-core flag与SDK保留flag、GM record上界、hTape、partial和系统workspace必须核实。新归并允许partial但不允许额外Reduce kernel；旧记录“不能分配partial”已废止。

## 6. 本轮文档变更审计

D2.4原设计和旧验证记录已保留。D3只重写设计、同步接口状态和验证计划，并新增从既有CSV提取的阶段JSON及源码hash核对。没有修改kernel、Host、Python测试或构建脚本。源码一致性记录见evidence/d3-code-unchanged-before.json及d3-document-review.json。

## 7. 保存线路专用检查（D3.1）

当前首要范围为disable_recompute=true，不修改代码。重算单kernel仍为最终合同，但新调度详设另行完成。先以CG1为基线，不能把多head吞吐提升冒充单head跨chunk流水成立。

- 扫描N=0/1/2/3/4/7及长序列，记录(r,physical_chunk,record_id,transfer_bank,stage)；record_id=r%3，而bank按传输批次%2，验证二者独立回绕。
- T65实际顺序为c2(M1)、c1(M32)、c0(M32)。P下一chunk提前计算；D只等dvLocal/Dnext；F当前chunk延后一拍。记录D_(r+1)不依赖F_r的事件关系。
- 用互不相同的chunk输入/状态和非零workspace，检查D_r只读的Dnext一直保留到F_r末次消费；Dprev直接写下一record的Dnext，无旧record别名，无覆盖P输出。
- 输出精度按原上游标杆；增加r%3回绕哨兵、F中间量跨Vector阶段暂存检查。只比较最终梯度不能代替快照/缓冲检查。
- 时间线分别标P、D、F及所属扫描序号，报告Cube/Vector/MTE/FIX重叠、D主线等待和填充/排空成本。N1无跨chunk重叠，不声称加速来自流水。
- 资源核对：3R+192KiB+scalar交换区/组，另加参数partial和SDK系统区；保存模式不得分配hTape。P/D/F暂存回读的字节量必须计入预算。

本次未执行设备原型或精度测试，不因设计更新改变P10历史结果。

## 8. 保存流水静态复审（D3.2）

本轮只改文档、执行本地抽象协议模型；未修改算子/Host/测试代码，未编译或跑卡。静态结果见[evidence/d3.2-saved-pipeline-static-check.json](evidence/d3.2-saved-pipeline-static-check.json)。

|检查|结果|证明边界|
|---|---|---|
|P_(r+1)/D_r/F_(r−1)依赖|存在四类合法跨pipe重叠窗口，正文2.2已列|不证明编译后实际重叠时长；首版宏步末F尾部不能声称被下一D完全隐藏|
|三record|N=0..129生命周期模型通过|未模拟物理DMA缓存，需设备哨兵验证|
|双bank信用|14组、4080抽象状态无死锁/残留credit|两AIV独立进展；输入包与输出tile不等数场景尚需设备/扩展协议模型|
|旧提交顺序反例|先填全部输入再消费结果出现死锁|正文改为消费者优先、有界outstanding，防实现重犯|
|持久化缺口|新增record.score4/5保存dA/middle；第三状态槽用于termQ|每次额外GM搬运计入预算，不能继续占公用relay|
|AIV归约|选择原生FP32 Cube固定矩阵中继|不新增未证实的AIV间flag API；性能和浮点误差待设备检查|
|record退休|flag8两AIV集合READY→AIC→flag9广播ACK|ACK完成才能回绕；不把input/result free当作record free|

下一设备验收必须补：不同Kr包数（K256）、危险band分支切换、norm开关的事务计数、两个AIV不同延迟、N=1/2/3及三record多轮回绕、RETIRE与SDK末尾barrier互不占用flag、真实HardEvent资源报告。确认GM禁L2、MTE3→ready→MTE2序列及有效字节边界。

本轮可以确认的是设计的依赖/存储分工及抽象信用闭环已补齐；不能宣布Prepare/Finalize在设备上已经隐藏到Dhu耗时内，也不能据模型宣布80/500μs预算通过。没有用修改精度、增加kernel或更换慢基线规避门禁。

## 9. D3.2 开发验证记录（2026-11-26，尚未完成交付）

本节之后已开始修改源码，前面“仅改文档”的表述仅描述对应历史轮次。

- `saved_contract.h` 实现三 record、独立双输入/结果 bank、checked workspace 分配和逆向 chunk 描述符。主机测试通过六组布局、溢出负例和长度 0..4097 的生命周期检查。
- `saved_operations.h` 实现 P/D/F 的十四个矩阵运算视图与输出 tile。192 组规格/状态布局/record 检查通过；DhuW 强制显式提供状态输出地址，不再隐式写最后一步的下一 record。该测试证明寻址和覆盖范围，不代表十四个设备阶段已集成。
- `saved_cube.h` 实现原生 FP32 Cube、四种转置组合、Kr128 分包累加、输入与结果独立信用及退休握手；`saved_pack.h` 实现物理连续行搬运和尾部补零，转置由 Cube 装载完成。
- A3、CANN 9.3.0.P009 上构建通过，80 个搬运/矩阵组件用例通过，最大绝对误差 `3.0517578125e-05`。包含 K256 两包归约、1/2/3/7/67 个输出连续复用、四种转置；日志为 [d32-transport10.log](evidence/d32-transport10.log)，构建日志为 [d32-build10.log](evidence/d32-build10.log)。本组实际矩阵维度均对齐，不能据此宣称 K60/100 有效字节边界已完成设备验证。
- 延后回收核内事件的早期版本发生退出/复用卡住，当前使用最后消费者处即时 Set/Wait 闭合；仅证明组件正确性，不能宣称设计中所有核内重叠均已实现。统一补零搬运曾在长循环出现误差，已补齐 MTE3→Vector 的 UB 复用依赖，修复后上述 80 例通过。失败日志 `d32-transport9.log` 保留，不覆盖为成功结果。

**尚未完成：** 新组件没有接入公开保存模式入口；完整 P/D/F 跨 chunk 调度、Finalize Vector 计算和参数归并仍需集成。尚未执行此版本的 fla-org 完整反向测试，也没有同规格 C64 性能结果。未运行 ATK，未安装新完整算子包，未提交或推送。上述组件通过不能作为完整单 kernel、80/500μs 预算或快于优化 C64 的验收结论。

## 10. 2026-11-26 性能复采（旧公开入口与新组件分开）

A3 物理设备14，CANN 9.3.0.P009。设备10有其他用户任务，未使用或终止该任务。完整入口使用既有安装版本，未切换新组件；T=65、H=2、scale=0.37，BF16组无初末状态梯度、raw safe gate，FP16/K256组有状态。沿用 profile_c32_matrix.py/profile_public_c32.py：2次预热、10次采样，统计每次全部设备任务时长之和的中位数，包含转置等任务，不是主机端到端耗时。

|规格/模式|旧C32 μs|C64 μs|C32/C64|设备任务数 C32/C64|
|---|---:|---:|---:|---:|
|BF16 K128 保存|12977.3505|97.9275|132.52|1/5|
|BF16 K128 重算|20832.2335|119.1975|174.77|1/7|
|FP16 K256 保存|25039.0495|24694.3365|1.014|1/19|
|FP16 K256 重算|51746.606|27561.519|1.877|1/21|

FP16/K256 的 C64 仍是慢通用实现，仅供诊断，不作为达标基线。旧入口结果与历史量级一致，不能归因于尚未接入的新组件。

新 saved_transport 单 group、原生FP32、包含两AIV输入搬运及结果写回，10次预热、20次采样；计时前与FP64 matmul对照通过。以下只含重复矩阵乘传输，不含完整KDA的Prepare/Finalize Vector、状态递推和参数归并。

|矩阵 M/N/K|同launch乘法数量|中位数 μs|范围 μs|平均每乘法 μs|
|---|---:|---:|---:|---:|
|32/64/128，无转置|3|7.4395|6.059–9.219|2.480|
|32/64/128，无转置|67|91.4185|88.898–94.778|1.364|
|64/64/256，双转置|3|9.2600|8.400–9.880|3.087|
|64/64/256，双转置|67|119.3580|115.378–122.177|1.781|

这是含搬运/同步的组件吞吐，不是裸Cube吞吐，也不能直接与完整C64相除形成加速比。短批有明显波动；尚无新完整C32性能结论。

证据：[公开入口汇总](evidence/d32-public-performance-20261126.json)、[组件汇总](evidence/d32-component-performance-20261126.json)、[原始CSV归档](evidence/d32-perf-raw-20261126.tar.gz)。组件so SHA256=`0db9e566af838562e4e5c8cabb0573a77cb32027e0a313e2e3ef1663c9952776`。

## 11. D3.3 C64 保存流水重设计

本轮从已有原始CSV提取逐任务中位数，新增 `evidence/d33-c64-save-stage-evidence.json`；未重新跑卡、未修改算子代码。design §2.12正文已替换，历史D3.2保存在 `evidence/design-d3.2-before-c64-save-redesign.md`。

确认诊断：K256 Prepare 4941.4115μs、Dhu 2216.796μs、Finalize主计算17404.062μs；优先优化完整Finalize，其次Prepare/Dhu。三段目标45/100/300μs，完整保存500μs是拟定门槛，不是可达性已通过。stage-major、多head及双generation的真实重叠、片上活跃量峰值、跨核信用资源均需实现后的完整阶段设备验证；已有saved_transport组件不能替代。

后续性能追踪按：Finalize完整主体→Prepare/Dhu→C64保存整链→C32保存单kernel。旧BF16/128快路径必须无显著回退，新增C32必须胜过优化后的同规格C64。补FP16 K128/K256相同state/gate的控制组，并记录实际tiling key；不把原BF16无状态与FP16有状态的差异全部解释为K翻倍。

## 12. D3.3 开发起点：Finalize 描述符

已新增 `chunk_kda_bwd_finalize/op_kernel/arch22/pipeline64/contract.h` 与 `products.h`，落实 CG1/2/4、generation 深度、checked workspace、参数 partial、独占 Rv/dRv、F1–F4 八个矩阵操作视图和 dL 发布前严格下三角要求。Host 构建函数与未来 device 共用描述符；尚未在公共 Host TilingKey 中启用。

按 PR833 的04阶段顺序，采用 `CHUNK_INDEPENDENT` V1 开发参考（来源a78efa9512dfa64c20b2a58a75e84e88a054d694）的独立work/生命周期组织；两个AIV的任务映射遵循当前V2设计，未照搬旧参考的整head承包方式。

`tests/test_pipeline64_contract.cpp` 在A3主机以 `g++ -O2 -std=c++17 -Wall -Wextra -Werror` 编译运行通过：288组布局，检查完整work覆盖、两状态布局矩阵视图、尾head/空chunk、dL mask契约及溢出拒绝。该测试不执行NPU，不代表实际mask指令、Cube/Vector流水或fla-org精度已通过。设备测试待各Stage接通；未安装新包、未改变旧专用key、未提交或推送。本条不改变§10的实测性能结论。

## 13. D3.3 Finalize 组件与完整 work 串接（2026-11-26）

A3 物理设备14、CANN 9.3.0.P009，独立测试库位于 `/root/wys/fla-compat-d33/source/tests/pipeline64/build/libkda_c64_stages.so`。全核并行编译。以下为开发期设备验证，不是公开接口的 fla-org 验收；未使用 ATK。

|阶段|实际设备检查|证据|
|---|---|---|
|F0 与 F1–F4|108组，三次重复执行，FP32/FP16/BF16 value，六组 K/V、M=1/63/64、两状态布局、exp/exp2；通过|[prepared-typed.log](evidence/prepared-typed.log)|
|Cube score 与梯度合并|20组 score、20组 merge，包含对角重合和非零梯度底值；通过|[score-stages.log](evidence/score-stages.log)、[score-merged.log](evidence/score-merged.log)|
|F5 直接贡献|108组，与独立 autograd 对照 q/k/v/G/beta 以及末行状态贡献；通过|[direct-contributions.log](evidence/direct-contributions.log)|
|保护 score 与跨 K panel 累加|56组，K=28/32/60/64/100/128/256 的组件宽度，M=1/16/63/64，强负 gate、beta=0；通过。28/32 为局部组件宽度，不扩展公开 K 契约|[direct-score-panels.log](evidence/direct-score-panels.log)|
|F8 控制流程|270组，五档 K、M=1/16/64，18种 norm/gate/beta 组合，FP16/BF16，检查反向 cumsum、norm、beta、参数 partial；通过。exp2/dtype 与 norm 相关选取，不代表完整笛卡尔积|[tail.log](evidence/tail.log)|
|完整单 work、单 launch|90组，K/V=60/60、64/32、100/100、128/128、256/256，M=1/16/64，每档6种分支；从 F0、八个 GEMM、直接贡献、保护 score 到尾部输出，对照独立 FP64 autograd 的全部梯度；通过|[finalize-work.log](evidence/finalize-work.log)|

`finalize_work.h` 使用 MatrixVector 的97 KiB缓冲与复用的82.25 KiB Vector arena，F5、保护 score、Tail 不同时占用独立 arena。F5 完成后两 AIV 退休握手，再复用 scratch；score 全 K panel 完成后握手再进入 Tail。beta partial 由 F5 初始化，score 每个 panel 只累加一次。Tail 每个 norm 行及最后参数归并通过独立10/11号交换握手。

**当前公开接入状态：** 常规安全 band 已接入 Cube，预激活且 Vp≥64 使用64行 band，否则使用16行；不安全 band 保留直接差值计算。固定长度的 Prepare、Dhu 和 Finalize 已接入公开链路，旧 BF16 专用路径保留。当前仍未实现跨 work 的完整流水，参数归并仍为独立 launch；不能据此宣称设计的全部流水已实现。

### 13.1 公开链路回归与实测

A3 物理设备14，同一 CANN 环境。新 Dhu wheel 安装后调用 fla-org 原始 `test_chunk` 函数及其断言，扩展 K=60/64/100/128/256、FP16/BF16、保存/重算、norm 开/关共40组，全部通过；B=2、T=65、H=HV=2、scale=0.37。证据：[upstream-dhu-expanded.log](evidence/upstream-dhu-expanded.log)。这不是整个上游测试套件全部通过的声明，也不包含 GVA。

Prepare 组件72组、Dhu组件72组均三次重复通过；开发组件检查不替代上述上游验收。证据：[prepare-work.log](evidence/prepare-work.log)、[dhu-work.log](evidence/dhu-work.log)。

公开入口性能仍采用2次预热、10次采样的每调用全部设备任务时长之和中位数。T=65、H=2；FP16/K256有状态、预激活 gate，BF16/K128无状态、raw safe gate，二者不是仅 K 不同的严格控制组。

|项目|旧版本 μs|本轮公开版本 μs|
|---|---:|---:|
|FP16 K256 保存整链|24694.3365|699.3855|
|其中 Prepare|4941.4115|37.3490|
|其中 Dhu|2216.7960|82.0885|
|其中 Finalize 主计算|17404.0620|455.5910|
|FP16 K256 重算整链|27561.5190|3617.7880|
|BF16 K128 保存旧专用路径|97.9275|约98.00|
|BF16 K128 重算旧专用路径|119.1975|约119.06|

各阶段独立中位数之和不必等于整链中位数。Finalize 另有约2.53μs参数归并，保存整链仍有19个设备任务。证据：[K256逐任务汇总](evidence/perf-k256-dhu-summary.json)、[BF16旧路径汇总](evidence/perf-bf128-dhu-summary.json)。BF16本项仅证明性能没有明显退化，不单独构成该 raw-safe 组合的精度验收。

Prepare/Dhu 分阶段45/100μs预算已达到；Finalize 300μs与整链500μs预算仍未达到。重算的 Recompute约1237μs、FwdH约1685μs尚未优化。packed及部分布局继续使用既有通用路径；本轮结果不能外推为这些路径的加速，也没有新的完整C32性能结论。未使用ATK，未提交或推送。

### 13.2 Finalize stage 重构的前一轮记录

同条件 BF16/K256、无状态、raw-safe 的旧16行版本主Finalize为1991.1605μs，整链2234.76μs；原BF16/K128主Finalize46.729μs。中点参考16行版本1513.5595μs，24行版本1057.2085μs；这些仅是调参结果，不代表stage流水已恢复。

本轮结构改动固定band参数，新增CG2 stage-major控制器、跨独立矩阵的两槽结果队列、复用F0 UB的四槽操作数缓存、按实际事务计数跳过危险band Cube运算。旧专用key保持原实现。当前UB缓存不能被描述为完整L1驻留，原目标中的跨work双generation尚未实现。

开发失败记录：最初同时放宽Cube内部三类等待的实验停在首个短尾块，已撤回；stage队列单独90例通过。双head公开验证首先发现head0 db错误，定位到ScoreBand::Beta写回源被下个head的Vector清零，补齐MTE3→Vector依赖后双head90组件例通过。随后预激活公开40例通过，raw-safe首例失败，定位到直接差值分支columns/columnStart传参颠倒，已修正；新增强gate、K60/K256双head定向4例通过。失败日志保留在A3 `/root/wys/fla-compat-d33`，不得用组件通过替代公开链路验收。

最终公开版本通过80组基本组合及8组H3/T129或1024的尾head、workspace复用组合，合计88 passed。上游原有13条warning保留，包括其允许warning的参数梯度比较；没有改动断言或容差。证据：[stagevec-upstream.log](evidence/stagevec-upstream.log)。新增强gate双head组件4例通过：[stagevec-test.log](evidence/stagevec-test.log)。未运行ATK。

固定CG2版本虽88例通过，但短例主Finalize为1610.8875μs，慢于24行旧调度1057.2085μs，故没有采用固定CG2。当前Host仅在配对后work数仍可铺满AIC时启用CG2，否则CG1；两者共享stage控制器。增加了仅动态队列使用的Fixpipe延后回收，以及F6指数差广播/F7整strip向量批处理。

A3设备14，同条件C64/T65/H2/BF16/K=V256/无状态/raw-safe，2次预热、每轮10次，按K256→K128交错执行三轮。主Finalize与整链分别统计，每行是该轮中位数，单位μs：

|轮次|K256保存Finalize|K256保存整链|K256重算Finalize|K256重算整链|旧K128保存Finalize|旧K128保存整链|
|---|---:|---:|---:|---:|---:|---:|
|1|792.2840|1039.4190|786.1545|3942.2210|46.4495|101.1470|
|2|787.6840|1027.5795|791.6945|3946.6710|46.5395|97.8675|
|3|787.5540|1033.2590|790.6640|3942.5415|46.9890|99.3490|

保存Finalize相对上一版24行1057.2085μs下降约25%，相对原raw-safe 1991.1605μs下降约60%。但仍为原K128专用Finalize约17倍，不能宣布性能泛化完成；300μs主Finalize/500μs整链预算仍未达到。当前片上复用主要为UB操作数缓存与梯度strip驻留，完整Akk/dW L1驻留、跨work双generation仍未恢复，且剩余GM打包/结果回读与Cube内部依赖等待仍需后续重构。

证据：[三轮逐阶段汇总及min/max](evidence/stagevec-performance.json)、[原始CSV](evidence/stagevec-raw-csv.tar.gz)、[源码及wheel SHA256](evidence/stagevec-provenance.json)。代码保留在本地feat/fla-compat工作区，未提交、推送或创建PR。

### 13.3 原高性能路径重构后的 C64 验收

A3设备14、CANN 9.3.0.P009，最终构建标识`predicate`。本轮从上节约788μs继续优化，性能目标始终对照原BF16/K128专用路径约46.5μs，不以慢通用路径作为达标基准。保持原FP32缓存、native FP32 MMAD（HF32关闭）、接口dtype和安全阈值32，没有用降精度换取性能。

实际接入：AIC直接GM→L1；八槽L1操作数复用；Assign结果由Fixpipe直接写最终内部面；CG1双record跨band执行下一块Cube与上一块Vector归并；dq/dk/dG按K面板驻留UB；两路key梯度在同一L0C累加；非对角矩形合并；原始gate单调性判定；对角块完整K面板重复判定并复用已读G；masked dS/dL×beta每work只生成一次；beta广播与行归约、F5固定树归约、norm关尾段8行批处理。CG2保留stage-major双head调度。公开参数归并仍为单独launch，未增加新的计算launch。

公开测试调用未经修改的fla-org `test_chunk`函数和断言：80组K=60/64/100/128/256、FP16/BF16、保存/重算、norm开关、预激活/raw-safe组合，加16组H=3、T=63/95/129/1024、FP16/BF16、保存/重算。**96 passed，13条上游原有warning**；包含上游本身允许warning的参数梯度检查，没有放宽任何容差。不是整个fla-org测试套件全部通过的声明。Prepare和Dhu共享消费者已重新构建后再做最终公开回归。未使用ATK。

补充开发检查：完整work90组、K60/K100/K256强gate6组、最终双head90组均通过。F5含M63等非整齐行数的108组组件检查也通过。组件仅补充同步/尾部定位，不替代上述fla-org结果。失败记录包括：最初批量尾段缺少一次Vector屏障导致dA偏差，已修复；梯度驻留时K100保护路径与安全路径列owner不同导致dq偏差，已统一分片。清零削减未形成可确认的独立性能收益，不将其单独宣传为加速来源。

A3同设备、C64/T65/H2/BF16/无状态/raw-safe/scale=0.37；每轮2次预热、10次设备采样，K256→原K128交错三轮。下表均为该轮中位数，单位μs；整链为每次调用全部设备任务时长之和：

|轮次|K256保存Finalize|K256保存整链|K256重算Finalize|K256重算整链|原K128保存Finalize|原K128保存整链|
|---|---:|---:|---:|---:|---:|---:|
|1|256.3750|493.9310|256.1650|3410.6625|46.9690|98.0980|
|2|257.3245|496.5805|256.6840|3412.9515|46.2090|96.3385|
|3|257.6450|495.2800|258.0850|3408.7425|46.7290|97.7485|

保存Finalize为同轮原K128的5.46–5.57倍，达到本轮4–6倍目标；相对上轮约788μs下降约67%。保存整链三轮均低于500μs，约为原K128保存整链的5.0–5.2倍。原K128性能无明显回退。阈值以多次采样的中位数判定，原始min/max均保留，不能声称每个单次样本都低于预算。

**范围限制：** 本结论只覆盖上述固定长度规格的C64保存主链和Finalize；重算整链仍约3.41ms，Recompute/FwdH尚是主要瓶颈。没有新的C32性能达标结论，也没有将结果外推到packed或所有布局。没有采集本轮逐指令timeline，因此不报告各流水单元利用率或推断的耗时占比。

证据：[上游96例](evidence/predicate-upstream.log)、[完整work](evidence/predicate-test.log)、[强gate及K100尾片](evidence/predicate-adversarial.log)、[双head](evidence/predicate-pair.log)、[F5组件](evidence/scalarcache-contributions.log)、[三轮逐阶段与min/max](evidence/predicate-performance.json)、[原始CSV](evidence/predicate-raw-csv.tar.gz)、[源码与wheel SHA256](evidence/predicate-provenance.json)。修改保留在本地`feat/fla-compat`，未提交或推送。
