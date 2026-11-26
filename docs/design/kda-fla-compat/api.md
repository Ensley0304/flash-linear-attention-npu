# KDA 第二批完整训练泛化：目标接口契约

版本 D3.2，2026-11-26。状态：目标接口保持；P10已有部分设备正确性证据，D3流水重设计尚未实现或验收。实现起点为本地 `feat/fla-compat` 的 `f2ed09dc9465fcc42e141560407c811e931811c7`。本文件是本批方案的接口事实源；不能用它替代当前已发布接口的支持声明。实现时将差异落回各受影响算子的 `docs/api.md`。

## 1. 本批范围

|项目|目标契约|
|---|---|
|平台|A2/A3（arch22）；A5保持既有分派和行为|
|计算链|C32前向/反向各单kernel，保存与重算均如此；C64维持组合链；不新增Recurrent/Graph/CP|
|chunk_size|C32融合前后向；C64既有独立阶段及组合链。独立阶段不新增C32公共能力|
|K/V|60/60、64/64、100/100、128/128、256/256、64/32|
|head|HK=HV=H；新增泛化路径不扩 GVA，既有 GVA 专用路线保留|
|主张量|q/k/v 同类型 FP16 或 BF16；o/dq/dk/dv/dO 同主类型|
|gate/beta|各自可为主类型或 FP32，梯度跟随各自原输入；不开放 FP32 q/k/v|
|布局|复用 BSND/BNSD/NTD/TND；packed B=1，Host offsets；Graph设备offsets不在本批|
|训练状态|可选 FP32 h0、FP32 dht、FP32 dh0；两种状态物理布局|
|gate模式|预激活 log-step、raw safe、raw softplus；raw模式允许 A_log 缺省，dt_bias 可选|
|其它组合|norm 开/关；有效 beta、sigmoid、2×sigmoid；保存/重算；exp/exp2单位一致|
|验证|fla-org 原测试及其标杆；不使用本地 ATK；A2/A3任一平台通过即可；本轮以A3验收（用户2026-11-25更新）|

标量 scale 沿现有显式 ACLNN 参数传递；fla-org 测试入口缺省值仍由已有语义适配层提供。此次不更改 C ABI 参数数量/顺序，不为可省略 scale 引入 NaN sentinel。完整 T=0 张量入口不纳入本批；允许 T>0 packed 中任意数量的空序列。

## 2. 层级和分派

Python 仅传参、路由和按输出描述分配；规格检查由 ACLNN 完成。不得在 Python 做 gate 激活、状态转置、默认参数 Tensor 合成或用其它计算后端回退来代替本批能力。现有返回 tuple 的槽数、顺序保持；没有输出的槽继续为 None。

A2/A3的C32调用统一走新增arch22融合入口：前向一次kernel、反向一次kernel，disable_recompute两态一致。现有C32 fused仅arch35/A5，是结构参考，不能称为A2/A3已有支持。其外部Exp等处理不得复制到新增单launch路线。norm/gate、dtype/layout处理、重算、状态和最终参数归并全在对应kernel内。C64保持原分阶段链；A5保持原行为。

前向沿既有输出档位导出训练缓存，普通推理不强制保留整套缓存；无论输出档位如何均一次计算launch。训练适配层必须请求匹配的Aqk/Akk及保存量；不得按输入形状猜测是否训练。测试用profiler计完整公开调用内计算kernel，不能只数命名包含Kda的任务。

反向由缓存 dtype、C、维度和完整标志组唯一选择新/旧路径。旧路径不支持的新状态/模式必须转新优化链，不能先命中旧 BF16/128 分支再报错。ACLNN 单独核验 Aqk/Akk 末维等于 C，不能用 C64 缓冲伪装 C32。

## 3. 状态、别名与可选量

序列数 S：dense 为 B；packed 为 len(cu_seqlens)-1。FP32 状态 shape 为 [S,H,K,V]，V-first 为 [S,H,V,K]。h0 为只读；ht 与 h0 不共享可写存储。已有 tuple 中透传 h0 的槽仍透传原对象，绝不把该槽当作 ht。通用训练链禁止输入输出状态重叠；不改变旧推理原位接口的既有约定。

|条件|前向|反向|
|---|---|---|
|h0 缺省|从零开始|公开 dh0 槽为 None；允许 dht 非空且传播到其它输入|
|h0 非空|读取对应序列状态|仅请求 dh0 时分配、写回 FP32 dh0；不请求也不影响其它梯度|
|output_final_state=False|不写 ht|dht 必须为空；训练封装沿用前向开关|
|dht 缺省|不适用|终端反向状态为零|
|空序列|ht=h0；无 h0 则 ht=0，无 chunk 快照|dh0=dht；无 dht 则 0；无 token/参数梯度贡献|

组合 ACLNN 单独调用没有前向 output_final_state 属性，故仅校验 dht 的描述符；“未返回 ht 时不能传 dht”由保存前向上下文的训练适配契约保证，不添加无从核实的 Host 校验。

qhat/khat、G、W/U、qg/qg_scaled/kg、Aqk/Akk、Vnew、h、dh 全部 FP32。h 是入口快照，dh 是更新前的末端梯度快照；运行状态 FP32，WH 使用相同 FP32 快照。新增训练 ht/dht/dh0 均 FP32，独立旧接口其它 StateT 保持原路由，不外推为新训练支持。

raw gate 的 A_log=None 等价 a=1；dt_bias=None 等价 0。kernel 用缺省分支读取常量，不要求 Python 创建零向量。缺省 A_log 时不输出 dA；缺省 bias 时不输出 dbias。预激活 gate 不消费 A_log/bias，原合法调用中的冗余参数不产生这些梯度；检查沿既有接口兼容策略，禁止偷偷改变预激活数学定义。

use_gate_in_kernel=False 时 g 是每步自然对数衰减，dg 是该量梯度；safe_gate 不触发第二次激活。raw safe 时 lower_bound ∈[-5,0)，默认 -5；raw softplus 不使用 lower_bound。norm 关闭时 rstd 缺省且不读取；norm 开启时保存 qhat/khat/rstd 的同一份值。sigmoid 关闭而 allow_neg_eigval=True 由 ACLNN 拒绝。

## 4. 缓存形状和保存责任

将主张量规范化为 head-major。dense 的 key cache=[B,H,T,K]、value cache=[B,H,T,V]、score cache=[B,H,T,C]；packed 分别为 [H,T,K/V/C]。标量缓存去掉末维；每条记录按实际逻辑维数寻址，不以最大256作为GM stride。

N_c=Σ_s ceil(L_s/C)，packed 状态快照 [N_c,H,K,V]，V-first 交换末两轴；dense 为 [B,ceil(T/C),H,K,V]。每序列 chunk 前缀保持原序列 id；空序列不生成 chunk，但拥有状态槽。

|模式|必须保留到反向|反向重新生成|
|---|---|---|
|保存|Aqk/Akk、G/W/qg/kg/Vnew/h；norm开启保存qhat/khat/rstd，关闭必须保存原始q/k；beta_eff（sigmoid）；原始g/beta/v、可选参数和h0上下文|仅阶段局部 Rk/Rv 等，不重复激活|
|重算|Aqk/Akk；qhat/khat/rstd（norm）；beta_eff（sigmoid）；原始输入/可选参数/h0/属性/offsets；预激活G按既有重算契约保留|G/W/U/qg/kg、h/Vnew；使用保存Akk，不重新求逆|

q/k反向输入在保存与重算两模式下都必须由训练上下文持有：norm开时消费前向FP32 qhat/khat和rstd；norm关时持有原始q/k并在核内提升。不能依赖调用者保留外部引用，不能从qg/kg除以指数恢复原输入。使用save_for_backward保存实际所需Tensor及版本信息；保存后的原始输入若被原位修改，应按框架版本检查报错，不读取静默变化后的值。

|门控对象|单位与形状|保存/重算生命周期|
|---|---|---|
|原始g|use_gate_in_kernel=False时为自然对数log-step；否则raw logits；原输入layout/dtype|保留原描述符用于dg dtype；raw模式保留值用于激活导数；不能绑定为累计G|
|累计G，即返回量gk|head-major FP32；exp模式G_i=Σ_{r=0..i}delta_r；exp2模式G_i=(1/ln2)Σdelta_r；每个chunk从0重置，实际FP32执行逐步delta换底再累加|保存模式由前向gk槽保留至反向末次消费；预激活重算也保存此槽；raw重算可不保存，由kernel重建|

gk在现有12槽基础返回tuple中为索引2，Aqk/Akk为索引3/4；训练适配层使用具名解包并测试槽位，扩展保存值接口不能混淆索引。读入G后不再次cumsum、不再次乘1/ln2；E导数乘b（ln2或1），回到原log-step的逆扫乘c_b（1/ln2或1）。预激活模式dg是原自然对数log-step梯度，与缓存单位不同。

C32融合kernel直接消费safe_gate/use_gate_in_kernel等属性，核内GateEval处理raw softplus，不调用独立Recompute或GateNormalize L0；kernel内部恢复G、W/U和状态。独立Recompute公共签名不扩展。C64若复用已有Recompute原型，则按其原接口单独处理，不能将该C64调用链用于C32。


## 5. ACLNN 校验与错误

检查平台、完整支持域、非空必选张量、rank/layout、各head和K/V一致、状态S维和布局、C32/64、缓存C维、rstd成对、beta_eff、各原始梯度dtype、连续性、合法标志组合、有限scale及相关scalar、Host offsets非递减且首0末T、canonical chunk_indices、64位分配/寻址溢出。

状态错误不得在 Python assert；延续项目结构化日志和 ACLNN_ERR_PARAM_INVALID/161002。检查标量不代表遍历设备Tensor校验有限性；任意输入的数学或FP32溢出不做无声裁剪。

## 6. 本轮文件与证据

旧设计：[D1.3](../KDA-算子内部泛化设计.md)。设计起始基线的差异位置（不是当前支持状态）：`chunk_kda_bwd/op_host/op_api/chunk_kda_bwd_general.h` 第31行拒绝状态，第33行拒绝raw non-safe；其组合调用硬编码64。FwdH/Dhu general kernel已有状态读写但遍历步长固定64。工作簿“接口参数对照”15/19/20/22/24–26/30–32/42–44/52/55/58/60/62/68–72行为本批差异依据。

## 7. 逐入口交付边界

|入口/组件|本批交付|
|---|---|
|公开npu_chunk_kda_fwd / ACLNN V2|新增A2/A3 C32融合前向及训练缓存；C64状态/gate契约保持配套|
|公开npu_chunk_kda_bwd / 组合ACLNN|新增A2/A3 C32融合反向（保存/重算单launch）；C64新增状态、V-first、raw softplus、可选参数|
|独立FwdPrepare/FwdFinalize/BwdPrepare/BwdFinalize/Recompute|不新增C32公共支持；本批需要的C64模式/optional输入消费者适配与直调回归|
|独立FwdH/Dhu|保留既有C域，本批补齐C64组合状态接线及布局直调回归；不据融合内函数宣称新增C32独立接口|
|C32内部Prepare/FwdH/Dhu/Finalize设备函数|共享数学实现和arena view，不是独立L0或公共入口|
|C64内部GateNormalize|仅组合重算的内部L0步骤，无Python公共入口，明确允许额外launch|

此表消除“独立阶段与组合链使用同一C”的扩大承诺。独立C32公共接口另列需求；这不减少C32融合前反向的既定输入/功能范围。

## 8. 当前实现与D3目标的区别

2026-11-26历史实现状态：C32反向已接入正式MIX_AIC_1_2入口，B1/B2用FP32 Cube，保存/重算共用一个kernel；前向key0保留AIV_ONLY。公开接口不变。P8原上游36项与扩展154项通过且九组单kernel验证通过；P9新增Dhu后扩展162项通过，最新主回归/性能以validation记录为准，不能将接口支持等同全规格性能达标。

P10最终包：原上游36项+扩展162项通过，安装后两节点复验通过；B1/B2 Cube输入已批量打包。当前性能限制见validation，未全规格超过C64。

D3只改变实现设计，不增加接口范围。现有AIV_ONLY前向、串行owner及部分Cube反向不是D3交付方案；新C32前反向均设计为两AIV真实参与的MIX流水，C64新规格采用高性能分阶段模板。CG与双页、参数partial及同launch归并以design.md为准；这些内部变化不增加公共参数。旧专用key的缓存精度合同保持，新增FP32合同不能通过隐式降精度适配旧消费者。性能判据以D3设计和验证计划为准，不能引用慢generic C64作为合格基线。

D3.1实施优先级：先设计/实现C32 disable_recompute=true，固定CG1、单head跨chunk流水；多head交错后续优化。disable_recompute=false的一次kernel合同不变，但本轮不宣称其新流水详设完成。接口范围不因实施顺序缩小。
