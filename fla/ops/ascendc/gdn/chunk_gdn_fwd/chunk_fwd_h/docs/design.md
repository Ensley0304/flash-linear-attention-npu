# ChunkFwdH 设计

## 2026-11-27：P32 K256独立分派增量与验收配置

规则版本V2。本增量按用户指令接入已验收原型commit
`557667b90fd21fe7dc62f9450d0d5918d7ddab22`；对应A2原型库SHA256
`1aecef5dfe79fa9749c555825e1242217dec32a445ac770505d874773d5bb74a`。
不是旧窗口的KDA_C64_FAST。此前整链预算失败仍有效；本次验收范围为新分派、
正式调用链的功能兼容性与FwdH阶段性能，不宣称仅此即可使完整前向超过Triton。

### 分派表与兼容性

新增模板BOOL `KDA_P32_256`，放在GENERAL_PATH之后，默认/全部旧选择均为0。
保持原字段顺序、位宽和取值列表；通过真实CANN GET_TPL_TILING_KEY编码器枚举
旧组合并比较新字段=0时结果，而不是仅比较源码。新字段=1只选择FP32 gate、
USE_GK=1、STATE_FP32=1、STATE_V_FIRST=0、DENSE_T16=0、GENERAL_PATH=0，
USE_EXP2/STORE_FINAL各两种，kernel type显式MIX_AIC_1_2。CANN编码器实测key：
exp/final=false为68126，exp2/final=false为69150，exp/final=true为76318，
exp2/final=true为77342；新位是bit16。144个旧选择与新位0逐项相等。
生产代码不得硬编码。原V_DIM=128继续作为历史模板token（现通用路径也使用它）；
真实K/V由tiling数据传递，新路径资格严格要求256，不修改此列表以免影响旧key。

| USE_EXP2 | STORE_FINAL | 生成key | kernel类型 |
|---:|---:|---:|---|
| 0 | 0 | 68126 | MIX_AIC_1_2 |
| 1 | 0 | 69150 | MIX_AIC_1_2 |
| 0 | 1 | 76318 | MIX_AIC_1_2 |
| 1 | 1 | 77342 | MIX_AIC_1_2 |

| Host输入条件 | 分派 |
|---|---|
| DAV_2201；g缺失/gk存在；C64；K=V256；HK=HV=1或2；B1；1<=T<=4096；内部BNSD；无cu_seqlens/chunk_indices；k(实际kg)/w/u/gk/H/Vnew均FP32 ND；initial若有为FP32，final若请求为FP32；stateVFirst=false；saveNewValue=true | KDA_P32_256=1，Gh1/Vtile64，MIX_AIC_1_2 |
| 其他合法FP32缓存规格，如H3、B2、T>4096、BF16 state、stateVFirst=true、varlen | 原GENERAL_PATH=1，AIV_ONLY |
| 原BF16 K128，包括KDA、GDN、dense/varlen及旧layout | 原模板与kernel映射，KDA_P32_256=0 |
| 非DAV_2201 | 新资格恒false；旧合法K128仍走原分支，旧契约不支持的FP32 K256仍报错 |
| 现有非法输入 | 继续由ACLNN/Host按原契约报错，不由Python补校验 |

初始/最终状态四组合及exp/exp2已在A2公开入口矩阵验证；4个冒烟与40个矩阵用例均核对
实际Host key和runtime launch。B/H/T白名单边界在Host做路由资格判断，不扩大公开接口合法范围。

KDA/GDN实际区分依据：ChunkKdaFwdV2与ChunkKdaBwdGeneral的recompute均调用
l0op::ChunkFwdH(kg,w,u,nullptr,G,...)；GDN公开链调用
l0op::ChunkFwdH(k,w,u,gCumsumBht,nullptr,...)。因此gk语义、g缺失与FP32缓存共同
限定KDA类递推；不能只看K/V。独立入口传相同参数时执行同一逐K门控数学语义，
不额外引入调用者名称属性或改变ABI。无需修改KDA/GDN调用点及Python。

### 资源与接口绑定

直接复用557667b9的Cube/Vector工作体，独立生产launch包装在arch22目录。
启用right驻留与Gh1 D分块流水。1AIC:2AIV；AIV0计算，AIV1为同步dummy。
tasks=B*H*4，blocks=min(tasks,AIC核数)，当前B1/H1或2分别4/8 blocks。
每block scratch=98304字节（P16384+right16384+D65536）；总用户scratch为
393216/786432字节，另加CANN libApi workspace（本次A2为16777216字节）；
实际总分配分别17170432/17563648字节，用户区偏移为0。
复用既有tiling的vWorkspaceOffset表达scratch起点，不改变tiling struct布局。
该偏移相对用户区为0：asc_opc生成入口已将workspace参数推进CANN保留区大小，
Host总分配仍包含保留区，但kernel不得再次加上该前缀。可选gk描述使用GetOptionalInputDesc按IR索引读取，
required输入使用GetRequiredInputDesc，避免缺省输入导致实例索引错位。
L1显式81920字节；L0A/B/C各32768；UB显式99328字节加TPipe元数据；
物理分区和事件账本沿用tests/fwdh256/README.md。H快照、Vnew及final直接写FP32输出，
不新增转换launch，不复用低精度运行状态。正式调用的既有转换/搬运仍纳入总计时。

| 物理资源 | 冻结配置 | 生命周期/用途 |
|---|---|---|
| R_run | 每活跃AIV 65536 B，FP32 UB | 完整chunk链驻留，禁止逐chunk BF16量化 |
| H_snapshot / H_cube | FP32 GM，同一快照存储 | snapshot按[B,NT,H,256,256]输出，Cube直接读取 |
| H_final | FP32 GM，可选 | [B,H,256,256]，只在请求时写回 |
| UB显式分配 | 99328 B/AIV，另加TPipe元数据 | state/gate/broadcast及双缓冲队列；AIV1为同步dummy |
| L1 | 81920 B/AIC | 含16384 B的S2 right驻留区 |
| L0A / L0B / L0C | 各32768 B/AIC | MMAD完整归约，不落GM部分和再串加 |
| scratch | 98304 B/block | P 16384 + right 16384 + D 65536 |
| blocks | H1为4，H2为8 | 每head四个V64 tile，Gh1，1AIC:2AIV |

生产可执行逻辑提交为`3aef396858ed46fdc52d77c23db30bd85664661d`；kernel工作体仍为
557667b9。完整构建后仅增量重建修正偏移的Host并重新打包，所有kernel对象hash不变。
wheel SHA256：`9e5a09caf5be5e91eea20223ea9a1db2d19a8fbda1f3f1e28384315ed208ec1a`。
这冻结的是已验证分派/精度配置；长序列超过Triton的性能目标仍未达成。

### 实际路由与回归矩阵

1. key兼容：旧128个BF16/FP32 gate布尔组合与16个GENERAL组合，新增位0逐项相等；
   新4个key唯一、与旧集合无交集；核类型核对生成JSON和实际任务。
2. 资格单测：逐项翻转arch、gate语义、chunk、K/V、dtype、layout、B/H/T、state、varlen，
   对生产资格函数断言；非目标架构仅Host模拟时须明确标记，不能冒充该设备运行。
3. 独立公开FwdH：T1/7/63/64/65/129/1024/4096，H1/2，exp/exp2，I/F四组合，
   FP32 reference与原阈值；读取日志中的key/block/workspace及profiler任务证据。
4. KDA正式前向与重算：原fla-org test_chunk，T65/H2、T1024/H1及尾块，保存/重算，
   原始输出/梯度断言。确认两个调用链实际都出现新key。
5. 反向路由：K128 KDA、K128 GDN、K256 H3/B2、BF16 state、stateVFirst、varlen；
   合法走旧分支，非法维持原错误。代表用例用原fla-org断言，不用ATK或放宽容差。
6. 同卡同输入10预热/10采样：正式FwdH阶段与原型、同规格Triton FwdH比较；
   同时保留公开调用全部设备任务CSV，不剔除转换/搬运。记录加载库/OPP hash和key。

共享Host/key/kernel入口文件单独提交；其他工作树未提交内容不复制、不覆盖。
检查通过按用户最新指令推送本分支，不创建PR。尚未验证的项目如实列出。

以下章节保留原K128路径的设计说明；P32 K256使用上方独立分派、资源和测试范围。

## 1. 原K128路径的目标与边界

`ChunkFwdH` 是独立的 Ascend C 算子，符号为 `ChunkFwdH`、`aclnnChunkFwdH` 和
`fla_npu.ops.ascendc.chunk_fwd_h`。它不复用、不修改 `ChunkGatedDeltaRuleFwdH`，也不提供
legacy `torch.ops.npu` 兼容入口。

当前支持 A2、A3、A5，固定 `K=V=128`、`chunk_size=64`，`k/w/u` 为 BF16。gate 和
state 分别支持 BF16/FP32，`state_v_first=true/false` 均由 kernel 原生处理。公开 tensor
descriptor 使用非私有 ND；输入可由 host 连续化，直接写入的输出必须连续。

## 2. 计算语义

记 `E(x)=exp(x)`，`use_exp2=true` 时为 `exp2(x)`。每个 value head、每个 chunk 执行：

```text
H_c = cast_BF16(R_c)
Pacc_c = W_c @ H_c
P_c = cast_PType(Pacc_c)  # A2 scalar-g preload 固定 FP32；其余路径随 StateT
V_new_fp32_c = fp32(U_c) - fp32(P_c)
V_new_c = cast_BF16(V_new_fp32_c)

g-only:
  V_new_g_c[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32_c[i,:])
  D_c = k_raw_c^T @ V_new_g_c
  R_next = E(g_last) * R_c + D_c

gk-only:
  D_c = kg_c^T @ V_new_c
  R_next[k,v] = E(gk_last[k]) * R_c[k,v] + D_c[k,v]
```

最终 chunk 不请求 `final_state` 时，只生成当前 `H` 和 `V_new`，跳过 Stage2/Stage3。

## 3. head round 规划

规划在 chunk 循环前完成。每个 AIC round 最多四个 head：AIV0 处理 round head 0/2，
AIV1 处理 1/3；两个 AIV 各有 local slot 0/1。

g-only 先按 `group_size=HV/HK` 划分共享 raw K 的 value-head 组：

- `HK:HV=1:3`：每轮三个 value head，一个 kg slot，只读一次 raw K。
- `HK:HV=1:2`：每轮四个 value head，两个 kg slot，各 raw K 只读一次。
- `HK:HV=1:6`：拆为 4+2 两轮；第二轮重新读取同一个 raw K，不跨 round 保留。

gk-only 每个 value head 对应自己的 prepared kg；本轮按 active head 数读取 1..4 份。
`FwdHHeadRoundPlan` 显式记录 `head -> hv -> kh -> kgSlot -> aiv -> localSlot`，Stage2 只遍历
`requiredKhCount`，不预留或加载 16 份 kg。

## 4. 分阶段实现

### S-1：FP32 初态转换

仅 `initial_state` 为 FP32 时执行 `H0=cast_BF16(initial_state)`。A2/A3 逐 tile 处理；A5
每个 head 使用一次完整 RegBase VF，并为两个 local head 分配独立 FP32 input/BF16 output
bank，以“预取当前 head、计算前一 head”的循环实现 ping-pong。全部 active H MTE3 drain
后统一发布本 phase 的 `H_READY`。

### Stage0：入口状态与第一次矩阵乘

AIC 将当前 `W[M,K]` 与 `H[K,V]` 从 GM 搬到四个 round-head L1 槽，随后执行
`P=W@H`。Stage0 不读取 kg/k_raw。tail chunk 先用 MTE2 `InitConstValue` 清零当前 W 槽，
再覆盖有效 ND 行；Cube M 取 `AlignUp(valid_tokens,16)`。

`W`、`U` 和 `g/gk` 的 GM 数据在一次算子执行中只消费一次，GM 读取使用 L2 cache bypass，
避免流式数据挤占递推数据的缓存空间。若当前 work unit 完整覆盖一个 key head 的全部
value-head consumer，则该份 `K` 只从 GM 读取一次并使用 bypass；consumer 跨 work unit 时
保留默认 L2 策略。`initial_state`、递推 `H` 和 Stage1 `right` 存在跨 AIV/AIC 的重复读取
或刚写即读关系，保留默认 L2 策略。

- A2/A3：L0C 经 Fixpipe 将对齐后的 M 行写入 P GM scratch，补齐行恒为零；AIV 再以
  MTE2 只读取 `valid_tokens` 个有效行。
- A5：L0C 经 Fixpipe 直接将有效行写入配对 AIV 的 local UB slot。

### Stage1：向量修正

AIV 读取 `U` 和 P，计算 `V_new`；g-only 同时生成带相对 gate 的 BF16 right，gk-only
直接以 BF16 `V_new` 作为 right。`V_new` 写 GM。right 必须经 MTE3 写 GM workspace，禁止
UB 直搬 L1。

A5 从 MTE2 后开始的全部向量计算由 RegBase VF 完成。g-only 的完整 chunk 先将 64 个
`exp(g_last-g_i)` 一次性向量化写入 gate-scale bank，Stage1 逐行广播该结果，不再重复执行
64 次向量 Exp；tail chunk 保留标量 gate 读取和单元素写回，避免读取 DataCopyPad 未初始化的
UB 余量。VF 只使用 `if constexpr` 模板分支和明确类型的 `for` 循环，并以两组 FP32 寄存器
覆盖 128 列。

### Stage2：第二次矩阵乘

AIC 此时才读取本 round 实际需要的 kg/k_raw，并等待每个 head 独立的 `RIGHT_READY`，把
GM ND right 搬为 L1 Cube 输入。计算 `D=k^T@right`；`state_v_first=true` 时使用等价转置式
`D_physical=right^T@k`，直接生成物理 `[V,K]`。tail chunk 对当前实际 kg/k_raw 槽和各 head
right 槽分别清零后再覆盖有效数据，Cube K 取 `AlignUp(valid_tokens,16)`。

- A2/A3：L0C 经 Fixpipe 写 D GM scratch，AIV MTE2 读取。
- A5：L0C 经 Fixpipe 直接写配对 AIV UB。

### Stage3：递推状态更新

AIV 用 FP32 算术执行 `R_next=decay*R+D`，按 StateT 保存 BF16 或 FP32 rolling state，并按需
写下一 chunk 的 H 或 final_state。A5 使用独立 RegBase VF，不调用 A2/A3 向量实现。

## 5. 存储布局

历史 H 使用 chunk-major：dense `[B,C,HV,K,V]`，公开 packed `[1,total_chunks,HV,K,V]`。
矩阵基址为 `((b*C+c)*HV+hv)*K*V`，packed 为 `(global_chunk*HV+hv)*K*V`；
`state_v_first` 继续只控制矩阵内部 K/V 顺序。A2/A3/A5 的 Cube/Vector 共用
`FwdHHOffset`，初始写出、下块写出和读取同时切换，状态递推计算顺序不变。

AIC L1 固定分区：W `[0,64) KiB`，保留空洞 `[64,128) KiB`，H/right `[128,256) KiB`，
kg `[256,320) KiB`。kg 区最多四个 16 KiB slot；每个 round 只占用 `requiredKhCount` 个。

A5 的 FP32 state 路径将物理数据槽和逻辑 head slot 分开管理。每个 AIV 的静态 UB 布局为：

| 地址范围 | 用途 |
| --- | --- |
| `[0,64) KiB` | 两个 local head 串行复用的 P/D 数据槽；P 占 `[0,32) KiB`，right 占 `[32,48) KiB`，D 占整个 64 KiB |
| `[64,128) KiB` | local slot 0 的 FP32 rolling state |
| `[128,192) KiB` | local slot 1 的 FP32 rolling state |
| `[192,208) KiB` / `[208,224) KiB` | 两个 head 的 BF16 U/V_new work bank |
| `[224,226) KiB` | gate、gate-scale 和 alpha bank |

四 head work unit 中，AIV0 处理 round head 0/2，AIV1 处理 1/3。每个 AIV 的两份 64 KiB
state 分别按逻辑 local slot 0/1 绑定，并在整个 chunk 循环内常驻；仅首块读取 initial state、
末块写 final state，不再逐 chunk 经 GM workspace 回写和恢复。P 与 D 只使用物理数据槽 0，
按 `P0 -> P1 -> D0 -> D1` 的生产消费顺序串行复用，不增加 head round。

`StateBf16Slot(0)` 的 `[128,160) KiB` 地址与第二份 FP32 state 的低半区重叠，但两个生命周期
互斥：它只在“首 chunk、无 initial state”时临时生成全零 H0；该次 H0 MTE3 完成后，Stage3
才初始化两份 FP32 state。存在 FP32 initial state 时先在 `[64,192) KiB` 完成 S-1 转换，
Stage1 的 `ZERO_STATE=false`，不会访问该 BF16 地址别名。

A5 的 BF16 state 路径维持两个原始 local slot。A2/A3 使用两个 32 KiB tile local slot 和
两个 32 KiB BF16 state slot，P/D 使用每核每 round-head 独立的 GM scratch。

A5 的 `gk` 路径在 `state_v_first=true` 时，物理 state 为 `[V,K]`，所有 V 行使用
同一条 128 维 `E(gk_last)`。Stage3 在行循环前按原有指数公式计算一次，保存在一对
FP32 向量寄存器中供 128 行复用；不改变 state 更新、BF16 舍入和写回顺序。
`state_v_first=false` 仍按当前 K 行加载对应 gate，scalar-g 路径保持原有 alpha 复用。

## 6. 同步协议

A5 每个 local slot 分别维护 `P_READY/P_FREE`、`D_READY/D_FREE`、
`RIGHT_READY/RIGHT_FREE`、`H_READY`。A2/A3 mode2 是 `AIC + 2*AIV` 集合同步：每个
pair 由 AIC set/wait 一次，两个 AIV 对同一 ID 各 wait/set 一次；尾 pair 缺 head 的 AIV
执行 dummy 同步但不访问数据。ready 由真实生产 pipe 发布，free 由最后消费者发布；同一
slot/pair 的事件复用前必须完成上一代 wait。

A5 FP32 state 的两个 local head 虽然共享一个 P/D 物理槽，跨核 flag 仍按逻辑 local slot
0/1 区分，ready/free 的 ID 和 set/wait 次数均不改变。Stage1 可以在 V pipe 完成后尽早发布
`P_FREE`，因为后一个 P 只覆盖低 32 KiB；但 D 会覆盖完整 64 KiB，必须等第二个 head 的
right MTE3 读完高 32 KiB。为此，同一 AIV 的两次 `RIGHT_READY` 保持原 ID 和次数，由
`PIPE_MTE3` 在两个 Stage1 都下发后统一发布。AIC 在搬入第一个 right 前等待 local slot 0 的
`RIGHT_READY`，因此首个 D 的 Fixpipe 不会与第二个 right 的 MTE3 形成 WAR。单 head 和
BF16 state 路径仍在各自 right 写回后立即发布 ready。

Stage 内按核内 head id 统一写一套流程，`headId&1` 选择 ping/pong L0 slot。当前 slot 的
MTE2 完成即可启动该 slot 的 VEC/Cube，不等待另一 slot；Cube->Fixpipe 和 VEC->MTE3 同理。

A5 的 FP32 scalar-g 单 head work unit额外启用跨 chunk lookahead。AIC 将 W 放在 L1 slot
0/1、K 放在 slot 2/3，按 chunk 奇偶轮转；当前 right 的 GM->L1 已下发后，立即预取下一
chunk 的 W/K，使下一块 MTE2 与当前 MTE1/MMAD/Fixpipe 重叠。AIV 同样以两个独立 input
bank 轮转 U/g，在消费当前 bank 前先下发下一 bank；Work bank 由 `MTE3_MTE2` free credit
保护，gate/alpha bank 由最终 Stage3 V consumer 发布 `V_MTE2` free credit。P/D/right/state
仍使用原 local slot 和跨核 ready/free 协议，递推 H 仍严格等待上一 chunk 的 `H_READY`。

A2/A3 在无初态、多 chunk 场景为第二个 chunk 的 P scratch 预置一次 free credit，因为首
chunk 没有 Stage0/P；后续 credit 由前一 chunk Stage1 产生。round 结束时 P 与 D 两条独立
scratch 链分别回收，不能用互斥分支漏掉其中一条。

跨 work unit 使用双向收口：两个 AIV 等本 unit 的 MTE3 全部完成后发布 `ROUND_DONE`；AIC 收到
完成信号并回收 P/D/right 后才回 ACK。A5 的 DONE 由 `PIPE_MTE3` 发布，DONE wait 和 ACK
使用 `PIPE_S` 作为 scalar/control gate；下一 unit 的 kg/H/W MTE2 因而不能越过 ACK，
不会与上一 unit 的未完成写回交叠。A2/A3 用 mode2 collective 完成同等的收口。

Host 将 `(sequence, value_head)` 展平为连续 head task。设有效 sequence 数为 `N`、物理 AIC
核数为 `C`，则 `totalHeadTasks=N*HV`、`headsPerCore=ceil(totalHeadTasks/C)`，实际启动核数为
`blockDim=ceil(totalHeadTasks/headsPerCore)`。第 `coreIdx` 个 AIC 处理连续区间
`[coreIdx*headsPerCore, min((coreIdx+1)*headsPerCore, totalHeadTasks))`，不再把同一 sequence
的全部 head round 绑定到一个核。核内区间跨越 sequence 边界时必须拆分 work unit；同一
sequence 内也按连续 value head 每四个一组拆分，因此每个 unit 最多四个 head。每个 head
的全部 chunk 递推仍固定在同一核并顺序执行，不依赖 block 启动顺序建立状态依赖。

A2 scalar-g 的 BF16 state 复用 gated preload，其 P workspace 固定为 FP32。
FP32 state 使用通用路径保留完整 rolling state，避免 preload 复用 BF16 H 快照导致
状态精度损失；host 的 `gatedEligible` 与 kernel 的 `FwdHGatedReusable` 保持相同条件。

## 7. 变长序列

变长模式要求 BNSD 容器 `B=1`。`cu_seqlens` 从 0 开始、以 T 结束且严格递增；
`chunk_indices` 若存在，必须是 sequence-major canonical `(seq_id, chunk_id)`。连续 head task
区间跨 sequence 时按边界拆为独立 work unit；同一 sequence 的不同 value head 也可分配到
不同核。state/final_state 的首维是 sequence 数，H 的 chunk 维使用全局 chunk 前缀，各 head
的 GM 输出区间互不重叠。

## 8. TilingData

`ChunkFwdHTilingData` 只保存 host 已校验并由 kernel 直接消费的数据，不在 tiling 中保存每个
head round 的展开数组。kernel 在进入 chunk 循环前，按 `kNumHead/vNumHead` 生成
`FwdHHeadRoundPlan`，因此每个 round 的 active head 数、实际 Hk 数和 head-to-kg-slot 映射
不会随 chunk 改变。

| 字段 | 语义 |
| --- | --- |
| `batch` | dense 时为逻辑 batch 数，varlen 时为 sequence 数 |
| `seqlen` | k/w/u 容器的 token 维长度 |
| `kNumHead` / `vNumHead` | Hk/Hv 数；g-only 要求 `vNumHead % kNumHead == 0` |
| `kHeadDim` / `vHeadDim` | 当前均固定为 128 |
| `chunkSize` | 当前固定为 64 |
| `useInitialState` | 是否读取 initial_state |
| `storeFinalState` | 是否生成并写 final_state |
| `isVariedLen` | 是否启用 varlen 调度 |
| `shapeBatch` | dense 的物理 batch；varlen 固定为 1 |
| `tokenBatch` | varlen 的 sequence 数；dense 固定为 1 |
| `vWorkspaceOffset` | A2/A3 P 的 GM scratch，按 FP32 slot stride 预留 `[blockDim,4,64,128]`；实际元素为 PType |
| `vUpdateWorkspaceOffset` | Stage1 BF16 right 的 GM workspace，形状为 `[blockDim,4,64,128]` |
| `kDecayWorkspaceOffset` | FP32 rolling state 的 GM workspace，形状为 `[blockDim,4,128,128]`；A5 使用每 AIV 两份常驻 state，不访问该段 |
| `hWorkspaceOffset` | A2/A3 D 的 FP32 GM scratch，形状为 `[blockDim,4,128,128]` |

workspace 的 core 维使用实际 `blockDim`，各段按 512 Byte 对齐，并在 CANN lib-api workspace
之后额外保留运行时安全区。
A5 的 P/D 走 L0C->AIV UB，不消费对应的 P/D GM scratch；为保持跨架构统一 tiling，offset
仍由 host 生成，但 A5 kernel 不访问这些地址。

## 9. 模板参数与 TilingKey

`ChunkFwdH` 的 kernel 全局入口按以下顺序声明六个编译期模板参数：

1. `D_T_G`：g/gk dtype，支持 BF16、FP32；
2. `V_DIM`：value head dim，当前仅注册 128；
3. `USE_GK`：scalar-g 或 key-gk；
4. `USE_EXP2`：exp 或 exp2；
5. `STATE_FP32`：rolling/final state 为 BF16 或 FP32；
6. `STATE_V_FIRST`：state 布局为 `[K,V]` 或 `[V,K]`。

模板声明共注册 `2 x 1 x 2 x 2 x 2 x 2 = 32` 个可达实例。host 在完成 dtype、shape、
gate 模式和 state 输出校验后，以完全相同的参数顺序调用 `GET_TPL_TILING_KEY`；kernel 直接由
对应实例构造 `GateT`、`FwdHCompilePolicy` 和 state offset，不再读取 tiling 数据做 gate mode、
指数函数、state dtype 或 state layout 的运行时分派。

state dtype 的模板选择保持运行语义优先级：存在 `initial_state` 时取 initial dtype；无 initial
但写 final state 时取 final output dtype；二者都不存在时 rolling state 固定选择 FP32。
`STATE_V_FIRST` 同时用于编译期 state 地址映射，避免热路径中的 layout 条件分支。

代码不维护手写数值 key，也不在 kernel 内使用 `TILING_KEY_IS` 做二次分派。TilingKey 数值由
目标 CANN 根据模板声明编码，只用于 host、binary metadata 和 runtime 之间的选择闭环，不作为
公开接口或跨版本稳定语义。
