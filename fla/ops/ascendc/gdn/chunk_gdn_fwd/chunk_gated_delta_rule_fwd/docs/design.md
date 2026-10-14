# A2/A3 megakernel 移植设计

方案设计规则版本：V2

## 1. 目标

接口合同见 [api.md](api.md)。本次移植既有固定实现，不重新设计算子公式或 tile 参数：
从 `95004a10` 成套引入 arch22 tiling/kernel 和 host 布局合同，使 A2/A3 按 DAV_2201
选择同一实现。只修改本算子的 24 个实现文件；公共组件、顶层 kernel/tiling、A5 arch35、
Python wrapper 和公开 ACLNN 头文件保持发布分支版本。

迁移后调用图为 public host → 同一 ChunkGatedDeltaRuleFwd L0 → arch22 Stage P/ABC/H/O。
A5 原 Prepare → H → O 调用图保留。新增可选私有属性以默认历史布局保持旧 L0 调用兼容。

性能验证使用已有陈曦五个模型 case：T=3072/438/355/6144/836，Hk=8、Hv=24、K=V=128、
BT64、BF16。相同冻结输入在发布基线与候选上按同卡 ABBA 顺序采集 msprof Task Duration(us)，
并单列 native 布局回归。单 kernel 时间与整个调用的 Cast/Transpose/Copy 时间分别统计。
T=6144/836 按原测试约定使用两条均匀序列，其余一条；这属于合成边界，不声称还原客户原始边界。

旧路径重复的门控布局搬运、O transpose 与串行 H/O 构成可减少的工作量；本次只有设备实测才
能判定收益，不从搬运字节数推导加速比。A3 没有当前执行环境，仅进行目标编译并保留实机门禁。

## 2. Stage 0–N 移植详设

令 G=Hv/Hk，Nc 为总 chunk 数，S 为有效序列数，C 为可用 Cube 核组数，VB=ceil(V/128)。
ABC 任务沿 value head/chunk 展开，H 沿 `(sequence,hv,v_block)` 传播依赖状态。
例如 dense B=1、T=3072、Hv=24、BT64 时，Nc=48，ABC 有1152个工作项；H 有24个生产任务。
实际 blockDim、ready bank 和 HO 是否启用均由来源 host tiling 与 device 校验确定。

| 阶段 | 固定来源组件 | 迁移内容与依赖 |
| --- | --- | --- |
| P，Vector | gdn_cumsum_prepare.hpp | BTH FP32 g 按 head tile 计算 chunk 内前缀并写 BHT workspace；可选 BTH 公开输出 |
| A，Cube | chunk_scaled_dot_kkt / chunk_kkt_solve_tri | 读取原生 QKV layout，GVA 映射共享 hk，生成 score |
| B，Cube/Vector 子阶段 | solve_tri_fp32/solve_tri | 保持固定来源的 FP32 Solve 运算序列、arena 和尾块处理 |
| C，Cube/Vector 子阶段 | recompute_w_u_fwd | 来源调度与 front-WU 缓冲；结果提供给 H/O |
| H，Cube/Vector 子阶段 | gdn_fwd_h_kernel.hpp | 依次执行状态投影、v_new、状态更新；同序列 chunk 不乱序 |
| O，Cube/Vector 子阶段 | gdn_fwd_o_kernel.hpp | 读取 H/v_new，按来源算法产生 O，支持直接 BTHV 写出 |

ABC/H/O 的计算、缓冲区地址、同步闭环和工作项顺序逐文件保留冻结来源，无二次改写。
私有 kernel/tiling 相互引用的 trailer、ready arena、front-WU 字段成套迁移，禁止只复制入口。
移植后通过源 blob 对照、源码审查与边界回归核对；详细源文件清单和差异记录在验证记录中。

Stage P 的 head tile 宽度为 min(Hv,64)，localStride 向8对齐，tileBytes=BT*localStride*4。
三个 FP32 buffer 和一个 uint32 Gather-offset buffer 的相对地址为 `[0,tileBytes)`、`[tileBytes,2*tileBytes)`、
`[2*tileBytes,3*tileBytes)`、`[3*tileBytes,4*tileBytes)`，累加区为
`[4*tileBytes,4*tileBytes+localStride*4)`；各区按32字节对齐。最大占用131328字节（128.25 KiB），
平台可用 UB 必须至少满足该容量。
单缓冲复用通过来源 MTE2/V/MTE3 事件闭合，不将其描述为 ping-pong。
GM prefix workspace 为 FP32 BHT，来源 tiling 独立预留，不与 score/solve/H/ready 区域别名。

HO 只在来源准入条件满足时启用：DAV_2201、无 gk、P=S*Hv*VB、0<P<C、P<=C-P，且存在
多 chunk 序列；varlen 的非空序列由 device 再核对。生产者位于核组前缀，消费者位于后缀；
H/O scratch 在预留触发时分离。ready 每 bank 为2*C个32字节槽，bankCount 在 dense 为
ceil(T/BT)，varlen 为总 chunk 数上界；总量为 AlignUp(bankCount*2*C*32,512)。
初始化后统一屏障；每个 H 子核发布自己的 `(chunk,sequence,hv,v_block)` 槽，
O 的每个 AIV 子核 s 等待同 `(sequence,hv,s)` 的 VB 个槽（V256 时两个），然后沿原 O barrier
继续；同槽不在后续 chunk 复用。条件不满足时执行来源串行 H/O 路径。

TilingKey 1/2 是历史 BHT raw-g 的 V128/V256，3/4 是 prepared BTH raw-g 的 V128/V256；
BT64/128 由 tiling data 表达。A2/A3 不按产品名分 key；不存在本次来源之外的 H 导出 key。
native QKV 与 O layout 在 trailer 中传递，所有访问按相同公共 shape 做偏移和尾部处理。

数值门禁：g/beta 在 A2/A3 转成 FP32 后再消费，避免按 FP32 指针读取 BF16 存储；不降低
累加 dtype、不近似 exp、不更换来源 Solve 算法。保留固定输入与 FP64 recurrence 标杆，
验证 O/final_state/有效 A/g_cumsum、有限值和布局等价。A5 只核对新增默认属性不改变原分支。
