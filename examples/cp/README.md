# GDN / KDA CP 示例

本目录分别提供 [GDN](chunk_gdn_cp.py) 和 [KDA](chunk_kda_cp.py) 的 CP 前后向调用示例。加载 CANN，并激活安装了匹配平台 `torch`、`torch_npu` 和 `fla_npu` 的环境，在仓库根目录执行下面的命令；设备号替换为分配到的空闲卡。

| 示例 | CP 调用方式 | 切分及长度要求 |
| --- | --- | --- |
| [GDN](#gdn) | `torchrun --nproc_per_node=N ...`，使用原有 AllGather 路径 | 连续等长切分，全局长度须整除 `64 × N` |
| [KDA](#kda) | `--communication none/allgather/ptp`，CP 卡数由 `torchrun` 决定 | 连续或 zigzag，支持 packed；连续全局长度整除 `N`，zigzag 整除 `2 × N`，无需局部长度按 64 对齐 |

两者都沿 token 维切分，每卡保留全部 head。以下分别说明各自的输入参数、接口和支持范围；本次更新未修改 GDN 接入代码。

## GDN

### 环境

在 Linux 昇腾服务器上加载 CANN，激活已安装 `torch`、`torch_npu`、`fla_npu` 的环境。算子包需包含对应平台的 GDN 小算子和 CP 算子，安装见[开发者指南](../../docs/开发者指南.md)。

```bash
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_STABLE_ABI=ctypes
export HCCL_EXEC_TIMEOUT=120
```

包需提供 `pre_process_fwd_kernel_merged`、`chunk_delta_h_bwd_preprocess`（反向预处理旧名）、`merge_fwd_bwd_kernel`。

GDN 均直接向 Dhu 传入 Merge 得到的 `dht`，算子包还需包含 [PR #933](https://github.com/flashserve/flash-linear-attention-npu/pull/933) 的修复；旧 Dhu 会忽略该参数，不能用于本示例。

### 运行

在仓库根目录执行，将设备 `0,1` 替换为分配到的卡。脚本自动识别平台；双卡时全局 T=256，每卡处理 128 个 token。

```bash
# GDN 双卡
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --accuracy-check

# GDN 双卡 GVA：Hk=2，Hv=6
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --value-heads 6 --accuracy-check

# GDN single NPU
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_gdn_cp.py \
  --device 0 --tokens 128 --heads 2 --accuracy-check

# GDN CPU reference only
python examples/cp/chunk_gdn_cp.py --tokens 128 --heads 2 --cpu-reference-only
```

双卡成功时打印 `gdn CP world=2 accuracy: PASS`，退出码为 0。`--accuracy-check` 检查输出、梯度及 h0/dht；不加该参数只检查结果有限。

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tokens` | `256` | 全局长度，须为 `64 × WORLD_SIZE` 的正整数倍 |
| `--heads` | `2` | Q/K head 数，范围 `[1,128]` |
| `--value-heads` | 等于 heads | 仅 GDN；Hv 在 `[1,128]`，Hv/Hk 为 1、2、3 或 4 |
| `--device` | `0` | 仅单进程使用；多进程由 LOCAL_RANK 选卡 |
| `--seed` | `20261010` | 随机种子 |
| `--accuracy-check` | 关闭 | 与完整序列 CPU 参考比较 |
| `--cpu-reference-only` | 关闭 | 只运行 CPU 参考，不使用 torchrun |
| `--output-tol` | `0.005` | 输出 atol/rtol |
| `--grad-tol` | `0.008` | dq/dk/dv atol/rtol |
| `--gate-tol` | `0.02` | 门控及其参数梯度 atol/rtol |
| `--boundary-tol` | `0.02` | h0/dht atol/rtol |

精度检查同时要求结果有限、普通量余弦相似度 ≥0.999、门控梯度 ≥0.99；`rel_l2` 仅供参考。

### 模型调用入参

内部入口为 `run_gdn(x, *, scale, rank, world, architecture="a5")`，同时执行前向和显式反向。`x` 为下表字典，`L=T/world`。

四维输入使用连续 **BNSD `[B,H,T,D]`** 布局；BSND 输入先做 `permute(0,2,1,3).contiguous()`。固定 `B=1`、`K=V=128`，Q/K 已做 L2 归一化。

#### GDN

| 键 | dtype | 本地 shape | 说明 |
| --- | --- | --- | --- |
| `q`、`k` | BF16 | `[1,Hk,L,128]` | Q/K |
| `v`、`do` | BF16 | `[1,Hv,L,128]` | value、上游输出梯度 |
| `g` | FP32 | `[1,Hv,L]` | 自然对数域逐 token gate，未做 cumsum |
| `beta` | FP32 | `[1,Hv,L]` | 已激活的更新系数，非 logits |

返回 `o/dq/dk/dv/dbeta/dg/h0/dht`。`o/dv` 使用 Hv，`dq/dk` 使用 Hk；GVA 的共享 Q/K 梯度求和。`h0/dht` 为 `[1,Hv,128,128]`。

#### 调用约定

- `scale=128**-0.5`；`architecture` 为 `"a5"` 或 `"a2_a3"`。
- 直接调用前加载 `fla_npu`、初始化 NPU，并将输入放到本 rank 设备；多卡先初始化 HCCL，`rank/world` 与默认进程组一致。有其他并行组时，需改为使用模型的 CP 组。
- 仅沿 token 维连续等长切分。
- 示例未封装 autograd。训练接入需拆开前后向并保存中间量；`dq/dk` 不含 L2Norm backward。

### 支持范围

- `chunk_size=64`，保存中间量、不开重计算。GDN 支持 GVA。
- GDN 直接传入 dht 的版本此前已在 A2 验证单卡、双卡和 GVA；本次 KDA 更新不修改 GDN 代码，也未重新做 GDN 的 NPU 验证。
- 不覆盖 varlen/packed、zigzag、不等长切分、外部非零初始状态、末状态损失和多机验证。
- Dhu 直接使用 `dht`，不追加虚拟 chunk；未做性能验收。

## KDA

The verified operator baseline is main commit `2bb7e74ecf4876acbc4527070e403bffeef06e75`
(PR #993). Use a matching platform wheel containing the staged KDA and CP
operators; do not mix the old backward ABI with this example.

### Supported path

- A2/A3/A5, B=1, Hk=Hv, K=V=128, BF16 Q/K/V, FP32 raw gate/beta/parameters.
- Equal contiguous or zigzag token shards, including packed and empty sequences.
- Fused QK L2Norm and safe gate (`lower_bound=-5`); beta sigmoid mode depends on the platform.
- Saved intermediates (`disable_recompute=True` / Prepare `backward_mode="save"`).
- Zero initial state, no final-state loss; explicit forward/backward calls.
- Output and all five token gradients are BSND (beta gradient is BSN).

`--beta-sigmoid auto` (default) uses fused sigmoid on A2/A3. A5's fused
sigmoid backward is reserved, so A5 computes `beta_eff = sigmoid(beta)` in
FP32 before the operator and applies `d_beta = d_beta_eff * beta_eff *
(1-beta_eff)` after backward. Both no-CP and CP use this same rule. The selected
mode is recorded in every result; timing/profiling includes these operations.
`--beta-sigmoid external` also allows comparison on A2/A3. Explicit `fused`
on A5 is rejected. This adapts the example, not the A5 kernel's capabilities.

### CP input contract

Partition metadata follows upstream FLA commit
[`f2ca71c0afb3`](https://github.com/fla-org/flash-linear-attention/blob/f2ca71c0afb3254b564471e152e73ba5d1cddcaa/fla/ops/cp/context.py).
The caller partitions the sequence dimension; each rank retains every head.

| Condition | This example and upstream FLA |
| --- | --- |
| Contiguous partition | Total tokens >= CP size and divisible by CP size |
| Zigzag partition | Total tokens >= 2 * CP size and divisible by 2 * CP size |
| Local length / individual packed sequence length | No multiple-of-64 requirement; partial chunks use real sequence boundaries |
| Head count | No divisibility requirement with CP size |
| Packed input | B=1; global cumulative lengths become local boundaries; empty segments are omitted |
| External initial state / final-state loss | Not supported in CP |

`--partition zigzag` assigns global intervals `r` and `2*world-1-r` to rank
`r`, concatenated as `[front; back]`. Each interval has `tokens/(2*world)`
tokens. An explicit local boundary separates the two intervals, even if they
belong to the same global sequence: the CP state carries their dependencies.
AllGather reorders summaries into global sequence order before merge. PTP
scans the same interval chain, copying same-rank edges locally and matching
cross-rank messages in global edge order. In either scan direction, PTP retains
only the local intervals' running summaries rather than gathering all summaries.

The example also validates its benchmark options (positive iteration count,
an output directory for profiling, and a size cap for the CPU autograd
reference). The CPU reference cap is not an operator input limit: use a saved
no-CP `--reference` for larger cases.

#### Operator capabilities outside this CP change

CP partition compatibility does not imply that all upstream KDA tensor modes
are implemented in the current AscendC operators. This change does not extend
the following capabilities:

| Capability | Upstream FLA | Current AscendC example |
| --- | --- | --- |
| Head dimensions | K <= 256 at the public entry, subject to backend/kernel constraints | K=V=128; also enforced by the CP pre-process kernels |
| Chunk size | 32 or 64 at the public entry | 64 |
| Grouped value heads | HV divisible by H | Hk=Hv in this backward composition |
| Q/K/V dtype | Backend-dependent, including FP16/BF16 | BF16 in the KDA backward stages |
| Gate / normalization options | Configurable | Fused L2Norm and safe gate with lower_bound=-5; A5 sigmoid is external |
| State representation | K-first or V-first | K-first internally |
| Recompute | Configurable | Saved intermediates, as requested for this example |
| Platforms / graph capture | Backend-dependent; Ascend CP graph mode currently rejected | A2/A3/A5 eager execution |

These are remaining operator or example-mode differences, not new restrictions
on sequence partitioning. Do not remove kernel guards to claim broader support.

### Communication selection

```bash
# Single NPU, total forward/backward public interfaces.
python examples/cp/chunk_kda_cp.py --communication none --accuracy-check

# Four NPUs, staged forward/backward with boundary exchange.
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication allgather --accuracy-check

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication ptp --accuracy-check

# Two 65-token intervals per rank, including unaligned packed boundaries.
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication ptp --partition zigzag \
  --tokens 520 --heads 3 --cu-seqlens 0,0,1,64,65,130,131,259,260,519,520 --accuracy-check
```

AllGather collects one `[E | M]` FP32 summary per interval (two per rank for
zigzag) and calls
`merge_fwd_bwd_kernel` on same-sequence predecessors (forward) or successors
(backward). PTP implements the inclusive/exclusive scan in
[MindSpeed issue #4, section 4.6](https://gitcode.com/Liz_/MindSpeed/issues/4):
distances 1, 2, 4, ...; each round sends the previous inclusive summary and
updates the local exclusive summary. Backward reverses the logical interval order.
It uses PyTorch FP32 matrix multiplication for summary composition; it is not
a new fused AscendC merge kernel and does not guarantee lower latency.

Packed resets zero the outgoing affine matrix when an interval contains a sequence
boundary; only its first local sequence can have incoming forward state, and
only its last can have incoming backward state. State exchange returns FP32
`h0` to FwdH and BF16 `dht` to Dhu. Shared parameter gradients are summed across
ranks in both communication modes.

### Current backward interface

The operator calls are kept together in `run_kda()`:

```text
forward:  FwdPrepare -> pre_process_fwd -> exchange_boundary -> FwdH(h0) -> FwdFinalize
backward: BwdPrepare -> pre_process_bwd -> exchange_boundary -> Dhu(dht) -> BwdFinalize
```

`exchange_boundary()` selects AllGather + Merge or PTP summary composition.
It produces the boundary state; FwdH/Dhu then propagate it through local chunks.
After backward, `run_kda()` sums `dA_log` and `ddt_bias` across the default
process group. Model integration must use its CP group and avoid reducing these
gradients twice. This example uses explicit backward calls, not an autograd layer.

`run_kda(x, plan, communication, fuse_beta_sigmoid=...)` takes local contiguous
head-major inputs. `plan` comes from `build_partition()` using global packed
offsets and the global token count; inputs are sliced in `plan["parts"]` order.

| `x` entry | dtype | Local shape | Meaning |
| --- | --- | --- | --- |
| `q`, `k`, `v`, `do` | BF16 | `[1,H,L,128]` | Q/K are not pre-normalized; `do` is the output gradient |
| `beta` | FP32 | `[1,H,L]` | Raw sigmoid logits, including external sigmoid mode |
| `g` | FP32 | `[1,H,L,128]` | Raw gate, before activation and cumsum |
| `A_log` | FP32 | `[H]` | Shared across CP ranks |
| `dt_bias` | FP32 | `[H,128]` | Shared across CP ranks |

Here `L = tokens / world`, including both local intervals in zigzag mode.
Direct A5 calls must set `fuse_beta_sigmoid=False`; the CLI selects this automatically.

The total backward (no CP) uses `layout="BSND"`: V and output token gradients
are BSND; `do`, raw gate and original beta are always token-major. Saved
normalized Q/K, reciprocal norms and `beta_eff` stay head-major. Both original
beta and FP32 `beta_eff` must be passed with fused sigmoid enabled. Without
fused sigmoid, pass the effective beta as `beta` and set `beta_eff=None`.

The CP path must use staged backward to inject `dht` into Dhu. Small Prepare uses
`layout="NTD"` for A5 variable-length support (also supported on A2/A3): remove
the unit batch axis from Aqk, Vnew, do and h, then restore it on the outputs.
These are views without data copies. Small Finalize
takes head-major tensors and **saved beta_eff**, with `output_layout="BSND"`
controlling all five token gradients. These are different contracts: do not
copy the total backward argument order/layout into small Finalize. No custom
ctypes ABI or model-side autograd adapter is used by this example.

### Packed validation and profiling

For CPU reference validation, keep `tokens * heads <= 8192`. The recurrence
checks forward, all seven gradients, and cross-rank h0/dht. Example:

```bash
torchrun --standalone --nproc_per_node=4 examples/cp/chunk_kda_cp.py \
  --communication ptp --tokens 512 --heads 2 --cu-seqlens 0,0,130,257,512 \
  --accuracy-check
```

Large cases use the same-seed no-CP NPU result as a separate comparison (not an
independent CPU accuracy certification). Warmup, timing and profiling exclude
input generation, accuracy checks, file writes and CPU copies. Timing includes
both forward/backward and shared parameter-gradient AllReduce, and reports the
maximum rank time; profiler collection is a separate warmed iteration.
Error norms and dot products accumulate in FP64 to avoid inaccurate FP32
reductions on long vectors. The original example's mixed tolerances and cosine
thresholds remain unchanged; failures are recorded and return a nonzero status.

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python examples/cp/chunk_kda_cp.py \
  --communication none --tokens 131072 --heads 32 \
  --cu-seqlens 0,70290,129130,131072 --warmup 5 --iterations 20 \
  --save-result /path/to/no_cp.pt --output-dir /path/to/none --profile

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication allgather \
  --tokens 131072 --heads 32 --cu-seqlens 0,70290,129130,131072 \
  --warmup 5 --iterations 20 --reference /path/to/no_cp.pt \
  --output-dir /path/to/allgather --profile
# Repeat the second command with --communication ptp and a separate output-dir.
```

`--profile` writes CANN data under `<output-dir>/rankN/profiling`. Export each
`PROF_*` directory with the installed CANN tool:

```bash
msprof --export=on --output=/path/to/PROF_directory --summary-format=csv
```

Keep `op_summary*.csv` for every rank, `result.json`, source revision and wheel
hash. Never sum overlapping HCCL/AICPU activity durations into wall time.
