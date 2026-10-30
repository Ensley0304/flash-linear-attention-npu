# KDA CP example

`chunk_kda_cp.py` adapts PR #866 to the current KDA interfaces. It changes only
the KDA example; GDN is outside this update.

## Supported path

- A2/A3, B=1, Hk=Hv, K=V=128, BF16 Q/K/V, FP32 raw gate/beta/parameters.
- Equal contiguous or zigzag token shards, including packed and empty sequences.
- Fused QK L2Norm, beta sigmoid and safe gate (`lower_bound=-5`).
- Saved intermediates (`disable_recompute=True` / Prepare `backward_mode="save"`).
- Zero initial state, no final-state loss; explicit forward/backward calls.
- Output and all five token gradients are BSND (beta gradient is BSN).

A5's fused beta sigmoid backward is reserved in the current API, so this
example rejects A5 rather than silently selecting different semantics.

## CP input contract

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

### Operator capabilities outside this CP change

CP partition compatibility does not imply that all upstream KDA tensor modes
are implemented in the current AscendC operators. This change does not extend
the following capabilities:

| Capability | Upstream FLA | Current AscendC example |
| --- | --- | --- |
| Head dimensions | K <= 256 at the public entry, subject to backend/kernel constraints | K=V=128; also enforced by the CP pre-process kernels |
| Chunk size | 32 or 64 at the public entry | 64 |
| Grouped value heads | HV divisible by H | Hk=Hv in this backward composition |
| Q/K/V dtype | Backend-dependent, including FP16/BF16 | BF16 in the KDA backward stages |
| Gate / normalization options | Configurable | Fused L2Norm, sigmoid and safe gate with lower_bound=-5 |
| State representation | K-first or V-first | K-first internally |
| Recompute | Configurable | Saved intermediates, as requested for this example |
| Platforms / graph capture | Backend-dependent; Ascend CP graph mode currently rejected | A2/A3 eager execution |

These are remaining operator or example-mode differences, not new restrictions
on sequence partitioning. Do not remove kernel guards to claim broader support.

## Communication selection

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

## Current backward interface

The total backward (no CP) uses `layout="BSND"`: V and output token gradients
are BSND; `do`, raw gate and original beta are always token-major. Saved
normalized Q/K, reciprocal norms and `beta_eff` stay head-major. Both original
beta and FP32 `beta_eff` must be passed with fused sigmoid enabled.

The CP path must use staged backward to inject `dht` into Dhu. Small Prepare's
explicit `layout="BNSD"` applies to Aqk, Vnew and do together. Small Finalize
takes head-major tensors and **saved beta_eff**, with `output_layout="BSND"`
controlling all five token gradients. These are different contracts: do not
copy the total backward argument order/layout into small Finalize. No custom
ctypes ABI or model-side autograd adapter is used by this example.

## Packed validation and profiling

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

CPU-only exchange/order/reset tests:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover -s tests -p test_kda_cp_example.py
```
