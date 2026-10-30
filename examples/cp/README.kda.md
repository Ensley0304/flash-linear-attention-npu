# KDA CP example

`chunk_kda_cp.py` adapts PR #866 to the current KDA interfaces. It changes only
the KDA example; GDN is outside this update.

## Supported path

- A2/A3, B=1, Hk=Hv, K=V=128, BF16 Q/K/V, FP32 raw gate/beta/parameters.
- Equal contiguous token shards, including packed sequences and empty sequences.
- Fused QK L2Norm, beta sigmoid and safe gate (`lower_bound=-5`).
- Saved intermediates (`disable_recompute=True` / Prepare `backward_mode="save"`).
- Zero initial state, no final-state loss; explicit forward/backward calls.
- Output and all five token gradients are BSND (beta gradient is BSN).

A5's fused beta sigmoid backward is reserved in the current API, so this
example rejects A5 rather than silently selecting different semantics.

## Communication selection

```bash
# Single NPU, total forward/backward public interfaces.
python examples/cp/chunk_kda_cp.py --communication none --accuracy-check

# Four NPUs, staged forward/backward with boundary exchange.
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication allgather --accuracy-check

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  examples/cp/chunk_kda_cp.py --communication ptp --accuracy-check
```

AllGather collects one `[E | M]` FP32 summary per rank and calls
`merge_fwd_bwd_kernel` on same-sequence predecessors (forward) or successors
(backward). PTP implements the inclusive/exclusive scan in
[MindSpeed issue #4, section 4.6](https://gitcode.com/Liz_/MindSpeed/issues/4):
distances 1, 2, 4, ...; each round sends the previous inclusive summary and
updates the local exclusive summary. Backward reverses the logical rank order.
It uses PyTorch FP32 matrix multiplication for summary composition; it is not
a new fused AscendC merge kernel and does not guarantee lower latency.

Packed resets zero the outgoing affine matrix when a rank contains a sequence
boundary; only the first local sequence can have incoming forward state, and
only the last can have incoming backward state. State exchange returns FP32
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
