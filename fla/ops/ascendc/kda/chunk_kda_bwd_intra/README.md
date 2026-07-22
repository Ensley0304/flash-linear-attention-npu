# ChunkKdaBwdIntra

AscendC implementation of the KDA chunk-local backward kernel. The operator consumes the gradients of the two intra-chunk matrices and updates `dq`, `dk`, `dbeta`, and per-feature `dg` in FP32.

Grouped key 23 defaults `KDA_GROUPED_TSCM_AB_DOUBLE_BUFFER=false` on the A2/910B delivery path. Its supported producer/consumer pipeline uses two GM workspace slots: while AIC consumes one slot, the two AIVs prepare the other. The main 179,936-byte UB slab is single-buffered, while CATLASS still uses its internal L1/L0 ping-pong. A source-level TSCM experiment is retained for non-A2 investigation, but it must not be enabled for DAV_2201: the AscendC A2 `UB -> TSCM` implementation is software-emulated through GM and requires a registered Matmul KFC client, which this direct CATLASS kernel does not own. Consequently it neither provides a physical on-chip AIV-to-AIC path nor removes the A/B GM round trip on A2.

The grouped source also coalesces each stage's equally spaced off-right C row tiles into one strided MTE2 request and one contiguous FP32 `MulAddDst`. On the producer side, adjacent `q*gate` and `beta*k*gate` scratch tiles are written into one right-B matrix by one strided MTE3 request; the two physical-AIV row windows of each left C are likewise gathered by one MTE2 request. These changes reduce independent API submissions without changing transferred bytes, gate references, operands, destinations, or per-element arithmetic. Each physical AIV now computes its invariant eight-row `rowStart` once at kernel entry and carries it through every task/stage; the three persistent right-outer blocks are explicitly unrolled so their UB offsets remain compile-time constants.

The primary delivery path is `safe_gate=true`; `safe_gate=false` remains a separate compiled branch for compatibility. General shapes use the 16-token AIV block-wise implementation. The exact DAV_2201 target domain (`BF16`, `BT=64`, `K=128`, dense full chunks, `H=HV`) selects grouped key 23: four late-block stages pack the causal work into 17 IEEE-FP32 Cube GEMMs, preserve a separate last-token reference for every off-right early block, compact the ten consumed references across bank-distinct UB colors and subtract them across eight rows with zero repeat stride, factor the six off-diagonal pair gates through K-wide bridges, broadcast each bridge across eight rows directly with zero repeat stride, batch feature loads and gate/vector epilogues in one UB-resident task, retain one 32-row beta broadcast for the entire task, and pipeline two workspace slots between 1 AIC and 2 AIVs.

The source default keeps `db` reduction and `dk_left*beta` in the final whole-task tail; the compile-time `KDA_GROUPED_OVERLAP_STAGE_EPILOGUE=true` candidate instead finalizes each 8-row block after its `ConsumeStage` so stages 0--2 may overlap that AIV work with the following AIC stage. The old whole-tail path remains the rollback.

`KDA_GROUPED_DOUBLE_BUFFER_PAIR_SCRATCH=true` is an independent candidate that adds only a second three-bank pair scratch set: its 192,256-byte UB slab remains below the 192-KiB DAV_2201 limit and overlaps a previous pair's MTE3 drain with the next pair's Vector work. The source default is `false`, retaining the 179,936-byte single-scratch path; its final 96 bytes keep the fixed per-AIV causal mask and zero vector resident across all tasks. This is local scratch ping-pong, not a full copy of the UB data path, and it does not change any gate formula or GEMM order.

The source now defaults `KDA_GROUPED_BATCH_TAIL_BLOCKS=true`. It gathers the four disjoint 8x128 output blocks with four strided MTE2 submissions, applies the same six Vector expressions once over 4096 FP32 elements, and scatters them with four MTE3 submissions. This reuses retired UB banks, preserves every element's FP32 operation order, and does not enlarge the UB slab. Setting it to `false` restores the per-block scalar rollback for clean-wheel bisection.

The source defaults `KDA_GROUPED_ENABLE_UNIT_FLAG=false` and `KDA_GROUPED_PERSISTENT_MMAD_ENGINES=false`. Each of the 17 logical GEMMs owns one conservative `preSetFlags -> operator() -> finalWaitFlags` transaction; the shared non-UnitFlag CATLASS branch supplies the complete `M_FIX/FIX_M` L0C lifecycle inside that envelope. A2 bisection proved that per-call UnitFlag envelopes still time out when grouped stage 1 submits several independent MMAD/Fixpipe transactions, while adding an external `FIX_M` event to the UnitFlag path made the 120-second grouped preflight time out as well. The manual event injection has therefore been removed rather than layered onto the fine-grained protocol. The persistent implementation remains compile-time-only and must not be used for delivery until a new protocol has independent device evidence. Disabling UnitFlag does not change GEMM order, workspace layout, gate formulas, or FP32 accumulation order, but its A2 exit/precision result and performance cost still require a clean-wheel run. The source additionally defaults `KDA_GROUPED_REUSE_VECTOR_MASK=true`: every aligned FP32 arithmetic envelope programs one 64-lane normal mask, uses the matching low-level `isSetMask=false` overloads without changing instruction or repeat order, and resets the mask before leaving the envelope. Setting it to `false` restores the per-call Level-2 API path for clean-wheel precision/performance bisection. A2 is the current validation target; A3 requires an independent device run. Key 23 and all scheduling candidates are under clean-build/NPU validation. Pair-wise key 15 remains compiled as a source-level rollback that requires disabling grouped dispatch and rebuilding; unsupported shapes and `safe_gate=false` continue to select keys 0--7 at runtime.

`KDA_GROUPED_OVERLAP_TASK_STORE=true` is an additional, default-off deep-pipeline candidate. With the batched whole-task tail enabled, it retains completed `dq/dk/dg` in the three accumulator banks, prepares and publishes the next task's first two Cube stages, and then drains the previous task's four output DMAs. It adds no UB allocation and preserves each output's FP32 operation order; `store-serial` and `store-overlap` still require separate clean-wheel precision and same-card profiling evidence.

`KDA_GROUPED_PACK_STAGE_A=true` is a precision-neutral, default-off transfer-fusion candidate. It writes the already-masked off-diagonal prefix and diagonal from the same raw dA slab into one `32 x prefix` RowMajor workspace matrix. The original per-path safe-gate references, all 17 logical GEMMs, FP32 accumulation order, two-slot workspace size, and UB budget remain unchanged; only two redundant AIV MTE3 row-copy submissions per nonzero stage are removed. The runner exposes the clean-wheel A/B as `--stage-a split|packed`.

`KDA_GROUPED_USE_HF32_CUBE` is a separate, default-off throughput experiment. `false` keeps the delivery baseline's IEEE-FP32 MMAD inputs; `true` lets A2 round both FP32 Cube operands to HF32 before MMAD. The runner exposes this as `--cube-mode ieee|hf32`. Because the mode changes numerical precision, an HF32 wheel is accepted only if the full 37-case suite, including the sparse cancellation and extreme safe-gate guards, passes before same-card profiling.

Public Python API:

```python
from fla_npu.ops import ascendc

dq, dk, db, dg = ascendc.chunk_kda_bwd_intra(
    q, k, g, beta, dAqk, dAkk, dq_acc, dk_acc, db_acc, dg_acc,
    chunk_size=64,
    layout="BSND",
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=True,
)
```

Supported layouts are uppercase `BSND`, `BNSD`, `TND`, and `NTD`. `q/k` must share FP16 or BF16 dtype. Public `g/beta` may be FP32 or BF16 and are normalized to FP32 before L0; matrix gradients, accumulated gradients, and outputs are FP32. `K` must be a multiple of 16 in `[16, 256]`; the kernel uses 32-feature tiles and a 16-feature tail when needed. Chunk size is 64 or 128.

`g` is already the finite chunk-local cumulative gate in base-2 log space: upstream applies `RCP_LN2=log2(e)` once before the cumulative sum. Its per-token increments are non-positive, so every feature is non-increasing inside a chunk. This operator must not apply `log2(e)` again; AscendC evaluates `exp2(g_i-g_j)` as `Exp((g_i-g_j)*ln(2))`. The grouped pair-bridge factorization relies on this cumulative-gate contract; `safe_gate=True` does not sanitize arbitrary non-monotonic tensors supplied to the standalone API.

Reproduce the primary `B=1, T=8192, H=HV=32, K=128, BT=64, BF16,
safe_gate=true` performance case with:

```bash
python3 torch_custom/fla_npu/test/benchmark_npu_chunk_kda_bwd_intra.py \
    --device 0 --warmup 3 --repeat 10
```

See [design.md](docs/design.md), [api.md](docs/api.md), [performance.md](docs/performance.md), and
[validation.md](docs/validation.md) for formulas, interfaces, optimization notes, and the verification matrix.
