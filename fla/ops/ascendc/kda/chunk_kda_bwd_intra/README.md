# ChunkKdaBwdIntra

AscendC implementation of the KDA chunk-local backward kernel. The operator consumes the gradients of the two intra-chunk matrices and updates `dq`, `dk`, `dbeta`, and per-feature `dg` in FP32.

**Safe baseline and current experiment (2026-07-23):** key7 retains the clean AIV block-wise snapshot from commit `75535cd`. That exact snapshot passed all 22 then-existing NPU tests and measured 48.660 ms kernel time (51.107 ms end-to-end) for `B=1,T=8192,H=HV=32,K=128,BT=64,BF16,safe_gate=true`. The single-launch key12 `AIV pack -> FP32 Cube -> AIV consume` fallback uses a 64-feature tile and measures 32.477 ms in the same profiling shape. Key13 keeps key12 intact, batches 16 contiguous source gates into each `Sub/Muls/Exp` repeat, and measures 31.034 ms kernel time. Key14 additionally batches row post-scale operations but measured 31.036 ms, so it is retained only as a no-gain experiment. Key15 remains quarantined after its endpoint guard timed out; keys 16/17/18 retain the later split-launch diagnostic path but are no longer selected publicly. The target shape now selects the isolated key19 experiment. Key19 copies PR190's outer execution skeleton: one `KERNEL_TYPE_MIX_AIC_1_2` launch, chunks distributed over logical AIC cores, value heads processed in pairs, four per-core workspace slots, and raw flag ids 2/4 for AIV-to-AIC and AIC-to-AIV handshakes. The original large multi-tile Cube implementation timed out in the first `left_prev` contraction; the current implementation retains the six semantic contractions but executes them as bounded IEEE-FP32 single-tile calls. Key19 still requires renewed endpoint and precision validation, while key13 remains the known-good fallback.

Key20 is a completion-only diagnostic for that endpoint shape. Setting `FLA_NPU_KDA_DIAG_MATMULS=0..6` selects key20 and runs AIV pack/ready followed by exactly that many of key19's six semantic Cube contractions; it deliberately skips the done wait and result consume so the launch can return normally. The first diagnostic run showed prefix 0 completing and prefix 1 timing out, isolating the failure to the original `M=96,N=128,K=96` multi-tile `left_prev` BlockMmad. Key19 therefore keeps the PR190 outer schedule but splits all six sparse contractions into non-overlapping calls bounded by `M<=64,N=64,K<=64`, using the same IEEE-FP32 `MmadPingpong<..., false, false>` configuration as the locally proven rowBlock3 path. Key20 remains available to validate every semantic contraction after this change. Without the environment variable, public dispatch remains key19.

The primary delivery path is `safe_gate=true`; `safe_gate=false` remains a separate compiled branch for compatibility. The stable safe fallback uses one AIV launch, 16-token block-wise accumulation, resident `[chunk, 32]` q/k/g feature tiles, FP32 accumulation, dense/GVA/varlen support, and no sequence-sized workspace. Key19 is restricted to dense `B=1,H=HV,BT=64,K=128,BF16`, full chunks, and at most 4096 `(chunk,head)` tasks. It allocates four 600 KiB slots per used logical AIC core rather than task-sized A/B/C tensors. Keys 16/17/18 and quarantined key15 remain compiled only for diagnosis. Key13 and key7 remain the immediate target-shape and general fallbacks.

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

Supported layouts are uppercase `BSND`, `BNSD`, `TND`, and `NTD`. `q/k` must share FP16 or BF16 dtype. Public `g/beta` may be FP32 or BF16 and are normalized to FP32 before L0; matrix gradients, accumulated gradients, and outputs are FP32. `K` must be a multiple of 16 in `[16, 256]`; the general paths use 32-feature tiles and a 16-feature tail when needed, while the exact key12-key15/key19 target shape uses 64-feature vector tiles. Chunk size is 64 or 128.

See [design.md](docs/design.md), [api.md](docs/api.md),
[performance.md](docs/performance.md), [validation.md](docs/validation.md), and
[key15_full_cube.md](docs/key15_full_cube.md) for formulas, interfaces,
optimization notes, and the verification matrix.
