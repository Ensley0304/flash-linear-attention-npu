# ChunkKdaBwdIntra

AscendC implementation of the KDA chunk-local backward kernel. The operator consumes the gradients of the two intra-chunk matrices and updates `dq`, `dk`, `dbeta`, and per-feature `dg` in FP32.

**Safe baseline and current experiment (2026-07-23):** key7 retains the clean AIV block-wise snapshot from commit `75535cd`. That exact snapshot passed all 22 then-existing NPU tests and measured 48.660 ms kernel time (51.107 ms end-to-end) for `B=1,T=8192,H=HV=32,K=128,BT=64,BF16,safe_gate=true`. The single-launch key12 `AIV pack -> FP32 Cube -> AIV consume` fallback now uses a 64-feature tile and measures 32.477 ms in the same profiling shape. The current key13 experiment keeps key12 intact and only batches 16 contiguous source gates into each `Sub/Muls/Exp` repeat; every non-eligible shape still dispatches to key7 or the existing legacy keys. Key13 requires a fresh build, full precision regression, repeated-launch check, and profiling before its performance is treated as validated.

The primary delivery path is `safe_gate=true`; `safe_gate=false` remains a separate compiled branch for compatibility. The stable safe fallback uses one AIV launch, 16-token block-wise accumulation, resident `[chunk, 32]` q/k/g feature tiles, FP32 accumulation, dense/GVA/varlen support, and no sequence-sized workspace. The narrowly eligible key12/key13 MIX paths use one launch, per-slot FP32 workspace, and a target-specific `[64,64]` source feature tile; key12 is the immediate target-shape rollback and key7 remains the general fallback.

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

Supported layouts are uppercase `BSND`, `BNSD`, `TND`, and `NTD`. `q/k` must share FP16 or BF16 dtype. Public `g/beta` may be FP32 or BF16 and are normalized to FP32 before L0; matrix gradients, accumulated gradients, and outputs are FP32. `K` must be a multiple of 16 in `[16, 256]`; the general paths use 32-feature tiles and a 16-feature tail when needed, while the exact key12/key13 target shape uses 64-feature tiles. Chunk size is 64 or 128.

See [design.md](docs/design.md), [api.md](docs/api.md), [performance.md](docs/performance.md), and
[validation.md](docs/validation.md) for formulas, interfaces, optimization notes, and the verification matrix.
