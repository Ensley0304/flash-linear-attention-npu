# ChunkKdaBwdIntra

AscendC implementation of the KDA chunk-local backward kernel. The operator consumes the gradients of the two intra-chunk matrices and updates `dq`, `dk`, `dbeta`, and per-feature `dg` in FP32.

**Rollback baseline (2026-07-22):** the implementation has been restored to the clean AIV block-wise snapshot from commit `75535cd`. That exact snapshot passed all 22 then-existing NPU tests and measured 48.660 ms kernel time (51.107 ms end-to-end) for `B=1,T=8192,H=HV=32,K=128,BT=64,BF16,safe_gate=true`. The later key-15/key-23 mixed-Cube experiments and their diagnostic infrastructure have been removed. One instruction-count-neutral correction is retained on top: diagonal safe-gate factorization uses directional endpoints to prevent a legal large BF16 value from producing `0 * Inf = NaN`. This corrected baseline must pass the original 22 cases plus the new endpoint-reassociation guard before its performance number is reconfirmed.

The primary delivery path is `safe_gate=true`; `safe_gate=false` remains a separate compiled branch for compatibility. The safe path uses one AIV launch, 16-token block-wise accumulation, resident `[chunk, 32]` q/k/g feature tiles, FP32 accumulation, dense/GVA/varlen support, and no sequence-sized workspace. The proven row-wise safe implementation remains compiled under legacy tiling keys for rollback. Only tiling keys 0/1/2/3/5/7 exist in this baseline.

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

See [design.md](docs/design.md), [api.md](docs/api.md), [performance.md](docs/performance.md), and
[validation.md](docs/validation.md) for formulas, interfaces, optimization notes, and the verification matrix.
