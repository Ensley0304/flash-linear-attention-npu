# ChunkKdaBwdIntra

AscendC implementation of the KDA chunk-local backward kernel. The operator consumes the gradients of the two intra-chunk matrices and updates `dq`, `dk`, `dbeta`, and per-feature `dg` in FP32.

The primary delivery path is `safe_gate=true`; `safe_gate=false` remains a separate compiled branch for compatibility. The safe path uses one AIV launch, 16-token block-wise accumulation, resident `[chunk, 32]` q/k/g feature tiles, FP32 accumulation, dense/GVA/varlen support, and no sequence-sized workspace. The proven row-wise safe implementation remains compiled under legacy tiling keys for rollback.

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
