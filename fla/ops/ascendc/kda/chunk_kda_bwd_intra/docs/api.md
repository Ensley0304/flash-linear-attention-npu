# ChunkKdaBwdIntra API

## Python

```python
chunk_kda_bwd_intra(
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
    chunk_size,
    *,
    layout="BSND",
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
) -> (dq_out, dk_out, db_out, dg_out)
```

The default follows the upstream helper. Production safe-gate KDA should pass `safe_gate=True` explicitly.

In varlen mode, `cu_seqlens` describes a flattened `B=1` token axis. `chunk_indices` may be omitted;
when supplied it must contain every `(sequence, local_chunk)` pair in canonical sequence-major order.

For BSND, shapes are `q/k=[B,T,H,K]`, `g/dq/dk/dg=[B,T,HV,K]`, `beta/db=[B,T,HV]`, and `dAqk/dAkk=[B,T,HV,BT]`. Other layouts permute the token/head axes consistently. `HV` must be divisible by `H`.

Variable-length input uses a flattened token axis and `cu_seqlens`. `chunk_indices`, when supplied, must contain canonical sequence-major `(sequence_id, local_chunk_id)` pairs. The wrapper creates them when omitted.

## aclnn

The stable symbol is `aclnnChunkKdaBwdIntra`. Its L0 inputs use contiguous BNSD layout and include scalar attributes `chunkSize`, `safeGate`, and `totalChunks`. Python performs public layout conversion before entering aclnn.

The operator creates four outputs and does not mutate the input accumulated gradients.
