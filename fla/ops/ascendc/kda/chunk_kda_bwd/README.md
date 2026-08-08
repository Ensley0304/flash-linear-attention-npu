# ChunkKdaBwd P0

`ChunkKdaBwd` is the single public L2 entry for the no-recompute KDA backward.
The executor registers the following complete-tensor L0 chain once:

1. `ChunkKdaBwdDAv` (`dAqk`, `dv0`)
2. key-wise `ChunkGatedDeltaRuleBwdDhu` (`dh`, `dv_scan`)
3. `ChunkKdaBwdWy` (base gradients, `dAkk`, final `dv`)
4. `ChunkKdaBwdIntra` (intra-chunk corrections)
5. `ChunkKdaBwdGatePost` (chunk-local reverse cumsum)

No Python or L2 token segmentation is used.  K3 writes directly to the final
head-major `dv` allocation; for BNSD, K4/K6 also write directly to the public
gradient allocations.

The default Python route is
`fla_npu.ops.ascendc.chunk_kda_bwd`, backed by the two-phase
`aclnnChunkKdaBwdGetWorkspaceSize/aclnnChunkKdaBwd` ABI.  Internal L0 canaries
remain available for stage isolation but are not the stable public API.

## Implemented P0 key

- dense `BSND` or `BNSD`; `d_o` remains sequence-major BSND;
- BF16 q/k/v and saved forward tensors, BF16 or FP32 beta, FP32 gk;
- `chunk_size=64`, `K=V=128`, `H=HV`, `safe_gate=true`;
- `use_gate_in_kernel=false`, `state_v_first=false`;
- `recompute_policy="NONE"`, without initial state, dht or raw gate backward.

The full optional interface is reserved now so later keys do not need a new
ABI. Unsupported combinations fail parameter validation rather than selecting
an unverified tiling key.

## Validation status

The source contains independent K1, K2-keywise, K3, K6 canaries and an L2
composition/layout test.  On the current Windows development host only Python
syntax and repository static checks can run; CANN compilation, NPU precision,
endpoint repetition and profiling are still required before this key is
considered validated.

Recommended board order:

1. build/install the custom-operator wheel;
2. run each L0 canary, starting with K2 and K3;
3. run `test/test_chunk_kda_bwd_p0.py` for BNSD and BSND;
4. run the full GPU-dump dual benchmark and 100 repeated launches;
5. profile K1-K6 separately before changing the proven decomposition.
