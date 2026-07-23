# Key15 full-Cube design and validation

## Scope and rollback

Key15 is restricted to the proven stage-4 eligibility:

- `safe_gate=true`
- BF16 `q/k`
- dense `B=1`, `H=HV`
- `BT=64`, `K=128`, complete chunks

It does not change the public API or the unsafe/general dispatches. Key13
remains compiled and is the immediate rollback. Changing
`KDA_STAGE4_TILING_KEY` from `KDA_FULL_CUBE_TILING_KEY` to
`KDA_ROW3_BATCHED_GATE_TILING_KEY` restores both the proven key13 kernel and
its original task-sized workspace formula.

## Contraction layout

The four causal contraction families are encoded as six sparse block-diagonal
FP32 GEMMs:

| Contraction | A shape | B shape | C shape |
|---|---:|---:|---:|
| left previous (`dAqk`, `dAkk` stacked) | 96x96 | 96x128 | 96x128 |
| left diagonal (`dAqk`, `dAkk` stacked) | 128x64 | 64x128 | 128x128 |
| right future `dAqk^T @ q` | 48x96 | 96x128 | 48x128 |
| right future `dAkk^T @ beta*k` | 48x96 | 96x128 | 48x128 |
| right diagonal `dAqk^T @ q` | 64x64 | 64x128 | 64x128 |
| right diagonal `dAkk^T @ beta*k` | 64x64 | 64x128 | 64x128 |

The safe factorization matches the CPU/Triton reference: previous-left uses
the block first token, both diagonal directions use the block midpoint, and
future-right uses the block last token. AIV applies the matching outer row
factor after Cube. No HF32, fused multiply-add, or `db` reduction-tree change
is introduced.

## CATLASS selection

| Component | Selection |
|---|---|
| Architecture | `AtlasA2` (`Ascend950` under arch310) |
| Dispatch policy | `MmadPingpong<ArchTag, false, false>` |
| L1/L0 tiles | 64x64x64 / 64x64x64 (required equal M/N) |
| A/B/C | FP32 row-major |
| Mainloop | direct `BlockMmadTla`, six calls |
| Epilogue | none; AIV consumes FP32 C |
| Kernel task type | `KERNEL_TYPE_MIX_AIC_1_2` |

## Workspace and synchronization

Each logical AIC owns one 600 KiB workspace slot. Its two AIV lanes pack row
blocks `{0,3}` and `{1,2}`, set one aggregated ready generation, wait for one
AIC done generation, and consume disjoint output rows. A lane cannot pack its
next task until it has consumed the current C, so one slot per core is enough.
For 20 AICs the operator workspace is about 12 MiB rather than the former
sequence-sized 184 MiB. This version intentionally has no double buffer.

The protocol remains:

```text
AIV0/AIV1: pack disjoint A/B rows -> ready
AIC:       wait ready -> six FP32 GEMMs -> done
AIV0/AIV1: wait done -> consume disjoint C rows -> next task
```

## Performance gate

Measured baselines for
`B1,T8192,H=HV=32,K128,BT64,BF16,safe_gate=true` are:

| Version | Kernel duration |
|---|---:|
| key7 AIV | 48.660 ms |
| key12 BK64 MIX | 32.477 ms |
| key13 batched source gates | 31.034 ms |
| key14 batched row post-scale | 31.036 ms |

Key15 is not considered accepted until a clean wheel passes precision and
repeated-launch gates and same-card profiling shows a material gain. The
initial performance target is below 22 ms kernel duration. A regression,
timeout, or precision failure requires switching the single stage-4 constant
back to key13 before further optimization.

## Validation gate

Do not relax the existing FP64-golden tolerances or skip existing cases.
Validation order:

1. single-operator fast build and wheel binary provenance check;
2. source contracts plus eight path canaries covering
   `{left previous, left diagonal, right future, right diagonal}` x
   `{dAqk, dAkk}`;
3. zero-dA, endpoint reassociation, dense random, and 100 repeated launches;
4. the complete `test_npu_chunk_kda_bwd_intra.py` regression;
5. same-card benchmark and `msprof`, with exactly one KDA MIX operator row.

The new path remains experimental until all five gates pass on NPU.
