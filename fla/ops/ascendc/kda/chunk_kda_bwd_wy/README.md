# ChunkKdaBwdWy

Fused KDA WY backward stage for the P0 `C=64,K=V=128` path.  It implements the
nine contractions and their dependent Vector work from section 7.3 of
`chunk_kda_bwd_a_design.md` in one MIX L0.

The implementation keeps the verified structural choices from PR190 and the
existing fused KDA kernels:

- a 12-head production window in the single-launch fused path; the diagnostic
  split-stage fallback retains the original two-head ordering;
- twelve 256 KiB GM slots per logical core in the fused path; the diagnostic
  fallback maps four of those slots as two generations x two heads;
- FP32 MMAD accumulation with explicit BF16 `dW`, `kE`, `Zb` and `T` boundaries;
- fixed saved-state layouts: public `h=[B,NT,H,K,V]`, internal
  `dh=[B,H,NT,K,V]`;
- dependency-only AIC/AIV ready/acknowledgement edges and no fake flag between
  same-AIC dependent GEMMs;
- strict-lower `Zb` and `dAkk` masks with a fixed physical 64-column stride for
  tail chunks.

The current canary drains a task group before reusing the next slot generation.
This keeps the first implementation's workspace lifetime simple; inter-task
generation overlap is a profiling-driven optimization, not a correctness
assumption.
