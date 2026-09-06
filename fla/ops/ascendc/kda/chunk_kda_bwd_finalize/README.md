# ChunkKdaBwdFinalize: Stage0-12

The Ascend950 implementation now produces all seven finalize gradients:
`dq`, `dk`, `dv`, `dBeta`, `dG`, `dALog` and `dDtBias`. Dense and packed
variable-length tails use zero-padded physical Cube tiles, with only valid
tokens written to outputs. This is not a claim of complete KDA backward
integration or merge readiness. Public Host/aclnn
ABI is unchanged; `raw_g` and `dt_bias` remain required FP32 inputs.

The existing Host scope is NQ=NV, K=V=128, chunk size 64, safe gate and
exp2 enabled, gate computed in-kernel, and non-transposed state layout.
Q/K normalization inputs must either both be present or both be absent.

## Implementation

- Stage0-5 form the base gradients and local operands.
- Stage6/7 compute the local Q contribution and optional Q normalization.
- Stage8 uses paired-BF16 left and concatenated-reduction right GEMMs.
  Each retains the high/high and two high/residual products in FP32.
- Stage9/10 combine the K/beta gradients and optional K normalization.
- Stage11 reverse-scans each chunk's gate gradient, applies the sigmoid
  chain rule, writes FP32 `dG`, and produces per-chunk parameter partials.
  It consumes Stage9's resident UB result without a GM round trip.
- Stage12 reduces all batch/chunk partials in fixed order, without atomics.
  AIV-only global synchronization uses flag 14; MIX synchronization would
  collide with the earlier unspent ZB_FREE credits on flags 12/13.
- Each scalar partial owns 32 bytes. Stage12 assigns eight adjacent heads
  to one AIV so final scalar output DMA blocks have a single writer.
- A task-wide AIC-to-AIV L1-free credit prevents an idle AIV in a one-head
  window from overwriting the preceding task's live Stage8 operands.
- Stage5 operands move directly from UB to L1 using strided vector-mode MTE3.
- Directional events protect overlapping UB/L1 lifetimes; Stage0 uses
  per-stream last-reader credits.
- Vector FP32-to-BF16 conversions use nearest-even rounding, including exact
  FP32 midpoint inputs. Shared conversion helpers are not modified.
- dW, kE, Zb, Tza and Stage5 operands retain BF16 high/residual planes.
  This avoids amplifying Cube/CPU FP32 accumulation differences at BF16
  midpoints. Two BF16 planes and three products are an approximation, not
  exact FP32 emulation. The BF16 exponent range is retained.
- A two-head window gives each AIV one physical FP32 dAkk handoff at
  UB [0,16) KiB. Residual handoffs hold the following phase's MTE2 credit
  until overlapping UB readers finish. Tza-ready also releases dAkk UB.
- kE/Zb are produced in ND and converted to NZ by MTE3 while copying to L1,
  removing the per-row Vector readback/transpose.
- Stage4 reuses the FP32 exponential-gradient product for Vector beta/gate
  gradients instead of consuming the rounded BF16 Cube operand `kE`.
- Cube uses fixed 64-row physical tiles and GM leading dimensions. Tail
  loads copy only the valid region after clearing L1 padding; AIV producers
  initialize the inactive rows of kE, Zb and Stage5 operands. This fixes
  unaligned Fixpipe dimensions, compressed Akk strides and stale padding.
- Scalar DMA padding stops at a block boundary rather than extending all
  the way to 64 elements. Stage2 explicitly masks inactive beta lanes;
  other scalar consumers use only valid rows. Very short tails therefore
  do not exceed the DMA per-side padding limit.

## Validation Scope

The authoritative matrix is `tests/st/aclnnChunkKdaBwdFinalize` at repository
root: 200 fixed ATK PyAclnn cases, 100 dense and 100 packed, checked against
the unmodified CPU FP64 and FP32/BF16 oracle. Thresholds remain
5 / 1.5 / 1.5, including the default small-value checks. The original intra
reference, not a factored operand simulation, defines CPU acceptance.
The final layout-optimized build passes all 200 cases and 1400 output
comparisons, including B1/H96/T8192 and T16384. All seven outputs are bytewise
deterministic over 20 repeated launches for B1/H96/T8192 and H3 packed
lengths [64,1]. Performance approval remains separate from precision.

### Historical Checks

The results below describe the earlier BF16 implementation and its diagnostic
models. They do not replace the original-oracle ATK gate above or establish
the current build's precision, bitwise identity, or performance.

CPU-only dual L1 checks cover K=V=128, chunk size 64 and these main cases:

- B=1, H/T=1/64, 4/128, 7/128 with Q/K normalization, 4/4096, 96/8192.
- B=2, H=3, T=192 with BF16 `a_log`.
- B=1, H=3, variable sequence lengths [0, 64, 128], with normalization.
- B=1, H=9, T=8256, exercising 129 reduction rows and a final head group.
- B=2, H=9, T=128, with raw gate values [-8, 8] and `a_log` [-1, 1].

`tests/tail_cases.json` adds dense boundaries 1, 7/8/9, 15/16/17, 31/32/33,
47/48/49, 63/65 and 127, a multi-batch BF16-a_log T=129 case, a packed
case containing an empty sequence and every length 1-64 plus 65/127/129,
and H=96/T=8193. Additional regressions cover packed lengths [0,1,63,65].

All 31 cases (217 output checks) pass CT 0.7.1's default L1 settings:
BF16 for the first
four and FP32 for the gate outputs. Tail matrix cases repeat twenty times,
except the all-tail packed case, which repeats fifty times. Before the L1
lifetime fix, both Stage0-10 and the new implementation could produce
unstable `dk`; the fixed H=9/T=8256 case previously passed 100 repeats.
These checks do not establish untested platform coverage or full GDN
regression, and are not GPU dual-benchmark or merge-readiness evidence.

The tail matrix also passes a separate bytewise repeated-launch check,
including signed-zero bit patterns. An isolated single-operator wheel
matches the validated 8K, multi-batch tail, all-tail packed and 8K+1 dumps
bytewise, with twenty repeats each. The aligned 8K seven-output dump is
bytewise unchanged from the implementation before the tail fix.

The H=1, T=64, seed=73 regression exposed extra Stage4 BF16 quantization:
`dBeta` had six small-value errors versus two in the CPU benchmark, failing
the default ratio limit of two. Removing the rounded `kE` dependency reduces
the count to two and passes without threshold changes. Across all five cases,
`dq`, `dk` and `dv` remain bitwise identical to the rounding-fixed baseline.

An exact, causal `dAqk` midpoint regression previously failed 1035 output
elements and passes after the rounding fix. The older FP64 stagewise
`allclose(atol=2e-5, rtol=0.02)` diagnostic still has 43 DK outliers on its
unmasked random 8K stress input; it has not been relabeled as passing. Sampled
outliers arise at FP32 exponent/BF16 rounding boundaries. CPU dual acceptance
and this diagnostic are different tests.

For saved inputs and outputs, run the independent CPU formula checker:

```bash
python tests/check_cpu_dual.py --oracle /path/to/kernel_ac_torch_ref.py \
  --inputs /path/to/inputs.pt --outputs /path/to/outputs.pt \
  --report /path/to/cpu_dual.json
```

The oracle supplies `kernel_c_base_torch`, `kernel_c_intra_torch`,
`reverse_chunk_cumsum` and `gate_backward_torch`. Golden uses FP64 without
simulated Cube downcasts. The low-precision CPU benchmark uses the original
FP32/BF16 base and unfactored intra oracle. It rounds dA only within intra.
Golden, the external oracle and CT thresholds are unchanged. The historical
factored `k_neg/q_pos/bk_pos` model is retained in `factored_outputs` as a
diagnostic, not an acceptance criterion. Inputs are never changed.
The checker records the oracle hash and raw RMSE ratios, in addition to CT's
ratios, whose denominators include dtype-dependent lower bounds.

`tests/check_bf16_midpoints.py --write-inputs inputs.pt` generates the exact
cast regression input. After an aclnn launch, use `--outputs outputs.pt` to
check all seven saved outputs bitwise. This is an exact rounding check,
separate from the numerical dual-benchmark tolerance.

## Performance

The paired correctness checkpoint measured approximately 7.31 ms versus
6.00 ms for the earlier BF16 implementation. Stage8 K concatenation and
fused MTE3 ND-to-NZ conversion reduce the candidate to about 6.51 ms.
Final A/B/B/A device measurements use two runs per build, three excluded
warmups and ten measured launches per run. Baseline median is 5.990660 ms
(5.940920--6.020410); final paired build median is 6.517315 ms
(6.501640--6.538010), a measured 8.79% regression. Both compute all seven
outputs. The final paired build passes the original 200-case ATK matrix;
the baseline did not. The measured regression has been accepted for this
precision fix; this does not imply upstream CI or merge approval.

Historically, an Ascend950 A/B/B/A comparison at B=1/H=96/T=8192 gave median
aicore time 5.994295 ms before the tail fix and 5.999095 ms after it:
0.08% higher time. Each version has twenty measured samples from two runs,
excluding three warmups per run. The ranges overlap: 5.969770-6.019850 ms
before and 5.978840-6.013120 ms after. Both versions compute all seven
outputs. Tail padding adds no GM intermediates or global barriers.

Measurements must compare complete Stage0-12 implementations with all seven
outputs. The previous Stage0-10 result (5.439135 ms at B=1, H=96, T=8192)
omits gate backward/reduction and is not an equivalent-work baseline.
Use three warmups followed by ten measured launches; exclude the first three
rows when msprof records both. Report aicore time, not enqueue or Task Wait
time, and distinguish kernel time from end-to-end backward latency.
