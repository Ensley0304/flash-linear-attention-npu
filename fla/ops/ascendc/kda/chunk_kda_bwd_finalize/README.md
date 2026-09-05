# ChunkKdaBwdFinalize: Stage0-10 Milestone

This is an incomplete Ascend950 backward fusion implementation, not a complete
KDA backward operator. Stage0-10 produce `dq`, `dk`, `dv` and `dBeta`.
Stage11/12 are not implemented: `dG`, `dALog` and `dDtBias` remain untouched
and must not be consumed as valid gradients. Public Host/aclnn ABI is unchanged.

## Implementation

- Stage0-5 form the base gradients and local operands.
- Stage6/7 compute the local Q contribution and optional Q normalization.
- Stage8 uses one left GEMM and a concatenated-reduction right GEMM.
- Stage9/10 combine the K/beta gradients and optional K normalization.
- Stage5 operands move directly from UB to L1 using strided vector-mode MTE3.
- Directional events protect overlapping UB/L1 lifetimes; Stage0 uses
  per-stream last-reader credits.
- Vector FP32-to-BF16 conversions use nearest-even rounding, including exact
  FP32 midpoint inputs. Shared conversion helpers are not modified.
- Stage4 reuses the FP32 exponential-gradient product for Vector beta/gate
  gradients instead of consuming the rounded BF16 Cube operand `kE`.

## Validation Scope

CPU-only dual L1 checks cover B=1, K=V=128, chunk size 64, H/T=1/64,
4/128, 7/128 with Q/K normalization, 4/4096, and 96/8192.
The four produced outputs pass
CT 0.7.1's default BF16 L1 settings. Five repeated launches are bitwise exact.
A rebuilt single-operator wheel reproduces the validated 8K outputs bitwise.
These results do not establish untested tail/varlen/platform coverage or full
GDN regression, and are not GPU dual-benchmark or merge-readiness evidence.

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

The oracle supplies `kernel_c_base_torch` and `kernel_c_intra_torch`. Golden
uses FP64 without simulated Cube downcasts; benchmark uses FP32 accumulation,
BF16 Cube boundaries and BF16 outputs. Inputs are never changed by this check.
The checker records the oracle hash and raw RMSE ratios, in addition to CT's
ratios, whose denominators include dtype-dependent lower bounds.

`tests/check_bf16_midpoints.py --write-inputs inputs.pt` generates the exact
cast regression input. After an aclnn launch, use `--outputs outputs.pt` to
check its four saved outputs bitwise. This is an exact rounding check,
separate from the numerical dual-benchmark tolerance.

## Performance

For B=1, H=96, T=8192, same-session hardware profiling with three warmups and
ten measured launches gives median aicore time 5.589530 ms before optimization
and 5.439135 ms after optimization and both precision fixes: 2.69% lower time.
The baseline is the preceding Stage0-10 implementation, not the Stage0-5
source milestone. This is kernel time, not end-to-end backward latency.
