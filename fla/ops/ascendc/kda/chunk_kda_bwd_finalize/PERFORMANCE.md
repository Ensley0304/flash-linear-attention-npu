# Two-head UB residency

The two-head schedule assigns one head to each AIV in a window. Keeping
that head's inputs in UB removes repeated GM transfers between Vector
stages. This change preserves the Cube products, all residual
products, FP32 arithmetic order, output types and public interfaces.

`RunStateAndBase` executes the original Stage3 and Stage4 Vector functions
consecutively. Stage4 consumes Stage3's FP32 UB results directly. Once
Stage3 finishes reading `gk` and `v`, a V-to-MTE2 event permits those regions
to receive `dKgb_raw` and `q`. The completed base gradients are then stored
for the later stages. No additional quantization or reassociation is used.

## UB lifetimes

Offsets below are per AIV. Intervals are half-open, in KiB, for a full
64-token chunk with K=V=128. Short chunks use the same layout and process
only valid rows; existing Cube tail padding remains unchanged.

| Data | Interval | Producer → final consumer |
|---|---|---|
| q (BF16) | [144,160) | Stage3/4 input → Stage7, then overwritten by dq |
| k (BF16) | [160,176) | Stage0 input → Stage9 |
| exp2(gk) (FP32) | [176,208) | Stage0 → Stage9 |
| beta (BF16) | [208,208.125) | Stage2 input → Stage9 |
| dbeta delta (FP32, two event slots) | [209,209.5) | Stage9 → Stage10 |
| q_rstd (FP32) | [212,212.25) | Stage7 only |
| dg (FP32) | [216,248) | Stage7 → Stage9 → Stage11 |

Stage0 temporarily uses [144,160) for kE, [208,208.5) for gk_last and
[209,209.5) for r_h. Their DMA readers finish before those regions change
meaning. Stage5 writes local operands in [16,144), below the retained
inputs, and reads dAqk in [216,232). Stage7 reuses that dead dAqk region
for dg. Stage11 reuses dead k/exp regions for gate inputs and scratch;
dbeta delta at 209 KiB remains intact until Stage10 consumes it.

The Stage7 dg result no longer makes a GM round trip before Stage9.
Stage0 no longer stores exp2(gk) or the unused kE workspace copy. Reserved
GM slot offsets and Host workspace allocation remain unchanged.

## Ownership and precision

- Retained inputs rely on one head per AIV per window, enforced by a
  compile-time assertion. This layout cannot be copied into a four-head
  window without a different lifetime plan.
- Generation and event-slot rotation remain unchanged, including odd-head
  and packed-tail windows. Slot numbers do not select independent copies
  of retained inputs.
- Existing MTE3-reader and cross-core READY/FREE handshakes remain in place.
  The new V-to-MTE2 event protects Stage3's last reads before Stage4 input
  DMA overwrites the same UB addresses.
- Stage9 needs no new MTE2 input. Cube READY flags protect its raw inputs,
  while Vector stage order protects retained inputs. Its V-to-MTE2 event
  remains necessary before Stage11 overwrites k/exp storage.
- Cube products and Vector function bodies match the precision baseline.
  Reusing L0 operands was evaluated separately and excluded because its
  measured benefit was negligible.

## Tail clear dependency

Mixed-profile repeated launches exposed intermittent zeroed column blocks
in Stage0's DVb path. The original two-head baseline also reproduced the
failure. In saved failing outputs, affected dv elements were exactly zero,
with errors propagating to dk, dbeta and gate gradients; dq was unchanged.
The same saved inputs produce the reference outputs when rerun.

`LoadGmToL1A/B` clears a full physical L1 tile before loading its valid
sub-tile. These writes overlap. A `PipeBarrier<PIPE_MTE2>()` immediately
after `InitConstValue` makes the clear complete before ND-to-NZ DMA writes
valid data. The barrier is inside the existing partial-tile condition;
full 64-token chunks do not execute it. No extra MMAD or global barrier is
added. A diagnostic build with only these two barriers added to the UB
candidate passed 10000 mixed tail launches; the original baseline had two
unstable launches in the same 10000-launch test.

The targeted regression can be repeated with the desired wheel selected:

```bash
python tests/check_repeatability.py --atk-root /path/to/aclnnChunkKdaBwdFinalize \
  --output /path/to/new-result-directory --repeats 10000 --device 0
```

It preserves the original ATK input generator and tests three seed offsets.
It checks all seven outputs bytewise, including signed zeros, and saves the
inputs plus expected/actual outputs on failure. Its nonzero exit status
prevents an unstable build from being accepted just because a single ATK
pass succeeded.

## Validation and measurements

The clean single-operator wheel passes the original 200-case ATK suite at
seed bases 20260906, 20270906 and 20280906: 600/600 case runs and 4200/4200
output comparisons. The original FP64 and FP32/BF16 oracle and thresholds
are unchanged. The same wheel passes 10000 mixed tail launches without
bytewise differences in any of the seven outputs. Each seed family includes
the original 8K and 16K cases. These results cover the tested seeds and
Ascend950 environment, rather than claiming all possible random inputs.

Wheel SHA256: `a508184f25949ea9b13b6b91768d82faef0e3462c7c5fc1d5f64eda5f7323c72`.
The four kernel headers embedded in the wheel match the source tree.

Performance uses complete Stage0–12 kernel time, all seven outputs,
and A/B/B/A runs on the same Ascend950 device. Each run excludes three
warmups and retains ten measured launches. Python enqueue time and Task
Wait are excluded.

| B / H / T | Baseline 53f1eb30 median | Optimized median | Latency reduction |
|---|---:|---:|---:|
| 1 / 96 / 8192 | 6.523005 ms | 6.429630 ms | 1.43% |
| 1 / 96 / 16384 | 13.076435 ms | 12.823220 ms | 1.94% |

All configurations use K=V=128 with Q/K normalization disabled. At 8K,
baseline samples span 6.507600–6.541890 ms and optimized samples span
6.401860–6.448430 ms. At 16K, the ranges are 13.048140–14.787780 ms and
12.774500–12.842350 ms. The baseline's high 16K sample is retained; the
table uses medians without removing outliers. These measurements describe
KernelC device time, not complete backward latency.
