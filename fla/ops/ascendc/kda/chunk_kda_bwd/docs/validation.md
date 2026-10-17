# Issue 874 repair validation

Base: 8c5dd635. A2 910B3, CANN 9.1, torch 2.7.1, torch_npu 2.7.1.post5.
This addresses the reproduced dense/segment path; the reported long packed
production call has not been reproduced from the synthetic inputs.

Baseline dense H1 T64, head padding disabled: workspace byte fill 255 causes
8192 nonfinite dk, 8192 dg, 63 dbeta, 1 dA and 128 dbias; dq/dv stay unchanged.
Fill 63 causes dk max difference 0.033333823 vs the initial zero-fill result.
Packed H1 and dense H2 controls are independent of these fills.
Warm baseline bisection excludes final dAkk as the initial poisoned source;
an internal scratch 512-byte region is sufficient to alter the result.
Do not infer an exact hardware synchronization mechanism from these tests.

Diagnostic build repeating S6/S7 twice on all generic windows passed all 12
workspace-fill cases and 40 Stable ABI calls, including unpadded short tails.
The production candidate limits the extra replay to lone-head full dense
windows. Existing packed/tail replay, A5 routing, ABI and head padding remain.

Completed candidate regression covers
H=1/2/3/4, T=63/64/65/128, packed controls, workspace fills 0/63/255,
and repeated Stable ABI calls. Check every output for finite values and exact
independence from workspace initialization. This is not full precision or
performance acceptance; no tolerance or golden input domain is changed.

Candidate build and focused regression completed: 96 workspace-fill calls
and 40 Stable ABI calls passed finite/repeat checks. All workspace-fill
differences were exactly zero. Full precision/performance acceptance remains pending.
