# FwdH K256 design-v2 gate probes

These are isolated development probes, not an enabled production route.
Authoritative input: FINAL_DESIGN.md version 2.0 dated 2026-11-27.
Base: 9fb8247a975618bcb1b496b0f8a60a527db00bbf. Original dirty trees preserved.

Candidate P32 uses FP32 operands and FP32 R_run; HF32 explicitly disabled.
S0 accumulates four K64 packets in one L0C result and publishes once.
S2 covers the complete valid token reduction in a padded K64 packet.
Only valid GM input rectangles are read; L1 is cleared before partial loads.

## Candidate allocation (bytes, not frozen production configuration)

| Storage | Absolute half-open range | Owner / lifetime |
|---|---|---|
| L1 | [0,32768), [32768,65536) | AB packet banks; each A/B occupies 16384 bytes; MTE2 write to MTE1 last read |
| L0A | [0,16384), [16384,32768) | packet banks, MTE1 write to MMAD last read |
| L0B | [0,16384), [16384,32768) | same |
| L0C | [0,16384), [16384,32768) | output banks, first MMAD through final Fixpipe read |
| UB | [0,65536) | one FP32 [256,64] R_run per active AIV |
| UB | [65536,66560) | FP32 gate/alpha [256] |
| UB | [66560,74752) | Brcb alpha [256,8], consumed by row-broadcast Mul |
| UB | [74752,82944), [82944,91136) | two input buffers; U/P or D, final Vector read before reuse |
| UB | [91136,95232), [95232,99328) | two output buffers; all Vnew/right or state writes before reuse |
| GM per core/head | [0,16384) | P [64,64] |
| GM per core/head | [16384,32768) | right [64,64] FP32 |
| GM per core/head | [32768,98304) | D [256,64] FP32 |

Physical offsets follow InitBuffer call order. Prototype TPipe queue metadata is
additional framework allocation; capacity must be verified in emitted metadata.
Gh=1 or 2, Vtile=64. lane0=head0, lane1=head1 or dummy. Each work owns all K.
R_run lives in UB throughout its work. Public FP32 snapshots supply the Cube
state view; snapshot writes are not optional, and no extra quantized state exists.

All six local event directions allocate two independent IDs dynamically and
release after drain. L1_FREE, AB_FREE and C_FREE start with one credit per bank.
MTE2_READY and MTE1_READY are each set/consumed once per packet. C_READY is set/
consumed once per completed reduction. L1 and L0AB rotate by packet, C by result.
Next packet MTE2 is issued before current packet LoadData/MMAD; MMAD retirement
is waited only when its AB bank is reused. Fixpipe retirement is waited only
when its C bank is reused. This code structure requires timeline verification.

MIX mode2 flags: H_READY=0, P_READY=1, P_FREE=2, RIGHT_READY=3, RIGHT_FREE=4,
D_READY=5, D_FREE=6, DONE=7, ACK=8. Both AIV lanes participate, including dummy.
This probe uses AIV-prepared first snapshot, so H_READY occurs C times (including
first chunk); it deliberately does not use the initial-state direct-read ledger.
P_READY/P_FREE each occur C-1+I times. RIGHT_READY/RIGHT_FREE/D_READY/D_FREE each
occur C-1+F times. DONE/ACK occur once per work. No seed P_FREE is needed; AIC
waits only if it actually produced a previous P. Final drains are independent.

CPU numerical controls do not qualify the production operator. `producer_gate.py`
uses mathematical producer caches and upstream recurrent assertions, while
`test_cube.py` and `test_mix.py` compare actual NPU outputs with FP64 controls.
Public KDA/gradient correctness, original producer outputs, timeline overlap,
remaining chain budget, installed package provenance and production dispatch
remain separate required gates.

## Long-sequence experiment (2026-11-27)

Two independent CMake switches default OFF, preserving the original probe:
`FWDH_RIGHT_RESIDENT=ON` reserves another 16384-byte L1 B1 region (L1 total
81920 bytes), copies S2 right once and reuses it across four output blocks.
Two extra dynamic local events own its MTE2/MTE1 ready/free lifecycle.
The last MTE1 read, not the end of MMAD, releases this L1 operand.

`FWDH_D_BLOCK_PIPELINE=ON` enables Gh1 block notifications: four D blocks use
mode2 IDs 5/9/10/11, one outstanding credit each per chunk. Both AIV lanes
consume each notification (including dummy). Whole-chunk D_FREE drains all
four before any next-chunk D overwrite. RIGHT_FREE is consumed after S3 blocks,
avoiding an all-S2 barrier before the first S3 block. H_READY still requires
all state updates/writes; no next-chunk partial state is consumed. Gh2 retains
the previous whole-D protocol. FP32 math and state storage are unchanged.

Run `test_mix.py --long --profile` for T1024/4096, H1/H2, Gh1, I0F0/I1F1.
All eight configurations passed FP64 output checks for baseline, resident-only,
and resident+block versions on A2 physical6. Each has 10 warmups and 10 tasks.
The final version reduced device duration 9.18%-10.48% from the original;
resident-only gave 0.21%-1.62%. One interrupted resident-only batch detected
an external device process and was discarded; use the `-retry` evidence.
T65 quick tests include Gh2 fallback and the M1 update tail with final state.
H1/T65 simulator completed and now shows 0.146668us Cube/Vector instruction
intersection (formerly zero); this is short-trace evidence, not a measured
long-sequence overlap fraction or hardware latency.

These remain isolated stage probes. Public dispatch and strict same-cache
Triton pairing have not been validated by this experiment.
