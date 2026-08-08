import math
import os

import torch
import torch_npu

from fla_npu.ops import ascendc


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    batch, heads, seqlen, dim, chunk = (
        1,
        int(os.getenv("KDA_BWD_H", "2")),
        int(os.getenv("KDA_BWD_T", "64")),
        128,
        64,
    )
    chunks = (seqlen + chunk - 1) // chunk
    device = "npu:0"
    fast_input = os.getenv("KDA_BWD_FAST_INPUT", "0") == "1"

    def rand(shape, scale=0.01):
        if fast_input:
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return (torch.randn(shape, dtype=torch.bfloat16) * scale).to(device)

    head_shape = (batch, heads, seqlen, dim)
    q_h = rand(head_shape)
    k_h = rand(head_shape)
    v_h = rand(head_shape)
    beta_dtype = (
        torch.float32
        if os.getenv("KDA_BWD_BETA_FP32", "0") == "1"
        else torch.bfloat16
    )
    if fast_input:
        beta_h = torch.zeros(
            (batch, heads, seqlen), dtype=beta_dtype, device=device
        )
        gk = torch.zeros(head_shape, dtype=torch.float32, device=device)
    else:
        beta_h = (
            torch.rand((batch, heads, seqlen), dtype=beta_dtype) * 0.1
        ).to(device)
        gk = (-torch.rand(head_shape, dtype=torch.float32) * 0.02).to(device)
    aqk = rand((batch, heads, seqlen, chunk))
    akk = rand((batch, heads, seqlen, chunk))
    w = rand(head_shape)
    qg = rand(head_shape)
    kg = rand(head_shape)
    v_new = rand(head_shape)
    h = rand((batch, chunks, heads, dim, dim))
    d_o = rand((batch, seqlen, heads, dim))
    scale = 1.0 / math.sqrt(dim)

    def run(layout, q, k, v, beta):
        event_timing = os.getenv("KDA_BWD_EVENT_TIMING", "0") == "1"
        if event_timing:
            torch.npu.synchronize()
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start_event.record()
        outputs = ascendc.chunk_kda_bwd(
            q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o,
            scale, raw_g=None, A_log=None, dt_bias=None,
            initial_state=None, dht=None, cu_seqlens=None,
            chunk_indices=None, layout=layout, chunk_size=chunk,
            safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=False,
            state_v_first=False, recompute_policy="NONE",
        )
        if event_timing:
            end_event.record()
        torch.npu.synchronize()
        if event_timing:
            print(
                f"L2 {layout} device_elapsed_ms="
                f"{start_event.elapsed_time(end_event):.6f}",
                flush=True,
            )
        if os.getenv("KDA_BWD_LAUNCH_ONLY", "0") == "1":
            print(f"L2 {layout} launch-only pass", flush=True)
            return outputs[:5]
        assert outputs[5:] == (None, None, None)
        names = ("dq", "dk", "dv", "db", "dg")
        for name, tensor in zip(names, outputs[:5]):
            finite = torch.isfinite(tensor)
            if not bool(finite.all().cpu()):
                bad = (~finite).nonzero()[:64].cpu().tolist()
                print(f"L2 {layout} {name} first_nonfinite={bad}", flush=True)
            print(
                f"L2 {layout} {name} finite="
                f"{int(finite.sum().cpu())}/{tensor.numel()} "
                f"finite_abs_max={tensor.float()[finite].abs().max().item():.6g}",
                flush=True,
            )
        if os.getenv("KDA_BWD_ALLOW_NONFINITE", "0") != "1":
            assert all(bool(torch.isfinite(x).all().cpu()) for x in outputs[:5])
        return outputs[:5]

    layout_only = os.getenv("KDA_BWD_LAYOUT_ONLY", "")
    bnsd = None
    if layout_only != "BSND":
        bnsd = run("BNSD", q_h, k_h, v_h, beta_h)
    if os.getenv("KDA_BWD_REPEAT_BNSD", "0") == "1":
        repeated = run("BNSD", q_h, k_h, v_h, beta_h)
        for name, first, second in zip(
            ("dq", "dk", "dv", "db", "dg"), bnsd, repeated
        ):
            diff = (first.float() - second.float()).abs().max().item()
            print(f"L2 repeat BNSD {name} max_diff={diff:.6g}", flush=True)
        return
    if layout_only == "BNSD":
        print("L2_SMOKE_PASS", flush=True)
        return
    bsnd = run(
        "BSND",
        q_h.transpose(1, 2).contiguous(),
        k_h.transpose(1, 2).contiguous(),
        v_h.transpose(1, 2).contiguous(),
        beta_h.transpose(1, 2).contiguous(),
    )

    if bnsd is None:
        print("L2_SMOKE_PASS", flush=True)
        return

    names = ("dq", "dk", "dv", "db", "dg")
    for name, got_h, got_s in zip(names, bnsd, bsnd):
        converted = got_s.transpose(1, 2).contiguous()
        diff = (got_h.float() - converted.float()).abs().max().item()
        raw = got_s.reshape(got_h.shape)
        raw_diff = (got_h.float() - raw.float()).abs().max().item()
        print(
            f"L2 {name} max_layout_diff={diff:.6g} "
            f"max_raw_storage_diff={raw_diff:.6g}",
            flush=True,
        )
        if os.getenv("KDA_BWD_ALLOW_NONFINITE", "0") != "1":
            torch.testing.assert_close(got_h, converted, rtol=2e-4, atol=2e-4)

    print("L2_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
