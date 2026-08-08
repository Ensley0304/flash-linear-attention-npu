import math
import os

import torch
import torch_npu

from fla_npu.ops import ascendc


def timed(name, fn):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    outputs = fn()
    end.record()
    torch.npu.synchronize()
    print(f"{name} device_elapsed_ms={start.elapsed_time(end):.6f}", flush=True)
    return outputs


def warmed_timed(name, fn):
    # Exclude first-use runtime/kernel-loading overhead from the device timing.
    fn()
    torch.npu.synchronize()
    outputs = None
    repeats = int(os.getenv("KDA_BWD_TIMING_REPEATS", "1"))
    for index in range(repeats):
        outputs = timed(f"{name}[{index}]", fn)
    return outputs


def main():
    torch.npu.set_device(0)
    batch = 1
    heads = int(os.getenv("KDA_BWD_H", "96"))
    seqlen = int(os.getenv("KDA_BWD_T", "18432"))
    dim = 128
    chunk = 64
    device = "npu:0"
    scale = 1.0 / math.sqrt(dim)
    which = os.getenv("KDA_BWD_KERNEL", "K2").upper()

    head_shape = (batch, heads, seqlen, dim)
    scalar_shape = (batch, heads, seqlen)
    matrix_shape = (batch, heads, seqlen, chunk)

    if which == "K1":
        from kda_bwd_dav_repeat import run_dav

        aqk = torch.zeros(matrix_shape, dtype=torch.bfloat16, device=device)
        v_new = torch.zeros(head_shape, dtype=torch.bfloat16, device=device)
        d_o = torch.zeros_like(v_new)
        warmed_timed("K1", lambda: run_dav(aqk, v_new, d_o))
    elif which == "K2":
        q = torch.zeros(head_shape, dtype=torch.bfloat16, device=device)
        k = torch.zeros_like(q)
        w = torch.zeros_like(q)
        d_o = torch.zeros_like(q)
        dv = torch.zeros_like(q)
        gk = torch.zeros(head_shape, dtype=torch.float32, device=device)
        warmed_timed(
            "K2",
            lambda: ascendc.chunk_gated_delta_rule_bwd_dhu(
                q,
                k,
                w,
                d_o,
                dv,
                scale,
                chunk,
                gK=gk,
            ),
        )
    elif which == "K4":
        q = torch.zeros(head_shape, dtype=torch.bfloat16, device=device)
        k = torch.zeros_like(q)
        gk = torch.zeros(head_shape, dtype=torch.float32, device=device)
        beta = torch.zeros(scalar_shape, dtype=torch.bfloat16, device=device)
        d_aqk = torch.zeros(matrix_shape, dtype=torch.float32, device=device)
        d_akk = torch.zeros_like(d_aqk)
        dq = torch.zeros(head_shape, dtype=torch.float32, device=device)
        dk = torch.zeros_like(dq)
        db = torch.zeros(scalar_shape, dtype=torch.float32, device=device)
        dg = torch.zeros_like(dq)
        warmed_timed(
            "K4",
            lambda: ascendc.chunk_kda_bwd_intra(
                q,
                k,
                gk,
                beta,
                d_aqk,
                d_akk,
                dq,
                dk,
                db,
                dg,
                chunk_size=chunk,
                safe_gate=True,
                layout="BNSD",
            ),
        )
    elif which == "K6":
        from kda_bwd_gate_post_repeat import run_gate_post

        dg_hv = torch.zeros(head_shape, dtype=torch.float32, device=device)
        warmed_timed("K6", lambda: run_gate_post(dg_hv))
    else:
        raise ValueError(
            f"Unsupported KDA_BWD_KERNEL={which}; use K1, K2, K4, or K6"
        )

    print(f"{which}_TARGET_PASS", flush=True)


if __name__ == "__main__":
    main()
