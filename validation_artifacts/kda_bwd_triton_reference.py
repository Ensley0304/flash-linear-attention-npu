import math
import os

import torch
import torch_npu

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.kda.chunk_bwd import (
    chunk_kda_bwd as triton_chunk_kda_bwd,
    chunk_kda_bwd_dAv,
    chunk_kda_bwd_wy_dqkg_fused,
)
from fla.ops.kda.chunk_fwd import chunk_kda_fwd as triton_chunk_kda_fwd
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.utils import chunk_local_cumsum


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    batch = 1
    seqlen = int(os.getenv("KDA_REF_T", "64"))
    heads = int(os.getenv("KDA_REF_H", "1"))
    dim = 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(dim)
    magnitude = float(os.getenv("KDA_REF_MAG", "0.01"))
    beta_scale = float(os.getenv("KDA_REF_BETA_SCALE", "0.1"))
    device = "npu:0"

    def randn(shape, magnitude=0.01):
        return (torch.randn(shape, dtype=torch.bfloat16) * magnitude).to(device)

    q = randn((batch, seqlen, heads, dim), magnitude)
    k = randn((batch, seqlen, heads, dim), magnitude)
    v = randn((batch, seqlen, heads, dim), magnitude)
    # The public non-fused gate is a per-token natural-log decay.  Keep it in
    # the safe-gate range expected by the P0 kernels.
    g = (-torch.rand((batch, seqlen, heads, dim), dtype=torch.float32) * 0.02).to(device)
    beta = (torch.rand((batch, seqlen, heads), dtype=torch.bfloat16) * beta_scale).to(device)
    do = randn((batch, seqlen, heads, dim), magnitude)

    print("triton forward begin", flush=True)
    forward = triton_chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        chunk_size=chunk_size,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        disable_recompute=True,
        return_intermediate_states=False,
        cp_context=None,
    )
    torch.npu.synchronize()
    print("triton forward end", flush=True)
    (
        _, _, gk, aqk, akk, w, u, qg, kg, v_new, h, initial_state
    ) = forward
    del u

    print("triton backward begin", flush=True)
    expected = triton_chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        Aqk=aqk,
        Akk=akk,
        scale=scale,
        initial_state=initial_state,
        do=do,
        dht=None,
        g=gk,
        g_org=None,
        state_v_first=False,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk_size,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        disable_recompute=True,
        cp_context=None,
        w=w,
        u=forward[6],
        qg=qg,
        kg=kg,
        v_new=v_new,
        h=h,
    )
    torch.npu.synchronize()
    print("triton backward end", flush=True)

    d_aqk, dv0 = chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do=do, A=aqk, scale=scale,
        cu_seqlens=None, chunk_size=chunk_size, chunk_indices=None,
    )
    dh, _, dv_scan = chunk_gated_delta_rule_bwd_dhu(
        q=qg, k=kg, w=w, gk=gk, h0=None, dht=None, do=do, dv=dv0,
        scale=scale, cu_seqlens=None, chunk_size=chunk_size,
        chunk_indices=None, state_v_first=False,
    )
    k3 = chunk_kda_bwd_wy_dqkg_fused(
        q=q, k=k, v=v, v_new=v_new, g=gk, beta=beta, A=akk,
        h=h, do=do, dh=dh, dv=dv_scan, scale=scale,
        cu_seqlens=None, chunk_size=chunk_size, chunk_indices=None,
        state_v_first=False,
    )
    k4 = chunk_kda_bwd_intra(
        q=q, k=k, g=gk, beta=beta, dAqk=d_aqk, dAkk=k3[5],
        dq=k3[0], dk=k3[1], db=k3[3], dg=k3[4], cu_seqlens=None,
        chunk_size=chunk_size, chunk_indices=None, safe_gate=True,
    )
    gate = chunk_local_cumsum(k4[3], chunk_size=chunk_size, reverse=True)
    torch.npu.synchronize()

    dump_path = os.getenv("KDA_REF_DUMP", "/tmp/kda_bwd_triton_reference.pt")
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "gk": gk,
        "aqk": aqk,
        "akk": akk,
        "w": w,
        "qg": qg,
        "kg": kg,
        "v_new": v_new,
        "h": h,
        "do": do,
        "expected": tuple(expected[:5]),
        "stages": {
            "k1": (d_aqk, dv0),
            "k2": (dh, dv_scan),
            "k3": tuple(k3),
            "k4": tuple(k4),
            "k6": (gate,),
        },
        "scale": scale,
    }
    torch.save(
        {
            key: (
                tuple(item.detach().cpu() for item in value)
                if isinstance(value, tuple)
                else {
                    stage: tuple(item.detach().cpu() for item in items)
                    for stage, items in value.items()
                }
                if isinstance(value, dict)
                else value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in tensors.items()
        },
        dump_path,
    )
    print(f"KDA_BWD_TRITON_DUMP={dump_path}", flush=True)


if __name__ == "__main__":
    main()
