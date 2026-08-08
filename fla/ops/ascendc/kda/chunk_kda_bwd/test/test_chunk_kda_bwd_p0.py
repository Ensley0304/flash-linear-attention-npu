import ctypes
import math
import os

import torch
import torch_npu
from fla_npu.ops import ascendc as fla_ascendc
from fla_npu.ops.ascendc import _aclnn_ctypes as aclnn


def nd_tensor(ctx, tensor, name):
    return ctx.tensor(
        tensor,
        name,
        acl_format_override=aclnn.ACL_FORMAT_ND,
        storage_shape_override=tuple(tensor.shape),
    )


def run_dav(aqk, v_new, d_o, scale):
    outputs = (
        torch.empty_like(aqk, dtype=torch.float32),
        torch.empty_like(v_new),
    )
    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdDav",
        lambda ctx: [
            nd_tensor(ctx, aqk, "Aqk"),
            nd_tensor(ctx, v_new, "v_new"),
            nd_tensor(ctx, d_o, "do"),
            ctypes.c_float(scale),
            ctypes.c_int64(64),
            nd_tensor(ctx, outputs[0], "dAqk"),
            nd_tensor(ctx, outputs[1], "dv0"),
        ],
        outputs,
    )


def run_wy(q, k, v, v_new, gk, beta, akk, h, d_o, dh, dv_scan, scale):
    inputs = (q, k, v, v_new, gk, beta, akk, h, d_o, dh, dv_scan)
    outputs = (
        torch.empty_like(q, dtype=torch.float32),
        torch.empty_like(k, dtype=torch.float32),
        torch.empty_like(v),
        torch.empty_like(beta, dtype=torch.float32),
        torch.empty_like(gk),
        torch.empty_like(akk, dtype=torch.float32),
    )
    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdWy",
        lambda ctx: [
            *(nd_tensor(ctx, tensor, name) for tensor, name in zip(
                inputs,
                (
                    "q", "k", "v", "v_new", "gk", "beta", "Akk", "h",
                    "do", "dh", "dv_scan",
                ),
            )),
            ctypes.c_float(scale),
            ctypes.c_int64(64),
            *(nd_tensor(ctx, tensor, name) for tensor, name in zip(
                outputs, ("dq", "dk", "dv", "db", "dg", "dAkk")
            )),
        ],
        outputs,
    )


def run_gate_post(dg_hv):
    output = torch.empty_like(dg_hv)
    return aclnn._call_aclnn(
        "aclnnChunkKdaBwdGatePost",
        lambda ctx: [
            nd_tensor(ctx, dg_hv, "dg_hv"),
            ctypes.c_int64(64),
            nd_tensor(ctx, output, "dg"),
        ],
        output,
    )


def run_low_level(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o):
    scale = 1.0 / math.sqrt(q.shape[-1])
    d_aqk, dv0 = run_dav(aqk, v_new, d_o, scale)
    torch.npu.synchronize()
    dh, _, dv_scan = fla_ascendc.chunk_gated_delta_rule_bwd_dhu(
        qg, kg, w, d_o, dv0, scale=scale, chunk_size=64,
        g=None, gK=gk, h0=None, dht=None, cu_seqlens=None,
        chunk_indices=None,
    )
    torch.npu.synchronize()
    dq0, dk0, dv, db0, dg0, d_akk = run_wy(
        q, k, v, v_new, gk, beta, akk, h, d_o, dh, dv_scan, scale
    )
    torch.npu.synchronize()
    if os.environ.get("KDA_BWD_STAGE_DIAG", "0") == "1":
        torch.npu.synchronize()
        print(
            "after K3 "
            f"dq={dq0.float().abs().max().cpu().item():.6g} "
            f"dk={dk0.float().abs().max().cpu().item():.6g} "
            f"dv={dv.float().abs().max().cpu().item():.6g} "
            f"db={db0.float().abs().max().cpu().item():.6g} "
            f"dg={dg0.float().abs().max().cpu().item():.6g} "
            f"dAkk={d_akk.float().abs().max().cpu().item():.6g}",
            flush=True,
        )
    dq, dk, db, dg_hv = fla_ascendc.chunk_kda_bwd_intra(
        q, k, gk, beta, d_aqk, d_akk, dq0, dk0, db0, dg0,
        cu_seqlens=None, chunk_indices=None, chunk_size=64,
        safe_gate=True, layout="BNSD",
    )
    torch.npu.synchronize()
    if os.environ.get("KDA_BWD_STAGE_DIAG", "0") == "1":
        torch.npu.synchronize()
        print(
            "after K4 "
            f"dq={dq.float().abs().max().cpu().item():.6g} "
            f"dk={dk.float().abs().max().cpu().item():.6g} "
            f"db={db.float().abs().max().cpu().item():.6g} "
            f"dg={dg_hv.float().abs().max().cpu().item():.6g}",
            flush=True,
        )
    dg = run_gate_post(dg_hv)
    torch.npu.synchronize()
    return dq, dk, dv, db, dg


def run_case(layout: str) -> None:
    torch.manual_seed(20260805)
    b, h_num, t, d = 1, 4, 128, 128
    nt = (t + 63) // 64
    shape = (b, h_num, t, d)
    q_h = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    k_h = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    v_h = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    beta_h = (torch.rand((b, h_num, t), dtype=torch.bfloat16) * 0.1).npu()
    gk = (-torch.rand(shape, dtype=torch.float32) * 0.02).npu()
    aqk = (torch.randn((b, h_num, t, 64), dtype=torch.bfloat16) * 0.01).npu()
    akk = (torch.randn((b, h_num, t, 64), dtype=torch.bfloat16) * 0.01).npu()
    w = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    qg = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    kg = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    v_new = (torch.randn(shape, dtype=torch.bfloat16) * 0.01).npu()
    state = (torch.randn((b, nt, h_num, d, d), dtype=torch.bfloat16) * 0.01).npu()
    d_o_s = (torch.randn((b, t, h_num, d), dtype=torch.bfloat16) * 0.01).npu()
    d_o_h = d_o_s.transpose(1, 2).contiguous()
    expected = run_low_level(
        q_h, k_h, v_h, beta_h, gk, aqk, akk, w, qg, kg, v_new,
        state, d_o_h,
    )
    if layout == "BSND":
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q_h, k_h, v_h)]
        beta = beta_h.transpose(1, 2).contiguous()
        expected = tuple(x.transpose(1, 2).contiguous() for x in expected)
    else:
        q, k, v, beta = q_h, k_h, v_h, beta_h
    actual = fla_ascendc.chunk_kda_bwd(
        q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, state, d_o_s,
        1.0 / math.sqrt(d), raw_g=None, A_log=None, dt_bias=None,
        initial_state=None, dht=None,
        cu_seqlens=None, chunk_indices=None, layout=layout,
        chunk_size=64, safe_gate=True,
        lower_bound=-5.0, use_gate_in_kernel=False, state_v_first=False,
        recompute_policy="NONE",
    )
    torch.npu.synchronize()
    for name, got, want in zip(("dq", "dk", "dv", "db", "dg"), actual[:5], expected):
        got_finite = int(torch.isfinite(got).sum().cpu())
        want_finite = int(torch.isfinite(want).sum().cpu())
        print(
            f"{layout} {name}: actual_finite={got_finite}/{got.numel()} "
            f"reference_finite={want_finite}/{want.numel()} "
            f"actual_abs_max={got.float().abs().max().cpu().item():.6g} "
            f"reference_abs_max={want.float().abs().max().cpu().item():.6g}",
            flush=True,
        )
        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
    assert actual[5] is None and actual[6] is None and actual[7] is None


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    run_case("BNSD")
    run_case("BSND")
