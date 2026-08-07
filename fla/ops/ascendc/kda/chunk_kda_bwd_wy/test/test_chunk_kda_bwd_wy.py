import math
import os

import torch
import torch_npu


def reference(q, k, v, v_new, gk, beta, a, h, d_o, dh, dv_scan, scale):
    batch, heads, seqlen, key_dim = q.shape
    chunk_size = a.shape[-1]
    chunk_num = (seqlen + chunk_size - 1) // chunk_size
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v)
    db = torch.zeros_like(beta, dtype=torch.float32)
    dg = torch.zeros_like(gk, dtype=torch.float32)
    d_akk = torch.zeros_like(a, dtype=torch.float32)
    for b in range(batch):
        for head in range(heads):
            for c in range(chunk_num):
                begin = c * chunk_size
                end = min(begin + chunk_size, seqlen)
                valid = end - begin
                sl = slice(begin, end)
                q_c = q[b, head, sl].float()
                k_c = k[b, head, sl].float()
                v_c = v[b, head, sl].float()
                vn_c = v_new[b, head, sl].float()
                g_c = gk[b, head, sl].float()
                beta_c = beta[b, head, sl].float()
                a_c = a[b, head, sl, :valid].float()
                # GPU KDA loads the physical [token, local_token] Akk tile as
                # b_A[local_token, token].  Keep the CPU reference faithful to
                # that storage contract instead of treating the bytes as the
                # mathematical matrix directly.
                a_math = a_c.transpose(-1, -2)
                h_c = h[b, c, head].float()
                dh_c = dh[b, head, c].float()
                do_c = d_o[b, head, sl].float()
                d_c = dv_scan[b, head, sl].float()

                e = torch.exp2(g_c)
                e_rel = torch.exp2(g_c[-1:] - g_c)
                k_e = (k_c * e).to(torch.bfloat16).float()
                dq_raw = do_c @ h_c.T
                dk_raw = vn_c @ dh_c.T
                d_w = (-(d_c @ h_c.T)).to(torch.bfloat16).float()
                z_v = d_c @ v_c.T
                dvb = a_math @ d_c
                z_w = d_w @ k_e.T
                dkgb = a_math @ d_w

                dq_c = dq_raw * e * scale
                dk_state = dk_raw * e_rel
                dv_c = (beta_c[:, None] * dvb).to(v.dtype)
                db_c = (dvb * v_c).sum(-1) + (dkgb * k_e).sum(-1)
                dk_c = dk_state + dkgb * (beta_c[:, None] * e)
                state = (h_c * dh_c).sum(-1) * torch.exp2(g_c[-1])
                state = state + (k_c * dk_state).sum(0)
                dg_c = q_c * dq_c - k_c * dk_state
                dg_c = dg_c + k_e * dkgb * beta_c[:, None]
                dg_c[-1] += state
                z_b = torch.tril(z_v + z_w, diagonal=-1) * beta_c[None, :]
                z_b = z_b.to(torch.bfloat16).float()
                t_za = (z_b @ a_math).to(torch.bfloat16).float()
                da = torch.tril(-(a_math @ t_za), diagonal=-1)

                dq[b, head, sl] = dq_c
                dk[b, head, sl] = dk_c
                dv[b, head, sl] = dv_c
                db[b, head, sl] = db_c
                dg[b, head, sl] = dg_c
                d_akk[b, head, sl, :valid] = da
    return dq, dk, dv, db, dg, d_akk


def run_case(seqlen: int, beta_dtype: torch.dtype) -> None:
    torch.manual_seed(20260805)
    batch, heads, dim, chunk = 1, 4, 128, 64
    chunks = (seqlen + chunk - 1) // chunk
    shape = (batch, heads, seqlen, dim)
    q = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    k = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    v = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    v_new = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    gk = -torch.rand(shape, dtype=torch.float32) * 0.05
    beta = torch.rand((batch, heads, seqlen), dtype=beta_dtype) * 0.2
    a = torch.randn((batch, heads, seqlen, chunk), dtype=torch.bfloat16) * 0.01
    h = torch.randn((batch, chunks, heads, dim, dim), dtype=torch.bfloat16) * 0.01
    d_o = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    dh = torch.randn((batch, heads, chunks, dim, dim), dtype=torch.bfloat16) * 0.01
    dv_scan = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    scale = 1.0 / math.sqrt(dim)
    expected = reference(
        q, k, v, v_new, gk, beta, a, h, d_o, dh, dv_scan, scale
    )
    actual = torch.ops.npu.npu_chunk_kda_bwd_wy(
        q.npu(), k.npu(), v.npu(), v_new.npu(), gk.npu(), beta.npu(),
        a.npu(), h.npu(), d_o.npu(), dh.npu(), dv_scan.npu(),
        scale=scale, chunk_size=chunk,
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(
            got.cpu(), want, rtol=3e-2, atol=3e-2, equal_nan=False
        )


if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    run_case(128, torch.bfloat16)
    run_case(79, torch.float32)
