import ctypes
import math
import os

import torch
import torch_npu
import fla_npu

from fla_npu.ops.ascendc._runtime import ACL_FORMAT_ND, call_aclnn


def reference(q, k, v, v_new, gk, beta, a, h, d_o, dh, dv_scan, scale):
    batch, heads, seqlen, _ = q.shape
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


def main():
    loaded = fla_npu.load_ascendc_opapi_libraries()
    print("FLA_NPU", fla_npu.__file__, flush=True)
    print("OPAPI", *(lib._name for lib in loaded), flush=True)
    torch.npu.set_device(0)
    torch.manual_seed(20260805)
    batch = 1
    heads = int(os.environ.get("CASE_HEADS", "1"))
    seqlen = int(os.environ.get("CASE_SEQLEN", "64"))
    beta_dtype = (
        torch.bfloat16
        if os.environ.get("CASE_BETA_DTYPE", "fp32") == "bf16"
        else torch.float32
    )
    dim, chunk = 128, 64
    chunks = (seqlen + chunk - 1) // chunk
    device = "npu:0"

    def rand(shape, dtype, scale=0.02):
        return (torch.randn(shape, dtype=dtype) * scale).to(device)

    shape = (batch, heads, seqlen, dim)
    q = rand(shape, torch.bfloat16)
    k = rand(shape, torch.bfloat16)
    v = rand(shape, torch.bfloat16)
    v_new = rand(shape, torch.bfloat16)
    gk = (-torch.rand(shape, dtype=torch.float32) * 0.05).to(device)
    beta = (torch.rand((batch, heads, seqlen), dtype=beta_dtype) * 0.2).to(device)
    a = rand((batch, heads, seqlen, chunk), torch.bfloat16, 0.01)
    h = rand((batch, chunks, heads, dim, dim), torch.bfloat16, 0.01)
    d_o = rand(shape, torch.bfloat16)
    dh = rand((batch, heads, chunks, dim, dim), torch.bfloat16, 0.01)
    dv_scan = rand(shape, torch.bfloat16)
    outputs = (
        torch.zeros_like(q, dtype=torch.float32),
        torch.zeros_like(k, dtype=torch.float32),
        torch.zeros_like(v),
        torch.zeros_like(beta, dtype=torch.float32),
        torch.zeros_like(gk),
        torch.zeros_like(a, dtype=torch.float32),
    )

    def nd(ctx, tensor, name):
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=ACL_FORMAT_ND,
            storage_shape_override=tuple(tensor.shape),
        )

    inputs = (q, k, v, v_new, gk, beta, a, h, d_o, dh, dv_scan)
    torch.npu.synchronize()
    print("INPUTS_SYNCHRONIZED", flush=True)
    scale = 1.0 / math.sqrt(dim)
    call_aclnn(
        "aclnnChunkKdaBwdWy",
        lambda ctx: [
            *(nd(ctx, tensor, f"input_{idx}") for idx, tensor in enumerate(inputs)),
            ctypes.c_float(scale),
            ctypes.c_int64(chunk),
            *(nd(ctx, tensor, f"output_{idx}") for idx, tensor in enumerate(outputs)),
        ],
        outputs,
    )
    print("LAUNCHED", flush=True)
    torch.npu.synchronize()
    print("SYNCHRONIZED", flush=True)
    print("FINITE", *(bool(torch.isfinite(tensor).all().cpu()) for tensor in outputs), flush=True)
    host_inputs = tuple(tensor.cpu() for tensor in inputs)
    expected = reference(*host_inputs, scale)
    names = ("dq", "dk", "dv", "db", "dg", "dAkk")
    failed = []
    for name, got, want in zip(names, outputs, expected):
        got_cpu = got.cpu()
        diff = (got_cpu.float() - want.float()).abs()
        max_abs = diff.max().item()
        split = seqlen // 2
        half0 = diff[:, :, :split].max().item() if split else 0.0
        half1 = diff[:, :, split:].max().item()
        got_half0 = got_cpu[:, :, :split].float().abs().max().item() if split else 0.0
        got_half1 = got_cpu[:, :, split:].float().abs().max().item()
        try:
            torch.testing.assert_close(
                got_cpu, want, rtol=3e-2, atol=3e-2, equal_nan=False
            )
            status = "PASS"
        except AssertionError:
            status = "FAIL"
            failed.append(name)
            flat_index = int(diff.reshape(-1).argmax().item())
            print(
                f"DEBUG {name} flat_index={flat_index} "
                f"got={got_cpu.float().reshape(-1)[flat_index].item():.6g} "
                f"want={want.float().reshape(-1)[flat_index].item():.6g}",
                flush=True,
            )
        print(
            f"{status} {name} max_abs={max_abs:.6g} "
            f"first_half={half0:.6g} second_half={half1:.6g} "
            f"got_first={got_half0:.6g} got_second={got_half1:.6g}",
            flush=True,
        )
    if failed:
        raise AssertionError(f"precision failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
