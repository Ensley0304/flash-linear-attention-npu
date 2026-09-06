from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class Chunk:
    batch: int
    state_index: int
    start: int
    end: int


def iter_chunks(B: int, T: int, chunk_size: int, cu_cpu: torch.Tensor | None) -> Iterable[Chunk]:
    if cu_cpu is None:
        nt = (T + chunk_size - 1) // chunk_size
        for b in range(B):
            for ci in range(nt):
                start = ci * chunk_size
                yield Chunk(b, ci, start, min(start + chunk_size, T))
        return
    state_index = 0
    values = cu_cpu.tolist()
    for seq in range(len(values) - 1):
        bos, eos = values[seq], values[seq + 1]
        local = 0
        while bos + local < eos:
            start = bos + local
            yield Chunk(0, state_index, start, min(start + chunk_size, eos))
            state_index += 1
            local += chunk_size


def dot_acc(a: torch.Tensor, b: torch.Tensor, acc_dtype: torch.dtype) -> torch.Tensor:
    return torch.matmul(a.to(acc_dtype), b.to(acc_dtype))


def kernel_a_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    Aqk: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_cpu: torch.Tensor | None,
    state_v_first: bool,
    acc_dtype: torch.dtype = torch.float32,
    preserve_output_dtypes: bool = True,
) -> dict[str, torch.Tensor]:
    B, T, H, K = k.shape
    HV, V = do.shape[2], do.shape[-1]
    G = HV // H
    dAqk = torch.zeros_like(Aqk, dtype=acc_dtype)
    dv0 = torch.empty_like(do) if preserve_output_dtypes else torch.empty_like(do, dtype=acc_dtype)
    dq_raw = torch.empty((B, T, HV, K), device=do.device, dtype=acc_dtype)
    for chunk in iter_chunks(B, T, chunk_size, cu_cpu):
        b, s, e, si = chunk.batch, chunk.start, chunk.end, chunk.state_index
        c = e - s
        for hv in range(HV):
            hk = hv // G
            A_storage = Aqk[b, s:e, hv, :c]
            A_t = A_storage.transpose(0, 1)
            do_c = do[b, s:e, hv]
            vn_c = v_new[b, s:e, hv]
            h_c = h[b, si, hv]
            if not state_v_first:
                h_c = h_c.transpose(-1, -2)
            raw = dot_acc(do_c, vn_c.transpose(0, 1), acc_dtype)
            dAqk[b, s:e, hv, :c] = torch.tril(raw) * scale
            dv_value = dot_acc(A_t, do_c, acc_dtype)
            dv0[b, s:e, hv] = dv_value.to(do.dtype) if preserve_output_dtypes else dv_value
            dq_raw[b, s:e, hv] = dot_acc(do_c, h_c, acc_dtype)
    return {"dAqk": dAqk, "dv0": dv0, "dq_raw": dq_raw}


def kernel_c_base_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv_scan: torch.Tensor,
    dq_raw: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_cpu: torch.Tensor | None,
    state_v_first: bool,
    acc_dtype: torch.dtype = torch.float32,
    preserve_output_dtypes: bool = True,
    emulate_triton_casts: bool = True,
) -> dict[str, torch.Tensor]:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    dq = torch.empty((B, T, HV, K), device=q.device, dtype=acc_dtype)
    dk = torch.empty_like(dq)
    dv = torch.empty_like(v) if preserve_output_dtypes else torch.empty_like(v, dtype=acc_dtype)
    db = torch.empty_like(beta, dtype=acc_dtype)
    dg = torch.empty_like(gk, dtype=acc_dtype)
    dAkk = torch.zeros_like(Akk, dtype=acc_dtype)
    for chunk in iter_chunks(B, T, chunk_size, cu_cpu):
        b, s, e, si = chunk.batch, chunk.start, chunk.end, chunk.state_index
        c = e - s
        for hv in range(HV):
            hk = hv // G
            q_c = q[b, s:e, hk]
            k_c = k[b, s:e, hk]
            v_c = v[b, s:e, hv]
            vn_c = v_new[b, s:e, hv]
            do_c = do[b, s:e, hv]
            dvs_c = dv_scan[b, s:e, hv]
            g_c = gk[b, s:e, hv].to(acc_dtype)
            beta_c = beta[b, s:e, hv].to(acc_dtype)
            h_c = h[b, si, hv]
            dh_c = dh[b, si, hv]
            if not state_v_first:
                h_c = h_c.transpose(-1, -2)
                dh_c = dh_c.transpose(-1, -2)

            A_storage = Akk[b, s:e, hv, :c]
            A_t_q = A_storage.transpose(0, 1)
            A_t = A_t_q.to(acc_dtype)
            g_exp = torch.exp2(g_c)
            g_last = g_c[-1]
            state_decay = torch.exp2(g_last.unsqueeze(0) - g_c)

            dk_state_raw = dot_acc(vn_c, dh_c, acc_dtype)
            dk_state = dk_state_raw * state_decay
            DVb = dot_acc(A_t_q, dvs_c, acc_dtype)
            dW_raw = dot_acc(dvs_c, h_c, acc_dtype)
            dw_neg_q = (-dW_raw).to(Akk.dtype) if emulate_triton_casts else -dW_raw
            kE = k_c.to(acc_dtype) * g_exp
            kE_q = kE.to(Akk.dtype) if emulate_triton_casts else kE
            dKgb_signed = dot_acc(A_t_q, dw_neg_q, acc_dtype)
            zV = dot_acc(dvs_c, v_c.transpose(0, 1), acc_dtype)
            zW_signed = dot_acc(dw_neg_q, kE_q.transpose(0, 1), acc_dtype)
            Zb = torch.tril(zV + zW_signed, diagonal=-1) * beta_c.unsqueeze(0)
            Zb_cube = Zb.to(Akk.dtype) if emulate_triton_casts else Zb
            Tza = dot_acc(Zb_cube, A_t_q, acc_dtype)
            Tza_cube = Tza.to(Akk.dtype) if emulate_triton_casts else Tza
            dA_raw = -dot_acc(A_t_q, Tza_cube, acc_dtype)
            dA = torch.tril(dA_raw, diagonal=-1)

            dq_base = dq_raw[b, s:e, hv] * g_exp * scale
            dv_value = DVb * beta_c.unsqueeze(1)
            dv_c_out = dv_value.to(v.dtype) if preserve_output_dtypes else dv_value
            dk_base = dk_state + dKgb_signed * (beta_c.unsqueeze(1) * g_exp)
            r_h = (h_c.to(acc_dtype) * dh_c.to(acc_dtype)).sum(dim=0)
            gate_state = r_h * torch.exp2(g_last) + (k_c.to(acc_dtype) * dk_state).sum(dim=0)
            db_base = (DVb * v_c.to(acc_dtype)).sum(dim=1) + (dKgb_signed * kE).sum(dim=1)
            dg_base = (
                q_c.to(acc_dtype) * dq_base
                - k_c.to(acc_dtype) * dk_state
                + beta_c.unsqueeze(1) * kE * dKgb_signed
            )
            dg_base[-1] += gate_state

            dq[b, s:e, hv] = dq_base
            dk[b, s:e, hv] = dk_base
            dv[b, s:e, hv] = dv_c_out
            db[b, s:e, hv] = db_base
            dg[b, s:e, hv] = dg_base
            dAkk[b, s:e, hv, :c] = dA
    return {"dq_base": dq, "dk_base": dk, "dv": dv, "db_base": db, "dg_base": dg, "dAkk": dAkk}


def kernel_c_intra_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    base: dict[str, torch.Tensor],
    chunk_size: int,
    cu_cpu: torch.Tensor | None,
    cube_input_dtype: torch.dtype | None,
    acc_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    B, T, H, K = k.shape
    HV = gk.shape[2]
    G = HV // H
    dq = torch.empty_like(base["dq_base"])
    dk = torch.empty_like(base["dk_base"])
    db = torch.empty_like(base["db_base"])
    dg = torch.empty_like(base["dg_base"])
    for chunk in iter_chunks(B, T, chunk_size, cu_cpu):
        b, s, e = chunk.batch, chunk.start, chunk.end
        c = e - s
        for hv in range(HV):
            hk = hv // G
            q_c = q[b, s:e, hk].to(acc_dtype)
            k_c = k[b, s:e, hk].to(acc_dtype)
            g_c = gk[b, s:e, hv].to(acc_dtype)
            beta_c = beta[b, s:e, hv].to(acc_dtype)
            daq = dAqk[b, s:e, hv, :c]
            dak = dAkk[b, s:e, hv, :c]
            if cube_input_dtype is not None:
                daq = daq.to(cube_input_dtype).to(acc_dtype)
                dak = dak.to(cube_input_dtype).to(acc_dtype)
            delta = g_c[:, None, :] - g_c[None, :, :]
            decay = torch.exp2(delta)
            lower_eq = torch.tril(torch.ones((c, c), device=q.device, dtype=torch.bool))
            lower_strict = torch.tril(torch.ones((c, c), device=q.device, dtype=torch.bool), diagonal=-1)
            upper_eq = lower_eq.transpose(0, 1)
            upper_strict = lower_strict.transpose(0, 1)

            left_q = (daq[:, :, None] * k_c[None, :, :] * decay)
            left_k = (dak[:, :, None] * k_c[None, :, :] * decay)
            dq_local = torch.where(lower_eq[:, :, None], left_q, 0).sum(dim=1)
            dk_left = torch.where(lower_strict[:, :, None], left_k, 0).sum(dim=1)

            decay_t = decay.transpose(0, 1)
            right_q = daq.transpose(0, 1)[:, :, None] * q_c[None, :, :] * decay_t
            right_k = (
                dak.transpose(0, 1)[:, :, None]
                * (beta_c[:, None] * k_c)[None, :, :]
                * decay_t
            )
            dk_right = torch.where(upper_eq[:, :, None], right_q, 0).sum(dim=1)
            dk_right += torch.where(upper_strict[:, :, None], right_k, 0).sum(dim=1)

            dq_c = base["dq_base"][b, s:e, hv] + dq_local
            dk_c = base["dk_base"][b, s:e, hv] + beta_c[:, None] * dk_left + dk_right
            db_c = base["db_base"][b, s:e, hv] + (dk_left * k_c).sum(dim=1)
            dg_c = (
                base["dg_base"][b, s:e, hv]
                + q_c * dq_local
                + (beta_c[:, None] * dk_left - dk_right) * k_c
            )
            dq[b, s:e, hv] = dq_c
            dk[b, s:e, hv] = dk_c
            db[b, s:e, hv] = db_c
            dg[b, s:e, hv] = dg_c
    return {"dq_hv": dq, "dk_hv": dk, "db": db, "dg_hv": dg}


def reverse_chunk_cumsum(value: torch.Tensor, chunk_size: int, cu_cpu: torch.Tensor | None) -> torch.Tensor:
    out = torch.empty_like(value)
    B, T = value.shape[:2]
    for chunk in iter_chunks(B, T, chunk_size, cu_cpu):
        x = value[chunk.batch, chunk.start:chunk.end]
        out[chunk.batch, chunk.start:chunk.end] = torch.flip(
            torch.cumsum(torch.flip(x, dims=(0,)), dim=0), dims=(0,)
        )
    return out


def gate_backward_torch(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    dg_act: torch.Tensor,
    lower_bound: float,
    acc_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    x = raw_g.to(acc_dtype) + dt_bias.to(acc_dtype).view(1, 1, raw_g.shape[2], raw_g.shape[3])
    e = torch.exp(A_log.to(acc_dtype)).view(1, 1, -1, 1)
    sig = torch.sigmoid(e * x)
    dg = dg_act.to(acc_dtype) * lower_bound * e * sig * (1 - sig)
    dA = (dg * x).sum(dim=(0, 1, 3))
    dbias = dg.sum(dim=(0, 1)).reshape(-1)
    return {"dg": dg, "dA_log": dA, "ddt_bias": dbias}
