# Copyright (c) 2026 Tianjin University, Ltd.
#
# Pure PyTorch reference for chunk KDA forward. This file is intentionally
# independent from torch_npu and Triton so ATK CPU golden code can reuse it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChunkKdaForwardResult:
    o: torch.Tensor
    final_state: Optional[torch.Tensor]
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    v_new: torch.Tensor
    h: torch.Tensor


@dataclass
class ChunkKdaBackwardIntraResult:
    dq: torch.Tensor
    dk: torch.Tensor
    db: torch.Tensor
    dg: torch.Tensor


def _chunk_spans(
    batch: int,
    total_t: int,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor],
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        spans: list[tuple[int, int, int]] = []
        for b in range(batch):
            for start in range(0, total_t, chunk_size):
                spans.append((b, start, min(start + chunk_size, total_t)))
        return spans

    if batch != 1:
        raise ValueError("varlen reference expects flattened batch B=1.")

    cu = cu_seqlens.detach().cpu().tolist()
    spans = []
    for n in range(len(cu) - 1):
        seq_start, seq_end = int(cu[n]), int(cu[n + 1])
        for start in range(seq_start, seq_end, chunk_size):
            spans.append((0, start, min(start + chunk_size, seq_end)))
    return spans


def _lower_inverse(mat: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def chunk_kda_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> ChunkKdaForwardResult:
    """Reference KDA forward for token-first tensors.

    Args:
        q: [B, T, H, K].
        k: [B, T, H, K].
        v: [B, T, HV, V].
        gk: [B, T, HV, K], cumulative key-wise gate in log2 space.
        beta: [B, T, HV].
        scale: attention scale.
        chunk_size: 64 or 128 in the intended NPU implementation.
        initial_state: optional [N, HV, K, V] state.
        output_final_state: whether to return final state.
        cu_seqlens: optional flattened varlen boundaries.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4-D token-first tensors.")
    if gk.dim() != 4 or beta.dim() != 3:
        raise ValueError("gk must be [B, T, HV, K] and beta must be [B, T, HV].")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shape.")
    bsz, total_t, hq, kdim = q.shape
    b_v, t_v, hv, vdim = v.shape
    if (b_v, t_v) != (bsz, total_t):
        raise ValueError("v shape prefix must match q/k.")
    if gk.shape != (bsz, total_t, hv, kdim):
        raise ValueError("gk shape must be [B, T, HV, K].")
    if beta.shape != (bsz, total_t, hv):
        raise ValueError("beta shape must be [B, T, HV].")
    if hv % hq != 0:
        raise ValueError("HV must be divisible by H.")

    spans = _chunk_spans(bsz, total_t, chunk_size, cu_seqlens)
    nt = len(spans) if cu_seqlens is not None else (total_t + chunk_size - 1) // chunk_size
    group = hv // hq
    num_seq = bsz if cu_seqlens is None else int(cu_seqlens.numel() - 1)

    out_dtype = v.dtype
    device = q.device
    o = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=out_dtype)
    Aqk = torch.zeros((bsz, total_t, hv, chunk_size), device=device, dtype=q.dtype)
    Akk = torch.zeros((bsz, total_t, hv, chunk_size), device=device, dtype=q.dtype)
    w = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=q.dtype)
    u = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=v.dtype)
    qg = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=q.dtype)
    kg = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=k.dtype)
    v_new = torch.zeros_like(v)
    h_out = torch.zeros((bsz, nt, hv, kdim, vdim), device=device, dtype=q.dtype)

    if initial_state is None:
        state = torch.zeros((num_seq, hv, kdim, vdim), device=device, dtype=torch.float32)
    else:
        state = initial_state.to(torch.float32).clone()

    seq_state_index = 0
    prev_b = None
    chunk_counter: dict[int, int] = {}
    if cu_seqlens is not None:
        cu_cpu = cu_seqlens.detach().cpu().tolist()
    else:
        cu_cpu = None

    for span_idx, (b, start, end) in enumerate(spans):
        if cu_cpu is not None:
            while seq_state_index + 1 < len(cu_cpu) - 1 and start >= cu_cpu[seq_state_index + 1]:
                seq_state_index += 1
            state_b = seq_state_index
            chunk_idx = span_idx
        else:
            if prev_b != b:
                chunk_counter[b] = 0
                prev_b = b
            state_b = b
            chunk_idx = chunk_counter[b]
            chunk_counter[b] += 1

        cur_t = end - start
        local_cols = slice(0, cur_t)
        for ihv in range(hv):
            ih = ihv // group
            q_blk = q[b, start:end, ih].to(torch.float32)
            k_blk = k[b, start:end, ih].to(torch.float32)
            v_blk = v[b, start:end, ihv].to(torch.float32)
            g_blk = gk[b, start:end, ihv].to(torch.float32)
            beta_blk = beta[b, start:end, ihv].to(torch.float32)

            causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril()
            strict_causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril(diagonal=-1)
            rel = g_blk[:, None, :] - g_blk[None, :, :]
            rel = rel.masked_fill(~causal[:, :, None], 0.0)
            gate = torch.exp2(rel)
            qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, gate) * float(scale)
            kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, gate)
            tril_qk = torch.where(causal, qk, torch.zeros_like(qk))
            tril_kk = torch.where(strict_causal, kk * beta_blk[:, None], torch.zeros_like(kk))
            inv_akk = _lower_inverse(tril_kk)

            k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
            v_beta = v_blk * beta_blk[:, None]
            w_blk = inv_akk @ k_beta_g
            u_blk = inv_akk @ v_beta

            last_g = g_blk[cur_t - 1]
            qg_blk = q_blk * torch.exp2(g_blk)
            kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)

            h_prev = state[state_b, ihv]
            h_out[b, chunk_idx, ihv] = h_prev.to(h_out.dtype)
            v_new_blk = u_blk - w_blk @ h_prev
            state[state_b, ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

            o_inter = qg_blk @ h_out[b, chunk_idx, ihv].to(torch.float32) * float(scale)
            o_local = tril_qk @ v_new_blk
            o[b, start:end, ihv] = (o_inter + o_local).to(out_dtype)

            Aqk[b, start:end, ihv, local_cols] = tril_qk.to(Aqk.dtype)
            Akk[b, start:end, ihv, local_cols] = inv_akk.to(Akk.dtype)
            w[b, start:end, ihv] = w_blk.to(w.dtype)
            u[b, start:end, ihv] = u_blk.to(u.dtype)
            qg[b, start:end, ihv] = qg_blk.to(qg.dtype)
            kg[b, start:end, ihv] = kg_blk.to(kg.dtype)
            v_new[b, start:end, ihv] = v_new_blk.to(v_new.dtype)

    final_state = state if output_final_state else None
    return ChunkKdaForwardResult(
        o=o,
        final_state=final_state,
        Aqk=Aqk,
        Akk=Akk,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        v_new=v_new,
        h=h_out,
    )


def chunk_kda_backward_intra_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
    safe_gate: bool = False,
    acc_dtype: torch.dtype = torch.float32,
) -> ChunkKdaBackwardIntraResult:
    """Token-first reference for ``chunk_kda_bwd_kernel_intra``.

    ``acc_dtype=torch.float32`` mirrors the kernel calculation order, while
    ``torch.float64`` provides the CPU golden used by precision validation.
    """
    if acc_dtype not in (torch.float32, torch.float64):
        raise ValueError("acc_dtype must be torch.float32 or torch.float64.")
    if q.shape != k.shape or q.dim() != 4:
        raise ValueError("q/k must be matching [B,T,H,K] tensors.")
    bsz, total_t, hq, kdim = q.shape
    if g.dim() != 4:
        raise ValueError("g must be [B,T,HV,K].")
    hv = g.shape[2]
    if g.shape != (bsz, total_t, hv, kdim) or hv % hq != 0:
        raise ValueError("g shape or grouped-value-head mapping is invalid.")
    if beta.shape != (bsz, total_t, hv):
        raise ValueError("beta must be [B,T,HV].")
    if dAqk.shape != (bsz, total_t, hv, chunk_size) or dAkk.shape != dAqk.shape:
        raise ValueError("dA tensors must be [B,T,HV,chunk_size].")

    dq_out = dq.to(acc_dtype).clone()
    dk_out = dk.to(acc_dtype).clone()
    db_out = db.to(acc_dtype).clone()
    dg_out = dg.to(acc_dtype).clone()
    group = hv // hq
    for b, start, end in _chunk_spans(bsz, total_t, chunk_size, cu_seqlens):
        cur_t = end - start
        for ihv in range(hv):
            ih = ihv // group
            q_blk = q[b, start:end, ih].to(acc_dtype)
            k_blk = k[b, start:end, ih].to(acc_dtype)
            g_blk = g[b, start:end, ihv].to(acc_dtype)
            beta_blk = beta[b, start:end, ihv].to(acc_dtype)
            aq = dAqk[b, start:end, ihv, :cur_t].to(acc_dtype)
            ak = dAkk[b, start:end, ihv, :cur_t].to(acc_dtype)
            if safe_gate:
                # Follow the Triton SAFE_GATE calculation order. Each 16-token
                # block factors long-range exponents around its first/middle/last
                # gate instead of clipping the exponent difference.
                dq_local = torch.zeros_like(g_blk)
                dk_left_pre = torch.zeros_like(g_blk)
                dk_right = torch.zeros_like(g_blk)
                for block_begin in range(0, cur_t, 16):
                    block_end = min(block_begin + 16, cur_t)
                    block_mid = block_begin + min(8, block_end - block_begin - 1)
                    target = slice(block_begin, block_end)
                    g_target = g_blk[target]
                    g_left_ref = g_blk[block_begin:block_begin + 1]
                    g_diag_ref = g_blk[block_mid:block_mid + 1]
                    g_right_ref = g_blk[block_end - 1:block_end]

                    if block_begin > 0:
                        left_k = k_blk[:block_begin] * torch.exp2(g_left_ref - g_blk[:block_begin])
                        dq_local[target] += (aq[target, :block_begin] @ left_k) * torch.exp2(
                            g_target - g_left_ref
                        )
                        dk_left_pre[target] += (ak[target, :block_begin] @ left_k) * torch.exp2(
                            g_target - g_left_ref
                        )

                    block_len = block_end - block_begin
                    lower = torch.ones((block_len, block_len), device=q.device, dtype=torch.bool).tril()
                    aq_diag = torch.where(lower, aq[target, block_begin:block_end], 0.0)
                    ak_diag = torch.where(lower, ak[target, block_begin:block_end], 0.0)
                    diag_k = k_blk[target] * torch.exp2(g_diag_ref - g_blk[target])
                    diag_outer_left = torch.exp2(g_target - g_diag_ref)
                    dq_local[target] += (aq_diag @ diag_k) * diag_outer_left
                    dk_left_pre[target] += (ak_diag @ diag_k) * diag_outer_left

                    diag_q = q_blk[target] * torch.exp2(g_blk[target] - g_diag_ref)
                    diag_k_beta = beta_blk[target, None] * k_blk[target] * torch.exp2(
                        g_blk[target] - g_diag_ref
                    )
                    diag_outer_right = torch.exp2(g_diag_ref - g_target)
                    dk_right[target] += (aq_diag.transpose(0, 1) @ diag_q) * diag_outer_right
                    dk_right[target] += (ak_diag.transpose(0, 1) @ diag_k_beta) * diag_outer_right

                    if block_end < cur_t:
                        future = slice(block_end, cur_t)
                        future_gate = torch.exp2(g_blk[future] - g_right_ref)
                        future_q = q_blk[future] * future_gate
                        future_k_beta = beta_blk[future, None] * k_blk[future] * future_gate
                        right_outer = torch.exp2(g_right_ref - g_target)
                        dk_right[target] += (aq[future, target].transpose(0, 1) @ future_q) * right_outer
                        dk_right[target] += (ak[future, target].transpose(0, 1) @ future_k_beta) * right_outer
            else:
                # Upstream keeps first/last-reference factorization for
                # inter-block terms even when SAFE_GATE is disabled. Only the
                # current 16x16 diagonal block uses direct pairwise exp2.
                dq_local = torch.zeros_like(g_blk)
                dk_left_pre = torch.zeros_like(g_blk)
                dk_right = torch.zeros_like(g_blk)
                for block_begin in range(0, cur_t, 16):
                    block_end = min(block_begin + 16, cur_t)
                    target = slice(block_begin, block_end)
                    g_target = g_blk[target]
                    g_left_ref = g_blk[block_begin:block_begin + 1]
                    g_right_ref = g_blk[block_end - 1:block_end]

                    if block_begin > 0:
                        left_k = k_blk[:block_begin] * torch.exp2(g_left_ref - g_blk[:block_begin])
                        dq_local[target] += (aq[target, :block_begin] @ left_k) * torch.exp2(
                            g_target - g_left_ref
                        )
                        dk_left_pre[target] += (ak[target, :block_begin] @ left_k) * torch.exp2(
                            g_target - g_left_ref
                        )

                    block_len = block_end - block_begin
                    lower = torch.ones((block_len, block_len), device=q.device, dtype=torch.bool).tril()
                    aq_diag = torch.where(lower, aq[target, block_begin:block_end], 0.0)
                    ak_diag = torch.where(lower, ak[target, block_begin:block_end], 0.0)
                    diag_diff = g_target[:, None, :] - g_target[None, :, :]
                    diag_gate = torch.where(lower[:, :, None], torch.exp2(diag_diff), 0.0)
                    dq_local[target] += torch.einsum("ij,jd,ijd->id", aq_diag, k_blk[target], diag_gate)
                    dk_left_pre[target] += torch.einsum(
                        "ij,jd,ijd->id", ak_diag, k_blk[target], diag_gate
                    )
                    dk_right[target] += torch.einsum(
                        "ji,jd,jid->id", aq_diag, q_blk[target], diag_gate
                    )
                    dk_right[target] += torch.einsum(
                        "ji,jd,jid->id", ak_diag, beta_blk[target, None] * k_blk[target], diag_gate
                    )

                    if block_end < cur_t:
                        future = slice(block_end, cur_t)
                        future_gate = torch.exp2(g_blk[future] - g_right_ref)
                        future_q = q_blk[future] * future_gate
                        future_k_beta = beta_blk[future, None] * k_blk[future] * future_gate
                        right_outer = torch.exp2(g_right_ref - g_target)
                        dk_right[target] += (aq[future, target].transpose(0, 1) @ future_q) * right_outer
                        dk_right[target] += (ak[future, target].transpose(0, 1) @ future_k_beta) * right_outer
            dk_left = beta_blk[:, None] * dk_left_pre
            dq_out[b, start:end, ihv] += dq_local
            dk_out[b, start:end, ihv] += dk_left + dk_right
            db_out[b, start:end, ihv] += (dk_left_pre * k_blk).sum(dim=-1)
            dg_out[b, start:end, ihv] += q_blk * dq_local + (dk_left - dk_right) * k_blk
    return ChunkKdaBackwardIntraResult(dq=dq_out, dk=dk_out, db=db_out, dg=dg_out)


def run_smoke() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 16, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(1, 32, 2, 32, dtype=torch.float32)
    gk = torch.randn(1, 32, 2, 16, dtype=torch.float32).cumsum(dim=1) * 0.01
    beta = torch.sigmoid(torch.randn(1, 32, 2, dtype=torch.float32))
    res = chunk_kda_forward_reference(q, k, v, gk, beta, scale=16 ** -0.5, chunk_size=16, output_final_state=True)
    for name, value in vars(res).items():
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf")


if __name__ == "__main__":
    run_smoke()
