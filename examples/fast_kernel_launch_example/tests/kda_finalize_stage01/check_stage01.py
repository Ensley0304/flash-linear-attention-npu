#!/usr/bin/env python3
"""Private Stage 0-6 checks; not an end-to-end Finalize accuracy claim."""
import argparse
import os
import signal
from pathlib import Path

import torch
import torch_npu  # noqa: F401

CAPTURE_BYTES = 384 * 1024
OFFSETS = {
    "dk_state_raw": (0, torch.float32, (64, 128)),
    "dv_b": (32 * 1024, torch.float32, (64, 128)),
    "zV": (64 * 1024, torch.float32, (64, 64)),
    "dW": (80 * 1024, torch.bfloat16, (64, 128)),
    "exp2_gk": (96 * 1024, torch.float32, (64, 128)),
    "kE": (128 * 1024, torch.bfloat16, (64, 128)),
    "rH": (144 * 1024, torch.float32, (128,)),
    "gk_last": (144 * 1024 + 512, torch.float32, (128,)),
    "dKgb_raw": (160 * 1024, torch.float32, (64, 128)),
    "zW": (192 * 1024, torch.float32, (64, 64)),
}


def make_case(batch, heads, tokens, lengths=None, seed=0, gate_mode="cumulative"):
    generator = torch.Generator().manual_seed(seed)
    variable = lengths is not None
    shape = (heads, tokens, 128) if variable else (batch, heads, tokens, 128)

    def rand(dims):
        return (torch.randn(dims, generator=generator) * 0.125).to(torch.bfloat16)

    k, v, v_new, dv_scan = [rand(shape) for _ in range(4)]
    akk = rand((*shape[:-1], 64))
    gk = torch.empty(shape, dtype=torch.float32)
    cu = [0]
    if variable:
        for length in lengths:
            cu.append(cu[-1] + length)
    chunks = []
    for sequence in range(len(lengths) if variable else batch):
        begin, end = (cu[sequence], cu[sequence + 1]) if variable else (0, tokens)
        for local, start in enumerate(range(begin, end, 64)):
            rows = min(64, end - start)
            chunks.append((sequence, local, start, rows))
            gate = -torch.rand((heads, rows, 128), generator=generator) * 5.0
            # FP32 cumsum is local to a chunk and is stored in base-2 units.
            if gate_mode == "cumulative":
                gate = gate.cumsum(1) / torch.log(torch.tensor(2.0))
            else:
                # Keep every row significant when testing the Stage 2 kE path.
                gate = gate * 0.4
            if variable:
                gk[:, start:start + rows] = gate
            else:
                gk[sequence, :, start:start + rows] = gate
    state_shape = (len(chunks), heads, 128, 128) if variable else (
        batch, (tokens + 63) // 64, heads, 128, 128)
    h, dh = rand(state_shape), rand(state_shape)
    return (k, v, gk, akk, v_new, h, dh, dv_scan), cu if variable else [], chunks


def reference(inputs, chunks, head, index, variable):
    k, v, gk, akk, v_new, h, dh, dv_scan = inputs
    sequence, local, start, rows = chunks[index]

    def token(x, width):
        out = torch.zeros((64, width), dtype=torch.float64)
        src = x[head] if variable else x[sequence, head]
        out[:rows] = src[start:start + rows].double()
        return out

    kk, vv, new, scan = (token(x, 128) for x in (k, v, v_new, dv_scan))
    a = token(akk, 64)
    a[:, rows:] = 0
    hh = (h[index, head] if variable else h[sequence, local, head]).double()
    dd = (dh[index, head] if variable else dh[sequence, local, head]).double()
    gate = token(gk, 128).float()
    exponent = torch.zeros_like(gate)
    exponent[:rows] = torch.exp2(gate[:rows])
    dw = (scan @ hh.T).float().bfloat16()
    ke = (kk.float() * exponent).bfloat16()
    # Propagate the already fixed BF16 intermediate tolerance. Stage 2 also
    # has a separate strict GEMM check using captured operands below.
    dw_error = 2e-5 + 8e-3 * dw.double().abs()
    ke_error = 2e-5 + 8e-3 * ke.double().abs()
    return {
        "dKgb_budget": (a.abs().T @ dw_error).float(),
        "zW_budget": (dw_error @ ke.double().abs().T +
                       dw.double().abs() @ ke_error.T + dw_error @ ke_error.T).float(),
        "dKgb_raw": (a.T @ dw.double()).float(),
        "zW": (dw.double() @ ke.double().T).float(),
        "dk_state_raw": (new @ dd.T).float(),
        "dv_b": (a.T @ scan).float(),
        "dW": dw,
        "zV": (scan @ vv.T).float(),
        "rH": (hh * dd).sum(-1).float(),
        "gk_last": gate[rows - 1],
        "exp2_gk": exponent,
        "kE": ke,
    }


def decode(record, name, workspace=False):
    offset, dtype, shape = OFFSETS[name]
    if workspace:
        offset = {"dKgb_raw": 80 * 1024, "zW": 112 * 1024}.get(name, offset)
    elements = 1
    for size in shape:
        elements *= size
    size = elements * torch.empty((), dtype=dtype).element_size()
    return record[offset:offset + size].clone().view(dtype).reshape(shape)


def workspace_works(heads, chunks, cores, stage):
    works = ((heads + 1) // 2) * chunks
    for core in range(cores):
        last = core + ((works - 1 - core) // cores) * cores
        yield core, last
        if stage >= 3 and last - cores >= core:
            yield core, last - cores


def check_case(args, batch, heads, tokens, lengths=None):
    inputs, cu, chunks = make_case(batch, heads, tokens, lengths, gate_mode=args.gate_mode)
    expected = None if args.workspace_only else [
        [reference(inputs, chunks, head, c, lengths is not None) for c in range(len(chunks))]
        for head in range(heads)]
    beta = torch.rand(inputs[0].shape[:-1], generator=torch.Generator().manual_seed(137)) * 2
    beta.flatten()[:min(4, beta.numel())] = torch.tensor([0.0, 1.0, 2.0, 0.99966466])[:min(4, beta.numel())]
    generator = torch.Generator().manual_seed(593)
    q = (torch.randn(inputs[0].shape, generator=generator) * .125).bfloat16()
    dq_raw = torch.randn(inputs[0].shape, generator=generator) * .125
    device_q, device_dq_raw = q.to(f"npu:{args.device}"), dq_raw.to(f"npu:{args.device}")
    device_beta = beta.to(f"npu:{args.device}")
    dv_out = torch.full_like(inputs[0], float("nan"), device=f"npu:{args.device}")
    device_inputs = [x.to(f"npu:{args.device}") for x in inputs]
    torch.npu.synchronize()
    for iteration in range(args.repeat):
        # The alarm covers kernel execution and the following device synchronize.
        signal.alarm(60)
        try:
            capture, trace, workspace = torch.ops.ascend_ops.kda_finalize_stage01(
                *device_inputs, cu, args.cores, args.stage, device_beta, dv_out, device_q, device_dq_raw, args.scale)
            torch.npu.synchronize()
        finally:
            signal.alarm(0)
        capture, trace, workspace = capture.cpu(), trace.cpu(), workspace.cpu()
        if args.stage >= 3:
            check_stage3(inputs, beta, chunks, lengths is not None, capture, workspace, dv_out.cpu(), args.workspace_only, expected)
        if args.stage >= 6:
            check_stage6(inputs, beta, chunks, lengths is not None, capture, workspace, args, expected)
        if args.stage >= 5:
            check_stage5(inputs, q, dq_raw, beta, chunks, lengths is not None, capture, workspace, args, expected)
        if args.stage >= 4 and not args.workspace_only:
            check_stage4(inputs, beta, chunks, lengths is not None, capture, expected)
            if args.stage == 4:
                assert (capture[:, :, 256 * 1024:] == 0xA5).all(), "unused Stage 5 capture guard"
        if args.workspace_only:
            assert capture.numel() == 1, "workspace-only requires a capture-disabled build"
            used_cores = workspace.numel() // (1024 * 1024)
            works = ((heads + 1) // 2) * len(chunks)
            for core, work in workspace_works(heads, len(chunks), used_cores, args.stage):
                c = work % len(chunks)
                for owner in range(2):
                    head = (work // len(chunks)) * 2 + owner
                    if head >= heads:
                        continue
                    wanted = reference(inputs, chunks, head, c, lengths is not None)
                    base = (2 * core + owner) * 512 * 1024 + ((work // used_cores) & 1) * 256 * 1024 * (args.stage >= 3)
                    record = workspace[base:base + (256 if args.stage >= 3 else 512) * 1024]
                    for name in ((("zV",) if args.stage >= 5 else ("dk_state_raw", "dv_b", "zV")) +
                                 (("dKgb_raw", "zW") if args.stage >= 2 else ())):
                        actual = decode(record, name, workspace=True)
                        assert torch.isfinite(actual).all(), name
                        if name in ("dKgb_raw", "zW"):
                            budget = wanted["dKgb_budget" if name == "dKgb_raw" else "zW_budget"]
                            assert ((actual - wanted[name]).abs() <= budget + 2e-6 + 3e-4 * wanted[name].abs()).all(), name
                        else:
                            torch.testing.assert_close(actual, wanted[name], rtol=3e-4, atol=2e-6)
                    if args.stage >= 1:
                        actual = record[144 * 1024:160 * 1024].clone().view(torch.bfloat16).reshape(64, 128)
                        assert torch.isfinite(actual).all(), "kE"
                        torch.testing.assert_close(actual.float(), wanted["kE"].float(), rtol=8e-3, atol=2e-5)
            print(f"PASS workspace stage={args.stage} B={batch} H={heads} T={tokens} "
                  f"varlen={lengths is not None} gate={args.gate_mode} cores={used_cores} repeat={iteration + 1}", flush=True)
            continue
        names = list(OFFSETS)[:(4, 8, 10, 10, 10, 10, 10)[args.stage]]
        # Batch comparisons keep the full 8K check practical without sampling or
        # changing the per-element tolerance / independent CPU reference.
        for name in names:
            offset, dtype, shape = OFFSETS[name]
            elements = 1
            for size in shape:
                elements *= size
            size = elements * torch.empty((), dtype=dtype).element_size()
            actual = capture[:, :, offset:offset + size].contiguous().view(dtype).reshape(
                heads, len(chunks), *shape)
            wanted = torch.stack([torch.stack([expected[head][c][name] for c in range(len(chunks))])
                                  for head in range(heads)])
            assert torch.isfinite(actual).all(), name
            if name in ("dW", "kE"):
                rtol, atol = 8e-3, 2e-5
            elif name in ("exp2_gk", "gk_last"):
                rtol, atol = 3e-5, 1e-7
            else:
                rtol, atol = 3e-4, 2e-6
            if name in ("dKgb_raw", "zW"):
                budget_name = "dKgb_budget" if name == "dKgb_raw" else "zW_budget"
                budget = torch.stack([torch.stack([expected[head][c][budget_name] for c in range(len(chunks))])
                                      for head in range(heads)])
                assert ((actual - wanted).abs() <= budget + atol + rtol * wanted.abs()).all(), name
            else:
                torch.testing.assert_close(
                    actual.float(), wanted.float(), rtol=rtol, atol=atol,
                    msg=lambda msg: f"{name}, indices=(head,chunk,...): {msg}")
        if args.stage >= 2:
            # Batch by head: compare every element without thousands of tiny assertions.
            for head in range(heads):
                a = torch.zeros((len(chunks), 64, 64), dtype=torch.float64)
                for c, (sequence, local, start, rows) in enumerate(chunks):
                    src = inputs[3][head] if lengths is not None else inputs[3][sequence, head]
                    a[c, :rows] = src[start:start + rows].double()
                    a[c, :, rows:] = 0
                dw = capture[head, :, 80 * 1024:96 * 1024].contiguous().view(
                    torch.bfloat16).reshape(len(chunks), 64, 128).double()
                ke = capture[head, :, 128 * 1024:144 * 1024].contiguous().view(
                    torch.bfloat16).reshape(len(chunks), 64, 128).double()
                for name, local_ref in (("dKgb_raw", a.transpose(1, 2) @ dw),
                                        ("zW", dw @ ke.transpose(1, 2))):
                    offset, dtype, shape = OFFSETS[name]
                    size = shape[0] * shape[1] * 4
                    actual = capture[head, :, offset:offset + size].contiguous().view(dtype).reshape(
                        len(chunks), *shape)
                    torch.testing.assert_close(actual, local_ref.float(), rtol=3e-4, atol=2e-6,
                                               msg=lambda msg: f"Stage 2 local GEMM: {msg}")
        assert (capture[:, :, 145 * 1024:160 * 1024] == 0xA5).all(), "capture guard overwritten"
        guard_start = 216 * 1024 + 256 if args.stage >= 3 else (208 if args.stage >= 2 else 160) * 1024
        assert (capture[:, :, guard_start:224 * 1024 if args.stage >= 4 else CAPTURE_BYTES] == 0xA5).all(), "capture guard overwritten"
        works = ((heads + 1) // 2) * len(chunks)
        assert trace.shape == (works * 2, 16)
        used_cores = int(trace[0, 6])
        for work in range(works):
            for owner in range(2):
                record = trace[2 * work + owner].tolist()
                head = (work // len(chunks)) * 2 + owner
                assert record[:7] == [
                    work, work % used_cores, owner, head, work % len(chunks),
                    work // used_cores + 1, used_cores,
                ], record
                assert record[7:11] == [
                    chunks[work % len(chunks)][3], int(head < heads), args.stage, 7,
                ], record
                assert record[11] in (used_cores, 2 * used_cores), record
                assert record[12:15] == [
                    2, 2 * (work % used_cores) + owner,
                    (2 * (work % used_cores) + owner) * 512 * 1024 + ((work // used_cores) & 1) * 256 * 1024 * (args.stage >= 3),
                ], record
        # Read the actual reusable slots, not just the extra capture writes.
        for core, work in workspace_works(heads, len(chunks), used_cores, args.stage):
            c = work % len(chunks)
            for owner in range(2):
                head = (work // len(chunks)) * 2 + owner
                if head >= heads:
                    continue
                base = (2 * core + owner) * 512 * 1024 + ((work // used_cores) & 1) * 256 * 1024 * (args.stage >= 3)
                record = workspace[base:base + (256 if args.stage >= 3 else 512) * 1024]
                for name in ((("zV",) if args.stage >= 5 else ("dk_state_raw", "dv_b", "zV")) +
                                 (("dKgb_raw", "zW") if args.stage >= 2 else ())):
                    torch.testing.assert_close(decode(record, name, workspace=True), decode(capture[head, c], name),
                                               rtol=0, atol=0)
                if args.stage >= 1:
                    actual_ke = record[144 * 1024:160 * 1024].clone().view(torch.bfloat16).reshape(64, 128)
                    torch.testing.assert_close(actual_ke, decode(capture[head, c], "kE"), rtol=0, atol=0)
                assert (record[(128 if args.stage >= 2 else 80) * 1024:144 * 1024] == 0x5A).all(), "workspace guard overwritten"
                if args.stage >= 5:
                    assert (record[200 * 1024 + 256:208 * 1024] == 0x5A).all(), "Stage 5 workspace guard"
                    assert (record[208 * 1024 + 512:216 * 1024 if args.stage >= 6 else record.numel()] == 0x5A).all(), "Stage 5 workspace guard"
                    if args.stage >= 6:
                        assert (record[232 * 1024:] == 0x5A).all(), "Stage 6 workspace guard"
                else:
                    assert (record[(168 if args.stage >= 3 else 160) * 1024:] == 0x5A).all(), "workspace guard overwritten"
        if args.cores == 1 and works > 1:
            assert trace[-1, 5] > 1, "workspace reuse was not exercised"
        print(f"PASS stage={args.stage} B={batch} H={heads} T={tokens} "
              f"varlen={lengths is not None} gate={args.gate_mode} cores={used_cores} repeat={iteration + 1}", flush=True)



def check_stage3(inputs, beta, chunks, variable, capture, workspace, dv, workspace_only, expected=None):
    """Check all dv outputs; capture builds additionally check every Zb and db_v.

    The full-path Zb bound propagates the established Stage 2 intermediate budget.
    A second local check isolates Stage 3 using the actual captured FP32 operands.
    """
    heads = inputs[0].shape[-3]
    assert torch.isfinite(dv).all(), "dv not fully written"
    for head in range(heads):
        refs, actual_dv, actual_db, local_db, actual_zb, local_zb, budgets = [], [], [], [], [], [], []
        dv_refs, db_refs = [], []
        for c, (sequence, _, start, rows) in enumerate(chunks):
            b = torch.zeros(64, dtype=torch.float64)
            b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows].double()
            v = torch.zeros((64, 128), dtype=torch.float64)
            v[:rows] = (inputs[1][head] if variable else inputs[1][sequence, head])[start:start + rows].double()
            if expected is not None:
                ref = expected[head][c]
            else:
                # Full-output dv only needs this GEMM. Other workspace quantities
                # are checked separately for both retained generations below.
                a = torch.zeros((64, 64), dtype=torch.float64)
                scan = torch.zeros((64, 128), dtype=torch.float64)
                src_a = inputs[3][head] if variable else inputs[3][sequence, head]
                src_scan = inputs[7][head] if variable else inputs[7][sequence, head]
                a[:rows, :rows] = src_a[start:start + rows, :rows].double()
                scan[:rows] = src_scan[start:start + rows].double()
                ref = {"dv_b": (a.T @ scan).float()}
            dv_refs.append((ref['dv_b'].double() * b[:, None])[:rows].float().bfloat16().float())
            actual_dv.append((dv[head] if variable else dv[sequence, head])[start:start + rows].float())
            if workspace_only:
                continue
            record = capture[head, c]
            zv, zw, dvb = (decode(record, name).double() for name in ('zV', 'zW', 'dv_b'))
            zb = record[208 * 1024:216 * 1024].clone().view(torch.bfloat16).reshape(64, 64).float()
            db = record[216 * 1024:216 * 1024 + 256].clone().view(torch.float32)
            actual_zb.append(zb)
            actual_db.append(db)
            local_zb.append(torch.tril((zv - zw) * b[None, :], -1).float().bfloat16().float())
            local_db.append((v * dvb).sum(-1).float())
            full_zb = torch.tril((ref['zV'].double() - ref['zW'].double()) * b[None, :], -1).float().bfloat16().float()
            refs.append(full_zb)
            budgets.append(torch.tril((ref['zW_budget'] + 2e-6 + 3e-4 * ref['zV'].abs()) * b[None, :], -1).float())
            db_refs.append((v * ref['dv_b'].double()).sum(-1).float())
            assert torch.count_nonzero(torch.triu(zb)) == 0, 'Zb must be strictly lower'
            assert torch.count_nonzero(zb[rows:]) == 0 and torch.count_nonzero(zb[:, rows:]) == 0, 'Zb tail'
            assert torch.count_nonzero(db[rows:]) == 0, 'db_v tail'
        torch.testing.assert_close(torch.cat(actual_dv), torch.cat(dv_refs), rtol=8e-3, atol=2e-5, msg='Stage 3 dv')
        if not workspace_only:
            got, wanted, budget = torch.stack(actual_zb), torch.stack(refs), torch.stack(budgets)
            assert torch.isfinite(got).all() and ((got - wanted).abs() <= budget + 2e-5 + 8e-3 * wanted.abs()).all(), 'Stage 3 full Zb'
            torch.testing.assert_close(got, torch.stack(local_zb), rtol=8e-3, atol=2e-5, msg='Stage 3 local Zb')
            torch.testing.assert_close(torch.stack(actual_db), torch.stack(local_db), rtol=3e-4, atol=2e-6, msg='Stage 3 local db_v')
            torch.testing.assert_close(torch.stack(actual_db), torch.stack(db_refs), rtol=3e-4, atol=2e-6, msg='Stage 3 full db_v')
    cores = workspace.numel() // (1024 * 1024)
    works = ((heads + 1) // 2) * len(chunks)
    for core, work in workspace_works(heads, len(chunks), cores, 3):
        c = work % len(chunks)
        for owner in range(2):
            head = (work // len(chunks)) * 2 + owner
            if head >= heads:
                continue
            base = (2 * core + owner) * 512 * 1024 + ((work // cores) & 1) * 256 * 1024 + 160 * 1024
            actual = workspace[base:base + 8192].clone().view(torch.bfloat16).reshape(64, 64)
            if not workspace_only:
                wanted = capture[head, c, 208 * 1024:216 * 1024].clone().view(torch.bfloat16).reshape(64, 64)
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            else:
                sequence, _, start, rows = chunks[c]
                b = torch.zeros(64)
                b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows]
                ref = reference(inputs, chunks, head, c, variable)
                wanted = torch.tril((ref['zV'] - ref['zW']) * b[None, :], -1).bfloat16().float()
                budget = torch.tril((ref['zW_budget'] + 2e-6 + 3e-4 * ref['zV'].abs()) * b[None, :], -1)
                assert torch.isfinite(actual).all() and ((actual.float() - wanted).abs() <= budget + 2e-5 + 8e-3 * wanted.abs()).all(), 'workspace Zb'


def check_stage4(inputs, beta, chunks, variable, capture, expected):
    """Isolate the Stage 4 GEMM and independently propagate the full-path budget."""
    for head in range(inputs[0].shape[-3]):
        actual, local, full, budgets, l1_checks, l1_refs = [], [], [], [], [], []
        for c, (sequence, _, start, rows) in enumerate(chunks):
            a = torch.zeros((64, 64), dtype=torch.float64)
            src = inputs[3][head] if variable else inputs[3][sequence, head]
            a[:rows, :rows] = src[start:start + rows, :rows].double()
            b = torch.zeros(64, dtype=torch.float64)
            b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows].double()
            record = capture[head, c]
            zb = record[208 * 1024:216 * 1024].clone().view(torch.bfloat16).reshape(64, 64).double()
            got = record[224 * 1024:232 * 1024].clone().view(torch.bfloat16).reshape(64, 64).float()
            ref = expected[head][c]
            zb_ref = torch.tril((ref['zV'].double() - ref['zW'].double()) * b[None, :], -1).float().bfloat16().double()
            zb_budget = torch.tril((ref['zW_budget'].double() + 2e-6 + 3e-4 * ref['zV'].double().abs()) * b[None, :], -1)
            zb_budget += 2e-5 + 8e-3 * zb_ref.abs()
            l1_checks.append(record[232 * 1024:248 * 1024].clone().view(torch.float32).reshape(64, 64))
            l1_refs.append((got.double() @ a.T).float())
            actual.append(got)
            local.append((zb @ a.T).float().bfloat16().float())
            full.append((zb_ref @ a.T).float().bfloat16().float())
            budgets.append((zb_budget @ a.abs().T).float())
            assert torch.count_nonzero(got[rows:]) == 0 and torch.count_nonzero(got[:, rows:]) == 0, 'Tza tail'
        got, wanted = torch.stack(actual), torch.stack(full)
        torch.testing.assert_close(torch.stack(l1_checks), torch.stack(l1_refs), rtol=3e-4, atol=2e-6, msg='Stage 4 NZ L1 consumer')
        assert torch.isfinite(got).all(), 'Tza nonfinite'
        torch.testing.assert_close(got, torch.stack(local), rtol=8e-3, atol=2e-5, msg='Stage 4 local Tza')
        assert ((got - wanted).abs() <= torch.stack(budgets) + 2e-5 + 8e-3 * wanted.abs()).all(), 'Stage 4 full Tza'
    assert (capture[:, :, 248 * 1024:256 * 1024] == 0xA5).all(), 'Tza capture guard'


def stage5_reference(inputs, q, dq_raw, beta, chunk, head, variable, ref, scale):
    sequence, _, start, rows = chunk
    def token(x):
        width = x.shape[-1]
        out = torch.zeros((64, width), dtype=torch.float64)
        out[:rows] = (x[head] if variable else x[sequence, head])[start:start + rows].double()
        return out
    k, query, raw = token(inputs[0]), token(q), token(dq_raw)
    g = token(inputs[2])
    b = torch.zeros(64, dtype=torch.float64)
    b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows].double()
    exponent = torch.zeros_like(g); exponent[:rows] = torch.exp2(g[:rows])
    ratio = torch.exp2(g[rows - 1] - g)
    state = ref['dk_state_raw'].double() * ratio
    gb = ref['dKgb_raw'].double() * exponent
    dq = scale * raw * exponent
    dk = state - b[:, None] * gb
    db = ref['db_v'].double() - (k * gb).sum(-1)
    dg = query * dq - k * state - b[:, None] * k * gb
    dg[rows - 1] += (k * state).sum(0) + torch.exp2(g[rows - 1]) * ref['rH'].double()
    return dict(dq_base=dq.float(), dk_base=dk.float(), dg_base=dg.float(), db_base=db.float())


def check_stage5(inputs, q, dq_raw, beta, chunks, variable, capture, workspace, args, expected):
    names = ('dq_base', 'dk_base', 'dg_base', 'db_base')
    cap_offsets = (256, 288, 320, 352)
    ws_offsets = (32, 0, 168, 200)
    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1024 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 512 * 1024 + ((work // cores) & 1) * 256 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 5) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        for c, chunk in enumerate(chunks):
            if args.workspace_only and (head, c) not in retained:
                continue
            sequence, _, start, rows = chunk
            ref = dict(expected[head][c] if expected is not None else reference(inputs, chunks, head, c, variable))
            v = torch.zeros((64, 128), dtype=torch.float64)
            v[:rows] = (inputs[1][head] if variable else inputs[1][sequence, head])[start:start + rows].double()
            ref['db_v'] = (v * ref['dv_b'].double()).sum(-1).float()
            full = stage5_reference(inputs, q, dq_raw, beta, chunk, head, variable, ref, args.scale)
            if not args.workspace_only:
                record = capture[head, c]
                local = {key: decode(record, key) for key in ('dk_state_raw', 'dKgb_raw', 'rH')}
                local['db_v'] = record[216 * 1024:216 * 1024 + 256].clone().view(torch.float32)
                wanted_local = stage5_reference(inputs, q, dq_raw, beta, chunk, head, variable, local, args.scale)
            # Independent full-path bound propagates the fixed Stage 2 budget.
            k = torch.zeros((64, 128), dtype=torch.float64)
            g = torch.zeros_like(k); b = torch.zeros(64, dtype=torch.float64)
            k[:rows] = (inputs[0][head] if variable else inputs[0][sequence, head])[start:start + rows].double()
            g[:rows] = (inputs[2][head] if variable else inputs[2][sequence, head])[start:start + rows].double()
            b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows].double()
            exp = torch.zeros_like(g); exp[:rows] = torch.exp2(g[:rows])
            state_err = (2e-6 + 3e-4 * ref['dk_state_raw'].double().abs()) * torch.exp2(g[rows - 1] - g)
            gb_err = (ref['dKgb_budget'].double() + 2e-6 + 3e-4 * ref['dKgb_raw'].double().abs()) * exp
            dg_err = k.abs() * (state_err + b[:, None] * gb_err)
            dg_err[rows - 1] += (k.abs() * state_err).sum(0) + torch.exp2(g[rows - 1]) * (2e-6 + 3e-4 * ref['rH'].double().abs())
            budgets = dict(dq_base=torch.zeros_like(g), dk_base=state_err + b[:, None] * gb_err,
                           dg_base=dg_err, db_base=(k.abs() * gb_err).sum(-1) + 2e-6 + 3e-4 * ref['db_v'].double().abs())
            for name, co, wo in zip(names, cap_offsets, ws_offsets):
                count = 64 if name == 'db_base' else 8192
                shape = (64,) if name == 'db_base' else (64, 128)
                if args.workspace_only:
                    base = retained[head, c] + wo * 1024
                    got = workspace[base:base + count * 4].clone().view(torch.float32).reshape(shape)
                else:
                    got = record[co * 1024:co * 1024 + count * 4].clone().view(torch.float32).reshape(shape)
                    torch.testing.assert_close(got, wanted_local[name], rtol=3e-4, atol=2e-6,
                                               msg=f'Stage 5 local {name} head={head} chunk={c}')
                    if (head, c) in retained:
                        base = retained[head, c] + wo * 1024
                        actual_ws = workspace[base:base + count * 4].clone().view(torch.float32).reshape(shape)
                        torch.testing.assert_close(actual_ws, got, rtol=0, atol=0)
                assert torch.isfinite(got).all(), name
                assert ((got - full[name]).abs() <= budgets[name] + 2e-6 + 3e-4 * full[name].abs()).all(), f'Stage 5 full {name}'
                assert torch.count_nonzero(got[rows:]) == 0, f'{name} tail'
    if not args.workspace_only:
        assert (capture[:, :, 352 * 1024 + 256:360 * 1024 if args.stage >= 6 else CAPTURE_BYTES] == 0xA5).all(), 'Stage 5 capture guard'


def check_stage6(inputs, beta, chunks, variable, capture, workspace, args, expected):
    """Check the real NZ L1 consumer and propagate the unchanged Stage 4 budget."""
    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1024 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 512 * 1024 + ((work // cores) & 1) * 256 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 6) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        actual, full, budgets, local = [], [], [], []
        for c, (sequence, _, start, rows) in enumerate(chunks):
            if args.workspace_only and (head, c) not in retained:
                continue
            a = torch.zeros((64, 64), dtype=torch.float64)
            src = inputs[3][head] if variable else inputs[3][sequence, head]
            a[:rows, :rows] = src[start:start + rows, :rows].double()
            b = torch.zeros(64, dtype=torch.float64)
            b[:rows] = (beta[head] if variable else beta[sequence, head])[start:start + rows].double()
            ref = expected[head][c] if expected is not None else reference(inputs, chunks, head, c, variable)
            zb_ref = torch.tril((ref['zV'].double() - ref['zW'].double()) * b[None, :], -1).float().bfloat16().double()
            zb_budget = torch.tril((ref['zW_budget'].double() + 2e-6 + 3e-4 * ref['zV'].double().abs()) * b[None, :], -1)
            zb_budget += 2e-5 + 8e-3 * zb_ref.abs()
            tza = (zb_ref @ a.T).float().bfloat16().double()
            tza_budget = zb_budget @ a.abs().T + 2e-5 + 8e-3 * tza.abs()
            full.append((a.T @ tza).float())
            budgets.append((a.abs().T @ tza_budget).float())
            if args.workspace_only:
                offset = retained[head, c] + 216 * 1024
                got = workspace[offset:offset + 16 * 1024].clone().view(torch.float32).reshape(64, 64)
            else:
                record = capture[head, c]
                got = record[360 * 1024:376 * 1024].clone().view(torch.float32).reshape(64, 64)
                saved_tza = record[224 * 1024:232 * 1024].clone().view(torch.bfloat16).reshape(64, 64).double()
                local.append((a.T @ saved_tza).float())
                if (head, c) in retained:
                    offset = retained[head, c] + 216 * 1024
                    torch.testing.assert_close(workspace[offset:offset + 16 * 1024].clone().view(torch.float32).reshape(64, 64),
                                               got, rtol=0, atol=0, msg='Stage 6 workspace/capture')
            actual.append(got)
            assert torch.count_nonzero(got[rows:]) == 0 and torch.count_nonzero(got[:, rows:]) == 0, 'dAkk_raw tail'
        if not actual:
            continue
        got, wanted = torch.stack(actual), torch.stack(full)
        assert torch.isfinite(got).all(), 'Stage 6 nonfinite'
        if not args.workspace_only:
            torch.testing.assert_close(got, torch.stack(local), rtol=3e-4, atol=2e-6, msg='Stage 6 local GEMM')
        assert ((got - wanted).abs() <= torch.stack(budgets) + 2e-6 + 3e-4 * wanted.abs()).all(), 'Stage 6 full dAkk_raw'
    if not args.workspace_only:
        assert (capture[:, :, 376 * 1024:] == 0xA5).all(), 'Stage 6 capture guard'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4, 5, 6), default=0)
    parser.add_argument("--scale", type=float, default=128 ** -.5)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--suite", choices=("smoke", "tails", "model"), default="smoke")
    parser.add_argument("--workspace-only", action="store_true")
    parser.add_argument("--gate-mode", choices=("cumulative", "flat"), default="cumulative")
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    cases = [(1, 3, 65, None), (1, 3, 99, [0, 1, 0, 33, 65, 0])]
    if args.suite == "tails":
        cases = [(1, 3, t, None) for t in (1, 16, 31, 32, 33, 63, 64, 129)]
        cases += [(2, 3, 65, None), (1, 96, 65, None), (1, 3, 289, [0, 1, 31, 32, 33, 63, 64, 65, 0])]
    elif args.suite == "model":
        cases = [(1, 96, 8192, None), (1, 96, 8192, [0, 1, 31, 64, 65, 4096, 3935, 0])]
    for case in cases:
        check_case(args, *case)


if __name__ == "__main__":
    main()
