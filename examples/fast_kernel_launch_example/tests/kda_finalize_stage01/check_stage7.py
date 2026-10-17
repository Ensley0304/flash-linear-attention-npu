"""Independent FP64 Stage 7 formulas and BF16 residual/zero-fill checks."""
import torch


def read(record, offset, shape, dtype=torch.float32):
    count = 1
    for n in shape:
        count *= n
    size = count * torch.empty((), dtype=dtype).element_size()
    return record[offset:offset + size].clone().view(dtype).reshape(shape)


def check_split(got, wanted, name, exact=False):
    assert torch.isfinite(got).all(), name + " nonfinite"
    hi, lo = got.double().unbind(0)
    target = wanted.double()
    if exact:
        expected_hi = target.float().bfloat16()
        expected_lo = (target.float() - expected_hi.float()).bfloat16()
        torch.testing.assert_close(got[0], expected_hi, rtol=0, atol=0, msg=name + " hi")
        torch.testing.assert_close(got[1], expected_lo, rtol=0, atol=0, msg=name + " lo")
    else:
        # FP32 centering/Exp plus two BF16 roundings, relative to the unrounded operand.
        # This rejects a missing/wrong-sign residual without relying on a large absolute atol.
        tolerance = 3e-5 * target.abs() + 2e-38
        assert ((hi + lo - target).abs() <= tolerance).all(), name + " reconstruction"
        assert (lo.abs() <= hi.abs() / 128 + 2e-38).all(), name + " residual magnitude"
    zero = target == 0
    assert (got[:, zero] == 0).all(), name + " inactive/zero rows"


def check_stage7(inputs, q, daqk, beta, chunks, variable, capture, workspace, args):
    from check_stage01 import workspace_works, reference
    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1472 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 736 * 1024 + ((work // cores) & 1) * 368 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 7) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        for c, (sequence, _, first, rows) in enumerate(chunks):
            if args.workspace_only and (head, c) not in retained:
                continue
            def token(x):
                out = torch.zeros((64, x.shape[-1]), dtype=torch.float64)
                out[:rows] = (x[head] if variable else x[sequence, head])[first:first + rows].double()
                return out
            k, query, gate, aq = (token(x) for x in (inputs[0], q, inputs[2], daqk))
            b = torch.zeros(64, dtype=torch.float64)
            b[:rows] = (beta[head] if variable else beta[sequence, head])[first:first + rows].double()
            ws = workspace[retained[head, c]:retained[head, c] + 320 * 1024] if (head, c) in retained else None
            record = ws if args.workspace_only else capture[head, c]
            qo, ko = (64 * 1024, 80 * 1024) if args.workspace_only else (384 * 1024, 400 * 1024)
            check_split(read(record, qo, (2, 64, 64), torch.bfloat16), torch.tril(aq, -1) * 65536, 'dAqk lower', True)
            if not args.workspace_only:
                torch.testing.assert_close(read(record, 416 * 1024, (64,)).double(), aq.diag(), rtol=0, atol=0)
            got_ak = read(record, ko, (2, 64, 64), torch.bfloat16)
            if not args.workspace_only:
                ak = read(record, 360 * 1024, (64, 64)).double()
                check_split(got_ak, -torch.tril(ak, -1) * 65536, 'dAkk lower', True)
            else:
                # The raw Stage 6 slot has been consumed. Preserve its established full-path error budget.
                ref = reference(inputs, chunks, head, c, variable)
                a = token(inputs[3]); a[:, rows:] = 0
                zb = torch.tril((ref['zV'].double() - ref['zW'].double()) * b[None, :], -1).float().bfloat16().double()
                zb_err = torch.tril((ref['zW_budget'].double() + 2e-6 + 3e-4 * ref['zV'].double().abs()) * b[None, :], -1)
                zb_err += 2e-5 + 8e-3 * zb.abs()
                tza = (zb @ a.T).float().bfloat16().double()
                ak = a.T @ tza
                budget = a.abs().T @ (zb_err @ a.abs().T + 2e-5 + 8e-3 * tza.abs())
                wanted = -torch.tril(ak, -1) * 65536
                allowance = torch.tril(budget + 2e-6 + 3e-4 * ak.abs(), -1) * 65536 + wanted.abs() * 2e-5
                assert torch.isfinite(got_ak).all()
                assert ((got_ak.double().sum(0) - wanted).abs() <= allowance).all(), 'dAkk full path'
            for start in ([0] if args.workspace_only else range(0, rows, 32)):
                end = min(start + 32, rows)
                center = (gate[start] + gate[end - 1]) / 2
                kn, qp, bp = (torch.zeros((64, 128), dtype=torch.float64) for _ in range(3))
                kn[:end] = k[:end] * torch.exp2(center - gate[:end])
                qp[start:rows] = query[start:rows] * torch.exp2(gate[start:rows] - center)
                bp[start:rows] = b[start:rows, None] * k[start:rows] * torch.exp2(gate[start:rows] - center)
                exp = torch.zeros((32, 128), dtype=torch.float64)
                exp[:end-start] = torch.exp2(gate[start:end] - center)
                saved = (424 + start // 32 * 116) * 1024
                offsets = (96 * 1024, 128 * 1024, 224 * 1024) if args.workspace_only else tuple(saved + i * 32768 for i in range(3))
                for offset, wanted, name in zip(offsets, (kn, qp, bp), ('kNeg', 'qPos', 'bkPos')):
                    if not (args.workspace_only and args.stage >= 10 and name == 'kNeg'):
                        check_split(read(record, offset, (2, 64, 128), torch.bfloat16), wanted, name)
                if not args.workspace_only:
                    eo, co = saved + 96 * 1024, saved + 112 * 1024
                    torch.testing.assert_close(read(record, co, (128,)).double(), center, rtol=2e-7, atol=0)
                    torch.testing.assert_close(read(record, eo, (32, 128)).double(), exp, rtol=2e-5, atol=2e-38)
                if not args.workspace_only and start == 0 and ws is not None:
                    for wo, so, count in ((64, 384, 16), (80, 400, 16), (96, 424, 32), (128, 456, 32), (224, 488, 32)):
                        if args.stage >= 10 and wo == 96:
                            continue  # Stage 10 reuses the consumed kNeg slot.
                        assert torch.equal(ws[wo * 1024:(wo + count) * 1024], record[so * 1024:(so + count) * 1024]), 'Stage 7 workspace/capture'
                if not args.workspace_only:
                    assert (record[saved + 112 * 1024 + 512:saved + 116 * 1024] == 0xA5).all(), 'band guard'
            if not args.workspace_only:
                assert (record[416 * 1024 + 256:424 * 1024] == 0xA5).all(), 'diagonal guard'
                if rows <= 32:
                    assert (record[540 * 1024:656 * 1024] == 0xA5).all(), 'unused band'
