"""Stage 9: band scaling, diagonal contribution, Q norm and live outputs."""
import torch

from check_stage7 import read


def check_stage9(inputs, q, dq_raw, beta, daqk, rstd, chunks, variable,
                 capture, workspace, dq_out, args, expected):
    from check_stage01 import reference, stage5_reference, workspace_works

    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1280 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 640 * 1024 + ((work // cores) & 1) * 320 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 9) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        for c, chunk in enumerate(chunks):
            sequence, _, first, rows = chunk

            def token(x):
                return (x[head] if variable else x[sequence, head])[first:first + rows].double()

            query, key, gate, aq = (token(x) for x in (q, inputs[0], inputs[2], daqk))
            diagonal = aq[torch.arange(rows), torch.arange(rows), None]
            # Direct causal differences avoid depending on device centering or saved exponent.
            causal = torch.ones((rows, rows), dtype=torch.bool).tril(-1)
            difference = gate[:, None, :] - gate[None, :, :]
            difference.masked_fill_(~causal[:, :, None], 0)
            terms = aq[:, :rows].tril(-1)[:, :, None] * key[None, :, :] * torch.exp2(difference)
            local = terms.sum(1)
            local_magnitude = terms.abs().sum(1)

            if not args.workspace_only:
                record = capture[head, c]
                base_q = read(record, 256 * 1024, (64, 128)).double()[:rows]
                base_g = read(record, 320 * 1024, (64, 128)).double()[:rows]
                base_q_error = torch.zeros_like(base_q)
                base_g_error = torch.zeros_like(base_g)
            else:
                ref = dict(reference(inputs, chunks, head, c, variable))
                values = torch.zeros((64, 128), dtype=torch.float64)
                values[:rows] = token(inputs[1])
                ref['db_v'] = (values * ref['dv_b'].double()).sum(-1).float()
                bases = stage5_reference(inputs, q, dq_raw, beta, chunk, head, variable, ref, args.scale)
                base_q, base_g = (bases[name].double()[:rows] for name in ('dq_base', 'dg_base'))
                base_q_error = 2e-6 + 3e-4 * base_q.abs()
                exp = torch.exp2(gate)
                state_error = (2e-6 + 3e-4 * ref['dk_state_raw'].double()[:rows].abs()) * torch.exp2(gate[-1] - gate)
                gb_error = (ref['dKgb_budget'].double()[:rows] + 2e-6 +
                            3e-4 * ref['dKgb_raw'].double()[:rows].abs()) * exp
                base_g_error = key.abs() * (state_error + token(beta)[:, None] * gb_error)
                base_g_error[-1] += (key.abs() * state_error).sum(0) + torch.exp2(gate[-1]) * (
                    2e-6 + 3e-4 * ref['rH'].double().abs())
                base_g_error += 2e-6 + 3e-4 * base_g.abs()

            pre = base_q + local + diagonal * key
            wanted_g = base_g + query * local
            local_error = 6e-5 * local_magnitude + 2e-6
            q_error = base_q_error + local_error + 3e-4 * pre.abs() + 2e-6
            g_error = base_g_error + query.abs() * local_error + 3e-4 * wanted_g.abs() + 2e-6
            wanted_q = pre
            if rstd is not None:
                scale = token(rstd)[:, None]
                wanted_q = (pre - query * (pre * query).sum(-1, keepdim=True)) * scale
                q_error = (q_error + query.abs() * (q_error * query.abs()).sum(-1, keepdim=True)) * scale.abs()
            got_dq = token(dq_out)
            assert torch.isfinite(got_dq).all(), 'Stage 9 dq nonfinite'
            assert ((got_dq - wanted_q).abs() <= q_error + 8e-3 * wanted_q.abs() + 2e-5).all(), 'Stage 9 full dq'

            if args.workspace_only and (head, c) not in retained:
                continue
            record = (workspace[retained[head, c]:retained[head, c] + 320 * 1024]
                      if args.workspace_only else capture[head, c])
            qo, go = (32, 168) if args.workspace_only else (720, 752)
            got_q = read(record, qo * 1024, (64, 128)).double()[:rows]
            got_g = read(record, go * 1024, (64, 128)).double()[:rows]
            assert torch.isfinite(got_q).all() and torch.isfinite(got_g).all(), 'Stage 9 nonfinite'
            assert ((got_q - wanted_q).abs() <= q_error).all(), 'Stage 9 FP64 dq'
            assert ((got_g - wanted_g).abs() <= g_error).all(), 'Stage 9 FP64 dg'
            torch.testing.assert_close(got_dq, got_q.float().bfloat16().double(), rtol=0, atol=0)
            if not args.workspace_only:
                for start in range(0, rows, 32):
                    end = min(start + 32, rows)
                    raw = read(record, 656 * 1024, (64, 128)).double()[start:end]
                    exp = read(record, (424 + start // 32 * 116 + 96) * 1024, (32, 128)).double()[:end-start]
                    delta = raw * exp / 65536
                    local_q = base_q[start:end] + delta + diagonal[start:end] * key[start:end]
                    local_g = base_g[start:end] + query[start:end] * delta
                    if rstd is not None:
                        qq = query[start:end]
                        local_q = (local_q - qq * (local_q * qq).sum(-1, keepdim=True)) * token(rstd)[start:end, None]
                    torch.testing.assert_close(got_q[start:end], local_q, rtol=3e-4, atol=2e-6)
                    torch.testing.assert_close(got_g[start:end], local_g, rtol=3e-4, atol=2e-6)
                if (head, c) in retained:
                    ws = workspace[retained[head, c]:retained[head, c] + 320 * 1024]
                    for offset, got in ((32, got_q), (168, got_g)):
                        torch.testing.assert_close(read(ws, offset * 1024, (64, 128))[:rows].double(), got, rtol=0, atol=0)
                padded = (rows + 31) // 32 * 32
                for offset in (720, 752):
                    assert (read(record, offset * 1024, (64, 128))[rows:padded] == 0).all(), 'Stage 9 tail'
                    assert (record[offset * 1024 + padded * 512:(offset + 32) * 1024] == 0xA5).all(), 'Stage 9 guard'
