"""Stage 8: independent band GEMMs, residual accumulation and retained workspace."""
import torch

from check_stage7 import read


def check_stage8(inputs, daqk, chunks, variable, capture, workspace, args):
    from check_stage01 import workspace_works
    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1280 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 640 * 1024 + ((work // cores) & 1) * 320 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 8) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        for c, (sequence, _, first, rows) in enumerate(chunks):
            if args.workspace_only and (head, c) not in retained:
                continue

            def token(x):
                out = torch.zeros((64, x.shape[-1]), dtype=torch.float64)
                out[:rows] = (x[head] if variable else x[sequence, head])[first:first + rows].double()
                return out

            k, gate, aq = (token(x) for x in (inputs[0], inputs[2], daqk))
            ws = workspace[retained[head, c]:retained[head, c] + 320 * 1024] if (head, c) in retained else None
            record = ws if args.workspace_only else capture[head, c]
            matrices = [read(record, offset * 1024, (2, 64, 64), torch.bfloat16).double()
                        for offset in ((64, 80) if args.workspace_only else (384, 400))]
            for start in range(0, rows, 32):
                end = min(start + 32, rows)
                center = (gate[start] + gate[end - 1]) / 2
                kn = torch.zeros((64, 128), dtype=torch.float64)
                kn[:end] = k[:end] * torch.exp2(center - gate[:end])
                if not args.workspace_only:
                    bh, bl = read(record, (424 + start // 32 * 116) * 1024,
                                  (2, 64, 128), torch.bfloat16).double().unbind(0)
                for matrix, name in enumerate(('dq_local_raw', 'left_raw')):
                    offset = ((256, 288) if args.workspace_only else (656, 688))[matrix] * 1024
                    got = read(record, offset + start * 128 * 4, (32, 128)).double()
                    assert torch.isfinite(got).all(), name + ' nonfinite'
                    ah, al = matrices[matrix][:, start:start + 32].unbind(0)
                    if not args.workspace_only:
                        local = ah @ bh + ah @ bl + al @ bh
                        budget = 2e-6 * (ah.abs() @ bh.abs() + ah.abs() @ bl.abs() + al.abs() @ bh.abs()) + 2e-38
                        assert ((got - local).abs() <= budget).all(), name + ' three-term GEMM'
                    # Stage 6/7 have separate full-path gates; this checks centering and GEMM
                    # against FP64 without reconstructing the exponent from the device result.
                    a = torch.tril(aq, -1)[start:start + 32] * 65536 if matrix == 0 else ah + al
                    wanted = a @ kn
                    # Propagate Stage 7's existing 2e-38 absolute operand budget through A.
                    # Tiny BF16 residuals can underflow before multiplication by the 2^16-scaled A.
                    budget = 6e-5 * (a.abs() @ kn.abs()) + a.abs().sum(-1, keepdim=True) * 2e-38 + 2e-38
                    assert ((got - wanted).abs() <= budget).all(), name + ' FP64 band formula'
                    assert (got[end-start:] == 0).all(), name + ' tail'
                    if not args.workspace_only and ws is not None:
                        actual = read(ws, (256 + matrix * 32) * 1024 + start * 128 * 4, (32, 128))
                        torch.testing.assert_close(actual.double(), got, rtol=0, atol=0)
            if not args.workspace_only and rows <= 32:
                for offset in (656, 688):
                    assert (record[(offset + 16) * 1024:(offset + 32) * 1024] == 0xA5).all(), 'uncomputed band guard'
