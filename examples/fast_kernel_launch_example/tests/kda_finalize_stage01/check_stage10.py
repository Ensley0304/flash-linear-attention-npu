"""Stage 10: six-term transposed GEMM and independently centered FP64 formula."""
import torch

from check_stage7 import read


def check_stage10(inputs, q, beta, daqk, chunks, variable, capture, workspace, args):
    from check_stage01 import workspace_works
    heads = inputs[0].shape[-3]
    cores = workspace.numel() // (1280 * 1024)
    retained = {(work // len(chunks) * 2 + owner, work % len(chunks)):
                (2 * core + owner) * 640 * 1024 + ((work // cores) & 1) * 320 * 1024
                for core, work in workspace_works(heads, len(chunks), cores, 10) for owner in range(2)
                if work // len(chunks) * 2 + owner < heads}
    for head in range(heads):
        for c, (sequence, _, first, rows) in enumerate(chunks):
            if args.workspace_only and (head, c) not in retained:
                continue

            def token(x):
                out = torch.zeros((64, *x.shape[x.ndim-1:]), dtype=torch.float64)
                out[:rows] = (x[head] if variable else x[sequence, head])[first:first + rows].double()
                return out

            key, query, gate, aq = (token(x) for x in (inputs[0], q, inputs[2], daqk))
            b = torch.zeros(64, dtype=torch.float64)
            b[:rows] = (beta[head] if variable else beta[sequence, head])[first:first + rows].double()
            ws = workspace[retained[head, c]:retained[head, c] + 320 * 1024] if (head, c) in retained else None
            record = ws if args.workspace_only else capture[head, c]
            matrices = [read(record, offset * 1024, (2, 64, 64), torch.bfloat16).double()
                        for offset in ((64, 80) if args.workspace_only else (384, 400))]
            for start in ([0] if args.workspace_only else range(0, rows, 32)):
                end = min(start + 32, rows)
                center = (gate[start] + gate[end - 1]) / 2
                pos = torch.zeros((64, 128), dtype=torch.float64)
                pos[start:rows] = torch.exp2(gate[start:rows] - center)
                operands = (query * pos, b[:, None] * key * pos)
                saved = 424 + start // 32 * 116
                offsets = (128, 224) if args.workspace_only else (saved + 32, saved + 64)
                got = read(record, (96 if args.workspace_only else 784) * 1024 + start * 512, (32, 128)).double()
                assert torch.isfinite(got).all(), 'right_raw nonfinite'
                local, magnitude, wanted, budget = (torch.zeros_like(got) for _ in range(4))
                for matrix, (offset, operand) in enumerate(zip(offsets, operands)):
                    ah, al = matrices[matrix][:, :, start:start + 32].transpose(-1, -2).unbind(0)
                    bh, bl = read(record, offset * 1024, (2, 64, 128), torch.bfloat16).double().unbind(0)
                    local += ah @ bh + ah @ bl + al @ bh
                    magnitude += ah.abs() @ bh.abs() + ah.abs() @ bl.abs() + al.abs() @ bh.abs()
                    # dAkk's Stage 0-7 full-path budget is checked separately. Here
                    # isolate transpose, sign, both branches, exponent and residuals.
                    a = torch.tril(aq, -1)[:, start:start + 32].T * 65536 if matrix == 0 else ah + al
                    wanted += a @ operand
                    budget += 6e-5 * (a.abs() @ operand.abs()) + a.abs().sum(-1, keepdim=True) * 2e-38
                assert ((got - local).abs() <= 3e-6 * magnitude + 2e-38).all(), 'right_raw six-term GEMM'
                assert ((got - wanted).abs() <= budget + 2e-38).all(), 'right_raw FP64 band formula'
                assert (got[end-start:] == 0).all(), 'right_raw tail'
                if not args.workspace_only and start == 0 and ws is not None:
                    torch.testing.assert_close(read(ws, 96 * 1024, (32, 128)).double(), got, rtol=0, atol=0)
            if not args.workspace_only and rows <= 32:
                assert (record[800 * 1024:816 * 1024] == 0xA5).all(), 'right_raw uncomputed band'
