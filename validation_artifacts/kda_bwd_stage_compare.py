import importlib.util
import os
from pathlib import Path

import torch
import torch_npu

from fla_npu.ops import ascendc


def head(tensor):
    return tensor.transpose(1, 2).contiguous()


def load_low_level_helpers():
    path = Path(__file__).parents[1] / "fla/ops/ascendc/kda/chunk_kda_bwd/test/test_chunk_kda_bwd_p0.py"
    spec = importlib.util.spec_from_file_location("kda_bwd_p0_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(stage, names, actual, expected):
    for name, got, want in zip(names, actual, expected):
        diff = (got.float().cpu() - want.float()).abs()
        print(
            f"{stage}.{name}: max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g}", flush=True
        )


def main():
    torch.npu.set_device(0)
    data = torch.load(
        os.getenv("KDA_REF_DUMP", "/tmp/kda_bwd_triton_reference_strong.pt"),
        map_location="cpu", weights_only=False,
    )
    npu = lambda name: data[name].to("npu:0")
    helpers = load_low_level_helpers()
    scale = float(data["scale"])

    q, k, v, beta = (head(npu(name)) for name in ("q", "k", "v", "beta"))
    gk, aqk, akk, w, qg, kg, v_new = (
        head(npu(name)) for name in ("gk", "aqk", "akk", "w", "qg", "kg", "v_new")
    )
    do = head(npu("do"))
    h = npu("h")

    k1 = helpers.run_dav(aqk, v_new, do, scale)
    torch.npu.synchronize()
    k2_all = ascendc.chunk_gated_delta_rule_bwd_dhu(
        qg, kg, w, do, k1[1], scale=scale, chunk_size=64,
        g=None, gK=gk, h0=None, dht=None, cu_seqlens=None,
        chunk_indices=None,
    )
    k2 = (k2_all[0], k2_all[2])
    torch.npu.synchronize()
    # K3 intentionally reuses dead K2 storage (including dv_scan) as scratch.
    # Snapshot K2 here so stage-local validation does not compare overwritten GM.
    k2_snapshot = tuple(item.detach().float().cpu().clone() for item in k2)
    k3 = helpers.run_wy(q, k, v, v_new, gk, beta, akk, h, do, k2[0], k2[1], scale)
    torch.npu.synchronize()
    stage_count = int(os.getenv("KDA_BWD_WY_STAGE_COUNT", "8"))
    if stage_count == 4:
        # Zb occupies the first contiguous BF16 half of each FP32 dAkk task
        # allocation; it is not interleaved as 64 BF16 columns per FP32 row.
        raw_zb = (
            k3[5].detach().cpu().view(torch.bfloat16).flatten()[: 64 * 64]
            .reshape(1, 1, 64, 64).float()
        )
        d = k2_snapshot[1]
        v_cpu = v.float().cpu()
        beta_cpu = beta.float().cpu()
        expected_zb = torch.matmul(d, v_cpu.transpose(-1, -2))
        expected_zb = torch.tril(expected_zb, diagonal=-1) * beta_cpu.unsqueeze(-2)
        expected_zb = expected_zb.to(torch.bfloat16).float()
        diff = (raw_zb - expected_zb).abs()
        print(
            f"K3.Zb: max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g} "
            f"actual_abs_max={raw_zb.abs().max().item():.6g} "
            f"expected_abs_max={expected_zb.abs().max().item():.6g}",
            flush=True,
        )
        upper = torch.triu(raw_zb, diagonal=0)
        lower_diff = torch.tril(raw_zb - expected_zb, diagonal=-1).abs()
        print(
            f"K3.Zb.regions: upper_actual_abs_max={upper.abs().max().item():.6g} "
            f"lower_diff_max={lower_diff.max().item():.6g}",
            flush=True,
        )
        upper_abs = upper.abs()[0, 0]
        flat_index = int(upper_abs.argmax().item())
        print(
            f"K3.Zb.upper_argmax=({flat_index // 64},{flat_index % 64}) "
            f"row0_abs_max={upper_abs[0].max().item():.6g} "
            f"rows1_63_abs_max={upper_abs[1:].max().item():.6g}",
            flush=True,
        )
        return
    k4 = ascendc.chunk_kda_bwd_intra(
        q, k, gk, beta, k1[0], k3[5], k3[0], k3[1], k3[3], k3[4],
        cu_seqlens=None, chunk_indices=None, chunk_size=64,
        safe_gate=True, layout="BNSD",
    )
    torch.npu.synchronize()
    k6 = (helpers.run_gate_post(k4[3]),)
    torch.npu.synchronize()

    stages = data["stages"]
    report("K1", ("dAqk", "dv0"), k1, tuple(head(x) for x in stages["k1"]))
    k2_expected = (stages["k2"][0].transpose(1, 2).contiguous(), head(stages["k2"][1]))
    report("K2", ("dh", "dv_scan"), k2_snapshot, k2_expected)
    dv_scan_diff = (k2_snapshot[1] - k1[1].float().cpu()).abs()
    print(
        "K2.dv_scan_vs_K1.dv0: "
        f"max_abs={dv_scan_diff.max().item():.6g} "
        f"mean_abs={dv_scan_diff.mean().item():.6g}",
        flush=True,
    )
    for token_begin, token_end in ((0, 32), (32, 64)):
        half_diff = dv_scan_diff[:, :, token_begin:token_end]
        print(
            f"K2.dv_scan_vs_K1.dv0.tokens[{token_begin}:{token_end}]: "
            f"max_abs={half_diff.max().item():.6g} "
            f"mean_abs={half_diff.mean().item():.6g}",
            flush=True,
        )
    report("K3", ("dq", "dk", "dv", "db", "dg", "dAkk"), k3,
           tuple(head(x) for x in stages["k3"]))
    # Single-full-chunk orientation probe for the K3 A @ dv_scan branch.
    if q.shape[2] == 64:
        a_cpu = akk.float().cpu()
        d_cpu = k2_snapshot[1]
        beta_cpu = beta.float().cpu().unsqueeze(-1)
        dv_a = torch.matmul(a_cpu, d_cpu) * beta_cpu
        dv_at = torch.matmul(a_cpu.transpose(-1, -2), d_cpu) * beta_cpu
        for label, candidate in (("A@D", dv_a), ("AT@D", dv_at)):
            diff = (k3[2].float().cpu() - candidate).abs()
            print(
                f"K3.dv_orientation.{label}: max_abs={diff.max().item():.6g} "
                f"mean_abs={diff.mean().item():.6g}",
                flush=True,
            )
        d_cpu = d_cpu[0, 0]
        v_cpu = v.float().cpu()[0, 0]
        p_cpu = a_cpu[0, 0]
        beta_vec = beta.float().cpu()[0, 0]
        z = torch.matmul(d_cpu, v_cpu.transpose(-1, -2))
        z = torch.tril(z, diagonal=-1) * beta_vec.unsqueeze(0)
        z = z.to(torch.bfloat16).float()
        got_da = k3[5].float().cpu()[0, 0]
        want_da = head(stages["k3"][5]).float()[0, 0]
        persisted_t = k2[1].float().cpu()[0, 0, :, :64]
        expected_t = torch.matmul(z, p_cpu.T).to(torch.bfloat16).float()
        t_diff = (persisted_t - expected_t).abs()
        print(
            f"K3.Tza: max_abs={t_diff.max().item():.6g} "
            f"mean_abs={t_diff.mean().item():.6g} "
            f"actual_abs_max={persisted_t.abs().max().item():.6g} "
            f"expected_abs_max={expected_t.abs().max().item():.6g}",
            flush=True,
        )
        print(
            f"K3.dA_magnitude: actual={got_da.abs().max().item():.6g} "
            f"triton={want_da.abs().max().item():.6g}",
            flush=True,
        )
        for left_name, left in (("P", p_cpu), ("PT", p_cpu.T)):
            for right_name, right in (("P", p_cpu), ("PT", p_cpu.T)):
                t = torch.matmul(z, right).to(torch.bfloat16).float()
                candidate = torch.tril(-torch.matmul(left, t), diagonal=-1)
                for target_name, target in (("actual", got_da), ("triton", want_da)):
                    diff = (target - candidate).abs()
                    print(
                        f"K3.dA_orientation.{target_name}.{left_name}Z{right_name}: "
                        f"max_abs={diff.max().item():.6g} "
                        f"mean_abs={diff.mean().item():.6g}",
                        flush=True,
                    )
    report("K4", ("dq", "dk", "db", "dg"), k4,
           tuple(head(x) for x in stages["k4"]))
    report("K6", ("dg",), k6, (head(stages["k6"][0]),))


if __name__ == "__main__":
    main()
