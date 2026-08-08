import os

import torch
import torch_npu

from fla_npu.ops import ascendc


def head_major(tensor):
    return tensor.transpose(1, 2).contiguous()


def main():
    torch.npu.set_device(0)
    device = "npu:0"
    dump_path = os.getenv("KDA_REF_DUMP", "/tmp/kda_bwd_triton_reference.pt")
    data = torch.load(dump_path, map_location="cpu", weights_only=False)

    def npu(name):
        return data[name].to(device)

    q, k, v, beta = (npu(name) for name in ("q", "k", "v", "beta"))
    actual = ascendc.chunk_kda_bwd(
        q,
        k,
        v,
        beta,
        *(head_major(npu(name)) for name in (
            "gk", "aqk", "akk", "w", "qg", "kg", "v_new"
        )),
        npu("h"),
        npu("do"),
        float(data["scale"]),
        raw_g=None,
        A_log=None,
        dt_bias=None,
        initial_state=None,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        layout="BSND",
        chunk_size=64,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
        state_v_first=False,
        recompute_policy="NONE",
    )
    torch.npu.synchronize()

    failures = []
    for name, got, want_cpu in zip(
        ("dq", "dk", "dv", "db", "dg"), actual[:5], data["expected"]
    ):
        got_f = got.float().cpu()
        want_f = want_cpu.float()
        diff = (got_f - want_f).abs()
        denom = want_f.abs().clamp_min(1e-7)
        print(
            f"{name}: max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g} "
            f"max_rel={torch.max(diff / denom).item():.6g} "
            f"got_abs_max={got_f.abs().max().item():.6g} "
            f"ref_abs_max={want_f.abs().max().item():.6g}",
            flush=True,
        )
        try:
            torch.testing.assert_close(got_f, want_f, rtol=2e-2, atol=2e-4)
        except AssertionError as error:
            failures.append(f"{name}: {error}")

    if failures:
        raise AssertionError("\n\n".join(failures))
    print("KDA_BWD_TRITON_REFERENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
