"""Dense A2 canary/benchmark for the connected KDA backward A+B+C path.

Kernel B is PR291's optimized ``chunk_gated_delta_rule_bwd_dhu`` gK path.
It is called with q=qg, k=kg and dv=dv0.  Its head-major ``dh`` storage is
consumed directly by Kernel C through C's internal dual-layout tiling branch;
there is no transpose or fourth device launch.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401

from a2_canary import ACL_SUCCESS, CanaryRuntime, _error


class Invocation:
    def __init__(self, name, query, launch, query_args, device="npu"):
        self.name = name
        self.query = query
        self.launch_fn = launch
        self.query_args = query_args
        self.device = device
        self.workspace = None

    def prepare(self):
        size = ctypes.c_uint64()
        executor = ctypes.c_void_p()
        status = self.query(
            *self.query_args, ctypes.byref(size), ctypes.byref(executor))
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{self.name} GetWorkspaceSize failed: {status}")
        if self.workspace is None:
            self.workspace = torch.empty(
                size.value, dtype=torch.uint8, device=self.device)
        elif self.workspace.numel() < size.value:
            raise RuntimeError(
                f"{self.name} workspace grew from {self.workspace.numel()} "
                f"to {size.value}")
        return size.value, executor

    def launch(self, prepared, stream):
        size, executor = prepared
        workspace = (ctypes.c_void_p(self.workspace.data_ptr())
                     if size else None)
        status = self.launch_fn(workspace, size, executor, stream)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{self.name} launch failed: {status}")


def _configure_symbols(rt):
    get_a = rt.op.aclnnChunkKdaBwdAGetWorkspaceSize
    get_a.restype = ctypes.c_int
    get_a.argtypes = ([ctypes.c_void_p] * 7 +
                      [ctypes.c_float, ctypes.c_int64] +
                      [ctypes.c_void_p] * 4 +
                      [ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_void_p)])
    launch_a = rt.op.aclnnChunkKdaBwdA

    get_b = rt.op.aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize
    get_b.restype = ctypes.c_int
    get_b.argtypes = ([ctypes.c_void_p] * 11 +
                      [ctypes.c_double, ctypes.c_int64] +
                      [ctypes.c_void_p] * 3 +
                      [ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_void_p)])
    launch_b = rt.op.aclnnChunkGatedDeltaRuleBwdDhu

    get_c = rt.op.aclnnChunkKdaBwdCGetWorkspaceSize
    get_c.restype = ctypes.c_int
    get_c.argtypes = ([ctypes.c_void_p] * 17 +
                      [ctypes.c_float, ctypes.c_int64, ctypes.c_bool,
                       ctypes.c_bool, ctypes.c_float] +
                      [ctypes.c_void_p] * 8 +
                      [ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_void_p)])
    launch_c = rt.op.aclnnChunkKdaBwdC

    for fn in (launch_a, launch_b, launch_c):
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                       ctypes.c_void_p, ctypes.c_void_p]
    return get_a, launch_a, get_b, launch_b, get_c, launch_c


def _cpu_inputs(value):
    return value.detach().cpu().float()


def _reference_a(aqk, qg, vnew, h, d_o, scale, data):
    bsz, heads, seqlen, chunk = aqk.shape
    chunks = seqlen // chunk
    key_dim, value_dim = qg.shape[-1], d_o.shape[-1]
    dv0 = torch.empty_like(d_o)
    q0 = torch.empty((bsz, chunks, heads, key_dim, value_dim))
    dq_raw = torch.empty((bsz, heads, seqlen, key_dim))
    daqk = torch.empty((bsz, heads, seqlen, chunk))
    for b in range(bsz):
        for head in range(heads):
            for ci in range(chunks):
                begin, end = ci * chunk, (ci + 1) * chunk
                do_c = d_o[b, head, begin:end]
                dv0[b, head, begin:end] = (
                    aqk[b, head, begin:end].T @ do_c)
                q0[b, ci, head] = scale * (
                    qg[b, head, begin:end].T @ do_c)
                dq_raw[b, head, begin:end] = do_c @ h[b, ci, head].T
                daqk[b, head, begin:end] = do_c @ vnew[b, head, begin:end].T
    return dv0.to(data).float(), q0, dq_raw, daqk


def _reference_b(qg, kg, w, d_o, dv0, gk, scale, data):
    bsz, heads, seqlen, key_dim = qg.shape
    chunk, value_dim = 64, d_o.shape[-1]
    chunks = seqlen // chunk
    dh = torch.empty((bsz, heads, chunks, key_dim, value_dim))
    dv_scan = torch.empty_like(d_o)
    for b in range(bsz):
        for head in range(heads):
            state = torch.zeros((key_dim, value_dim), dtype=torch.float32)
            for ci in range(chunks - 1, -1, -1):
                begin, end = ci * chunk, (ci + 1) * chunk
                dh_dt = state.to(data).float()
                dh[b, head, ci] = dh_dt
                dv_state = kg[b, head, begin:end] @ dh_dt
                dv2 = (dv_state + dv0[b, head, begin:end]).to(data).float()
                dv_scan[b, head, begin:end] = dv2
                term_q = (qg[b, head, begin:end].T @
                          d_o[b, head, begin:end]).to(data).float()
                term_w = (w[b, head, begin:end].T @ dv2).to(data).float()
                decay = torch.exp2(gk[b, head, end - 1]).unsqueeze(-1)
                state = state * decay + term_q * scale - term_w
    return dh.to(data).float(), dv_scan.to(data).float()


def _reference_c_reduced(h, dh, dq_raw, vnew, scale):
    """Reference for the nontrivial reduced-path precision canary.

    The canary intentionally sets original q/k/v/beta/Akk/gk to zero.  The
    complete C graph then has closed-form outputs while still consuming both
    PR291 outputs: dh drives dk and dv_scan traverses the full WY pipeline.
    """
    bsz, heads, seqlen, key_dim = dq_raw.shape
    chunk, value_dim = 64, vnew.shape[-1]
    chunks = seqlen // chunk
    dq = dq_raw * scale
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    for b in range(bsz):
        for head in range(heads):
            for ci in range(chunks):
                begin, end = ci * chunk, (ci + 1) * chunk
                dh_c = dh[b, head, ci]
                dk[b, head, begin:end] = vnew[b, head, begin:end] @ dh_c.T
                gate_state = (h[b, ci, head] * dh_c).sum(dim=-1)
                dg[b, head, begin:end] = gate_state
    return dq, dk, dg


def run(args):
    if args.seqlen % 64:
        raise ValueError("connected dense benchmark currently requires T%64==0")
    torch.set_num_threads(1)
    torch.manual_seed(20260811)
    device = "npu"
    data = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    bsz, heads, seqlen = 1, args.heads, args.seqlen
    chunk, key_dim, value_dim = 64, 128, args.value_dim
    chunks = seqlen // chunk
    check = args.check

    def rand(shape, gain=0.03, dtype=data):
        if not check:
            return torch.empty(shape, dtype=dtype, device=device)
        return (torch.randn(shape) * gain).to(dtype).to(device)

    def zero(shape, dtype=data):
        return torch.zeros(shape, dtype=dtype, device=device)

    aqk = rand((bsz, heads, seqlen, chunk))
    qg = rand((bsz, heads, seqlen, key_dim))
    kg = rand((bsz, heads, seqlen, key_dim))
    w = rand((bsz, heads, seqlen, key_dim))
    vnew = rand((bsz, heads, seqlen, value_dim))
    h = rand((bsz, chunks, heads, key_dim, value_dim), gain=0.02)
    d_o = rand((bsz, heads, seqlen, value_dim))
    gk = zero((bsz, heads, seqlen, key_dim), torch.float32)

    if check:
        q = zero((bsz, heads, seqlen, key_dim))
        k = zero((bsz, heads, seqlen, key_dim))
        v = zero((bsz, heads, seqlen, value_dim))
        beta = zero((bsz, heads, seqlen), torch.float32)
        akk = zero((bsz, heads, seqlen, chunk))
    else:
        # Reuse storage in the performance-only path; values do not change the
        # fixed instruction path and this keeps the large-shape HBM footprint
        # close to the real saved-for-backward set.
        q, k, v = qg, kg, vnew
        beta = torch.empty((bsz, heads, seqlen), dtype=torch.float32,
                           device=device)
        akk = aqk

    dv0 = torch.empty_like(d_o)
    q0 = torch.empty((bsz, chunks, heads, key_dim, value_dim),
                     dtype=torch.float32, device=device)
    dq_raw = torch.empty((bsz, heads, seqlen, key_dim),
                         dtype=torch.float32, device=device)
    daqk = torch.empty((bsz, heads, seqlen, chunk),
                       dtype=torch.float32, device=device)
    dh = torch.empty((bsz, heads, chunks, key_dim, value_dim),
                     dtype=data, device=device)
    dv_scan = torch.empty_like(d_o)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v)
    db = torch.empty((bsz, heads, seqlen), dtype=torch.float32, device=device)
    dg = torch.empty_like(gk)
    dakk = torch.empty_like(daqk)

    rt = CanaryRuntime(args.op_api)
    get_a, launch_a, get_b, launch_b, get_c, launch_c = _configure_symbols(rt)
    values = [aqk, qg, kg, w, vnew, h, d_o, gk, q, k, v, beta, akk,
              dv0, q0, dq_raw, daqk, dh, dv_scan, dq, dk, dv, db, dg, dakk]
    desc = {id(value): rt.tensor(value) for value in values}
    hd = lambda value: desc[id(value)].handle

    a = Invocation("A", get_a, launch_a, [
        hd(aqk), hd(qg), hd(vnew), hd(h), hd(d_o), None, None,
        args.scale, chunk, hd(dv0), hd(q0), hd(dq_raw), hd(daqk)])
    b = Invocation("B", get_b, launch_b, [
        hd(qg), hd(kg), hd(w), hd(d_o), hd(dv0), None, hd(gk),
        None, None, None, None, args.scale, chunk, hd(dh), None,
        hd(dv_scan)])
    c = Invocation("C", get_c, launch_c, [
        hd(q), hd(k), hd(v), hd(vnew), hd(gk), hd(beta), hd(akk),
        hd(h), hd(dh), hd(dv_scan), hd(dq_raw), hd(daqk),
        None, None, None, None, None,
        args.scale, chunk, True, False, args.lower_bound,
        hd(dq), hd(dk), hd(dv), hd(db), hd(dg), hd(dakk), None, None])
    invocations = [a, b, c]
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)

    def launch_chain():
        prepared = [op.prepare() for op in invocations]
        for op, item in zip(invocations, prepared):
            op.launch(item, stream)

    launch_chain()
    torch.npu.synchronize()

    if check:
        aqk_c, qg_c, kg_c, w_c = map(_cpu_inputs, (aqk, qg, kg, w))
        vnew_c, h_c, do_c, gk_c = map(_cpu_inputs, (vnew, h, d_o, gk))
        dv0_r, q0_r, dqraw_r, daqk_r = _reference_a(
            aqk_c, qg_c, vnew_c, h_c, do_c, args.scale, data)
        dh_r, dvscan_r = _reference_b(
            qg_c, kg_c, w_c, do_c, dv0_r, gk_c, args.scale, data)
        dq_r, dk_r, dg_r = _reference_c_reduced(
            h_c, dh_r, dqraw_r, vnew_c, args.scale)
        reports = {
            "A.dv0": _error("A.dv0", dv0, dv0_r),
            "A.Q0": _error("A.Q0", q0, q0_r),
            "A.dq_raw": _error("A.dq_raw", dq_raw, dqraw_r),
            "A.dAqk": _error("A.dAqk", daqk, daqk_r),
            "B.dh": _error("B.dh", dh, dh_r),
            "B.dv_scan": _error("B.dv_scan", dv_scan, dvscan_r),
            "C.dq": _error("C.dq", dq, dq_r),
            "C.dk": _error("C.dk", dk, dk_r),
            "C.dg": _error("C.dg", dg, dg_r),
        }
        zero_reports = {
            name: float(value.detach().float().abs().max().cpu())
            for name, value in (("C.dv", dv), ("C.db", db),
                                ("C.dAkk", dakk))
        }
        print("ABC_PRECISION", reports, zero_reports, flush=True)
        failed = [name for name, report in reports.items()
                  if report["cos"] < 0.99 or report["mean_abs"] > 3e-3]
        failed += [name for name, maximum in zero_reports.items()
                   if maximum > 1e-6]
        if failed:
            raise AssertionError(f"A+B+C precision failed: {failed}")

    for _ in range(args.warmup):
        launch_chain()
        torch.npu.synchronize()

    event_ms, host_ms = [], []
    for _ in range(args.repeat):
        prepared = [op.prepare() for op in invocations]
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        start.record()
        for op, item in zip(invocations, prepared):
            op.launch(item, stream)
        end.record()
        torch.npu.synchronize()
        host_ms.append((time.perf_counter_ns() - t0) / 1e6)
        event_ms.append(start.elapsed_time(end))

    result = {
        "shape": [bsz, heads, seqlen, key_dim, value_dim],
        "dtype": args.dtype,
        "launches": 3,
        "event_ms": {"median": statistics.median(event_ms),
                     "min": min(event_ms), "max": max(event_ms)},
        "host_ms": {"median": statistics.median(host_ms),
                    "min": min(host_ms), "max": max(host_ms)},
        "workspace_bytes": {op.name: op.workspace.numel()
                            for op in invocations},
        "repeat": args.repeat,
    }
    print("ABC_PERF", result, flush=True)
    rt.destroy(*desc.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-api", default=os.environ.get(
        "KDA_BWD_OP_API", "libcust_opapi.so"))
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--scale", type=float, default=0.08838834764831845)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
