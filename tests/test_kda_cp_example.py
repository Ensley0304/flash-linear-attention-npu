"""CPU checks of ordered PTP exchange, packed resets and recurrence gradients."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
# assert_close lazily imports DTensor on newer torch. Load its registrations
# before patch.dict(sys.modules), which would otherwise remove imported modules.
import torch.distributed.tensor

spec = importlib.util.spec_from_file_location(
    "kda_cp", Path(__file__).resolve().parents[1] / "examples/cp/chunk_kda_cp.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
torch.set_num_threads(1)


class KdaCpTests(unittest.TestCase):
    def test_external_beta_sigmoid_chain_rule(self):
        """Check the operator's effective-beta gradient becomes a logits gradient."""
        inputs = example.make_inputs(5, 3, 29)
        logits = inputs["beta"].clone().requires_grad_(True)
        effective = logits.sigmoid()
        (effective.square().sum() / 2).backward()
        token_shape = (1, 5, 3, 128)

        def forward(q, k, v, g, beta, *args, **kwargs):
            self.assertFalse(kwargs["use_beta_sigmoid_in_kernel"])
            torch.testing.assert_close(beta, effective.detach())
            return (torch.zeros(token_shape),) + (None,) * 11

        def backward(q, k, v, beta, *args, **kwargs):
            self.assertFalse(kwargs["use_beta_sigmoid_in_kernel"])
            self.assertIsNone(kwargs["beta_eff"])
            torch.testing.assert_close(beta, effective.detach().transpose(1, 2))
            return (torch.zeros(token_shape), torch.zeros(token_shape), torch.zeros(token_shape),
                    beta.clone(), torch.zeros(token_shape), None, torch.zeros(3), torch.zeros(3, 128))

        fake = types.ModuleType("fla_npu.ops")
        fake.ascendc = types.SimpleNamespace(chunk_kda_fwd=forward, chunk_kda_bwd=backward)
        with patch.dict(sys.modules, {"fla_npu.ops": fake}):
            result = example.run_kda(inputs, {"cu": [0, 5]}, "none", fuse_beta_sigmoid=False)
        torch.testing.assert_close(result["dbeta"], logits.grad.transpose(1, 2))

    def test_allgather_only_merges_same_sequence_neighbours(self):
        summaries = []
        for r in range(4):
            matrix = torch.eye(128)[None]
            matrix[0, r, (r + 1) % 4] = r + 1
            summaries.append(torch.cat((torch.full_like(matrix, float(r + 1)), matrix), -1))

        def gather(outputs, local):
            for output, summary in zip(outputs, summaries):
                output[0].copy_(summary)

        def merge(state, gathered, count, rank, *, forward, state_v_first):
            indices = range(rank - count, rank) if forward else range(rank + count, rank, -1)
            for i in indices:
                state.copy_(gathered[i, ..., 128:] @ state + gathered[i, ..., :128])

        fake = types.ModuleType("fla_npu.ops.ascendc")
        fake.merge_fwd_bwd_kernel = merge
        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}), \
                patch.object(example.dist, "all_gather", side_effect=gather) as transport:
            for forward in (True, False):
                for rank in range(4):
                    part = dict(slot=rank, first=0, last=1, prefix=min(rank, 1), suffix=min(3-rank, 1))
                    plan = dict(rank=rank, world=4, num_parts=4, cu=[0, 17, 64], parts=[part])
                    actual = example.exchange_boundary(summaries[rank][None], plan,
                                                       forward=forward, communication="allgather")
                    count = part["prefix" if forward else "suffix"]
                    expected = summaries[rank-1 if forward else rank+1][..., :128] if count else torch.zeros(1,128,128)
                    torch.testing.assert_close(actual[0 if forward else -1], expected if forward else expected.bfloat16())
                    self.assertEqual(actual[-1 if forward else 0].count_nonzero(), 0)
            self.assertEqual(transport.call_count, 8)

    def test_ptp_noncommuting_prefix_suffix_and_resets(self):
        """Validate every round's send snapshot and result against a serial recurrence.

        The transport supplies independently precomputed inclusive prefixes. In
        particular a wrong multiplication order or same-round reuse cannot pass.
        """
        generator = torch.Generator().manual_seed(71)
        for world in (1, 2, 3, 4, 6, 8):
            for forward in (True, False):
                for reset_at in (None, world // 2):
                    # Model a rank containing two sequences: outgoing M is zero.
                    originals = []
                    for _ in range(world):
                        matrix = torch.eye(128)[None] * 0.7 + torch.randn(1, 128, 128, generator=generator) * 0.01
                        state = torch.randn(1, 128, 128, generator=generator) * 0.01
                        originals.append(torch.cat((state, matrix), -1))
                    maps = [x.clone() for x in originals]
                    maps[0][..., 128:] = 0
                    if reset_at is not None:
                        maps[reset_at][..., 128:] = 0
                    snapshots = []
                    distance = 1
                    while distance < world:
                        snapshots.append([x.clone() for x in maps])
                        updated = [x.clone() for x in maps]
                        for r in range(distance, world):
                            updated[r] = torch.cat((
                                maps[r][..., 128:] @ maps[r-distance][..., :128] + maps[r][..., :128],
                                maps[r][..., 128:] @ maps[r-distance][..., 128:],
                            ), -1)
                        maps = updated
                        distance *= 2
                    expected = torch.zeros(1, 128, 128)
                    for logical in range(world):
                        rank = logical if forward else world - 1 - logical
                        part = dict(slot=rank, first=0, last=int(logical == reset_at),
                                    prefix=logical if forward else 0, suffix=logical if not forward else 0)
                        plan = dict(rank=rank, world=world, num_parts=world,
                                    cu=[0, 32, 64] if logical == reset_at else [0, 64], parts=[part])
                        step = [0]

                        def exchange(operations):
                            d = 2 ** step[0]
                            for operation in operations:
                                if operation.op == example.dist.isend:
                                    torch.testing.assert_close(operation.tensor, snapshots[step[0]][logical])
                                    self.assertEqual(operation.peer, rank + d if forward else rank - d)
                                else:
                                    self.assertEqual(operation.peer, rank - d if forward else rank + d)
                                    operation.tensor.copy_(snapshots[step[0]][logical-d])
                            step[0] += 1
                            return [types.SimpleNamespace(wait=lambda: None) for _ in operations]

                        before = originals[logical].clone()
                        with patch.object(example.dist, "batch_isend_irecv", side_effect=exchange), \
                                patch.object(example.dist, "P2POp", side_effect=lambda op, tensor, peer:
                                             types.SimpleNamespace(op=op, tensor=tensor, peer=peer)):
                            actual = example.exchange_boundary(originals[logical][None], plan, forward=forward, communication="ptp")
                        torch.testing.assert_close(originals[logical], before)
                        boundary = actual[0 if forward else -1]
                        torch.testing.assert_close(boundary, expected if forward else expected.bfloat16())
                        if len(plan["cu"]) > 2:
                            self.assertEqual(actual[-1 if forward else 0].count_nonzero(), 0)
                        matrix = before[..., 128:] if logical != reset_at else torch.zeros_like(before[..., 128:])
                        expected = matrix @ expected + before[..., :128]

    def test_future_loss_and_packed_reset(self):
        inputs = example.make_inputs(128, 1, 19)
        inputs["do"][:, :, :64] = 0
        contiguous, _, states = example.recurrent_reference(inputs, [0, 128], 64)
        self.assertGreater(states[1].abs().max(), 1e-5)
        self.assertGreater(contiguous["dv"][:, :64].abs().max(), 1e-5)
        packed, _, states = example.recurrent_reference(inputs, [0, 64, 128], 64)
        self.assertEqual(states[1].count_nonzero(), 0)
        self.assertEqual(packed["dv"][:, :64].count_nonzero(), 0)

    def test_arguments(self):
        args = example.parse_args(["--communication", "ptp", "--tokens", "257", "--cu-seqlens", "0,0,131,257"])
        self.assertEqual(args.cu, [0, 0, 131, 257])
        with patch.dict("os.environ", {"WORLD_SIZE": "4"}):
            self.assertEqual(example.parse_args(["--tokens", "4", "--heads", "129"]).heads, 129)
            for mode in ("allgather", "ptp"):
                for tokens in (4, 12, 252, 260):
                    self.assertEqual(example.parse_args(["--communication", mode, "--tokens", str(tokens),
                                                         "--heads", "3"]).tokens, tokens)
                self.assertEqual(example.parse_args(["--communication", mode, "--tokens", "520",
                                                     "--partition", "zigzag"]).tokens, 520)
            for argv in (["--communication", "none"], ["--tokens", "257"]):
                with self.assertRaises(SystemExit):
                    example.parse_args(argv)
            with self.assertRaises(SystemExit):
                example.parse_args(["--tokens", "260", "--partition", "zigzag"])

    def test_partition_boundaries(self):
        for layout in ("contiguous", "zigzag"):
            for world in (1, 2, 3, 4, 6, 8):
                count = world * (2 if layout == "zigzag" else 1)
                for length in (1, 3, 63, 65):
                    tokens = count * length
                    cu = sorted([0, 0, 1, tokens // 2, tokens - 1, tokens, tokens])
                    all_tokens = []
                    for rank in range(world):
                        plan = example.build_partition(cu, tokens, rank, world, layout)
                        local = []
                        expected_cu = [0]
                        for p in plan["parts"]:
                            self.assertEqual(p["lo"], p["slot"] * length)
                            local.extend(range(p["lo"], p["hi"]))
                            for token in range(p["lo"], p["hi"]):
                                if token + 1 in cu or token + 1 == p["hi"]:
                                    expected_cu.append(len(local) - p["hi"] + token + 1)
                        self.assertEqual(plan["cu"], expected_cu)
                        self.assertEqual(len(local), tokens // world)
                        self.assertEqual(plan["cu"][-1], len(local))
                        self.assertEqual(len(plan["indices"]) // 2,
                                         sum((b - a + 63) // 64 for a, b in zip(expected_cu, expected_cu[1:])))
                        all_tokens.extend(local)
                    self.assertEqual(sorted(all_tokens), list(range(tokens)))

    def test_zigzag_exchange(self):
        generator = torch.Generator().manual_seed(291)
        for world in (1, 2, 3, 4, 6, 8):
            count, length = 2 * world, 65
            for cu in ([0, count * length], [0, 0, 17, count * length // 2 + 1, count * length]):
                plans = [example.build_partition(cu, count * length, r, world, "zigzag") for r in range(world)]
                by_slot = {p["slot"]: p for plan in plans for p in plan["parts"]}
                originals = [torch.cat((torch.randn(1, 128, 128, generator=generator) * 0.01,
                                        torch.eye(128)[None] * 0.7 + torch.randn(1, 128, 128, generator=generator) * 0.01), -1)
                             for _ in range(count)]
                owners = list(range(world)) + list(range(world - 1, -1, -1))
                for forward in (True, False):
                    order = list(range(count)) if forward else list(range(count - 1, -1, -1))
                    maps = [originals[s].clone() for s in order]
                    for i, slot in enumerate(order):
                        p = by_slot[slot]
                        if p["first"] != p["last"] or p["prefix" if forward else "suffix"] == 0:
                            maps[i][..., 128:] = 0
                    snapshots = []
                    for step in range((count - 1).bit_length()):
                        snapshots.append([x.clone() for x in maps])
                        distance = 2 ** step
                        updated = [x.clone() for x in maps]
                        for i in range(distance, count):
                            updated[i] = torch.cat((maps[i][..., 128:] @ maps[i-distance][..., :128] + maps[i][..., :128],
                                                    maps[i][..., 128:] @ maps[i-distance][..., 128:]), -1)
                        maps = updated
                    for plan in plans:
                        rank = plan["rank"]
                        local = torch.stack([originals[p["slot"]] for p in plan["parts"]])
                        step = [0]

                        def transport(operations):
                            # Enumerate every global edge independently; local-only rounds
                            # do not call the transport and must be skipped here.
                            while True:
                                distance = 2 ** step[0]
                                edges = [(i, i + distance) for i in range(count - distance)
                                         if owners[order[i]] != owners[order[i + distance]]
                                         and rank in (owners[order[i]], owners[order[i + distance]])]
                                if edges:
                                    break
                                step[0] += 1
                            self.assertEqual(len(operations), len(edges))
                            for op, (source, target) in zip(operations, edges):
                                if owners[order[source]] == rank:
                                    self.assertEqual(op.op, example.dist.isend)
                                    self.assertEqual(op.peer, owners[order[target]])
                                    torch.testing.assert_close(op.tensor, snapshots[step[0]][source])
                                else:
                                    self.assertEqual(op.op, example.dist.irecv)
                                    self.assertEqual(op.peer, owners[order[source]])
                                    op.tensor.copy_(snapshots[step[0]][source])
                            step[0] += 1
                            return [types.SimpleNamespace(wait=lambda: None) for _ in operations]

                        def gather(outputs, source):
                            for r, output in enumerate(outputs):
                                output.copy_(torch.stack((originals[r], originals[count - 1 - r])))

                        def merge(state, chain, n, slot, *, forward, state_v_first):
                            indices = range(slot - n, slot) if forward else range(slot + n, slot, -1)
                            for s in indices:
                                state.copy_(chain[s, ..., 128:] @ state + chain[s, ..., :128])

                        fake = types.ModuleType("fla_npu.ops.ascendc")
                        fake.merge_fwd_bwd_kernel = merge
                        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}), \
                                patch.object(example.dist, "all_gather", side_effect=gather), \
                                patch.object(example.dist, "batch_isend_irecv", side_effect=transport), \
                                patch.object(example.dist, "P2POp", side_effect=lambda op, tensor, peer:
                                             types.SimpleNamespace(op=op, tensor=tensor, peer=peer)):
                            ag = example.exchange_boundary(local, plan, forward=forward, communication="allgather")
                            ptp = example.exchange_boundary(local, plan, forward=forward, communication="ptp")
                        torch.testing.assert_close(ptp, ag, atol=1e-4 if not forward else 1e-6, rtol=0.01 if not forward else 1e-5)


if __name__ == "__main__":
    unittest.main()
