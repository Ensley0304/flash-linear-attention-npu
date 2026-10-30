"""CPU checks of ordered PTP exchange, packed resets and recurrence gradients."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "kda_cp", Path(__file__).resolve().parents[1] / "examples/cp/chunk_kda_cp.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
torch.set_num_threads(1)


class KdaCpTests(unittest.TestCase):
    def test_allgather_only_merges_same_sequence_neighbours(self):
        summaries = []
        for r in range(4):
            matrix = torch.eye(128)[None]
            matrix[0, r, (r + 1) % 4] = r + 1
            summaries.append(torch.cat((torch.full_like(matrix, float(r + 1)), matrix), -1))

        def gather(outputs, local):
            for output, summary in zip(outputs, summaries):
                output.copy_(summary)

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
                    plan = dict(rank=rank, world=4, cu=[0, 17, 64],
                                prefix=min(rank, 1), suffix=min(3-rank, 1))
                    actual = example.exchange_boundary(summaries[rank], plan,
                                                       forward=forward, communication="allgather")
                    count = plan["prefix" if forward else "suffix"]
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
                        plan = dict(rank=rank, world=world, cu=[0, 32, 64] if logical == reset_at else [0, 64],
                                    prefix=logical if forward else 0, suffix=logical if not forward else 0)
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
                            actual = example.exchange_boundary(originals[logical], plan, forward=forward, communication="ptp")
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
            for argv in (["--communication", "none"], ["--tokens", "257"]):
                with self.assertRaises(SystemExit):
                    example.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
