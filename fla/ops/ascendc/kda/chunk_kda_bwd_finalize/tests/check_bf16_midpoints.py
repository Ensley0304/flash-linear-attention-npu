"""Generate a causal exact-cast regression, or check its saved NPU outputs."""
import argparse
from pathlib import Path

import torch


def make_inputs():
    vector = (1, 1, 64, 128)
    inputs = {
        name: torch.zeros(vector, dtype=torch.bfloat16)
        for name in ('q', 'k', 'v', 'v_new', 'dv_scan')
    }
    inputs.update({name: torch.zeros(vector) for name in ('gk', 'dq_raw')})
    inputs['beta'] = torch.zeros((1, 1, 64), dtype=torch.bfloat16)
    inputs['akk'] = torch.zeros((1, 1, 64, 64), dtype=torch.bfloat16)
    inputs['d_aqk'] = torch.zeros((1, 1, 64, 64))
    for name in ('h', 'dh'):
        inputs[name] = torch.zeros((1, 1, 1, 128, 128), dtype=torch.bfloat16)
    inputs.update(q_rstd=None, k_rstd=None)
    inputs.update(raw_g=torch.zeros(vector), a_log=torch.zeros(1),
                  dt_bias=torch.zeros(1, 128))
    inputs['q'][0, 0, :, :64] = torch.eye(64, dtype=torch.bfloat16)
    midpoints = (0.096923828125, -0.096923828125,
                 0.097412109375, -0.097412109375)
    for row in range(4, 64):
        for col in range(row + 1):
            inputs['d_aqk'][0, 0, row, col] = midpoints[(row + col) % 4]
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-inputs', type=Path)
    parser.add_argument('--outputs', type=Path)
    args = parser.parse_args()
    if args.write_inputs is None and args.outputs is None:
        parser.error('provide --write-inputs or --outputs')
    inputs = make_inputs()
    if args.write_inputs:
        torch.save(inputs, args.write_inputs)
    if args.outputs:
        outputs = torch.load(args.outputs, map_location='cpu', weights_only=False)
        expected = torch.zeros_like(inputs['q'])
        expected[0, 0, :, :64] = inputs['d_aqk'][0, 0].bfloat16().T
        assert torch.equal(outputs[1], expected), 'DK must match nearest-even exactly'
        for index in (0, 2, 3, 4, 5, 6):
            assert torch.count_nonzero(outputs[index]) == 0
        print('BF16_MIDPOINT_EXACT_PASS')


if __name__ == '__main__':
    main()
