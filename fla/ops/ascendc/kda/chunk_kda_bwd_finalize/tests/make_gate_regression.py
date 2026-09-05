"""Generate a multi-batch/chunk gate regression with sigmoid saturation."""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(311)
    b, h, t, d = 2, 9, 128, 128
    shape = (b, h, t, d)

    def random(size, dtype):
        return (torch.randn(size) * .05).to(dtype)

    inputs = {name: random(shape, torch.bfloat16)
              for name in ('q', 'k', 'v', 'v_new', 'dv_scan')}
    inputs.update({name: random(shape, torch.float32) for name in ('gk', 'dq_raw')})
    inputs['beta'] = random((b, h, t), torch.bfloat16)
    inputs['akk'] = random((b, h, t, 64), torch.bfloat16)
    inputs['d_aqk'] = random((b, h, 2, 64, 64), torch.float32).tril().reshape(b, h, t, 64)
    for name in ('h', 'dh'):
        inputs[name] = random((b, h, 2, d, d), torch.bfloat16)
    inputs['raw_g'] = torch.linspace(-8., 8., t * d).reshape(1, 1, t, d).repeat(b, h, 1, 1)
    inputs['a_log'] = torch.linspace(-1., 1., h)
    inputs['dt_bias'] = torch.linspace(-.5, .5, h * d).reshape(h, d)
    inputs.update(q_rstd=None, k_rstd=None, cu_seqlens=None)
    torch.save(inputs, args.output)


if __name__ == '__main__':
    main()
