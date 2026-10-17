"""ATK triple execution: A2 native kernel, CPU FP32 benchmark, CPU FP64 golden."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from gen_chunk_kda_bwd_finalize import PROFILES
from reference import build_inputs, reference

torch.set_num_threads(1)


def run_npu(data, spec, cu, chunks, device):
    import torch_npu  # noqa: F401
    torch.ops.load_library(os.environ['KDA_FINALIZE_LIBRARY'])
    dev = {name: value.to(device) for name, value in data.items()}
    shape = list(data['q'].shape)
    if spec['output_layout'] in ('BSND', 'TND'):
        shape[-3], shape[-2] = shape[-2], shape[-3]
    gate_shape = list(data['q'].shape)
    gate_shape[-3], gate_shape[-2] = gate_shape[-2], gate_shape[-3]
    dq, dk, dv = [torch.full(shape, float('nan'), device=device, dtype=torch.bfloat16) for _ in range(3)]
    db = torch.full(gate_shape[:-1], float('nan'), device=device)
    dg = torch.full(gate_shape, float('nan'), device=device)
    da = torch.full((spec['H'],), float('nan'), device=device)
    bias = torch.full((spec['H'], 128), float('nan'), device=device)
    partial = torch.full((spec['H'], len(chunks), 136), float('nan'), device=device)
    torch.ops.ascend_ops.kda_finalize_stage01(
        *(dev[n] for n in ('k', 'v', 'gk', 'akk', 'v_new', 'h', 'dh', 'dv_scan')),
        cu, spec['cores'], 12, dev['beta'], dv, dev['q'], dev['dq_raw'], spec['scale'],
        dev['dAqk'], dq, dev.get('q_rstd'), raw_g=dev['raw_g'], a_log=dev['a_log'],
        dt_bias=dev['dt_bias'], k_rstd=dev.get('k_rstd'), dk_out=dk, dg_out=dg, db_out=db,
        partial_out=partial, da_log_out=da, ddt_bias_out=bias, lower_bound=spec['lower_bound'],
        beta_mode=spec['beta_mode'], output_layout=spec['output_layout'])
    torch.npu.synchronize()
    return tuple(x.cpu() for x in (dq, dk, dv, db, dg, da, bias))


try:
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.base_api import BaseApi
except ModuleNotFoundError as exc:
    if exc.name != 'atk':
        raise
else:
    @register('executor_chunk_kda_bwd_finalize')
    class FunctionApi(BaseApi):
        def __call__(self, input_data, with_output=False):
            index = int(input_data.kwargs['case_id'])
            spec = PROFILES[index]
            seed = 20261013 + index + 10000 * int(os.getenv('KDA_SEED_ROUND', '0'))
            data, cu, chunks = build_inputs(spec, seed)
            print(f'FINALIZE_ROLE case={index} seed={seed} backend={self.device} '
                  f'golden={bool(self.task_result.is_benchmark_task)} layout={spec["output_layout"]}', flush=True)
            if self.device in ('npu', 'pyaclnn'):
                output = run_npu(data, spec, cu, chunks, input_data.kwargs['marker'].device)
            else:
                output = reference(data, spec, chunks, bool(self.task_result.is_benchmark_task))
            for tensor in output:
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f'case {index}: nonfinite output')
            return output
