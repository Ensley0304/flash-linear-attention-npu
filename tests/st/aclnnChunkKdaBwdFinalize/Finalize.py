"""ATK CPU dual executor; every DUT invocation uses the real finalize aclnn ABI."""
import ctypes
import hashlib
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
from generate_finalize import PROFILES
import kernel_ac_torch_ref as oracle
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

torch.set_num_threads(max(1, int(os.environ.get('KDA_FINALIZE_CPU_THREADS', '1'))))
ORACLE_SHA256 = 'fb8c6f9fc1b19031c9487e6394c2792b1f1d4e5b5cfed38d8f51772284d820c4'
assert hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest() == ORACLE_SHA256
INPUTS = ('q', 'k', 'v', 'gk', 'raw_g', 'beta', 'a_log', 'dt_bias', 'akk',
          'v_new', 'h', 'dh', 'dv_scan', 'd_aqk', 'dq_raw', 'q_rstd', 'k_rstd')
OUTPUTS = ('dq', 'dk', 'dv', 'd_beta', 'd_g', 'd_a_log', 'd_dt_bias')


def build_inputs(s):
    b, h, t, d = s['B'], s['H'], s['T'], 128
    lengths = s['lengths']
    nt = (t+63)//64 if lengths is None else sum((n+63)//64 for n in lengths)
    g = torch.Generator().manual_seed(s['seed'])
    def rand(shape, dtype=torch.float32, magnitude=None):
        return (torch.randn(shape, generator=g) * (s['magnitude'] if magnitude is None else magnitude)).to(dtype)
    vector = (b, h, t, d)
    x = {n: rand(vector, torch.bfloat16) for n in ('q', 'k', 'v', 'v_new', 'dv_scan')}
    x.update({n: rand(vector) for n in ('gk', 'dq_raw')})
    x['raw_g'] = rand(vector, magnitude=1.)
    x['beta'] = torch.sigmoid(rand((b, h, t), magnitude=1.)).bfloat16()
    x['a_log'] = rand((h,), torch.bfloat16 if s['alog_bf16'] else torch.float32, .5)
    x['dt_bias'] = rand((h, d), magnitude=.25)
    x['akk'] = rand((b, h, t, 64), torch.bfloat16)
    for n in ('h', 'dh'):
        x[n] = rand((b, h, nt, d, d), torch.bfloat16)
    x['d_aqk'] = rand((b, h, t, 64))
    local_rows = torch.cat([torch.arange(n) for n in lengths]) if lengths else torch.arange(t)
    x['d_aqk'] *= (local_rows[:, None] % 64 >= torch.arange(64)[None, :])
    if s['rstd']:
        for n in ('q', 'k'):
            raw = x[n].float()
            inv = raw.square().sum(-1).add(1e-6).rsqrt()
            x[n] = (raw*inv[..., None]).bfloat16()
            x[n+'_rstd'] = inv
    else:
        x.update(q_rstd=None, k_rstd=None)
    return x


def chunks(s):
    if s['lengths'] is None:
        for b in range(s['B']):
            for si, start in enumerate(range(0, s['T'], 64)):
                yield b, start, min(start+64, s['T']), si
    else:
        offset = si = 0
        for length in s['lengths']:
            for start in range(offset, offset+length, 64):
                yield 0, start, min(start+64, offset+length), si
                si += 1
            offset += length


def cpu_reference(x, s, high):
    dtype = torch.float64 if high else torch.float32
    b, h, t = s['B'], s['H'], s['T']
    out = [torch.empty((b, h, t, 128), dtype=dtype) for _ in range(3)]
    out += [torch.empty((b, h, t), dtype=dtype), torch.empty((b, h, t, 128), dtype=dtype),
            torch.zeros(h, dtype=dtype), torch.zeros((h, 128), dtype=dtype)]
    for bi, start, end, si in chunks(s):
        for hi in range(h):
            def token(n):
                return x[n][bi, hi, start:end][None, :, None, ...]
            def state(n):
                return x[n][bi, hi, si][None, None, None, ...]
            q, k, v, vn, gk, beta = (token(n) for n in ('q', 'k', 'v', 'v_new', 'gk', 'beta'))
            base = oracle.kernel_c_base_torch(q, k, v, vn, gk, beta, token('akk'), state('h'),
                torch.zeros_like(v), state('dh'), token('dv_scan'), token('dq_raw'),
                s['scale'], 64, None, False, acc_dtype=dtype,
                preserve_output_dtypes=False, emulate_triton_casts=not high)
            intra = oracle.kernel_c_intra_torch(q, k, gk, beta, token('d_aqk'), base['dAkk'],
                base, 64, None, None if high else torch.bfloat16, acc_dtype=dtype)
            values = [intra['dq_hv'], intra['dk_hv'], base['dv'], intra['db']]
            for idx, n in ((0, 'q'), (1, 'k')):
                if s['rstd']:
                    y = token(n).to(dtype)
                    values[idx] = token(n+'_rstd').to(dtype)[..., None] * (
                        values[idx] - y*(values[idx]*y).sum(-1, keepdim=True))
            gate = oracle.gate_backward_torch(token('raw_g'), x['a_log'][hi:hi+1],
                x['dt_bias'][hi:hi+1], oracle.reverse_chunk_cumsum(intra['dg_hv'], 64, None),
                s['lower_bound'], acc_dtype=dtype)
            values.append(gate['dg'])
            for idx, value in enumerate(values):
                out[idx][bi, hi, start:end] = value[0, :, 0]
            out[5][hi] += gate['dA_log'][0]
            out[6][hi] += gate['ddt_bias']
    if not high:
        out = [value.to(torch.bfloat16 if i < 4 else torch.float32) for i, value in enumerate(out)]
    if s['lengths'] is not None:
        out[:5] = [value.squeeze(0) for value in out[:5]]
    return tuple(out)


def npu_call(x, s, device):
    from fla_npu.ops.ascendc._runtime import call_aclnn, ACL_FORMAT_ND
    values = {n: None if x[n] is None else x[n].to(device) for n in INPUTS}
    if s['lengths'] is not None:
        for n in INPUTS:
            if n not in ('a_log', 'dt_bias') and values[n] is not None:
                values[n] = values[n].squeeze(0)
    output = tuple(torch.empty_like(values[n], dtype=torch.bfloat16 if i < 4 else torch.float32)
                   for i, n in enumerate(('q', 'k', 'v', 'beta', 'gk', 'a_log', 'dt_bias')))
    cu = indices = None
    if s['lengths'] is not None:
        cu, indices = [0], []
        for seq, length in enumerate(s['lengths']):
            cu.append(cu[-1]+length)
            for ci in range((length+63)//64):
                indices.extend((seq, ci))
    p = ctypes.c_void_p
    # The order is the public aclnnChunkKdaBwdFinalizeGetWorkspaceSize header.
    types = [p for _ in INPUTS]  # q through kRstdOptional, 17 tensor pointers
    types += [p, p]  # cuSeqlensOptional, chunkIndicesOptional (aclIntArray)
    types += [ctypes.c_double, ctypes.c_double, ctypes.c_int64]  # scale, lowerBound, chunkSize
    types += [ctypes.c_bool]*4  # safeGate, useGateInKernel, useExp2, stateVFirst
    types += [p for _ in OUTPUTS]  # dqOut through dDtBiasOut
    types += [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(p)]  # workspaceSize, executor
    def build(ctx):
        def tensor(value, name):
            return p() if value is None else ctx.tensor(value, name,
                acl_format_override=ACL_FORMAT_ND, storage_shape_override=tuple(value.shape))
        return [*(tensor(values[n], n) for n in INPUTS), ctx.int_array(cu), ctx.int_array(indices),
                ctypes.c_double(s['scale']), ctypes.c_double(s['lower_bound']), ctypes.c_int64(64),
                ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(False),
                *(tensor(value, name) for name, value in zip(OUTPUTS, output))]
    call_aclnn('aclnnChunkKdaBwdFinalize', build, output, get_workspace_argtypes=types)
    torch.npu.synchronize()
    return output


@register('executor_finalize')
class FinalizeApi(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.high = self.device == 'cpu' and bool(task_result.is_benchmark_task)

    def __call__(self, input_data, with_output=False):
        case_id = int(input_data.kwargs['case_id'])
        assert 0 <= case_id < len(PROFILES)
        s = PROFILES[case_id]
        x = build_inputs(s)
        if self.device == 'cpu':
            outputs = cpu_reference(x, s, self.high)
        elif self.device in ('npu', 'pyaclnn'):
            marker = input_data.kwargs['low_precision_marker']
            device = marker.device if isinstance(marker, torch.Tensor) and marker.device.type == 'npu' else torch.device('npu:0')
            outputs = npu_call(x, s, device)
        else:
            raise RuntimeError(f'Unsupported backend: {self.device}')
        if not all(torch.isfinite(value).all().item() for value in outputs):
            raise AssertionError(f'Non-finite output in case {case_id}')
        return outputs


@register('executor_finalize_aclnn')
class FinalizeAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        from fla_npu.ops.ascendc._runtime import _CallContext, _AclnnRuntime
        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
        s = PROFILES[int(input_data.kwargs['case_id'])]
        x = build_inputs(s)
        device = torch.device('npu:0')
        self.values = {n: None if x[n] is None else x[n].to(device) for n in INPUTS}
        if s['lengths'] is not None:
            for n in INPUTS:
                if n not in ('a_log', 'dt_bias') and self.values[n] is not None:
                    self.values[n] = self.values[n].squeeze(0)
        self.outputs = tuple(torch.empty_like(self.values[n],
            dtype=torch.bfloat16 if i < 4 else torch.float32)
            for i, n in enumerate(('q', 'k', 'v', 'beta', 'gk', 'a_log', 'dt_bias')))
        self.context = _CallContext(_AclnnRuntime(), device)
        cu = indices = None
        if s['lengths'] is not None:
            cu, indices = [0], []
            for seq, length in enumerate(s['lengths']):
                cu.append(cu[-1]+length)
                for ci in range((length+63)//64):
                    indices.extend((seq, ci))
        args = [ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(AclTensor))
                if self.values[n] is None else self.torch_tensor_to_acl(self.values[n]) for n in INPUTS]
        args += [self.context.int_array(cu), self.context.int_array(indices),
                 ctypes.c_double(s['scale']), ctypes.c_double(s['lower_bound']), ctypes.c_int64(64),
                 ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(False)]
        packages = [self.torch_tensor_to_acl(y) for y in self.outputs]
        self.input_args = args + packages
        return self.input_args, packages

    def after_call(self, output_packages):
        torch.npu.synchronize()
        return tuple(y.cpu() for y in self.outputs)
