"""Unchanged upstream training tests via installed public forward/backward ACLNN."""
import pytest
import torch
import torch_npu
from fla_npu.ops.ascendc import npu_chunk_kda_fwd,npu_chunk_kda_bwd

class PublicChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta,a,bias,h0,options):
        B,T,H,K=q.shape
        chunk_size=options.get('chunk_size',64)
        general=(chunk_size==32 or q.dtype==torch.float16 or K!=128 or v.shape[-1]!=128 or h0 is not None or options.get('state_v_first',False) or (options.get('use_gate_in_kernel',False) and (not options.get('safe_gate',False) or a is None)))
        cache_dtype=torch.float32 if general else q.dtype
        norm=options.get('use_qk_l2norm_in_kernel',False);sigmoid=options.get('use_beta_sigmoid_in_kernel',False)
        def empty(shape,dtype):return torch.empty(shape,device=q.device,dtype=dtype)
        qh=empty((B,H,T,K),cache_dtype) if norm else None
        kh=empty((B,H,T,K),cache_dtype) if norm else None
        qr=empty((B,H,T),torch.float32) if norm else None
        kr=empty((B,H,T),torch.float32) if norm else None
        be=empty((B,H,T),torch.float32) if sigmoid else None
        opts=dict(options)
        cu=opts.pop('cu_seqlens_cpu',None)
        if cu is None:cu=opts.get('cu_seqlens')
        cu=None if cu is None else tuple(int(x) for x in cu)
        indices=None if cu is None else tuple(x for si,(s,e) in enumerate(zip(cu,cu[1:])) for c in range((e-s+chunk_size-1)//chunk_size) for x in (si,c))
        opts['cu_seqlens']=cu;opts['chunk_indices']=indices
        scale=opts.pop('scale',None);scale=K**-.5 if scale is None else scale
        opts.setdefault('disable_recompute',False)
        # Upstream also spells per-head A_log as [1,1,H,1]; ACLNN uses [H].
        result=npu_chunk_kda_fwd(q,k,v,g,beta,scale,A_log=None if a is None else a.reshape(-1),dt_bias=bias,initial_state=h0,q_hat_out=qh,k_hat_out=kh,q_rstd_out=qr,k_rstd_out=kr,beta_eff_out=be,**opts)
        ctx.save_for_backward(q,k,v,g,beta,a,bias,h0,qh,kh,qr,kr,be,*result[2:11]);ctx.options=opts;ctx.scale=scale
        ctx.set_materialize_grads(False)
        torch.npu.synchronize()
        intermediate=result[10] if opts.get('return_intermediate_states',False) else None
        if intermediate is not None:ctx.mark_non_differentiable(intermediate)
        return result[0],result[1],intermediate
    @staticmethod
    def backward(ctx,do,dht,_intermediate_grad):
        q,k,v,g,beta,a,bias,h0,qh,kh,qr,kr,be,gk,aqk,akk,w,u,qg,kg,vn,h=ctx.saved_tensors
        opts=ctx.options
        if do is None:do=torch.zeros_like(v)
        result=npu_chunk_kda_bwd(q if qh is None else qh,k if kh is None else kh,v,beta,gk,aqk,akk,w,qg,kg,vn,h,do.contiguous(),ctx.scale,
            initial_state=h0,dht=None if dht is None else dht.contiguous(),chunk_size=opts.get('chunk_size',64),state_v_first=opts.get('state_v_first',False),
            raw_g=g,A_log=None if a is None else a.reshape(-1),dt_bias=bias,cu_seqlens=opts['cu_seqlens'],chunk_indices=opts['chunk_indices'],layout='BSND',
            safe_gate=opts.get('safe_gate',False),lower_bound=opts.get('lower_bound') or -5.,use_gate_in_kernel=opts.get('use_gate_in_kernel',False),
            disable_recompute=opts['disable_recompute'],use_exp2=opts.get('use_exp2',True),q_rstd=qr,k_rstd=kr,beta_eff=be,
            use_beta_sigmoid_in_kernel=opts.get('use_beta_sigmoid_in_kernel',False),allow_neg_eigval=opts.get('allow_neg_eigval',False))
        torch.npu.synchronize()
        dq,dk,dv,db,dg,dh0,da,dbias=result
        if da is not None:da=da.reshape_as(a)
        if dbias is not None:dbias=dbias.reshape_as(bias)
        return dq,dk,dv,dg,db,da,dbias,dh0,None

def public_chunk(q,k,v,g,beta,A_log=None,dt_bias=None,**kwargs):
    initial_state=kwargs.pop('initial_state',None)
    result=PublicChunk.apply(q,k,v,g,beta,A_log,dt_bias,initial_state,kwargs)
    return result if kwargs.get('return_intermediate_states',False) else result[:2]
@pytest.fixture(autouse=True)
def use_public_backward(request,monkeypatch):
    if request.node.originalname in ('test_chunk','test_chunk_varlen','test_chunk_causality','test_chunk_lower_bound','test_chunk_state_v_first','test_chunk_use_beta_sigmoid_in_kernel','test_chunk_varlen_prefill','test_chunk_collinear_keys_l2norm','test_chunk_varlen_fused_training_contract','test_chunk_return_intermediate_states'):
        monkeypatch.setattr(request.module,'chunk_kda',public_chunk)
    yield
