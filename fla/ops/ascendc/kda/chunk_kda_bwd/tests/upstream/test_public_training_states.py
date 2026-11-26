"""State/optional-parameter extensions against fla-org's recurrent autograd.

These are additional parameters, not original upstream pytest nodes. No ATK.
"""
import pytest
import torch
import torch.nn.functional as F
from public_backward_plugin import public_chunk
from test_public_backward_expanded import upstream


@pytest.mark.parametrize('chunk_size', [32, 64])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('save', [False, True])
@pytest.mark.parametrize('v_first', [False, True])
@pytest.mark.parametrize('gate_mode', [0, 1, 2])
def test_training_states(chunk_size, dtype, save, v_first, gate_mode):
    torch.manual_seed(123)
    H, T, K, V = 3, 65, 64, 32
    cu = [0, 0, 1, 34, 34, 65]
    device = upstream.device
    norm = bool((gate_mode + int(save)) % 2)
    beta_mode = (gate_mode + int(v_first)) % 3
    exp2 = dtype == torch.bfloat16
    q = F.normalize(torch.randn(1,T,H,K), dim=-1).to(device,dtype).requires_grad_()
    k = F.normalize(torch.randn(1,T,H,K), dim=-1).to(device,dtype).requires_grad_()
    v = (torch.randn(1,T,H,V)*.3).to(device,dtype).requires_grad_()
    g = ((torch.randn(1,T,H,K)*.4) if gate_mode else -torch.rand(1,T,H,K)*.2).to(device,torch.float32).requires_grad_()
    beta = (torch.randn(1,T,H)*.3 if beta_mode else torch.rand(1,T,H)*.5).to(device,dtype).requires_grad_()
    hs = (len(cu)-1,H,V,K) if v_first else (len(cu)-1,H,K,V)
    h0 = (torch.randn(hs)*.2).to(device).requires_grad_()
    a = (torch.randn(H)*.1).to(device).requires_grad_() if gate_mode and save else None
    bias = (torch.randn(H*K)*.2).to(device).requires_grad_() if gate_mode and v_first else None
    inputs = [q,k,v,g,beta,h0] + ([] if a is None else [a]) + ([] if bias is None else [bias])
    original_h0 = h0.detach().clone()
    o, ht = public_chunk(q,k,v,g,beta,A_log=a,dt_bias=bias,initial_state=h0,
        output_final_state=True,state_v_first=v_first,scale=.37,chunk_size=chunk_size,
        cu_seqlens=cu,use_gate_in_kernel=bool(gate_mode),safe_gate=gate_mode==2,
        lower_bound=-2.,use_qk_l2norm_in_kernel=norm,use_beta_sigmoid_in_kernel=bool(beta_mode),
        allow_neg_eigval=beta_mode==2,disable_recompute=save,use_exp2=exp2)
    assert torch.equal(h0.detach(),original_h0)
    do, dht = torch.randn_like(o), torch.randn_like(ht)
    actual = torch.autograd.grad((o,ht),inputs,(do,dht))
    qr = F.normalize(q.float(),dim=-1) if norm else q.float()
    kr = F.normalize(k.float(),dim=-1) if norm else k.float()
    br = beta.float().sigmoid()*(2 if beta_mode==2 else 1) if beta_mode else beta.float()
    x = g if bias is None else g + bias.reshape(H,K)
    factor = 1 if a is None else a.exp().reshape(H,1)
    gr = -2.*(factor*x).sigmoid() if gate_mode==2 else (-factor*F.softplus(x) if gate_mode==1 else g)
    initial = h0.transpose(-1,-2) if v_first else h0
    outputs, states = [], []
    for n,(s,e) in enumerate(zip(cu,cu[1:])):
        if s==e:
            states.append(initial[n:n+1])
            continue
        ro,rh = upstream.naive_recurrent_kda(qr[:,s:e],kr[:,s:e],v[:,s:e].float(),gr[:,s:e],br[:,s:e],
            scale=.37,initial_state=initial[n:n+1],output_final_state=True)
        outputs.append(ro);states.append(rh)
    expected_o, expected_h = torch.cat(outputs,1), torch.cat(states,0)
    if v_first:expected_h=expected_h.transpose(-1,-2)
    expected = torch.autograd.grad((expected_o,expected_h),inputs,(do.float(),dht))
    upstream.assert_close('o',expected_o.to(dtype),o,.008)
    upstream.assert_close('ht',expected_h,ht,.008)
    for n,(reference,result) in enumerate(zip(expected,actual)):
        assert result.dtype==reference.dtype and torch.isfinite(result).all()
        upstream.assert_close(f'gradient_{n}',reference,result,.008)
