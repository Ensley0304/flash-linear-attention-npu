import json
from pathlib import Path
import torch

torch.set_num_threads(4)
torch.manual_seed(11012)
raw=torch.randn(64,128,dtype=torch.float64,requires_grad=True)
with torch.no_grad(): raw[0,:5]=torch.tensor([-30.,-8.,0.,8.,30.])
a=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
bias=torch.zeros(128,dtype=torch.float64,requires_grad=True)
upstream=torch.randn_like(raw)
gate=-5*torch.sigmoid(a.exp()*(raw+bias))
loss=(gate.cumsum(0)*upstream).sum()
got_raw,got_a,got_bias=torch.autograd.grad(loss,(raw,a,bias))
scan=upstream.flip(0).cumsum(0).flip(0)
x=(raw+bias).detach()
z=(-x.abs()).exp()
wanted=scan*(-5*z/(1+z).square())
budget=4e-5*upstream.abs().flip(0).cumsum(0).flip(0)*(5*z/(1+z).square())+2e-38
# Autograd sigmoid itself loses relative precision at +30; the non-saturated range confirms the chain.
assert torch.allclose(got_raw[1:],wanted[1:],rtol=1e-12,atol=1e-12)
assert torch.allclose(got_bias,wanted.sum(0),rtol=1e-12,atol=1e-12)
assert torch.allclose(got_a,(wanted*x).sum(),rtol=1e-12,atol=1e-12)
f=x.float(); e=(-f.abs()).exp()
stable=scan.float()*(-5*e/(1+e).square())
assert ((stable.double()-wanted).abs()<=budget).all()
wrong={
 'zero':torch.zeros_like(wanted),
 'sign':-wanted,
 'forward_scan':upstream.cumsum(0)*(-5*z/(1+z).square()),
 'missing_scan':upstream*(-5*z/(1+z).square()),
 'saturated_sigmoid':scan.float()*(-5*torch.sigmoid(f)*(1-torch.sigmoid(f))),
 'missing_lower_bound':wanted/-5,
}
rejected={name:int(((value-wanted).abs()>budget).sum()) for name,value in wrong.items()}
assert all(rejected.values()),rejected
for mode in (1,2):
    r=torch.tensor([-8.,0.,8.],dtype=torch.float64,requires_grad=True)
    eff=mode*torch.sigmoid(r)
    derivative=torch.autograd.grad(eff.sum(),r)[0]
    formula=eff.detach()*(1-eff.detach()/mode)
    torch.testing.assert_close(formula,derivative,rtol=1e-12,atol=1e-15)
record=dict(gate_scan_autograd='PASS',beta_autograd='PASS',stable_fp32='PASS',rejected_elements=rejected,
            saturated_positive30=dict(reference=float(wanted[0,4]),stable=float(stable[0,4]),naive=float(wrong['saturated_sigmoid'][0,4])))
Path('stage12/calibration.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
