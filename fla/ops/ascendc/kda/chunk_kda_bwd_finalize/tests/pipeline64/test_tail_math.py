import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3308)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
run=lib.run_c64_tail_math
run.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*4+[ctypes.c_float]*4+[ctypes.c_void_p]
run.restype=None
cases=[]
for gate in range(3):
 for beta_mode in range(3):
  for norm in range(2):
   for a in [.5,1.,2.]:
    n=128
    x=torch.linspace(-10,10,n)
    x[:8]=torch.tensor([-80.,-20.,-2.772589,-.001,.001,2.772589,20.,80.])/a
    delta=torch.ones(n)
    beta=torch.sigmoid(torch.randn(n))*(2 if beta_mode==2 else 1)
    beta[0]=0;beta[1]=1
    bg=torch.randn(n);bg[1]=1
    q=torch.nn.functional.normalize(torch.randn(n),dim=0)
    ng=torch.randn(n)
    projection=(q*ng).sum().item();rstd=.7
    src=torch.stack([x,delta,beta,bg,q,ng]).npu()
    out=torch.empty(4,n,device='npu')
    run(src.data_ptr(),out.data_ptr(),n,gate,beta_mode,norm,a,-5.,projection,rstd,torch.npu.current_stream().npu_stream)
    torch.npu.synchronize()
    b=beta.double();expected_b=bg.double()
    if beta_mode:expected_b=expected_b*b*(1-b/(2 if beta_mode==2 else 1))
    expected_n=ng.double() if not norm else rstd*(ng.double()-q.double()*projection)
    xd=x.double()
    if gate==0: dg=delta.double();da=torch.zeros(n,dtype=torch.float64)
    elif gate==2:
     t=torch.exp(-(a*xd).abs());dg=-5*a*t/(1+t).square();da=dg*xd
    else:dg=-a*torch.sigmoid(xd);da=-a*torch.nn.functional.softplus(xd)
    expected=torch.stack([expected_b,expected_n,dg,da])
    actual=out.cpu().double()
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    if gate==2:
     assert actual[2,1]!=0 and actual[2,6]!=0, 'safe gate +/-20 derivative must remain nonzero'
    if beta_mode==0:assert actual[0,1]==1
    if beta_mode==2:assert actual[0,1]==.5
    cases.append(dict(gate=gate,beta=beta_mode,norm=norm,a=a,max_abs=(actual-expected).abs().max().item()))
print(json.dumps(dict(result='PASS',cases=cases)),flush=True)
