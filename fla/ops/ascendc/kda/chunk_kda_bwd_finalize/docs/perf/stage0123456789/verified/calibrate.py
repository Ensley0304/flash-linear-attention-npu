import json
import torch
torch.set_num_threads(4)
torch.manual_seed(593)
q=(torch.randn(64,128)*.125).bfloat16().double()
k=(torch.randn(64,128)*.125).bfloat16().double()
g=-torch.rand(64,128).double().cumsum(0)*.2
a=torch.randn(64,64).double()*.125
base=torch.randn(64,128).double()*.02
diag=a.diagonal()[:,None]*k
local=torch.stack([(a[t,:t,None]*k[:t]*torch.exp2(g[t]-g[:t])).sum(0) for t in range(64)])
expected=base+diag+local
budget=2e-5+8e-3*expected.abs()
wrong={'zero':torch.zeros_like(expected),'omit_diagonal':base+local,'omit_local':base+diag,
       'wrong_local_scale':base+diag+local*2}
result={name:int(((value-expected).abs()>budget).sum()) for name,value in wrong.items()}
qn=torch.nn.functional.normalize(q,dim=-1)
norm=(expected-qn*(expected*qn).sum(-1,keepdim=True))*1.25
result['omit_q_norm']=int(((expected-norm).abs()>2e-5+8e-3*norm.abs()).sum())
assert all(n>0 for n in result.values())
assert ((expected.float().bfloat16().double()-expected).abs()<=budget).all()
print(json.dumps(dict(correct_bf16_pass=True,rejected_elements=result),indent=2))
