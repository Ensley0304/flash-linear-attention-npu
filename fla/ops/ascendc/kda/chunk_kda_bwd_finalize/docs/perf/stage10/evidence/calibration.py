import json
from pathlib import Path
import torch
torch.set_num_threads(2)
g=torch.Generator().manual_seed(593)
a=[torch.tril(torch.randn(64,64,generator=g)*.125,-1)*65536 for _ in range(2)]
b=[torch.randn(64,128,generator=g)*.125 for _ in range(2)]
def split(x):
 h=x.bfloat16().double()
 return h,(x.double()-h).float().bfloat16().double()
ahal=[split(x) for x in a]
bhbl=[split(x) for x in b]
terms=[]
mag=torch.zeros(32,128,dtype=torch.float64)
for (ah,al),(bh,bl) in zip(ahal,bhbl):
 ah,al=ah[:,:32].T,al[:,:32].T
 terms.extend((ah@bh,ah@bl,al@bh))
 mag+=ah.abs()@bh.abs()+ah.abs()@bl.abs()+al.abs()@bh.abs()
y=sum(terms)
budget=3e-6*mag+2e-38
wrong={'zero':torch.zeros_like(y),'missing_query':sum(terms[3:]),'missing_beta_key':sum(terms[:3]),'missing_residuals':terms[0]+terms[3],'wrong_sign':sum(terms[:3])-sum(terms[3:]),'wrong_scale':y/65536,'wrong_transpose':sum(x[:32].double()@z.double() for x,z in zip(a,b))}
result={}
for name,x in wrong.items():
 bad=(x-y).abs()>budget
 assert bad.any(),name
 result[name]={'rejected_elements':int(bad.sum()),'elements':y.numel()}
Path('stage10/calibration.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
