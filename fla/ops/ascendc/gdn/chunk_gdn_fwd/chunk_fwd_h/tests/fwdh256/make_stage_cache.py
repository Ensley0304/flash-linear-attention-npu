"""CPU producer caches for paired stage measurement, from frozen public inputs."""
import sys
from pathlib import Path
import torch
from producer_gate import chunked
torch.set_num_threads(4)
data=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
batch,tokens,heads,dim=data['q'].shape;assert batch==1 and dim==256
out={key:[] for key in ('w','u','kg','G')}
for head in range(heads):
    _,_,caches,_=chunked(*[data[key][0,:,head] for key in ('q','k','v','g','beta')],
        data['A_log'][head],data['dt_bias'].reshape(heads,256)[head],torch.zeros(256,256),'P32')
    for dest,src in (('w','w'),('u','u'),('kg','kg'),('G','g')):
        out[dest].append(torch.cat([c[src] for c in caches]))
out={key:torch.stack(value)[None].contiguous() for key,value in out.items()}
assert not Path(sys.argv[2]).exists()
torch.save(out,sys.argv[2])
print('CPU producer cache',list(out['w'].shape),flush=True)
