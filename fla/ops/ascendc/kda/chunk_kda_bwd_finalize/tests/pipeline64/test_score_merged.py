"""Safe-band Cube gradients checked against direct exp(Gi-Gj) derivatives."""
import ctypes
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)
torch.manual_seed(3306)
lib=ctypes.CDLL(str(Path(__file__).parent/'build/libkda_c64_stages.so'))
lib.c64_workspace_bytes.argtypes=[ctypes.c_uint32]*2
lib.c64_workspace_bytes.restype=ctypes.c_uint64
lib.c64_offset.argtypes=[ctypes.c_uint32]*4
lib.c64_offset.restype=ctypes.c_uint64
run=lib.run_c64_score_merged
run.argtypes=[ctypes.c_void_p]*11+[ctypes.c_uint32]*4+[ctypes.c_float,ctypes.c_void_p]
run.restype=None
reports=[]
for m,n in [(1,1),(16,1),(16,16),(63,64),(64,64)]:
 for d in [28,32,60,64]:
  print('BEGIN',m,n,d,flush=True)
  q,kq,kk=[torch.randn(*s)*.1 for s in [(m,d),(m,d),(n,d)]]
  gq,gk=torch.randn(m,d)*.2,torch.randn(n,d)*.2
  ds,dl=torch.randn(m,n)*.1,torch.randn(m,n)*.1
  if m==n: ds=torch.tril(ds);dl=torch.tril(dl,-1)
  beta=torch.rand(m);beta[0]=0
  z=dl*beta[:,None]
  qp,kp,km=q*gq.exp(),kq*gq.exp(),kk*(-gk).exp()
  tensors=[t.npu() for t in [ds,z,qp,kp,km]]
  w=torch.full((lib.c64_workspace_bytes(256,256)//4,),-937.25,device='npu')
  qstart=0 if m==n else n
  length=max(n,qstart+m)
  seeds=[torch.randn(length,d)*.01 for _ in range(3)]
  grads=[t.npu() for t in seeds]
  exps=[gq.exp().npu(),(-gk).exp().npu()]
  base=0.6931471805599453 if d==28 else 1.
  run(w.data_ptr(),*[t.data_ptr() for t in tensors],*[t.data_ptr() for t in exps],
      *[t.data_ptr() for t in grads],m,n,d,qstart,base,torch.npu.current_stream().npu_stream)
  torch.npu.synchronize()
  def get(index,rows,cols):
   off=lib.c64_offset(256,256,0,index)//4
   whole=w[off:off+4096].view(64,64).cpu()
   assert torch.all(whole[rows:]==-937.25)
   assert torch.all(whole[:rows,cols:]==-937.25)
   return whole[:rows,:cols].double()
  a,b,c,e,r=get(4,m,d),get(5,n,d),get(0,m,d),get(1,n,d),get(3,m,n)
  factor=(gq.double()[:,None,:]-gk.double()[None,:,:]).exp()
  dq=(ds.double()[:,:,None]*kk.double()[None,:,:]*factor).sum(1)
  dkq=(z.double()[:,:,None]*kk.double()[None,:,:]*factor).sum(1)
  dkk=(ds.double()[:,:,None]*q.double()[:,None,:]*factor+
       z.double()[:,:,None]*kq.double()[:,None,:]*factor).sum(0)
  rkk=(kq.double()[:,None,:]*kk.double()[None,:,:]*factor).sum(2)
  actual=[a*gq.double().exp(),c*gq.double().exp(),(b+e)*(-gk.double()).exp(),r]
  expected=[dq,dkq,dkk,rkk]
  for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-6,rtol=2e-4)
  expected_grads=[x.double().clone() for x in seeds]
  expected_grads[0][qstart:qstart+m]+=dq
  expected_grads[1][qstart:qstart+m]+=dkq
  expected_grads[1][:n]+=dkk
  zg=(ds.double()[:,:,None]*q.double()[:,None,:]+z.double()[:,:,None]*kq.double()[:,None,:])*kk.double()[None,:,:]*factor
  expected_grads[2][qstart:qstart+m]+=base*zg.sum(1)
  expected_grads[2][:n]-=base*zg.sum(0)
  for x,y in zip(grads,expected_grads):torch.testing.assert_close(x.cpu().double(),y,atol=1e-6,rtol=2e-4)
  reports.append(dict(m=m,n=n,d=d,overlap=qstart==0,max_abs=max((x.cpu().double()-y).abs().max().item() for x,y in zip(grads,expected_grads))))
print(json.dumps(dict(result='PASS',cases=reports)),flush=True)
