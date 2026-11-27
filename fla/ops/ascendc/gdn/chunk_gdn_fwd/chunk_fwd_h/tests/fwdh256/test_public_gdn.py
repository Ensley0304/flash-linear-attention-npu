"""Representative unchanged GDN public-forward contract using upstream reference/assertions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import torch
import torch.nn.functional as F
import torch_npu
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.utils import assert_close
from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.npu.set_device(0);records=[]
def check(label):
    info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',os.environ['ASCEND_RT_VISIBLE_DEVICES'],'-c','0'],text=True)
    (a.output/(label+'.txt')).write_text(info)
    ids=[int(x) for x in re.findall(r'Process id:\s*(\d+)',info)]
    assert 'No process in device' in info or ids
    assert all(pid==os.getpid() for pid in ids),info
for tokens,heads,initial in ((65,2,False),(65,2,True),(256,1,True)):
    torch.manual_seed(42);shape=(1,tokens,heads,128)
    q=F.normalize(torch.rand(shape),dim=-1).bfloat16().npu()
    k=F.normalize(torch.rand(shape),dim=-1).bfloat16().npu()
    v=torch.rand(shape).bfloat16().npu()
    beta=torch.rand(shape[:-1]).sigmoid().npu()
    g=F.logsigmoid(torch.rand(shape[:-1])).npu()
    h0=torch.randn(1,heads,128,128,device='npu')*.1 if initial else None
    opts=dict(scale=.37,initial_state=h0,output_final_state=True)
    ref,ht=naive_recurrent_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,**opts)
    label=f't{tokens}-h{heads}-i{int(initial)}';check(label+'-before')
    print('GDN_ROUTE_BEGIN',label,flush=True)
    actual=npu_chunk_gated_delta_rule_fwd(q,k,v,g,beta,chunk_size=64,layout='BSND',**opts)
    torch.npu.synchronize()
    assert_close('o',ref,actual[0],.005);assert_close('ht',ht,actual[1],.005)
    check(label+'-after');records.append(dict(case=label,status='passed',scope='public_forward_only'))
    (a.output/'results.json').write_text(json.dumps(records,indent=2))
    print('GDN_ROUTE_PASS',label,flush=True)
