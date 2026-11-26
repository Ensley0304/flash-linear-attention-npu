"""Build-only copies of f2ed09dc stages for an isolated single-owner timing gate.

Does not modify production headers. Generated classes share one pipe/arena;
full-sequence caches intentionally bound the prototype scope (not final workspace).
"""
from pathlib import Path
import re

here=Path(__file__).parent
repo=next(p for p in here.parents if (p/'AGENTS.md').is_file())
(here/'softplus_math.h').write_bytes((repo/'fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/arch22/softplus_math.h').read_bytes())
paths={
 'prepare':'fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_general.h',
 'state':'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch22/chunk_fwd_h_general.h',
 'bp':'fla/ops/ascendc/kda/chunk_kda_bwd_prepare/op_kernel/arch22/chunk_kda_bwd_prepare_general.h',
 'dhu':'fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/arch22/chunk_gated_delta_rule_bwd_dhu_general.h',
 'bf':'fla/ops/ascendc/kda/chunk_kda_bwd_finalize/op_kernel/arch22/chunk_kda_bwd_finalize_general.h',
}
maps={
 'prepare':{'matrix_':0,'vectors_':2048},
 'state':{'stateBuf_':0,'kgBuf_':8192,'valueBuf_':16384,'rowBuf_':17408},
 'bp':{'buffer_':0},
 'dhu':{'states_':0,'rows_':24576},
 'bf':{'matrices_':0,'vectors_':4096},
}
for name,path in paths.items():
 s=(repo/path).read_text(encoding='utf-8')
 s=s.replace('../../../chunk_kda_fwd_prepare/op_kernel/arch22/softplus_math.h','softplus_math.h')
 # uint64_t names are untouched: only literal chunk extent 64 changes.
 s=re.sub(r'\b64\b','32',s)
 s=re.sub(r'\b64U\b','32U',s)
 s=re.sub(r'\b63U\b','31U',s)
 s=s.replace('tokens+63','tokens+31')
 s=re.sub(r'(TQue<[^;]+>) (input_|output_);',r'\1 *\2;',s)
 s=s.replace('input_.','input_->').replace('output_.','output_->')
 signature=re.search(r'void Init\(AscendC::TPipe\s*&pipe\)\s*\{',s)
 assert signature,name
 s=s[:signature.start()]+'''void Init(AscendC::TPipe &pipe, F arena,
        AscendC::TQue<AscendC::QuePosition::VECIN,1> &in,
        AscendC::TQue<AscendC::QuePosition::VECOUT,1> &out) {
        input_=&in; output_=&out;
'''+s[signature.end():]
 s=re.sub(r'pipe.InitBuffer\([^;]+;', '', s)
 for buf,offset in maps[name].items():s=s.replace(buf+'.Get<float>()',f'arena[{offset}]')
 if name=='prepare':
  s=s.replace('inverse_=lower_[4096]','inverse_=lower_[1024]')
  s=s.replace('lower_, 0.0F, 4096','lower_, 0.0F, 1024').replace('inverse_, 0.0F, 4096','inverse_, 0.0F, 1024')
 if name=='bf':
  s=s.replace('dA_=inverse_[4096];middle_=inverse_[8192];dL_=inverse_[12288]', 'dA_=inverse_[1024];middle_=inverse_[2048];dL_=inverse_[3072]')
  for field in ['inverse_','dA_','middle_','dL_']:s=s.replace(field+',0.0F,4096',field+',0.0F,1024')
 # Explicit completion for all private GM write/read transitions.
 s=s.replace('gm.SetGlobalBuffer(', 'gm.SetGlobalBuffer(')
 s=re.sub(r'(gm.SetGlobalBuffer\([^;]+;)',r'\1\n        gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);',s)
 # Every read waits for preceding MTE3; there is no other producer owner.
 read=re.search(r'void Read\([^\{]+\{',s)
 assert read,name
 s=s[:read.end()]+'''
        auto fenceId=static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(fenceId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(fenceId);
'''+s[read.end():]
 (here/f'probe_{name}.h').write_text(s,encoding='utf-8')
 print(name,'generated')
