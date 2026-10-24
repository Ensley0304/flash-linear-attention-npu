"""Reviewed 200-case A2 Finalize matrix; three independent seed rounds."""
import itertools
import json
from pathlib import Path

OP_NAME = "chunk_kda_bwd_finalize"
PROFILES = []
for variable, norm, beta_mode, token_major in itertools.product((False, True), (False, True), range(3), (False, True)):
    for boundary, tokens in enumerate((1, 16, 31, 32, 33, 63, 64, 129)):
        index = len(PROFILES)
        lengths = [0, tokens, 0, 65, 1] if variable else None
        PROFILES.append(dict(case_id=index, B=1 if variable else (2 if boundary == 7 else 1),
                             H=(1, 2, 3, 5, 7, 16, 4, 3)[boundary],
                             T=sum(lengths) if variable else tokens, lengths=lengths,
                             norm=norm, beta_mode=beta_mode,
                             output_layout=("TND" if token_major else "NTD") if variable else
                                           ("BSND" if token_major else "BNSD"),
                             lower_bound=(-5., -1., -.5, -.01)[boundary % 4],
                             scale=(128**-.5, 0., -.125, 1.)[boundary % 4],
                             gate_mode="saturated" if boundary == 4 else "ordinary",
                             cores=1 if boundary == 7 else 0))
for variable, norm, token_major in itertools.product((False, True), repeat=3):
    PROFILES.append(dict(case_id=len(PROFILES), B=1, H=96, T=8192,
                         lengths=[0, 1, 31, 64, 65, 4096, 3935, 0] if variable else None,
                         norm=norm, beta_mode=2 if token_major else 1,
                         output_layout=("TND" if token_major else "NTD") if variable else
                                       ("BSND" if token_major else "BNSD"),
                         lower_bound=-5., scale=128**-.5, gate_mode="ordinary", cores=0))
assert len(PROFILES) == 200

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
except ModuleNotFoundError as exc:
    if exc.name != 'atk':
        raise
else:
    @GENERATOR_REGISTRY.register('generator_chunk_kda_bwd_finalize')
    class Generator(CaseGenerator):
        def after_case_config(self, case_config):
            index = max(int(self.index) - 1, 0)
            if index >= len(PROFILES):
                raise ValueError('Finalize matrix has exactly 200 cases')
            case_config.id = index
            case_config.default_seed = 20261013 + index
            case_config.name = f'finalize_{index:03d}'
            for item in case_config.inputs:
                config = item[0] if isinstance(item, list) else item
                if config.name == 'case_id':
                    config.range_values = index
            return case_config


def records():
    result = []
    for spec in PROFILES:
        result.append(dict(id=spec['case_id'], default_seed=20261013 + spec['case_id'],
                           name=f"finalize_{spec['case_id']:03d}", api="pytorch", version="v2.1",
                           api_type="executor_chunk_kda_bwd_finalize", backward=False,
                           standard={"acc": {"cv_fused_double_benchmark": {
                               "max_re_ratio": 5, "avg_re_ratio": 1.5, "root_mean_squared_ratio": 1.5}},
                               "perf": "not_key"}, outputs=None,
                           inputs=[dict(name="marker", type="tensor", required=True, dtype="bf16",
                                        shape=[1], range_values=[0, 0], backward=False),
                                   dict(name="case_id", type="attr", required=True, dtype="int",
                                        shape=None, range_values=spec['case_id'], backward=False)]))
    return result


if __name__ == "__main__":
    root = Path(__file__).parent
    (root / 'atk_chunk_kda_bwd_finalize.json').write_text(json.dumps(records(), indent=2) + '\n')
    (root / 'case_matrix.json').write_text(json.dumps(PROFILES, indent=2) + '\n')
