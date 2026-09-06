"""200-case CPU dual matrix, modeled on the prepare ATK delivery."""
import argparse
import json
from pathlib import Path

COUNT = 200
TOKENS = (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 255, 257, 511, 1024)
HEADS = (1, 2, 3, 4, 5, 7, 8, 9, 16)
SEQUENCES = ((1,), (0, 1), (1, 0), (63, 1), (64, 1), (65, 63),
             (128, 1, 63), (64, 65, 127), (127, 128, 129), (1, 255),
             (129, 64, 63), (192, 1), (31, 33, 65), (128, 128),
             (7, 57, 64), (255, 65), (1, 1, 1, 1), (65, 128, 1),
             (0, 64, 1), (64, 0, 65), (1,) * 32, tuple(range(65)))
STANDARD = {'cv_fused_double_benchmark': {'max_re_ratio': 5, 'avg_re_ratio': 1.5,
                                       'root_mean_squared_ratio': 1.5}}


def profiles():
    result = []
    for i in range(COUNT):
        variable = 98 <= i < 198
        lengths = list(SEQUENCES[(i-98) % len(SEQUENCES)]) if variable else None
        result.append(dict(case_id=i, seed=20260906+i, B=1 if variable else (2 if i % 7 == 0 else 1),
                           H=HEADS[i % len(HEADS)], T=sum(lengths) if variable else TOKENS[i % len(TOKENS)],
                           lengths=lengths, rstd=bool(i % 2), alog_bf16=bool((i // 2) % 2),
                           scale=(0., .125, -.125, 128**-.5, 1.)[i % 5],
                           lower_bound=(-5., -1., -10.)[i % 3], magnitude=(.05, .1, .2)[i % 3]))
    for i, t in ((198, 8192), (199, 16384)):
        result[i].update(B=1, H=96, T=t, lengths=None, scale=128**-.5, lower_bound=-5., magnitude=.05)
    return result


PROFILES = profiles()


def records():
    return [dict(id=s['case_id'], default_seed=s['seed'], name=f"finalize_{s['case_id']:03d}",
                 aclnn_name='ChunkKdaBwdFinalize', version='v2.1', api='pytorch',
                 api_type='executor_finalize', aclnn_api_type='executor_finalize_aclnn',
                 expected_error_msg=None, backward=False, standard={'acc': STANDARD, 'perf': 'not_key'},
                 outputs=None, inputs=[
                     dict(name='low_precision_marker', type='tensor', required=True, dtype='bf16',
                          shape=[1], range_values=[0, 0], backward=False),
                     dict(name='fp32_marker', type='tensor', required=True, dtype='fp32',
                          shape=[1], range_values=[0, 0], backward=False),
                     dict(name='case_id', type='attr', required=True, dtype='int', shape=None,
                          range_values=s['case_id'], backward=False)]) for s in PROFILES]


def validate_records(data):
    if data != records():
        raise ValueError('ATK cases differ from the fixed input matrix or precision standard')
    assert len(data) == COUNT
    assert [s['case_id'] for s in PROFILES] == list(range(COUNT))
    assert len({s['seed'] for s in PROFILES}) == COUNT
    assert sum(s['lengths'] is not None for s in PROFILES) == 100
    for s in PROFILES:
        assert s['T'] > 0 and s['H'] > 0 and s['B'] > 0
        if s['lengths'] is not None:
            assert s['B'] == 1 and sum(s['lengths']) == s['T']
            assert all(n >= 0 for n in s['lengths'])
    assert [(s['B'], s['H'], s['T']) for s in PROFILES[-2:]] == [(1, 96, 8192), (1, 96, 16384)]


try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
except ModuleNotFoundError as exc:
    if exc.name != 'atk':
        raise
else:
    @GENERATOR_REGISTRY.register('generator_finalize')
    class Generator(CaseGenerator):
        def after_case_config(self, case_config):
            i = max(int(self.index)-1, 0)
            case_config.id = i
            case_config.default_seed = PROFILES[i]['seed']
            case_config.name = f'finalize_{i:03d}'
            for item in case_config.inputs:
                config = item[0] if isinstance(item, list) else item
                if config.name == 'case_id':
                    config.range_values = i
            return case_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        validate_records(json.loads((args.output_dir / 'atk_finalize.json').read_text()))
        print('200/200 fixed cases, seeds, and precision standards validated')
        raise SystemExit(0)
    data = records()
    validate_records(data)
    for filename, value in [('atk_finalize.json', data), ('smoke_case.json', data[:1]),
                            ('case_profiles.json', PROFILES)]:
        (args.output_dir / filename).write_text(json.dumps(value, indent=2)+'\n')
