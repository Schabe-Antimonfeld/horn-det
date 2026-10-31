"""Freeze validation-only SRP pruning choices and report matched runtime/quality."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))
import improve_band as frozen
from horn.recovery import camera_rig, digest, write_json
from ps.configuration import load_config
from ps.dsp import beamform, candidates, features, spatial_map
from ps.geometry import AcousticEvidence, Association, associate, route_cameras
from ps.search import Search

REFERENCE = frozen.reference
OUTPUT = PROTO / 'outputs/srp_optimization_v2_20261031'
VARIANTS = dict(cached91=dict(algorithm='cached'),
                mixed35=dict(algorithm='cached', pair_mode='mixed35'),
                hier35=dict(algorithm='hierarchical', pair_mode='mixed35', bounds=False),
                bounded35=dict(algorithm='hierarchical', pair_mode='mixed35'),
                folded35=dict(algorithm='hierarchical', pair_mode='mixed35', fft_fold=4),
                compact35=dict(algorithm='hierarchical', pair_mode='mixed35', fft_fold=8),
                folded91=dict(algorithm='hierarchical', pair_mode='all', fft_fold=4))
SETTINGS = dict(variants=VARIANTS, regions=6, coarse_azimuth=6, coarse_depression=15,
                maximum_candidate_loss=1, maximum_direction_loss=2, maximum_f1_loss=0.02,
                maximum_capture_loss=0.02, maximum_extra_negative=2, maximum_visual_loss=2,
                relative_time_target=0.1, absolute_steady_target_seconds=[0.325, 0.65],
                direction_threshold=0.95, capture_threshold=0.9, margin=0,
                evidence='reused public sources and synthetic arrays/camera, no new blind road or ARM timing proof')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan():
    frozen.check_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [PROTO / 'improve_detector.py', PROTO / 'improve_band.py', PROTO / 'improve_panns.py',
             frozen.OUTPUT / 'freeze.json', frozen.OUTPUT / 'model-combined/manifest.json',
             frozen.OUTPUT / 'model-combined/weights.npz', REFERENCE.OUTPUT / 'plan.json',
             PROTO.parent / 'deployment/pynq_z2/ps/dsp.py', PROTO.parent / 'deployment/pynq_z2/ps/search.py',
             PROTO.parent / 'deployment/pynq_z2/ps/band.py', PROTO.parent / 'deployment/pynq_z2/ps/geometry.py']
    value = dict(format=1, settings=SETTINGS,
                 frozen={str(path.relative_to(PROTO.parent)): digest(path) for path in paths},
                 selection='validation original/visible both pass predefined quality guards; minimize median paired SRP runtime ratio; no gate/model retraining',
                 limits='phase times are PC measurements, independent windows; all global coarse samples measured, unrefined fine directions never filled with zeros')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('SRP plan is already frozen')
    write_json(path, value)
    print('Frozen SRP optimization plan', flush=True)


def check_plan():
    value = read_json(OUTPUT / 'plan.json')
    if value['settings'] != SETTINGS:
        raise ValueError('SRP experiment settings changed')
    for name, checksum in value['frozen'].items():
        if digest(PROTO.parent / name) != checksum:
            raise ValueError('Frozen input changed: ' + name)
    return value


def legacy_run(audio, config):
    original = np.fft.irfft
    marker = {}
    def traced(*args, **kwargs):
        value = original(*args, **kwargs)
        marker['gcc_done'] = time.perf_counter()
        return value
    began = time.perf_counter()
    with patch.object(np.fft, 'irfft', side_effect=traced):
        grid = spatial_map(audio, config['audio'], config['search'])
    ended = time.perf_counter()
    return grid, candidates(grid, config['search']['candidates'], config['search']['separation']), dict(
        gcc_seconds=marker['gcc_done'] - began, search_seconds=ended - marker['gcc_done'], srp_seconds=ended - began,
        setup_seconds=0, pair_count=91, fft_length=1024, spectral_bins=513, full_directions=5580,
        evaluated_directions=5580, direction_pair_evaluations=507780, omitted_directions=0,
        regional_bound_prunes=0, boundary_expansions=0, full_fallback=False, evidence_grid='complete_fine', global_search='full')


def geometry_result(case, bearings, scores, grid, visual, rig, cameras):
    if not case['label']:
        return False
    detections, truth = visual
    groups = []
    arrival = 0.5 + np.linalg.norm(np.r_[case['points'][0], rig.horn_height - rig.height]) / 343
    for bearing, score in zip(bearings, scores):
        if score < SETTINGS['capture_threshold']:
            continue
        routed = route_cameras(rig, bearing.azimuth, cameras=cameras)
        selected = [item for item in detections if item.camera in routed]
        result = associate(rig, bearing.azimuth, arrival, selected, Association(), cameras=cameras,
                           evidence=AcousticEvidence(bearing, grid))
        groups += result['groups']
    return bool(REFERENCE.evaluate_match(dict(state='ambiguous' if groups else 'unmatched', groups=groups), truth)['target_retained'])


def prepare(split):
    check_plan()
    if split not in ('validation', 'test', 'stress'):
        raise ValueError('Unknown SRP source split')
    if split != 'validation' and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Freeze search choice before held-out evaluation')
    config = load_config(PROTO.parent / 'deployment/pynq_z2/config/band_temporal.json')
    engines = {name: Search(config['audio'], dict(config['search'], regions=SETTINGS['regions'],
                    coarse_azimuth=SETTINGS['coarse_azimuth'], coarse_depression=SETTINGS['coarse_depression'], **settings))
               for name, settings in VARIANTS.items()}
    detector = frozen.BandDetector(frozen.OUTPUT / 'model-combined')
    source_plan = REFERENCE.check_plan()
    rig = camera_rig()
    cameras = rig.cameras()
    for protocol in ('original', 'visible'):
        key = split + '-' + protocol
        path = OUTPUT / (key + '.json')
        if path.exists():
            load_result(key)
            continue
        cases, _ = REFERENCE.load_cache(key)
        names = {case[name] for case in cases for name in ('target', 'interferer')}
        sources = {record['file']: REFERENCE.waveform(record) for record in source_plan['sources'] if record['file'] in names}
        records = {name: [] for name in ['legacy'] + list(VARIANTS)}
        began = time.monotonic()
        for index, case in enumerate(cases):
            audio, pcm, _, _ = REFERENCE.synthesize(case, sources, source_plan['profile'], stress=split == 'stress')
            if pcm['pcm_sha256'] != case['pcm']['pcm_sha256']:
                raise ValueError('SRP scene PCM differs from frozen protocol')
            if not index:
                legacy_run(audio, config)
                for engine in engines.values():
                    engine.run(audio)
            visual = REFERENCE.observations(rig, cameras, np.asarray(case['points']), [np.zeros(2), np.zeros(2)],
                                            0.5, np.random.default_rng(case['visual_seed']))
            beams = {}
            order = ['legacy'] + list(engines)
            order = order[index % len(order):] + order[:index % len(order)]
            for name in order:
                grid, bearings, timing = legacy_run(audio, config) if name == 'legacy' else engines[name].run(audio)
                probabilities = []
                for bearing in bearings:
                    identity = (bearing.azimuth, bearing.depression)
                    if identity not in beams:
                        waveform = beamform(audio, bearing, config['audio'])[0]
                        beams[identity] = detector.score(features(waveform))['score']
                    probabilities.append(beams[identity])
                retained = geometry_result(case, bearings, probabilities, grid, visual, rig, cameras)
                rows = [dict(azimuth=bearing.azimuth, depression=bearing.depression, response=bearing.response) for bearing in bearings]
                while len(rows) < 2:
                    rows.append(dict(azimuth=None, depression=90, response=0))
                    probabilities.append(0)
                records[name].append(dict(bearings=rows, scores=probabilities, timing=timing,
                                           candidate_count=len(bearings), visual_target_retained=retained))
            if (index + 1) % 24 == 0 or index + 1 == len(cases):
                print(key, index + 1, '/', len(cases), 'seconds', round(time.monotonic() - began), flush=True)
        write_json(path, dict(plan_sha256=digest(OUTPUT / 'plan.json'), rows_sha256=digest(REFERENCE.OUTPUT / (key + '.json')),
                   pcm_cache_sha256=digest(REFERENCE.OUTPUT / (key + '.npz')), models=records,
                   scope='one rotated-order warm measurement per method/scene on PC; beam inference cache not timed as full pipeline'))


def load_result(key):
    check_plan()
    if not key.startswith('validation-') and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Held-out SRP results require frozen selection')
    value = read_json(OUTPUT / (key + '.json'))
    if value['plan_sha256'] != digest(OUTPUT / 'plan.json') or value['rows_sha256'] != digest(REFERENCE.OUTPUT / (key + '.json')) or value['pcm_cache_sha256'] != digest(REFERENCE.OUTPUT / (key + '.npz')):
        raise ValueError('SRP result provenance mismatch')
    cases = read_json(REFERENCE.OUTPUT / (key + '.json'))['rows']
    if set(value['models']) != set(['legacy'] + list(VARIANTS)) or any(len(rows) != len(cases) for rows in value['models'].values()):
        raise ValueError('Incomplete matched SRP results')
    return cases, value['models']


def summarize(cases, rows, baseline):
    paired = [dict(case, bearings=row['bearings']) for case, row in zip(cases, rows)]
    scores = np.asarray([row['scores'] for row in rows])
    joint = frozen.evaluation.outcomes(paired, scores, SETTINGS['direction_threshold'], SETTINGS['margin'])
    capture = frozen.evaluation.outcomes(paired, scores, SETTINGS['capture_threshold'], SETTINGS['margin'])
    timing = {key: dict(median=float(np.median([row['timing'][key] for row in rows])),
                       p90=float(np.percentile([row['timing'][key] for row in rows], 90)))
              for key in ('gcc_seconds', 'search_seconds', 'srp_seconds')}
    ratio = {key: dict(median=float(np.median([row['timing'][key] / previous['timing'][key] for row, previous in zip(rows, baseline)])),
                       p90=float(np.percentile([row['timing'][key] / previous['timing'][key] for row, previous in zip(rows, baseline)], 90)))
             for key in timing}
    return dict(joint=joint, capture=capture, timing=timing, relative_time=ratio,
                visual_target_retained=sum(row['visual_target_retained'] for row in rows),
                median_directions=float(np.median([row['timing']['evaluated_directions'] for row in rows])),
                p90_directions=float(np.percentile([row['timing']['evaluated_directions'] for row in rows], 90)),
                full_fallbacks=sum(row['timing']['full_fallback'] for row in rows),
                median_pair_evaluations=float(np.median([row['timing']['direction_pair_evaluations'] for row in rows])),
                bounded_prunes=sum(row['timing']['regional_bound_prunes'] for row in rows))


def quality_guard(candidate, baseline):
    reasons = []
    if candidate['joint']['candidate_hits'] < baseline['joint']['candidate_hits'] - SETTINGS['maximum_candidate_loss']:
        reasons.append('candidate_recall')
    if candidate['joint']['tp'] < baseline['joint']['tp'] - SETTINGS['maximum_direction_loss']:
        reasons.append('direction_recall')
    if candidate['joint']['f1'] < baseline['joint']['f1'] - SETTINGS['maximum_f1_loss']:
        reasons.append('joint_f1')
    if candidate['capture']['detection_recall'] < baseline['capture']['detection_recall'] - SETTINGS['maximum_capture_loss']:
        reasons.append('capture_recall')
    if candidate['capture']['accepted_negative'] > baseline['capture']['accepted_negative'] + SETTINGS['maximum_extra_negative']:
        reasons.append('negative_capture')
    if candidate['visual_target_retained'] < baseline['visual_target_retained'] - SETTINGS['maximum_visual_loss']:
        reasons.append('visual_target_retention')
    return reasons


def validation_summary():
    protocols = {}
    for protocol in ('original', 'visible'):
        cases, methods = load_result('validation-' + protocol)
        if any(case['split'] != 'validation' for case in cases):
            raise ValueError('Search selection requires validation sources only')
        protocols[protocol] = {name: summarize(cases, rows, methods['legacy']) for name, rows in methods.items()}
    return protocols


def select():
    check_plan()
    if (OUTPUT / 'freeze.json').exists():
        raise ValueError('Search choice is frozen')
    protocols = validation_summary()
    choices = {}
    for name in VARIANTS:
        reasons = {protocol: quality_guard(methods[name], methods['legacy']) for protocol, methods in protocols.items()}
        ratio = float(np.mean([methods[name]['relative_time']['srp_seconds']['median'] for methods in protocols.values()]))
        choices[name] = dict(quality_met=not any(reasons.values()), reasons=reasons, mean_relative_time=ratio,
                             time_target_met=ratio <= SETTINGS['relative_time_target'])
    feasible = [name for name, value in choices.items() if value['quality_met']]
    selected = min(feasible, key=lambda name: choices[name]['mean_relative_time']) if feasible else 'legacy'
    write_json(OUTPUT / 'freeze.json', dict(plan_sha256=digest(OUTPUT / 'plan.json'), selected=selected, choices=choices,
               validation=protocols, validation_sha256={protocol: digest(OUTPUT / ('validation-' + protocol + '.json')) for protocol in protocols},
               scope='frozen before test and stress; PC relative SRP time, no ARM end-to-end acceptance'))
    print(json.dumps(dict(selected=selected, choices=choices)), flush=True)


def evaluate():
    check_plan()
    frozen_choice = read_json(OUTPUT / 'freeze.json')
    if frozen_choice['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Frozen SRP selection changed')
    reports = {}
    for phase in ('test', 'stress'):
        reports[phase] = {}
        for protocol in ('original', 'visible'):
            cases, methods = load_result(phase + '-' + protocol)
            reports[phase][protocol] = {name: summarize(cases, rows, methods['legacy']) for name, rows in methods.items()}
    write_json(OUTPUT / 'results.json', dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), protocols=reports,
               selected=frozen_choice['selected'], limits=SETTINGS['evidence']))
    report()


def report():
    choice = read_json(OUTPUT / 'freeze.json')
    result = read_json(OUTPUT / 'results.json') if (OUTPUT / 'results.json').exists() else None
    lines = ['# SRP 混合基线与区域剪枝', '', '验证集选型：' + choice['selected'] + '。所有模型/鸣笛门限保持冻结。', '',
             '折叠FFT保留全部1024点Hann时间窗和全部帧，仅计算等间隔频率梳；GCC发生周期缩短和混响/噪声混叠，不等同完整PHAT。', '',
             '区间上界只保证每个选定细搜区域最大值；全局粗峰筛选仍是启发式。未搜索细网格不填零，关联使用实际测量粗网格，范围不足保持歧义。', '',
             '| 组 | 协议 | 方法 | 联合F1 | TP/84 | 候选命中/84 | 抓拍正例/84 | 负例抓拍/84 | 视觉目标保留 | SRP时间比例 | 空间搜索比例 | 中位方向数 | 全搜回退 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    phases = dict(validation=choice['validation'])
    if result:
        phases.update(result['protocols'])
    for phase, protocols in phases.items():
        for protocol, methods in protocols.items():
            for name, value in methods.items():
                joint, capture = value['joint'], value['capture']
                lines.append(f"| {phase} | {protocol} | {name} | {joint['f1']:.4f} | {joint['tp']} | {joint['candidate_hits']} | {capture['accepted_positive']} | {capture['accepted_negative']} | {value['visual_target_retained']} | {value['relative_time']['srp_seconds']['median']:.3f} | {value['relative_time']['search_seconds']['median']:.3f} | {value['median_directions']:.0f} | {value['full_fallbacks']} |")
    lines += ['', 'SRP时间含FFT/PHAT/GCC和空间搜索；空间搜索比例单列。比例为相同场景配对时间比中位数，不能外推为ARM全流程或视觉耗时比例。', '',
              '每协议84正例/84负例；复用来源与父录音重叠限制不变，不是新盲测或连续事件FP/hour。真实相机、ARM、DMA和冷启动仍需合作者验收。', '',
              '复现：plan；prepare --split validation；select；prepare --split test；prepare --split stress；evaluate；verify。新产物被Git忽略，不修改旧实验。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUTPUT / 'README.md', flush=True)


def verify():
    choice = read_json(OUTPUT / 'freeze.json')
    if choice['plan_sha256'] != digest(OUTPUT / 'plan.json') or choice['validation'] != validation_summary():
        raise ValueError('SRP validation freeze mismatch')
    for protocol, checksum in choice['validation_sha256'].items():
        if digest(OUTPUT / ('validation-' + protocol + '.json')) != checksum:
            raise ValueError('Frozen validation result changed')
    comparisons = 0
    for phase in ('test', 'stress'):
        for protocol in ('original', 'visible'):
            cases, methods = load_result(phase + '-' + protocol)
            original = read_json(frozen.OUTPUT / 'results.json')['protocols'][phase][protocol]['primary']['combined']['all']
            current = summarize(cases, methods['legacy'], methods['legacy'])['joint']
            for key in ('tp', 'fp', 'fn', 'candidate_hits'):
                if current[key] != original[key]:
                    raise ValueError('Legacy SRP baseline no longer reproduces the frozen experiment: ' + key)
            for rows in methods.values():
                if any(row['timing']['evaluated_directions'] > 5580 for row in rows):
                    raise ValueError('Unique scoring count exceeds full grid')
                comparisons += len(rows)
    write_json(OUTPUT / 'verification.json', dict(result='passed', matched_scene_methods=comparisons,
               old_inputs_unchanged=True, selected=choice['selected'], board_validated=False))
    print('SRP verification passed', comparisons, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'select', 'evaluate', 'report', 'verify'))
    parser.add_argument('--split', choices=('validation', 'test', 'stress'), default='validation')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.split)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
