"""Compare validation-frozen capture gates without changing direction confirmation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))
import improve_band as band
from horn.recovery import camera_rig, digest, write_json
from ps.geometry import route_cameras

REFERENCE = band.reference
OUTPUT = PROTO / 'outputs/capture_policy_v1'
SETTINGS = dict(budgets=[0.1, 0.2, 0.3], recall_targets=[0.7, 0.8, 0.9],
                thresholds=[float(value) for value in np.linspace(0, 1, 21)],
                models=['baseline', 'combined'], domains=['HornBase', 'ESC50'],
                selection='maximize worst-domain capture recall, then macro recall, then reduce worst-domain false capture; validation only',
                scope='completed independent windows; no event, camera-delivery or road FP/hour claim')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan():
    band.check_plan()
    frozen = read_json(band.OUTPUT / 'freeze.json')
    if frozen['validation_selected'] != 'combined':
        raise ValueError('This experiment requires the previously selected combined model')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [band.OUTPUT / 'plan.json', band.OUTPUT / 'freeze.json',
             band.OUTPUT / 'model-combined/manifest.json', band.OUTPUT / 'model-combined/weights.npz',
             band.PREVIOUS / 'spectral_model/manifest.json', band.PREVIOUS / 'spectral_model/weights.npz',
             PROTO / 'improve_band.py', PROTO / 'improve_detector.py', PROTO / 'improve_panns.py',
             PROTO.parent / 'deployment/pynq_z2/ps/dsp.py',
             PROTO.parent / 'deployment/pynq_z2/ps/geometry.py', PROTO / 'profiles/gc2093_single.json']
    value = dict(format=1, settings=SETTINGS,
                 frozen={str(path.relative_to(PROTO.parent)): digest(path) for path in paths},
                 direction_gates=dict(baseline=frozen['original_gate'], combined=frozen['selected']['combined']['gates']['primary']),
                 evidence='reused explored recordings, synthetic array and camera; inherited parent-recording overlap; no new blind test',
                 deployment='optional capture_threshold <= detector.threshold; default unchanged, no automatic model/gate promotion')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Capture experiment plan is already frozen')
    write_json(path, value)
    print('Frozen capture policy plan', flush=True)


def check_plan():
    value = read_json(OUTPUT / 'plan.json')
    if value['settings'] != SETTINGS:
        raise ValueError('Capture settings changed')
    for name, checksum in value['frozen'].items():
        if digest(PROTO.parent / name) != checksum:
            raise ValueError('Frozen input changed: ' + name)
    return value


def checked_scores(cases, probabilities):
    values = np.asarray(probabilities, dtype=float)
    if not cases or values.shape != (len(cases), 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Need two finite candidate probabilities for each scene')
    return values


def capture_metrics(cases, probabilities, threshold):
    """Count capture opportunities and retained candidate directions, not final DOA."""
    values = checked_scores(cases, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Capture threshold must lie in [0, 1]')
    labels = np.asarray([bool(case['label']) for case in cases])
    kept = values >= threshold
    captured = kept.any(axis=1)
    correct = np.asarray([[item['azimuth'] is not None and abs(band.reference.wrap_angle(item['azimuth'] - case['angle'])) < 10
                           for item in case['bearings']] for case in cases])
    tp = int(np.sum(labels & captured))
    fp = int(np.sum(~labels & captured))
    positives, negatives = int(labels.sum()), int((~labels).sum())
    return dict(positives=positives, negatives=negatives, captured_positive=tp, captured_negative=fp,
                capture_recall=tp / max(1, positives), negative_capture_rate=fp / max(1, negatives),
                capture_precision=tp / max(1, tp + fp), capture_f1=2 * tp / max(1, tp + fp + positives),
                retained_correct_direction=int(np.sum(labels & (kept & correct).any(axis=1))),
                direction_candidate_recall=int(np.sum(labels & (kept & correct).any(axis=1))) / max(1, positives))


def domain_metrics(cases, probabilities, threshold):
    values = checked_scores(cases, probabilities)
    result = {}
    for domain in SETTINGS['domains']:
        mask = np.asarray([case['dataset'] == domain for case in cases])
        selected = [case for case, keep in zip(cases, mask) if keep]
        if {case['label'] for case in selected} != {0, 1}:
            raise ValueError('Each source domain needs both labels')
        result[domain] = capture_metrics(selected, values[mask], threshold)
    return result


def validation_curve(cases, probabilities, direction_threshold):
    """Keep all curve selection on validation, with fixed source-balanced criteria."""
    if not cases or any(case['split'] != 'validation' for case in cases):
        raise ValueError('Capture selection requires validation sources only')
    if not np.isfinite(direction_threshold) or not 0 <= direction_threshold <= 1:
        raise ValueError('Invalid frozen direction gate')
    thresholds = sorted(set([value for value in SETTINGS['thresholds'] if value <= direction_threshold] + [direction_threshold]))
    curve = []
    for threshold in thresholds:
        domains = domain_metrics(cases, probabilities, threshold)
        curve.append(dict(threshold=threshold, metrics=capture_metrics(cases, probabilities, threshold), domains=domains,
                          worst_domain_acceptance=max(item['negative_capture_rate'] for item in domains.values()),
                          worst_domain_recall=min(item['capture_recall'] for item in domains.values()),
                          macro_recall=float(np.mean([item['capture_recall'] for item in domains.values()]))))
    return curve


def choose_policies(cases, probabilities, direction_threshold):
    curve = validation_curve(cases, probabilities, direction_threshold)
    choices = {}
    for budget in SETTINGS['budgets']:
        eligible = [item for item in curve if item['worst_domain_acceptance'] <= budget]
        selected = max(eligible, key=lambda item: (item['worst_domain_recall'], item['macro_recall'],
                                                  -item['worst_domain_acceptance'], item['threshold'])) if eligible else None
        choices['budget-' + str(round(budget * 100))] = dict(budget=budget, feasible=selected is not None, selected=selected)
    targets = {}
    for target in SETTINGS['recall_targets']:
        eligible = [item for item in curve if item['worst_domain_recall'] >= target]
        selected = min(eligible, key=lambda item: (item['worst_domain_acceptance'], -item['threshold'])) if eligible else None
        targets[str(round(target * 100))] = dict(target=target, feasible=selected is not None, selected=selected)
    return dict(curve=curve, policies=choices, recall_targets=targets)


def prepare(split):
    check_plan()
    if split not in ('validation', 'test', 'stress'):
        raise ValueError('Unknown scene split')
    if split != 'validation' and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Freeze capture gates before test predictions')
    models = dict(baseline=band.SpectralDetector(band.PREVIOUS / 'spectral_model'),
                  combined=band.BandDetector(band.OUTPUT / 'model-combined'))
    for protocol in ('original', 'visible'):
        key = split + '-' + protocol
        path = OUTPUT / (key + '.npz')
        if path.exists():
            load_cache(key)
            continue
        cases, data = REFERENCE.load_cache(key)
        scores = {name: np.asarray([[model.score(beam)['score'] for beam in scene] for scene in data['contexts']])
                  for name, model in models.items()}
        np.savez_compressed(path, **scores)
        write_json(path.with_suffix('.json'), dict(plan_sha256=digest(OUTPUT / 'plan.json'), cache_sha256=digest(path),
                   source_sha256=digest(band.PREVIOUS / (key + '.npz')), rows_sha256=digest(band.PREVIOUS / (key + '.json')),
                   rows=cases))
        print('Prepared frozen scores', key, flush=True)


def load_cache(key):
    check_plan()
    if key.split('-')[0] != 'validation' and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Test scores are blocked before gate freeze')
    metadata = read_json(OUTPUT / (key + '.json'))
    expected = dict(plan_sha256=digest(OUTPUT / 'plan.json'), cache_sha256=digest(OUTPUT / (key + '.npz')),
                    source_sha256=digest(band.PREVIOUS / (key + '.npz')), rows_sha256=digest(band.PREVIOUS / (key + '.json')))
    if any(metadata[name] != checksum for name, checksum in expected.items()):
        raise ValueError('Capture cache provenance mismatch')
    cases = read_json(band.PREVIOUS / (key + '.json'))['rows']
    if cases != metadata['rows']:
        raise ValueError('Capture rows changed')
    with np.load(OUTPUT / (key + '.npz'), allow_pickle=False) as archive:
        values = {name: checked_scores(cases, archive[name]) for name in SETTINGS['models']}
    return cases, values


def select():
    planned = check_plan()
    if (OUTPUT / 'freeze.json').exists():
        raise ValueError('Capture gates are frozen; further selection needs another experiment')
    rows, arrays = [], []
    for protocol in ('original', 'visible'):
        cases, values = load_cache('validation-' + protocol)
        rows += cases
        arrays.append(values)
    result = {name: choose_policies(rows, np.concatenate([item[name] for item in arrays]),
                                    planned['direction_gates'][name]['threshold']) for name in SETTINGS['models']}
    caches = {protocol: digest(OUTPUT / ('validation-' + protocol + '.npz')) for protocol in ('original', 'visible')}
    write_json(OUTPUT / 'freeze.json', dict(plan_sha256=digest(OUTPUT / 'plan.json'), validation_caches=caches, models=result,
               selection='all displayed operating points selected before current test predictions; no default recommendation from test'))
    print(json.dumps({name: {key: None if value['selected'] is None else value['selected']['threshold']
                            for key, value in result[name]['policies'].items()} for name in result}), flush=True)


def routing_masks(cases, rig):
    """Cache nominal camera coverage independently of classifier thresholds."""
    cameras = rig.cameras()
    requested, target = [], []
    for case in cases:
        routes = [set(route_cameras(rig, item['azimuth'], cameras=cameras)) for item in case['bearings']]
        point = [case['radius'] * np.cos(np.deg2rad(case['angle'])), case['radius'] * np.sin(np.deg2rad(case['angle'])), 0]
        visible = {camera.identity for camera in cameras if camera.project([point])[1][0]}
        requested.append([bool(route) for route in routes])
        target.append([bool(route & visible) and bool(case['visible']) for route in routes])
    return np.asarray(requested), np.asarray(target)


def photo_metrics(cases, probabilities, threshold, masks):
    values = checked_scores(cases, probabilities)
    kept = values >= threshold
    requested, target = masks
    labels = np.asarray([bool(case['label']) for case in cases])
    visible = np.asarray([bool(case['visible']) for case in cases])
    photos = (kept & requested).any(axis=1)
    return dict(positive_photo_opportunities=int(np.sum(labels & photos)),
                negative_photo_opportunities=int(np.sum(~labels & photos)),
                visible_positive=int(np.sum(labels & visible)),
                nominal_target_in_requested_camera=int(np.sum(labels & (kept & target).any(axis=1))),
                scope='nominal direction routing only; not successful exposure, vehicle detection or attribution')


def check_freeze():
    planned = check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Frozen capture plan changed')
    for protocol, checksum in frozen['validation_caches'].items():
        cases, values = load_cache('validation-' + protocol)
        if digest(OUTPUT / ('validation-' + protocol + '.npz')) != checksum:
            raise ValueError('Frozen validation scores changed')
    return planned, frozen


def evaluate():
    planned, frozen = check_freeze()
    reports = {}
    for phase in ('test', 'stress'):
        reports[phase] = {}
        for protocol in ('original', 'visible'):
            cases, values = load_cache(phase + '-' + protocol)
            masks = routing_masks(cases, camera_rig())
            models = {}
            for name in SETTINGS['models']:
                gate = planned['direction_gates'][name]
                points = dict(inherited=gate['threshold'])
                points.update({key: item['selected']['threshold'] for key, item in frozen['models'][name]['policies'].items() if item['feasible']})
                strict = band.evaluation.outcomes(cases, values[name], gate['threshold'], gate['margin'])
                points_result = {}
                for key, threshold in points.items():
                    metric = capture_metrics(cases, values[name], threshold)
                    if metric['captured_positive'] < strict['accepted_positive']:
                        raise ValueError('Capture must retain all strict positive decisions')
                    points_result[key] = dict(threshold=threshold, capture=metric,
                                             domains=domain_metrics(cases, values[name], threshold),
                                             photo=photo_metrics(cases, values[name], threshold, masks))
                models[name] = dict(strict_direction_gate=dict(threshold=gate['threshold'], margin=gate['margin']),
                                    strict_direction=strict, policies=points_result)
            reports[phase][protocol] = models
    write_json(OUTPUT / 'results.json', dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), protocols=reports,
               scope=SETTINGS['scope'], evidence=planned['evidence']))
    report()


def report():
    _, frozen = check_freeze()
    result = read_json(OUTPUT / 'results.json')
    if result['freeze_sha256'] != digest(OUTPUT / 'freeze.json'):
        raise ValueError('Results and capture freeze disagree')
    lines = ['# 独立抓拍门限实验', '',
             '固定旧BandMLP与时间＋回放组合权重，方向门限保持0.90/0.95；仅验证集选择抓拍门限。默认部署未降低门限。', '',
             '抓拍召回是独立窗口中出现鸣笛抓拍候选的比例；候选方向召回允许保留多个方向；二者都不是严格联合召回或车辆识别成功率。inherited仅表示沿用原方向门限，候选覆盖列仍使用本轮候选并集，不等同于旧版抓拍路由。', '',
             '预算限制HornBase、ESC50各自验证负例抓拍率，跨原/可见协议共用门限；按最差来源召回、宏平均召回、最差来源误接受和较高门限依次选择。', '',
             '| 模型 | 验证预算 | 抓拍门限 | 最差来源召回 | 最差来源负例抓拍率 |', '|---|---|---:|---:|---:|']
    for name, item in frozen['models'].items():
        for key, policy in item['policies'].items():
            point = policy['selected']
            if point is None:
                lines.append(f'| {name} | {key} | 无可行点 | — | — |')
            else:
                lines.append(f"| {name} | {key} | {point['threshold']:.2f} | {point['worst_domain_recall']:.4f} | {point['worst_domain_acceptance']:.4f} |")
    lines += ['', '| 模型 | 验证最差来源召回目标 | 所需门限 | 最差来源负例抓拍率 |', '|---|---:|---:|---:|']
    for name, item in frozen['models'].items():
        for key, target in item['recall_targets'].items():
            point = target['selected']
            lines.append(f'| {name} | {key}% | 无可行点 | — |' if point is None else
                         f"| {name} | {key}% | {point['threshold']:.2f} | {point['worst_domain_acceptance']:.4f} |")
    lines += ['', '| 组 | 协议 | 模型 | 工作点 | 抓拍正例/84 | 负例抓拍/84 | 保留正确方向候选/84 | 名义画面覆盖目标 |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for phase, protocols in result['protocols'].items():
        for protocol, models in protocols.items():
            for name, model in models.items():
                for key, policy in model['policies'].items():
                    metric, photo = policy['capture'], policy['photo']
                    lines.append(f"| {phase} | {protocol} | {name} | {key} | {metric['captured_positive']} | {metric['captured_negative']} | {metric['retained_correct_direction']} | {photo['nominal_target_in_requested_camera']}/{photo['visible_positive']} |")
    lines += ['', '本轮不重训，不改变严格方向工作点；降低抓拍门限不会自动提高严格联合F1。所有负例鸣笛抓拍候选均计误接受，画面外候选也不从声学误接受分母移除。', '',
              '名义画面覆盖仅表示候选方向请求了目标所在模拟相机，不表示拍摄成功、鸣笛车辆已定位或目标已标框。独立窗口不能证明确认计数、去重及连续事件召回，不能估计道路FP/hour。', '',
              '模型和来源已被先前探索，当前只是验证集冻结规则下的后续诊断，不是新的盲测；真实镜头、连续道路录音和ARM负载仍待验证。', '',
              '复现：plan；prepare --split validation；select；prepare --split test；prepare --split stress；evaluate；verify。产物在忽略目录capture_policy_v1，无新依赖。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUTPUT / 'README.md', flush=True)


def verify():
    _, frozen = check_freeze()
    result = read_json(OUTPUT / 'results.json')
    if result['freeze_sha256'] != digest(OUTPUT / 'freeze.json'):
        raise ValueError('Frozen capture results changed')
    rows, arrays = [], []
    for protocol in ('original', 'visible'):
        cases, values = load_cache('validation-' + protocol)
        rows += cases
        arrays.append(values)
    for name in SETTINGS['models']:
        recomputed = choose_policies(rows, np.concatenate([item[name] for item in arrays]),
                                    read_json(OUTPUT / 'plan.json')['direction_gates'][name]['threshold'])
        if recomputed != frozen['models'][name]:
            raise ValueError('Frozen operating points do not match validation')
    comparisons = 0
    for phase, protocols in result['protocols'].items():
        for protocol, models in protocols.items():
            cases, arrays = load_cache(phase + '-' + protocol)
            for name, model in models.items():
                strict = model['strict_direction']
                original = read_json(band.OUTPUT / 'results.json')['protocols'][phase][protocol]
                expected = original['inherited_baseline']['all'] if name == 'baseline' else original['primary']['combined']['all']
                if strict != expected:
                    raise ValueError('Strict direction decisions changed')
                labels = np.asarray([bool(case['label']) for case in cases])
                for policy in model['policies'].values():
                    accepted = np.max(arrays[name], axis=1) >= policy['threshold']
                    metric = policy['capture']
                    if metric['positives'] != 84 or metric['negatives'] != 84 or metric['captured_positive'] != int((accepted & labels).sum()) or metric['captured_negative'] != int((accepted & ~labels).sum()):
                        raise ValueError('Capture counts disagree with independent presence decisions')
                    comparisons += 1
    write_json(OUTPUT / 'verification.json', dict(result='passed', operating_points=comparisons,
               old_inputs_unchanged=True, strict_direction_unchanged=True, no_test_curve_selection=True,
               source_sha256=digest(Path(__file__)), freeze_sha256=digest(OUTPUT / 'freeze.json'),
               results_sha256=digest(OUTPUT / 'results.json'), board_validated=False))
    print('Capture policy verification passed:', comparisons, 'operating points', flush=True)


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
