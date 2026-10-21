"""Ablate temporal band statistics and original-recording replay on frozen sources."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))
import improve_detector as reference
import improve_panns as evaluation
from horn.calibration import remove_dc
from horn.recovery import digest, write_json, camera_rig
from ps.dsp import band_features, features
from ps.detector import SpectralDetector

BASE = reference.BASE
PREVIOUS = reference.OUTPUT
OUTPUT = PROTO / 'outputs/band_improvements_v1'
SCHEMES = dict(static=('static', False), temporal=('temporal', False),
               replay=('static', True), combined=('temporal', True))
SETTINGS = dict(seeds=[0, 17], epochs=30, patience=8, batch_scenes=8,
                learning_rate=0.0003, teacher_weight=0.5, ranking_weight=0.1,
                replay_weight=0.5, hidden=48, segments=3,
                primary_domain_acceptance=0.1, strict_domain_acceptance=0.05,
                scope='fixed factorial mechanisms; no teacher/model/threshold retuning from test outcomes')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sequence_features(contexts):
    """Pool ordered thirds and peak statistics, preserving relative energy envelopes."""
    values = np.asarray(contexts, dtype=float)
    if values.ndim < 3 or values.shape[-3:] != (7, 40, 24) or not np.isfinite(values).all():
        raise ValueError('Temporal statistics require seven finite 40x24 contexts')
    statistics = band_features(values)
    energy = values.mean(axis=(-2, -1))
    statistics[..., :40] += (energy - energy.mean(axis=-1, keepdims=True))[..., None]
    segments = np.array_split(np.arange(7), 3)
    pooled = [statistics[..., segment, :].mean(axis=-2) for segment in segments]
    return np.concatenate(pooled + [statistics.max(axis=-2)], axis=-1)


def model_features(contexts, mode):
    if mode == 'static':
        return band_features(contexts)
    if mode == 'temporal':
        return sequence_features(contexts)
    raise ValueError('Unknown band feature mode')


def numpy_scores(weights, inputs):
    values = np.asarray(inputs, dtype=float)
    hidden = np.maximum(0, values @ weights['weight0'].T + weights['bias0'])
    logits = hidden @ weights['weight1'].T + weights['bias1']
    return 1 / (1 + np.exp(-np.clip(logits[..., 1] - logits[..., 0], -80, 80)))


class BandDetector:
    """NumPy-only scoring of completed beam contexts with a checked exported MLP."""

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        self.mode = self.metadata.get('feature_mode', 'static')
        architectures = dict(static=('BandMLP-7826', 160), temporal=('BandTemporalMLP-30866', 640))
        if self.mode not in architectures or self.metadata.get('format') != 1:
            raise ValueError('Unsupported band model format')
        architecture, incoming = architectures[self.mode]
        if self.metadata.get('architecture') != architecture:
            raise ValueError('Model architecture and feature mode disagree')
        checksum = hashlib.sha256((directory / 'weights.npz').read_bytes()).hexdigest()
        if self.metadata.get('sha256') != {'weights.npz': checksum} or self.metadata.get('model_tag') != int(checksum[:8], 16):
            raise ValueError('Band model checksum mismatch')
        with np.load(str(directory / 'weights.npz'), allow_pickle=False) as archive:
            self.weights = {key: archive[key] for key in archive.files}
        expected = dict(weight0=(48, incoming), bias0=(48,), weight1=(2, 48), bias1=(2,))
        if set(self.weights) != set(expected) or any(self.weights[key].shape != shape or not np.isfinite(self.weights[key]).all()
                                                   for key, shape in expected.items()):
            raise ValueError('Invalid band weights')
        self.frontend = 'logmel'

    def score(self, contexts):
        if len(contexts) == 0:
            return dict(score=0.0, clipped=False, contexts=0, backend='band-numpy')
        values = np.asarray(contexts)
        if values.ndim != 3 or values.shape[1:] != (40, 24):
            raise ValueError('Band scoring needs N x 40 x 24 contexts')
        inputs = model_features(values, self.mode)
        probabilities = numpy_scores(self.weights, inputs)
        return dict(score=float(np.max(probabilities)), clipped=False, contexts=len(values), backend='band-numpy')

    def score_waveform(self, waveform):
        """Score a caller-conditioned mono beam; no additional DC filter is applied."""
        values = np.asarray(waveform, dtype=float)
        if values.shape not in ((48000,), (48128,)) or not np.isfinite(values).all():
            raise ValueError('Need one finite completed mono beam window')
        return self.score(features(values))


def plan():
    reference.check_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [BASE / 'clips.json', BASE / 'panns-horn.pt', PREVIOUS / 'plan.json', PREVIOUS / 'freeze.json',
             PREVIOUS / 'spectral_model/manifest.json', PREVIOUS / 'spectral_model/weights.npz',
             PROTO / 'improve_detector.py', PROTO / 'improve_panns.py',
             PROTO.parent / 'deployment/pynq_z2/ps/dsp.py', PROTO.parent / 'deployment/pynq_z2/ps/detector.py']
    value = dict(format=1, settings=SETTINGS, schemes={name: list(value) for name, value in SCHEMES.items()},
                 frozen={str(path.relative_to(PROTO.parent)): digest(path) for path in paths},
                 selection='validation only, shared gate across original/visible; count audible negatives even if direction is ambiguous',
                 limits='reused explored public sources and synthetic array/camera; shared HornBase parent recordings, not a new blind road test',
                 licenses='research/noncommercial; original source restrictions retained')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Experiment plan is already frozen')
    write_json(path, value)
    print('Frozen temporal/replay factorial plan', flush=True)


def check_plan():
    value = read_json(OUTPUT / 'plan.json')
    if value['settings'] != SETTINGS or value['schemes'] != {name: list(value) for name, value in SCHEMES.items()}:
        raise ValueError('Settings differ from the frozen experiment')
    for name, checksum in value['frozen'].items():
        if digest(PROTO.parent / name) != checksum:
            raise ValueError('Frozen input changed: ' + name)
    return value


def prepare_replay(split):
    """Keep complete one-second windows in weakly labeled recording bags."""
    check_plan()
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown replay source split')
    if split != 'train' and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Freeze models before held-out original-clip diagnostics')
    path = OUTPUT / ('replay-' + split + '.npz')
    if path.exists():
        load_replay(split)
        print('Reusing original replay', split, flush=True)
        return
    records = [record for record in read_json(BASE / 'clips.json') if record['split'] == split]
    static, temporal, offsets, starts = [], [], [0], []
    began = time.monotonic()
    for number, record in enumerate(records, 1):
        if digest(record['path']) != record['sha256']:
            raise ValueError('Original recording checksum changed')
        audio = reference.read_audio(record['path'], 48000, channel=0)
        audio, _ = remove_dc(audio)
        if len(audio) < 48000:
            raise ValueError('Original replay needs complete seconds')
        positions = sorted(set(range(0, len(audio) - 48000 + 1, 24000)) | {len(audio) - 48000})
        for start in positions:
            contexts = features(audio[start:start + 48000, 0])
            if contexts.shape != (7, 40, 24):
                raise ValueError('Replay frontend and beam frontend disagree')
            static.append(model_features(contexts, 'static').astype(np.float32))
            temporal.append(model_features(contexts, 'temporal').astype(np.float32))
        offsets.append(offsets[-1] + len(positions))
        starts.append(positions)
        if number % 100 == 0 or number == len(records):
            print('Original replay', split, number, '/', len(records), 'seconds', round(time.monotonic() - began), flush=True)
    np.savez_compressed(path, static=np.asarray(static), temporal=np.asarray(temporal), offsets=np.asarray(offsets))
    write_json(OUTPUT / ('replay-' + split + '.json'), dict(records=records, starts=starts, plan_sha256=digest(OUTPUT / 'plan.json'),
               cache_sha256=digest(path), frontend='causal DC once per recording, completed 48000-sample windows/24000 step, seven board Log-Mel contexts; labels remain recording-level'))


def load_replay(split):
    check_plan()
    path = OUTPUT / ('replay-' + split + '.npz')
    metadata = read_json(path.with_suffix('.json'))
    if metadata['cache_sha256'] != digest(path) or metadata['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Replay cache provenance mismatch')
    expected = [record for record in read_json(BASE / 'clips.json') if record['split'] == split]
    if metadata['records'] != expected:
        raise ValueError('Replay recording split changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data['offsets']) != len(expected) + 1 or data['offsets'][0] != 0 or np.any(np.diff(data['offsets']) <= 0):
        raise ValueError('Invalid replay recording bags')
    count = data['offsets'][-1]
    if data['static'].shape != (count, 7, 160) or data['temporal'].shape != (count, 640):
        raise ValueError('Replay feature dimensions disagree')
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite replay features')
    return metadata['records'], data


def scene_data(split):
    cases, values, teacher = [], [], []
    for protocol in ('original', 'visible'):
        if split == 'train' and protocol != 'original':
            continue
        rows, data = reference.load_cache(split + '-' + protocol)
        cases += rows
        values.append(data['contexts'])
        teacher.append(data['teacher'])
    return cases, np.concatenate(values), np.concatenate(teacher)


def torch_scores(model, inputs, mode):
    torch = reference.configure()
    with torch.inference_mode():
        logits = model(torch.as_tensor(inputs, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=-1)[..., 1]
        if mode == 'static':
            probabilities = probabilities.max(dim=-1).values
    return probabilities.numpy()


def choose_gate(cases, probabilities, budget):
    """Freeze one validation gate, with a separate source-domain presence budget."""
    if not cases or any(case['split'] != 'validation' for case in cases) or not 0 <= budget <= 1:
        raise ValueError('Operating points require validation sources and a valid budget')
    domains = {name: np.asarray([case['dataset'] == name for case in cases]) for name in ('HornBase', 'ESC50')}
    if any({case['label'] for case in cases if case['dataset'] == name} != {0, 1} for name in domains):
        raise ValueError('Each validation domain needs both labels')
    values, options = np.asarray(probabilities), []
    for threshold in evaluation.SETTINGS['thresholds']:
        for margin in evaluation.SETTINGS['margins']:
            metrics = {name: evaluation.outcomes([case for case in cases if case['dataset'] == name], values[mask], threshold, margin)
                       for name, mask in domains.items()}
            worst = max(metric['negative_acceptance'] for metric in metrics.values())
            macro = float(np.mean([metric['f1'] for metric in metrics.values()]))
            combined = evaluation.outcomes(cases, values, threshold, margin)
            rank = (worst <= budget, macro, -worst, combined['f1'], threshold, margin)
            options.append((rank, dict(threshold=threshold, margin=margin, constraint_met=worst <= budget,
                             macro_f1=macro, worst_domain_acceptance=worst, metrics=combined, domains=metrics)))
    return max(options, key=lambda item: item[0])[1]


def export_runtime():
    """Generate the inference-only artifact without PC prototype or Torch imports."""
    sources = [(PROTO.parent / 'deployment/pynq_z2/ps/dsp.py', {'frames', 'features', 'band_features'}),
               (Path(__file__).resolve(), {'sequence_features', 'model_features', 'numpy_scores', 'BandDetector'})]
    parts = ['"""Generated NumPy band inference; no training or board performance claim."""',
             'import hashlib\nimport json\nfrom pathlib import Path\nimport numpy as np']
    for path, names in sources:
        text = path.read_text(encoding='utf-8')
        nodes = [node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise ValueError('Runtime export could not find required definitions')
        parts.extend(ast.get_source_segment(text, node) for node in nodes)
    path = OUTPUT / 'numpy_runtime.py'
    path.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('exported_band_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_model(model, mode, scheme, selection):
    directory = OUTPUT / ('model-' + scheme)
    directory.mkdir(exist_ok=True)
    state = model.state_dict()
    np.savez(directory / 'weights.npz', weight0=state['0.weight'].numpy(), bias0=state['0.bias'].numpy(),
             weight1=state['2.weight'].numpy(), bias1=state['2.bias'].numpy())
    checksum = digest(directory / 'weights.npz')
    metadata = dict(format=1, architecture='BandMLP-7826' if mode == 'static' else 'BandTemporalMLP-30866',
                    feature_mode=mode, model_tag=int(checksum[:8], 16), input_shape=[40, 24],
                    contexts=7, sha256={'weights.npz': checksum}, selection=selection,
                    provenance=dict(frontend='logmel', scheme=scheme, plan_sha256=digest(OUTPUT / 'plan.json'),
                                    license_scope='research/noncommercial', evidence='reused source experiment; no road/board claim'))
    write_json(directory / 'manifest.json', metadata)
    return BandDetector(directory)


def train():
    torch = reference.configure()
    check_plan()
    if (OUTPUT / 'freeze.json').exists():
        raise ValueError('Models are frozen; further training needs a separate experiment')
    cases, raw, teacher = scene_data('train')
    held, held_raw, _ = scene_data('validation')
    records, replay = load_replay('train')
    groups = [np.flatnonzero([case['dataset'] == domain and case['label'] == label for case in cases])
              for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    replay_groups = [np.flatnonzero([record['dataset'] == domain and record['label'] == label for record in records])
                     for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    labels = torch.tensor([case['label'] for case in cases], dtype=torch.float32)
    soft = torch.from_numpy(teacher.astype(np.float32))
    candidates = []
    selected = {}
    runtime = export_runtime()
    for scheme, (mode, use_replay) in SCHEMES.items():
        inputs = torch.from_numpy(model_features(raw, mode).astype(np.float32))
        validation = model_features(held_raw, mode).astype(np.float32)
        for seed in SETTINGS['seeds']:
            path = OUTPUT / f'{scheme}-{seed}.pt'
            metadata_path = path.with_suffix('.json')
            if path.exists() and metadata_path.exists():
                candidate = read_json(metadata_path)
                if candidate['sha256'] != digest(path):
                    raise ValueError('Candidate checkpoint changed')
                candidates.append(candidate)
                continue
            torch.manual_seed(seed)
            generator = np.random.default_rng(seed)
            replay_generator = np.random.default_rng(seed + 100000)
            incoming = 160 if mode == 'static' else 640
            model = torch.nn.Sequential(torch.nn.Linear(incoming, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2))
            optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'])
            best, state, stale, chosen, history = None, None, 0, None, []
            for epoch in range(SETTINGS['epochs'] + 1):
                model.train()
                losses = []
                if epoch:
                    for batch in range(len(cases) // SETTINGS['batch_scenes']):
                        indices = np.concatenate([generator.choice(group, 2) for group in groups])
                        logits = model(inputs[indices])
                        margins = logits[..., 1] - logits[..., 0]
                        if mode == 'static':
                            margins = margins.max(dim=-1).values
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(margins.max(dim=1).values, labels[indices])
                        loss = loss + SETTINGS['teacher_weight'] * torch.nn.functional.binary_cross_entropy_with_logits(margins, soft[indices])
                        ranking, count = margins.sum() * 0, 0
                        for row, index in enumerate(indices):
                            errors = [180 if item['azimuth'] is None else abs(reference.wrap_angle(item['azimuth'] - cases[index]['angle']))
                                      for item in cases[index]['bearings']]
                            target = int(np.argmin(errors))
                            if cases[index]['label'] and errors[target] < 10 and teacher[index, target] > teacher[index, 1 - target]:
                                ranking = ranking + torch.nn.functional.softplus(margins[row, 1 - target] - margins[row, target])
                                count += 1
                        loss = loss + SETTINGS['ranking_weight'] * ranking / max(1, count)
                        if use_replay:
                            chosen_clips = np.concatenate([replay_generator.choice(group, 2) for group in replay_groups])
                            parts = [replay[mode][replay['offsets'][index]:replay['offsets'][index + 1]] for index in chosen_clips]
                            replay_logits = model(torch.from_numpy(np.concatenate(parts)))
                            replay_margins = replay_logits[..., 1] - replay_logits[..., 0]
                            if mode == 'static':
                                replay_margins = replay_margins.max(dim=-1).values
                            bags = torch.stack([part.max() for part in replay_margins.split([len(part) for part in parts])])
                            targets = torch.tensor([records[index]['label'] for index in chosen_clips], dtype=torch.float32)
                            loss = loss + SETTINGS['replay_weight'] * torch.nn.functional.binary_cross_entropy_with_logits(bags, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                model.eval()
                probabilities = torch_scores(model, validation, mode)
                gate = choose_gate(held, probabilities, SETTINGS['primary_domain_acceptance'])
                rank = (gate['constraint_met'], gate['macro_f1'], -gate['worst_domain_acceptance'])
                history.append(dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, gate=gate))
                if best is None or rank > best:
                    best, state, stale, chosen = rank, deepcopy(model.state_dict()), 0, dict(epoch=epoch, gate=gate)
                else:
                    stale += 1
                if epoch % 5 == 0:
                    print(scheme, seed, epoch, 'validation macro F1', round(gate['macro_f1'], 4), flush=True)
                if stale >= SETTINGS['patience']:
                    break
            torch.save(dict(state=state, mode=mode, scheme=scheme, selection=chosen), path)
            candidate = dict(scheme=scheme, mode=mode, replay=use_replay, seed=seed, checkpoint=path.name, sha256=digest(path), **chosen)
            write_json(metadata_path, candidate)
            write_json(OUTPUT / f'{scheme}-{seed}-history.json', history)
            candidates.append(candidate)
        chosen = max([candidate for candidate in candidates if candidate['scheme'] == scheme],
                     key=lambda item: (item['gate']['constraint_met'], item['gate']['macro_f1'], -item['gate']['worst_domain_acceptance']))
        model = torch.nn.Sequential(torch.nn.Linear(160 if mode == 'static' else 640, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2)).eval()
        model.load_state_dict(torch.load(OUTPUT / chosen['checkpoint'], map_location='cpu', weights_only=True)['state'])
        exported = export_model(model, mode, scheme, chosen)
        actual = np.asarray([[exported.score(beam)['score'] for beam in scene] for scene in held_raw])
        expected = torch_scores(model, validation, mode)
        error = float(abs(actual - expected).max())
        if error > 1e-5:
            raise ValueError('NumPy band inference differs from Torch')
        generated = runtime.BandDetector(OUTPUT / ('model-' + scheme))
        if any(abs(generated.score(beam)['score'] - exported.score(beam)['score']) > 1e-12 for beam in held_raw[0]):
            raise ValueError('Generated runtime differs from the training reference')
        gates = {name: choose_gate(held, actual, SETTINGS[key]) for name, key in
                 [('primary', 'primary_domain_acceptance'), ('strict', 'strict_domain_acceptance')]}
        selected[scheme] = dict(chosen, gates=gates, numpy_maximum_error=error, model_tag=exported.metadata['model_tag'],
                                 manifest_sha256=digest(OUTPUT / ('model-' + scheme) / 'manifest.json'))
    old = SpectralDetector(PREVIOUS / 'spectral_model')
    previous_scores = np.asarray([[old.score(beam)['score'] for beam in scene] for scene in held_raw])
    baseline_gates = {name: choose_gate(held, previous_scores, SETTINGS[key]) for name, key in
                      [('primary', 'primary_domain_acceptance'), ('strict', 'strict_domain_acceptance')]}
    best = max(selected, key=lambda name: (selected[name]['gates']['primary']['constraint_met'],
                     selected[name]['gates']['primary']['macro_f1'], -selected[name]['gates']['primary']['worst_domain_acceptance']))
    frozen = dict(plan_sha256=digest(OUTPUT / 'plan.json'), selected=selected, candidates=candidates,
                  baseline_gates=baseline_gates, original_gate=read_json(PREVIOUS / 'freeze.json')['gates']['spectral_numpy'],
                  validation_selected=best, policy='all schemes and both source-domain operating budgets frozen before test predictions; no default promotion')
    write_json(OUTPUT / 'freeze.json', frozen)
    print('Frozen validation choice', best, flush=True)


def metrics(cases, probabilities, gate, grid):
    values = np.asarray(probabilities)
    domains = {name: evaluation.outcomes([case for case in cases if case['dataset'] == name],
                  values[[case['dataset'] == name for case in cases]], gate['threshold'], gate['margin']) for name in ('HornBase', 'ESC50')}
    association = reference.association_metrics(cases, grid, values, dict(gate, margin=max(gate['margin'], np.finfo(float).eps)), camera_rig())
    ranked = evaluation.outcomes(cases, values, 0, 0)['tp']
    return dict(all=evaluation.outcomes(cases, values, gate['threshold'], gate['margin']), domains=domains,
                unconditionally_ranked_correct=ranked, wide_camera=association,
                gate=dict(threshold=gate['threshold'], margin=gate['margin']))


def evaluate():
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Frozen plan changed')
    models = {name: BandDetector(OUTPUT / ('model-' + name)) for name in SCHEMES}
    for name, model in models.items():
        if digest(OUTPUT / ('model-' + name) / 'manifest.json') != frozen['selected'][name]['manifest_sha256']:
            raise ValueError('Frozen model changed')
    models['baseline'] = SpectralDetector(PREVIOUS / 'spectral_model')
    reports = {}
    for phase in ('test', 'stress'):
        reports[phase] = {}
        for protocol in ('original', 'visible'):
            key = phase + '-' + protocol
            cases, data = reference.load_cache(key)
            predictions = {name: np.asarray([[model.score(beam)['score'] for beam in scene] for scene in data['contexts']])
                           for name, model in models.items()}
            result = dict(primary={}, strict={})
            for budget in ('primary', 'strict'):
                for name, probabilities in predictions.items():
                    gate = frozen['baseline_gates'][budget] if name == 'baseline' else frozen['selected'][name]['gates'][budget]
                    result[budget][name] = metrics(cases, probabilities, gate, data)
            result['inherited_baseline'] = metrics(cases, predictions['baseline'], frozen['original_gate'], data)
            reports[phase][protocol] = result
            np.savez_compressed(OUTPUT / (key + '-predictions.npz'), **predictions)
    write_json(OUTPUT / 'results.json', dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), protocols=reports,
               validation_selected=frozen['validation_selected'], limits=check_plan()['limits']))
    report()


def report():
    value = read_json(OUTPUT / 'results.json')
    frozen = read_json(OUTPUT / 'freeze.json')
    lines = ['# BandMLP 时间特征与录音回放消融', '', '日期：2026-10-21。科研/非商业；复用来源诊断，不是新的盲测。', '',
             '四方案共享训练预算、冻结教师和验证准则：静态重训对照、分段时间特征、原始录音回放、两者结合。旧模型保留继承门限及按相同预算重校准两种对照。', '',
             '误接受按鸣笛存在计数，方向歧义不能隐藏负例；联合TP还要求接受的方向误差<10°。每测试协议84正例＋84负例。', '',
             f"验证集预选方案：{frozen['validation_selected']}。主预算为每来源验证负例接受率≤10%；另冻结同一权重下≤5%的严格门限，不按测试结果选择。", '',
             '| 组 | 协议 | 预算 | 方案 | 联合F1 | 正确方向/84 | 负例鸣笛接受/84 |',
             '|---|---|---|---|---:|---:|---:|']
    for phase, protocols in value['protocols'].items():
        for protocol, result in protocols.items():
            for budget in ('primary', 'strict'):
                for name, item in result[budget].items():
                    metric = item['all']
                    lines.append(f"| {phase} | {protocol} | {budget} | {name} | {metric['f1']:.4f} | {metric['tp']} | {metric['accepted_negative']} |")
            metric = result['inherited_baseline']['all']
            lines.append(f"| {phase} | {protocol} | inherited | baseline | {metric['f1']:.4f} | {metric['tp']} | {metric['accepted_negative']} |")
    lines += ['', '时间方案将七个完成上下文的频带统计按顺序三段平均，并保留全段峰值；频带均值加回一秒内的相对能量包络，共640输入、48隐藏、2输出，30866参数。该对照同时改变统计与一秒级聚合，不能将全部收益归因于时间顺序。静态方案为160→48→2，共7826参数，按上下文最大分数聚合。', '',
              '回放先对整条原始录音做一次因果DC，再提取完整一秒窗口、0.5秒步幅；每窗口七个Log-Mel上下文与场景前端相同。标签仍是整条录音弱标签，跨完整窗口做MIL，没有把每个短窗硬标为鸣笛。', '',
              '新的numpy_runtime.py仅依赖NumPy与标准库。静态导出保持已有BandMLP架构；时间模型使用独立架构，未接入部署run.py的后端配置，不能直接替换原model目录。所有方案都未提升默认，ARM时延与真实道路效果仍未知。', '',
              '本轮两项结合主组F1为0.6818/0.6308，旧模型继承门限为0.5691/0.5645；正确方向35/35→45/41，负例接受2/3→1/2。压力组F1为0.6512/0.6190，旧0.5968/0.5920；负例均由2/84降为1/84。单项改动没有稳定提升，不继续用测试集修改网络或门限。', '',
              '严格预算的结果不能省略：预定网格内，旧模型、仅时间及两者结合的≤5%逐来源验证门限都是1，测试召回为0；只证明全部拒绝，不能称为达标检测器，也不能据此断言连续门限上不存在工作点。主预算的低测试误接受不保证新道路来源同样低。', '',
              '模型选择仍受复用录音与共享父录音限制。两个固定种子、最多30轮、耐心8；场景与回放抽样使用独立随机流，避免有无回放改变后续场景抽样。', '',
              '复现：plan；prepare-replay --split train；train；evaluate；prepare-replay --split test；evaluate-replay；benchmark；verify；package。已有horn_detector_v1/horn_recovery_v1作为只读输入，新产物被忽略，不随Git发布。', '']
    if (OUTPUT / 'original-test.json').exists():
        original = read_json(OUTPUT / 'original-test.json')
        lines += ['## 原始整条录音检测诊断', '',
                  '冻结后复用304条测试录音（116正、188负），完整一秒窗口取最大分数，沿用波束验证门限。下表是整条录音检测F1，不是方向联合F1，也不是连续事件或FP/hour。', '',
                  '| 方案 | 检测F1 | TP | FP | FN |', '|---|---:|---:|---:|---:|']
        for name, item in original['models'].items():
            metric = item['metrics']['all']
            lines.append(f"| {name} | {metric['f1']:.4f} | {metric['tp']} | {metric['fp']} | {metric['fn']} |")
        lines += ['', '原始录音与阵列场景复用来源，不构成第二套独立测试集；长录音最大聚合会改变误接受机会。', '']
    if (OUTPUT / 'benchmark.json').exists():
        benchmarked = read_json(OUTPUT / 'benchmark.json')['results']
        lines += ['## PC 时延诊断', '',
                  '四窗预热、十二窗计时；包含NumPy SRP、候选波束、特征、分类和事件，排除采集/DC、相机、视觉和I/O。', '']
        for name, item in benchmarked.items():
            lines.append(f"- {name}：中位{item['median_seconds'] * 1000:.1f} ms，P90 {item['p90_seconds'] * 1000:.1f} ms。")
        lines += ['', '这是PC上的48000样本窗口诊断，不能外推为ARM实时证据，也不能据此声称新方案更快。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUTPUT / 'README.md', flush=True)


def evaluate_replay():
    """Check weak clip decisions after freezing beam-domain operating points."""
    check_plan()
    records, data = load_replay('test')
    frozen = read_json(OUTPUT / 'freeze.json')
    models = {name: BandDetector(OUTPUT / ('model-' + name)) for name in SCHEMES}
    models['baseline'] = SpectralDetector(PREVIOUS / 'spectral_model')
    result = {}
    labels = np.asarray([record['label'] for record in records], dtype=bool)
    for name, model in models.items():
        mode = 'static' if name == 'baseline' else model.mode
        probabilities = numpy_scores(model.weights, data[mode])
        if mode == 'static':
            probabilities = probabilities.max(axis=-1)
        clip_probabilities = np.asarray([probabilities[data['offsets'][index]:data['offsets'][index + 1]].max()
                                         for index in range(len(records))])
        gate = frozen['original_gate'] if name == 'baseline' else frozen['selected'][name]['gates']['primary']
        summaries = {}
        for domain in ('all', 'HornBase', 'ESC50'):
            mask = np.ones(len(records), dtype=bool) if domain == 'all' else np.asarray([record['dataset'] == domain for record in records])
            accepted = clip_probabilities[mask] >= gate['threshold']
            target = labels[mask]
            tp, fp, fn = int(np.sum(target & accepted)), int(np.sum(~target & accepted)), int(np.sum(target & ~accepted))
            summaries[domain] = dict(tp=tp, fp=fp, fn=fn, positives=int(target.sum()), negatives=int((~target).sum()),
                                     f1=2 * tp / max(1, 2 * tp + fp + fn))
        result[name] = dict(metrics=summaries, threshold=gate['threshold'], scores=clip_probabilities.tolist())
    write_json(OUTPUT / 'original-test.json', dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), models=result,
               scope='304 reused weakly labeled whole clips; maximum over complete seconds; inherited beam-calibrated gates, no new threshold selection; not event FP/hour'))
    print(json.dumps({name: value['metrics']['all'] for name, value in result.items()}), flush=True)


def benchmark():
    from ps.configuration import load_config
    from ps.pipeline import Pipeline
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    cases, _ = reference.load_cache('test-original')
    indices = [next(index for index, case in enumerate(cases) if case['dataset'] == domain and case['label'] == label)
               for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    plan = reference.check_plan()
    names = {cases[index][name] for index in indices for name in ('target', 'interferer')}
    sources = {record['file']: reference.waveform(record) for record in plan['sources'] if record['file'] in names}
    signals = [reference.synthesize(cases[index], sources, plan['profile'])[0] for index in indices]
    durations = {}
    for name in ('baseline', frozen['validation_selected']):
        model = SpectralDetector(PREVIOUS / 'spectral_model') if name == 'baseline' else BandDetector(OUTPUT / ('model-' + name))
        gate = frozen['original_gate'] if name == 'baseline' else frozen['selected'][name]['gates']['primary']
        config = load_config()
        config['detector'].update(threshold=gate['threshold'], margin=gate['margin'], backend='spectral')
        pipeline = Pipeline(config, model)
        times = []
        for repeat in range(4):
            for index, signal in enumerate(signals):
                started = time.perf_counter()
                event, _ = pipeline.analyze(dict(audio=signal, first=0, end=48000 + index * config['audio']['update_samples'], epoch=repeat, clipped=False))
                if repeat:
                    times.append(time.perf_counter() - started)
        durations[name] = dict(median_seconds=float(np.median(times)), p90_seconds=float(np.percentile(times, 90)),
                               windows=len(times), warmup_windows=len(signals), classifier=event['classifier'])
    write_json(OUTPUT / 'benchmark.json', dict(results=durations, numpy=np.__version__, python=sys.version,
             scope='PC NumPy SRP + beam + features + detector + event; 48000 samples, excludes capture/conditioning/camera/vision/I-O; no ARM/PL timing claim'))
    print(json.dumps(durations, indent=2), flush=True)


def package():
    """Bundle the validation-selected inference artifact without training dependencies."""
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    if not (OUTPUT / 'verification.json').exists():
        raise ValueError('Verify raw-wave inference before packaging')
    name = frozen['validation_selected']
    source = OUTPUT / ('model-' + name)
    destination = OUTPUT / 'inference-package'
    if destination.exists():
        raise ValueError('Inference package already exists; preserve the earlier artifact')
    destination.mkdir()
    (destination / 'model').mkdir()
    runtime = export_runtime()
    for original, relative in [(OUTPUT / 'numpy_runtime.py', 'numpy_runtime.py'), (source / 'weights.npz', 'model/weights.npz'),
                                (source / 'manifest.json', 'model/manifest.json'), (OUTPUT / 'README.md', 'EXPERIMENT.md')]:
        (destination / relative).write_bytes(original.read_bytes())
    metadata = frozen['selected'][name]
    operating = dict(format=1, selected=name, selected_on='validation', model='model', module='numpy_runtime.py',
                      model_tag=metadata['model_tag'], gate={key: metadata['gates']['primary'][key] for key in ('threshold', 'margin')},
                      context_contract='seven completed 40x24 board Log-Mel contexts from a conditioned 48000/48128-sample mono candidate beam',
                      primary_validation_budget=SETTINGS['primary_domain_acceptance'], board_validated=False,
                      deployment_cli_integrated=metadata['mode'] == 'static', firmware_included=False, license_scope='research/noncommercial')
    write_json(destination / 'operating-point.json', operating)
    (destination / 'README.md').write_text(
        '# NumPy band inference candidate\n\n'
        'Load BandDetector("model") from numpy_runtime.py. Call score(contexts) or score_waveform(conditioned_beam). '
        'The waveform is decoded, already DC-conditioned mono audio; this module does not apply a second DC filter. '
        'Use the frozen threshold in operating-point.json and preserve direction ambiguity.\n\n'
        'Only NumPy and Python standard libraries are needed for inference. Temporal models require explicit deployment '
        'integration; do not replace the existing model directory. No FPGA firmware, camera calibration, road validation '
        'or ARM timing proof is included. See EXPERIMENT.md for results and limitations. Research/noncommercial.\n', encoding='utf-8')
    files = {path.relative_to(destination).as_posix(): digest(path) for path in destination.rglob('*') if path.is_file()}
    write_json(destination / 'bundle.json', dict(format=1, files=files, model_tag=metadata['model_tag'], board_validated=False))
    archive = OUTPUT / 'band-inference-candidate.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in destination.rglob('*'):
            if path.is_file():
                stream.write(path, arcname=path.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as stream:
        for relative, checksum in files.items():
            if hashlib.sha256(stream.read(relative)).hexdigest() != checksum:
                raise ValueError('Inference archive checksum mismatch')
    runtime.BandDetector(source)
    print(json.dumps(dict(archive=str(archive), model_tag=metadata['model_tag'], files=len(files))), flush=True)


def verify():
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    value = read_json(OUTPUT / 'results.json')
    if value['freeze_sha256'] != digest(OUTPUT / 'freeze.json') or frozen['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Frozen evaluation provenance mismatch')
    generated = export_runtime()
    cases, data = reference.load_cache('test-original')
    plan = reference.check_plan()
    indices = [next(index for index, case in enumerate(cases) if case['dataset'] == domain and case['label'] == label)
               for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    names = {cases[index][name] for index in indices for name in ('target', 'interferer')}
    sources = {record['file']: reference.waveform(record) for record in plan['sources'] if record['file'] in names}
    raw = []
    for index in indices:
        audio = reference.synthesize(cases[index], sources, plan['profile'])[0]
        raw.append([features(reference.beamform(audio, reference.Bearing(**item),
                         dict(channels=14, diameter=0.24, sample_rate=48000, speed=343.0))[0]) for item in cases[index]['bearings']])
    replay = {}
    for name in SCHEMES:
        model = BandDetector(OUTPUT / ('model-' + name))
        runtime = generated.BandDetector(OUTPUT / ('model-' + name))
        error = max(abs(runtime.score(beam)['score'] - model.score(data['contexts'][index, candidate])['score'])
                    for index, beams in zip(indices, raw) for candidate, beam in enumerate(beams))
        if error > 1e-5:
            raise ValueError('Raw-wave standalone inference disagrees with the frozen frontend')
        replay[name] = dict(raw_beams=8, maximum_score_error=float(error), model_tag=model.metadata['model_tag'])
    for phase in value['protocols'].values():
        for result in phase.values():
            for budget in ('primary', 'strict'):
                for item in result[budget].values():
                    if item['all']['positives'] != 84 or item['all']['negatives'] != 84:
                        raise ValueError('Evaluation denominators changed')
    artifacts = {path.relative_to(OUTPUT).as_posix(): digest(path) for path in OUTPUT.rglob('*')
                 if path.is_file() and path.name != 'verification.json'}
    write_json(OUTPUT / 'verification.json', dict(result='passed', replay=replay, artifacts=artifacts,
                   old_inputs_unchanged=True, board_validated=False))
    print(json.dumps(dict(result='passed', replay=replay)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare-replay', 'train', 'evaluate', 'report', 'verify', 'evaluate-replay', 'benchmark', 'package'))
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    args = parser.parse_args()
    if args.command == 'plan':
        plan()
    elif args.command == 'prepare-replay':
        prepare_replay(args.split)
    elif args.command == 'train':
        train()
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'report':
        report()
    elif args.command == 'verify':
        verify()
    elif args.command == 'evaluate-replay':
        evaluate_replay()
    elif args.command == 'benchmark':
        benchmark()
    elif args.command == 'package':
        package()


if __name__ == '__main__':
    main()
