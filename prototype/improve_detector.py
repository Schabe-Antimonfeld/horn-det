"""Reproduce paired PCM beam distillation without tuning on test outcomes."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

PROTO = Path(__file__).resolve().parent
ROOT = PROTO.parent
sys.path.insert(0, str(PROTO / 'src'))
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(ROOT / 'deployment/pynq_z2'))

from horn.calibration import fractional_delay
from horn.config import Config
from horn.frontend import read_audio
from horn.geometry import circular_array
from horn.recovery import camera_rig, choose_gate, digest, joint_metrics, pcm_roundtrip, scene_plan, visible_target, write_json
from horn.surround import (AcousticEvidence, Association, Bearing, FrozenPanns, Rig, associate,
                           bearing_candidates, propagate_audio, route_cameras, select_bearing, wrap_angle)
from ps.dsp import Conditioning, band_features, beamform, features, spatial_map
from ps.detector import SpectralDetector
from simulate_surround import evaluate_match, observations

BASE = PROTO / 'outputs/horn_detector_v1'
OUTPUT = PROTO / 'outputs/horn_recovery_v1'
PROFILE = PROTO / 'profiles/horn_recovery.json'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure():
    import torch
    torch.set_num_threads(2)
    return torch


def fixed_module():
    spec = importlib.util.spec_from_file_location('recovery_cnn', ROOT / 'fpga/cnn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_plan():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = load_json(PROFILE)
    records = load_json(BASE / 'clips.json')
    plans = {split + '-' + protocol: scene_plan(records, split, protocol, profile['seed'])
             for split in ('train', 'validation', 'test') for protocol in profile['protocols']
             if split != 'train' or protocol == 'original'}
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in [PROFILE, BASE / 'clips.json',
              BASE / 'tiny-horn.pt', BASE / 'panns-horn.pt', PROTO / 'profiles/gc2093_single.json']}
    for path in (BASE / 'fpga_model').glob('*.mem'):
        frozen[str(path.relative_to(ROOT))] = digest(path)
    used = {case[name] for cases in plans.values() for case in cases for name in ('target', 'interferer')}
    sources = [record for record in records if record['file'] in used]
    for record in sources:
        if digest(record['path']) != record['sha256']:
            raise ValueError('Source checksum mismatch: ' + record['file'])
    grouping = {}
    for record in sources:
        grouping.setdefault(record['group'], set()).add(record['split'])
    if any(len(splits) != 1 for splits in grouping.values()):
        raise ValueError('A declared recording group crosses source splits')
    value = dict(format=1, profile=profile, frozen=frozen, plans=plans, sources=sources,
                 limits='Inherited HornBase scenario split shares parent recordings; reused explored sources, not a new blind road test')
    path = OUTPUT / 'plan.json'
    if path.exists() and load_json(path) != value:
        raise ValueError('Frozen plan changed; choose a new experiment version')
    write_json(path, value)
    print(json.dumps({key: len(cases) for key, cases in plans.items()}), flush=True)


def check_plan():
    plan = load_json(OUTPUT / 'plan.json')
    for name, checksum in plan['frozen'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Frozen input changed: ' + name)
    return plan


def waveform(record):
    if digest(record['path']) != record['sha256']:
        raise ValueError('Source changed: ' + record['file'])
    signal = read_audio(record['path'], sample_rate=48000, channel=0)[:, 0]
    if len(signal) < 48000:
        raise ValueError('Sources need a complete second')
    starts = sorted(set(range(0, len(signal) - 48000 + 1, 24000)) | {len(signal) - 48000})
    start = max(starts, key=lambda offset: np.mean(signal[offset:offset + 48000] ** 2))
    signal = signal[start:start + 48000].copy()
    signal -= signal.mean()
    rms = np.sqrt(np.mean(signal ** 2))
    if rms <= 0:
        raise ValueError('Silent source')
    return signal / rms, start


def synthesize(case, sources, profile, stress=False, condition=True):
    rig, config = camera_rig(), Config(microphones=14, diameter=0.24)
    coordinates = circular_array(config)
    points = [radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
              for angle, radius in ((case['angle'], case['radius']), (case['other'], case['radius'] + 3))]
    components = [propagate_audio(sources[name][0], np.r_[point, rig.horn_height - rig.height], coordinates, config)[0]
                  for name, point in zip((case['target'], case['interferer']), points)]
    components[1] *= 10 ** (-case['sir_db'] / 20)
    rng = np.random.default_rng(case['noise_seed'])
    noise = rng.normal(0, 10 ** (-profile['noise_snr_db'] / 20), components[0].shape)
    augmented = stress or (case['split'] == 'train' and case['pair'] % 2 == 1)
    if augmented:
        settings = profile['stress']
        reflection = np.zeros_like(noise)
        for name, angle, level in ((case['target'], case['angle'] + 65, 1),
                                   (case['interferer'], case['other'] - 70, 10 ** (-case['sir_db'] / 20))):
            reflected = case['radius'] * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0])
            reflected[2] = rig.horn_height - rig.height
            echo = fractional_delay(sources[name][0], 0.025 * config.sample_rate)
            reflection += settings['reflection_gain'] * level * propagate_audio(echo, reflected, coordinates, config)[0]
        noise += reflection
    requested = case['target_rms']
    if stress:
        requested *= 10 ** (profile['stress']['gain_db'][case['pair'] % 2] / 20)
    decoded, pcm = pcm_roundtrip(*components, noise, requested, profile['headroom'])
    if not condition:
        if augmented:
            raise ValueError('Raw PCM replay export is only defined for unaugmented scenes')
        return decoded, pcm, points, augmented
    audio = Conditioning(14).feed(decoded * 32768)
    if augmented:
        settings = profile['stress']
        gains = 10 ** (rng.uniform(-settings['channel_gain_db'], settings['channel_gain_db'], 14) / 20)
        delays = rng.uniform(-settings['residual_delay_samples'], settings['residual_delay_samples'], 14)
        audio = np.column_stack([fractional_delay(audio[:, channel], 2 + delays[channel]) * gains[channel]
                                 for channel in range(14)])
    return audio, pcm, points, augmented


def prepare(keys):
    configure()
    plan = check_plan()
    teacher = FrozenPanns(BASE / 'panns-horn.pt')
    from horn.models import TinyCNN
    torch = configure()
    old = TinyCNN().eval()
    old.load_state_dict(torch.load(BASE / 'tiny-horn.pt', map_location='cpu', weights_only=True)['state'])
    names = {case[name] for key in keys for case in plan['plans'][key.replace('stress-', 'test-')] for name in ('target', 'interferer')}
    sources = {record['file']: waveform(record) for record in plan['sources'] if record['file'] in names}
    settings = dict(channels=14, diameter=0.24, sample_rate=48000, speed=343.0)
    search = dict(azimuth_step=2, depression_step=3, block_directions=128)
    for key in keys:
        path = OUTPUT / (key + '.npz')
        if path.exists():
            print('Reusing', key, flush=True)
            load_cache(key)
            continue
        contexts, grids, rows, soft, raw = [], [], [], [], []
        began = time.monotonic()
        planned = plan['plans'][key.replace('stress-', 'test-')]
        for number, case in enumerate(planned, 1):
            audio, pcm, points, augmented = synthesize(case, sources, plan['profile'], stress=key.startswith('stress-'))
            started = time.perf_counter()
            grid = spatial_map(audio, settings, search)
            bearings = bearing_candidates(*grid, count=plan['profile']['candidate_count'])
            if len(bearings) != 2:
                raise ValueError('This paired cache expects two spatial proposals')
            beams = [beamform(audio, bearing, settings)[0] for bearing in bearings]
            values = np.asarray([features(beam) for beam in beams], dtype=np.float32)
            scores = teacher.score(beams)
            with torch.inference_mode():
                predictions = torch.softmax(old(torch.from_numpy(values.reshape(-1, 1, 40, 24))), dim=1)[:, 1]
            elapsed = time.perf_counter() - started
            rows.append(dict(case, bearings=[asdict(item) for item in bearings], pcm=pcm,
                             visible=visible_target(camera_rig(), case['angle'], case['radius']),
                             target_crop=sources[case['target']][1], interferer_crop=sources[case['interferer']][1],
                             augmented=augmented, processing_seconds=elapsed, points=[point.tolist() for point in points]))
            contexts.append(values)
            grids.append(grid[2].astype(np.float32))
            soft.append(scores)
            raw.append(predictions.numpy().reshape(2, -1).max(axis=1))
            if number % 20 == 0 or number == len(planned):
                print(key, number, '/', len(planned), 'seconds', round(time.monotonic() - began), flush=True)
        np.savez_compressed(path, contexts=np.asarray(contexts), grids=np.asarray(grids), teacher=np.asarray(soft),
                            old_float=np.asarray(raw), azimuths=grid[0], depressions=grid[1])
        write_json(OUTPUT / (key + '.json'), dict(plan_sha256=digest(OUTPUT / 'plan.json'),
                   cache_sha256=digest(path), rows=rows, synthesis_sha256=digest(Path(__file__))))


def load_cache(key):
    plan = check_plan()
    metadata = load_json(OUTPUT / (key + '.json'))
    if metadata['plan_sha256'] != digest(OUTPUT / 'plan.json') or metadata['cache_sha256'] != digest(OUTPUT / (key + '.npz')):
        raise ValueError('Cache provenance mismatch')
    planned = plan['plans'][key.replace('stress-', 'test-')]
    if len(metadata['rows']) != len(planned) or [{name: row[name] for name in case} for row, case in zip(metadata['rows'], planned)] != planned:
        raise ValueError('Cached scene plan mismatch')
    with np.load(OUTPUT / (key + '.npz'), allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data['contexts']) != len(metadata['rows']) or data['contexts'].shape[1:] != (2, 7, 40, 24):
        raise ValueError('Cache feature dimensions disagree with the frozen frontend')
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Cache contains nonfinite values')
    return metadata['rows'], data


def transform(contexts, frontend):
    values = np.asarray(contexts, dtype=np.float32)
    if frontend == 'centered_logmel':
        return values - values.mean(axis=(-2, -1), keepdims=True)
    if frontend != 'logmel':
        raise ValueError('Unsupported frontend')
    return values


def predict(model, contexts, frontend):
    torch = configure()
    values = transform(contexts, frontend)
    flat = values.reshape(-1, 1, 40, 24)
    output = []
    with torch.inference_mode():
        for start in range(0, len(flat), 128):
            output.extend(torch.softmax(model(torch.from_numpy(flat[start:start + 128])), dim=1)[:, 1].tolist())
    return np.asarray(output).reshape(values.shape[:-2]).max(axis=-1)


def validation_data():
    rows, contexts = [], []
    for protocol in check_plan()['profile']['protocols']:
        cases, data = load_cache('validation-' + protocol)
        rows += cases
        contexts.append(data['contexts'])
    return rows, np.concatenate(contexts)


def train():
    torch = configure()
    from horn.models import TinyCNN
    settings = check_plan()['profile']['training']
    cases, data = load_cache('train-original')
    validation, contexts = validation_data()
    history, candidates = [], []
    for frontend in check_plan()['profile']['frontends']:
        for seed in settings['seeds']:
            path = OUTPUT / f'{frontend}-{seed}.pt'
            selection_path = path.with_suffix('.json')
            if path.exists() and selection_path.exists():
                candidates.append(load_json(selection_path))
                continue
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = TinyCNN()
            model.load_state_dict(torch.load(BASE / 'tiny-horn.pt', map_location='cpu', weights_only=True)['state'])
            optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
            values = transform(data['contexts'], frontend)
            labels = np.asarray([case['label'] for case in cases], dtype=np.float32)
            domains = [np.flatnonzero(np.asarray([case['dataset'] == domain and case['label'] == label for case in cases]))
                       for domain in ('HornBase', 'ESC50') for label in (0, 1)]
            best, state, stale, selected = None, None, 0, None
            for epoch in range(settings['epochs']):
                model.train()
                losses = []
                for batch in range(max(1, len(cases) // settings['batch_scenes'])):
                    indices = np.concatenate([rng.choice(group, settings['batch_scenes'] // 4) for group in domains])
                    starts = np.unique(np.linspace(0, values.shape[2] - 1, settings['contexts_per_beam']).astype(int))
                    chosen = torch.from_numpy(values[indices][:, :, starts])
                    logits = model(chosen.reshape(-1, 1, 40, 24)).reshape(len(indices), 2, len(starts), 2)
                    differences = logits[..., 1] - logits[..., 0]
                    beam_logits = differences.max(dim=2).values
                    scene_logits = beam_logits.max(dim=1).values
                    hard = torch.nn.functional.binary_cross_entropy_with_logits(scene_logits, torch.from_numpy(labels[indices]))
                    teacher = torch.from_numpy(data['teacher'][indices].astype(np.float32))
                    soft = torch.nn.functional.binary_cross_entropy_with_logits(beam_logits, teacher)
                    ranking = scene_logits.sum() * 0
                    count = 0
                    for row, index in enumerate(indices):
                        errors = [180 if item['azimuth'] is None else abs(wrap_angle(item['azimuth'] - cases[index]['angle']))
                                  for item in cases[index]['bearings']]
                        target = int(np.argmin(errors))
                        other = 1 - target
                        if labels[index] and errors[target] < 10 and teacher[row, target] > teacher[row, other]:
                            ranking = ranking + torch.nn.functional.softplus(beam_logits[row, other] - beam_logits[row, target])
                            count += 1
                    loss = hard + settings['teacher_weight'] * soft + settings['ranking_weight'] * ranking / max(1, count)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                scores = predict(model, contexts, frontend)
                gate = choose_gate(validation, scores, 'validation')
                rank = (gate['constraint_met'], gate['macro_f1'], -gate['metrics']['negative_acceptance'])
                history.append(dict(frontend=frontend, seed=seed, epoch=epoch + 1, loss=float(np.mean(losses)), gate=gate))
                print(frontend, seed, epoch + 1, 'loss', round(np.mean(losses), 4), 'validation macro F1', round(gate['macro_f1'], 4), flush=True)
                if best is None or rank > best:
                    best, state, stale = rank, deepcopy(model.state_dict()), 0
                    selected = dict(frontend=frontend, seed=seed, epoch=epoch + 1, gate=gate, checkpoint=path.name)
                else:
                    stale += 1
                if stale >= settings['patience']:
                    break
            torch.save(dict(state=state, metadata=selected), path)
            selected['sha256'] = digest(path)
            write_json(selection_path, selected)
            candidates.append(selected)
            write_json(OUTPUT / 'history.json', history)
    selected = max(candidates, key=lambda item: (item['gate']['constraint_met'], item['gate']['macro_f1'],
                   -item['gate']['metrics']['negative_acceptance']))
    selection = dict(selected=selected, candidates=candidates, plan_sha256=digest(OUTPUT / 'plan.json'),
                     policy='validation domains/protocols only; two frontend mechanisms, two fixed seeds, no test selection')
    path = OUTPUT / 'selection.json'
    if path.exists() and load_json(path) != selection:
        raise ValueError('Selection is already frozen')
    write_json(path, selection)


def score_fixed(key, directory, frontend, name):
    cnn = fixed_module()
    model = cnn.load_fixed(directory)
    path = OUTPUT / (key + '-' + name + '.json')
    if path.exists():
        result = load_json(path)
        if result['model_tag'] != model['model_tag'] or result['cache_sha256'] != digest(OUTPUT / (key + '.npz')):
            raise ValueError('Integer prediction provenance mismatch')
        return result
    cases, data = load_cache(key)
    values = transform(data['contexts'], frontend)
    scores, clipped, errors, invalid = [], 0, 0, 0
    started = time.monotonic()
    for index, scene in enumerate(values):
        beam_scores = []
        unreliable = False
        for contexts in scene:
            codes, input_clips = cnn.input_codes(contexts, model['input_fraction'])
            if input_clips:
                errors += input_clips
                unreliable = True
                beam_scores.append(0.0)
                continue
            results = [cnn.fixed_forward(model, code) for code in codes]
            clipped += sum(result['flags'] for result in results)
            unreliable = unreliable or any(result['flags'] for result in results)
            beam_scores.append(max(result['probability'] for result in results) / 65536)
        invalid += int(unreliable)
        scores.append([0.0] * len(beam_scores) if unreliable else beam_scores)
        if (index + 1) % 40 == 0 or index + 1 == len(cases):
            print('Integer', name, key, index + 1, '/', len(cases), 'seconds', round(time.monotonic() - started), flush=True)
    result = dict(model_tag=model['model_tag'], frontend=frontend, scores=scores, activation_clipped_contexts=clipped,
                  input_clips=errors, invalid_scenes=invalid, cache_sha256=digest(OUTPUT / (key + '.npz')),
                  processing_seconds=time.monotonic() - started)
    write_json(path, result)
    return result


def baseline():
    rows, predictions = [], dict(old_integer=[], panns=[])
    for protocol in check_plan()['profile']['protocols']:
        key = 'validation-' + protocol
        cases, data = load_cache(key)
        rows += cases
        predictions['panns'].extend(data['teacher'].tolist())
        predictions['old_integer'].extend(score_fixed(key, BASE / 'fpga_model', 'logmel', 'old_integer')['scores'])
    result = {name: dict(selected=choose_gate(rows, scores, 'validation'),
                         inherited=joint_metrics(rows, scores, 0.65 if name == 'old_integer' else 0.2,
                                                 0.05 if name == 'old_integer' else 0))
              for name, scores in predictions.items()}
    write_json(OUTPUT / 'baseline-validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def train_spectral():
    """Try one fixed band-aware architecture before any test prediction."""
    torch = configure()
    if (OUTPUT / 'freeze.json').exists():
        raise ValueError('Do not train additional candidates after test authorization')
    if (OUTPUT / 'spectral_model/manifest.json').exists():
        SpectralDetector(OUTPUT / 'spectral_model')
        print('Reusing frozen spectral candidate', flush=True)
        return
    cases, data = load_cache('train-original')
    validation, raw = validation_data()
    values = band_features(data['contexts']).astype(np.float32)
    valid = torch.from_numpy(band_features(raw).astype(np.float32))
    teacher = torch.from_numpy(data['teacher'].astype(np.float32))
    labels = torch.tensor([case['label'] for case in cases], dtype=torch.float32)
    groups = [np.flatnonzero(np.asarray([case['dataset'] == domain and case['label'] == label for case in cases]))
              for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = torch.nn.Sequential(torch.nn.Linear(160, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2))
    settings = check_plan()['profile']['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    history, best, state, selected, stale = [], None, None, None, 0
    for epoch in range(settings['epochs']):
        model.train()
        losses = []
        for batch in range(len(cases) // settings['batch_scenes']):
            indices = np.concatenate([rng.choice(group, settings['batch_scenes'] // 4) for group in groups])
            chosen = torch.from_numpy(values[indices][:, :, [0, 3, 6]])
            logits = model(chosen)
            differences = logits[..., 1] - logits[..., 0]
            beams = differences.max(dim=2).values
            scene = beams.max(dim=1).values
            hard = torch.nn.functional.binary_cross_entropy_with_logits(scene, labels[indices])
            soft = torch.nn.functional.binary_cross_entropy_with_logits(beams, teacher[indices])
            ranking, count = scene.sum() * 0, 0
            for row, index in enumerate(indices):
                errors = [180 if item['azimuth'] is None else abs(wrap_angle(item['azimuth'] - cases[index]['angle']))
                          for item in cases[index]['bearings']]
                target = int(np.argmin(errors))
                if labels[index] and errors[target] < 10 and teacher[index, target] > teacher[index, 1 - target]:
                    ranking = ranking + torch.nn.functional.softplus(beams[row, 1 - target] - beams[row, target])
                    count += 1
            loss = hard + settings['teacher_weight'] * soft + settings['ranking_weight'] * ranking / max(1, count)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            scores = torch.softmax(model(valid), dim=-1)[..., 1].max(dim=2).values.numpy()
        gate = choose_gate(validation, scores, 'validation')
        rank = (gate['constraint_met'], gate['macro_f1'], -gate['metrics']['negative_acceptance'])
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), gate=gate))
        print('BandMLP', epoch + 1, 'validation macro F1', round(gate['macro_f1'], 4), flush=True)
        if best is None or rank > best:
            best, state, stale, selected = rank, deepcopy(model.state_dict()), 0, dict(epoch=epoch + 1, gate=gate)
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    model.load_state_dict(state)
    model.eval()
    directory = OUTPUT / 'spectral_model'
    directory.mkdir(exist_ok=True)
    np.savez(directory / 'weights.npz', weight0=state['0.weight'].numpy(), bias0=state['0.bias'].numpy(),
             weight1=state['2.weight'].numpy(), bias1=state['2.bias'].numpy())
    checksum = digest(directory / 'weights.npz')
    metadata = dict(format=1, architecture='BandMLP-7826', model_tag=int(checksum[:8], 16), input_shape=[40, 24],
                    sha256={'weights.npz': checksum}, selection=selected,
                    provenance=dict(frontend='logmel', feature_transform='centered band mean/std/absolute-delta/range',
                                    selection_split='validation', plan_sha256=digest(OUTPUT / 'plan.json'),
                                    license_scope='research/noncommercial', evidence='synthetic beam distillation; no board/road validation'))
    write_json(directory / 'manifest.json', metadata)
    runtime = SpectralDetector(directory)
    with torch.inference_mode():
        reference = torch.softmax(model(valid), dim=-1)[..., 1].max(dim=2).values.numpy()
    actual = np.asarray([[runtime.score(beam)['score'] for beam in scene] for scene in raw])
    error = float(np.max(np.abs(actual - reference)))
    if error > 1e-5:
        raise ValueError('NumPy spectral inference disagrees with Torch')
    write_json(OUTPUT / 'spectral-validation.json', dict(history=history, selected=selected, numpy_maximum_error=error))


def export():
    torch = configure()
    from horn.models import TinyCNN
    selection = load_json(OUTPUT / 'selection.json')['selected']
    checkpoint = OUTPUT / selection['checkpoint']
    if digest(checkpoint) != selection['sha256']:
        raise ValueError('Selected checkpoint changed')
    model = TinyCNN().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state'])
    rows, values = validation_data()
    calibration = transform(values, selection['frontend'])
    examples = [calibration[next(index for index, case in enumerate(rows) if case['dataset'] == domain and case['label'] == label), 0, 0]
                for domain in ('HornBase', 'ESC50') for label in (1, 0)]
    cnn = fixed_module()
    directory = OUTPUT / 'fpga_model'
    if directory.exists():
        metadata = cnn.check_export(directory)
        if metadata['provenance'].get('checkpoint_sha256') != digest(checkpoint):
            raise ValueError('Existing export is from another checkpoint')
    else:
        metadata = cnn.export_model(model, np.concatenate((examples, calibration.reshape(-1, 40, 24))), directory, 4,
                    dict(frontend=selection['frontend'], checkpoint_sha256=digest(checkpoint),
                         plan_sha256=digest(OUTPUT / 'plan.json'), calibration_split='validation',
                         training='weak scene MIL + frozen PANNs beam distillation + teacher-agreed direction ranking',
                         license_scope='research/noncommercial; inherited HornBase and ESC50 source restrictions',
                         evidence='public real waveforms in synthetic array; no road/board accuracy claim'))
    log = cnn.verify_rtl(directory)
    write_json(OUTPUT / 'rtl.json', dict(model_tag=metadata['model_tag'], vectors=4, output=log,
                manifest_sha256=digest(directory / 'manifest.json')))
    print('Exported and RTL verified', metadata['model_tag'], flush=True)


def freeze():
    check_plan()
    if not (OUTPUT / 'rtl.json').exists():
        raise ValueError('Verify exported RTL before freezing integer operating points')
    if (OUTPUT / 'freeze.json').exists():
        print('Operating points already frozen', flush=True)
        return
    selection = load_json(OUTPUT / 'selection.json')['selected']
    spectral = SpectralDetector(OUTPUT / 'spectral_model')
    rows, predictions = [], {name: [] for name in ('old_integer', 'panns', 'recovered_integer', 'spectral_numpy')}
    inherited, provenance = {}, {}
    for protocol in check_plan()['profile']['protocols']:
        key = 'validation-' + protocol
        cases, data = load_cache(key)
        rows += cases
        predictions['panns'].extend(data['teacher'].tolist())
        predictions['spectral_numpy'].extend([[spectral.score(beam)['score'] for beam in scene] for scene in data['contexts']])
        for name, directory, frontend in [('old_integer', BASE / 'fpga_model', 'logmel'),
                                         ('recovered_integer', OUTPUT / 'fpga_model', selection['frontend'])]:
            result = score_fixed(key, directory, frontend, name)
            predictions[name].extend(result['scores'])
            provenance[name] = dict(model_tag=result['model_tag'], frontend=frontend,
                                    manifest_sha256=digest(directory / 'manifest.json'))
    gates = {name: choose_gate(rows, scores, 'validation') for name, scores in predictions.items()}
    inherited['old_integer'] = joint_metrics(rows, predictions['old_integer'], 0.65, 0.05)
    inherited['panns'] = joint_metrics(rows, predictions['panns'], 0.2, 0.0)
    provenance['spectral_numpy'] = dict(model_tag=spectral.metadata['model_tag'], frontend='logmel',
                                        manifest_sha256=digest(OUTPUT / 'spectral_model/manifest.json'))
    value = dict(plan_sha256=digest(OUTPUT / 'plan.json'), selection_sha256=digest(OUTPUT / 'selection.json'),
                 models=provenance, gates=gates, inherited_validation=inherited,
                 policy='shared validation-selected integer gates across original and visible; all test predictions deferred; no promotion from test scores')
    write_json(OUTPUT / 'freeze.json', value)
    for name, directory in [('recovered_integer', OUTPUT / 'fpga_model'), ('spectral_numpy', OUTPUT / 'spectral_model')]:
        config = load_json(ROOT / 'deployment/pynq_z2/config/default.json')
        config['detector'].update(model_directory=str(directory), threshold=gates[name]['threshold'], margin=gates[name]['margin'],
                                  backend='spectral' if name == 'spectral_numpy' else 'integer')
        write_json(OUTPUT / (name + '-config.json'), config)
    print(json.dumps(gates, indent=2), flush=True)


def detection_metrics(cases, scores, threshold):
    labels = np.asarray([case['label'] for case in cases], dtype=bool)
    accepted = np.max(scores, axis=1) >= threshold
    tp, fp, fn = int(np.sum(labels & accepted)), int(np.sum(~labels & accepted)), int(np.sum(labels & ~accepted))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=int(np.sum(~labels & ~accepted)), precision=precision, recall=recall,
                f1=2 * precision * recall / max(1e-12, precision + recall))


def association_metrics(cases, data, scores, gate, rig):
    cameras = rig.cameras()
    counts = dict(positive_count=0, visible_positive=0, correct_unique=0, incorrect_unique=0, ambiguous=0,
                  unmatched=0, accepted_correct_outside=0, target_retained=0, missed_collinear_rejected=0,
                  missed_collinear_trials=0)
    details = []
    for index, (case, probabilities) in enumerate(zip(cases, scores)):
        if not case['label']:
            continue
        counts['positive_count'] += 1
        counts['visible_positive'] += int(visible_target(rig, case['angle'], case['radius']))
        bearings = [Bearing(**item) for item in case['bearings']]
        decision = select_bearing(bearings, probabilities, gate['threshold'], gate['margin'])
        bearing = bearings[decision['index']] if decision['valid'] else None
        center, other = np.asarray(case['points'])
        detections, truth = observations(rig, cameras, [center, other], [np.zeros(2), np.zeros(2)], 0.5,
                                         np.random.default_rng(case['visual_seed']))
        if bearing is not None:
            routed = route_cameras(rig, bearing.azimuth, cameras=cameras)
            detections = [item for item in detections if item.camera in routed]
            grid = (data['azimuths'], data['depressions'], data['grids'][index])
            evidence = AcousticEvidence(bearing, grid)
            arrival = 0.5 + np.linalg.norm(np.r_[center, rig.horn_height - rig.height]) / 343
            result = associate(rig, bearing.azimuth, arrival, detections, Association(), cameras=cameras, evidence=evidence)
            evaluation = evaluate_match(result, truth)
            counts['correct_unique'] += int(evaluation['correct_unique'])
            counts['incorrect_unique'] += int(evaluation['incorrect_unique'])
            counts['target_retained'] += int(evaluation['target_retained'])
            if result['state'] in ('ambiguous', 'unmatched'):
                counts[result['state']] += 1
            counts['accepted_correct_outside'] += int(not visible_target(rig, case['angle'], case['radius']) and bearing.azimuth is not None and abs(wrap_angle(bearing.azimuth - case['angle'])) < 10)
            if bearing.azimuth is not None and abs(wrap_angle(bearing.azimuth - case['angle'])) < 10:
                hidden, _ = observations(rig, cameras, [center, center * 1.7], [np.zeros(2), np.zeros(2)], 0.5,
                                          np.random.default_rng(case['visual_seed']), drop_target=True)
                hidden = [item for item in hidden if item.camera in routed]
                pressure = associate(rig, bearing.azimuth, arrival, hidden, Association(), cameras=cameras, evidence=evidence)
                counts['missed_collinear_trials'] += 1
                counts['missed_collinear_rejected'] += int(pressure['state'] != 'unique')
            details.append(dict(id=case['id'], state=result['state'], evaluation=evaluation))
        else:
            counts['unmatched'] += 1
    return dict(counts, details=details)


def evaluate(stress=False):
    torch = configure()
    frozen = load_json(OUTPUT / 'freeze.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json') or frozen['selection_sha256'] != digest(OUTPUT / 'selection.json'):
        raise ValueError('Frozen selection changed')
    selection = load_json(OUTPUT / 'selection.json')['selected']
    from horn.models import TinyCNN
    model = TinyCNN().eval()
    model.load_state_dict(torch.load(OUTPUT / selection['checkpoint'], map_location='cpu', weights_only=True)['state'])
    reports = {}
    for protocol in check_plan()['profile']['protocols']:
        key = ('stress-' if stress else 'test-') + protocol
        cases, data = load_cache(key)
        predictions = dict(panns=data['teacher'])
        spectral = SpectralDetector(OUTPUT / 'spectral_model')
        if digest(OUTPUT / 'spectral_model/manifest.json') != frozen['models']['spectral_numpy']['manifest_sha256']:
            raise ValueError('Frozen spectral model changed')
        predictions['spectral_numpy'] = np.asarray([[spectral.score(beam)['score'] for beam in scene] for scene in data['contexts']])
        integer = {}
        for name, directory, frontend in [('old_integer', BASE / 'fpga_model', 'logmel'),
                                         ('recovered_integer', OUTPUT / 'fpga_model', selection['frontend'])]:
            if digest(directory / 'manifest.json') != frozen['models'][name]['manifest_sha256']:
                raise ValueError('Frozen model manifest changed')
            integer[name] = score_fixed(key, directory, frontend, name)
            predictions[name] = np.asarray(integer[name]['scores'])
        float_scores = predict(model, data['contexts'], selection['frontend'])
        result = dict(count=len(cases), positives=sum(case['label'] for case in cases),
                      visible_positives=sum(case['label'] and case['visible'] for case in cases), models={})
        for name, scores in predictions.items():
            gate = frozen['gates'][name]
            domains = {}
            for dataset in ('HornBase', 'ESC50'):
                mask = np.asarray([case['dataset'] == dataset for case in cases])
                domains[dataset] = joint_metrics([case for case in cases if case['dataset'] == dataset], scores[mask], gate['threshold'], gate['margin'])
            main = joint_metrics(cases, scores, gate['threshold'], gate['margin'])
            no_gate = joint_metrics(cases, scores, 0, 0)
            item = dict(joint=main, detection=detection_metrics(cases, scores, gate['threshold']), domains=domains,
                        unconditionally_ranked_correct=no_gate['tp'],
                        wide_camera=association_metrics(cases, data, scores, gate, camera_rig()),
                        old_ring_camera=association_metrics(cases, data, scores, gate, Rig()),
                        gate=dict(threshold=gate['threshold'], margin=gate['margin']))
            if name in integer:
                item['integer_health'] = {key: value for key, value in integer[name].items() if key != 'scores'}
            if name == 'recovered_integer':
                item['probability_mae'] = float(np.abs(scores - float_scores).mean())
                item['maximum_probability_error'] = float(np.abs(scores - float_scores).max())
                item['float_at_integer_gate'] = joint_metrics(cases, float_scores, gate['threshold'], gate['margin'])
            result['models'][name] = item
        reports[protocol] = result
    write_json(OUTPUT / ('stress-results.json' if stress else 'results.json'), dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), protocols=reports,
               limits=check_plan()['limits'], evidence='synthetic array/camera/boxes from weakly labeled real clips; not FP/hour, live latency, or road validation'))
    report()


def report():
    result = load_json(OUTPUT / 'results.json')
    frozen = load_json(OUTPUT / 'freeze.json')
    selection = load_json(OUTPUT / 'selection.json')['selected']
    lines = ['# PCM 波束蒸馏与广角覆盖恢复实验', '', '日期：2026-10-21。科研/非商业，CPU 离线仿真。', '',
             '模型、前端和统一门限仅按验证来源选定并在两套测试评分前冻结。来源沿用已有划分且曾被探索，不是新的盲测；HornBase 场景分组仍共享父录音。', '',
             '原协议包含相机视野外目标；可见协议在合成前共同旋转目标和干扰，保留相对夹角、来源、SIR、RMS、噪声种子及距离，不按结果筛选。有限阵列采样使旋转后音频不同。', '',
             '每协议 84 个弱标签鸣笛目标和 84 个负例，14 条正来源复用。数字 RMS 不等于 SPL，合成球面直达场、假镜头、理想框不能证明实际归属。', '',
             f"选定前端 `{selection['frontend']}`，种子 {selection['seed']}，第 {selection['epoch']} 轮，模型标签 {frozen['models']['recovered_integer']['model_tag']}。", '',
             '| 协议 | 模型 | 联合 F1 | 正确接受方向/84 | 指错方向 | 负例接受/84 | 候选命中/84 | 广角正确唯一 | 原环阵相机正确唯一 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for protocol, values in result['protocols'].items():
        for name, item in values['models'].items():
            metric = item['joint']
            lines.append(f"| {protocol} | {name} | {metric['f1']:.4f} | {metric['tp']} | {metric['wrong_direction']} | {metric['accepted_negative']} | {metric['candidate_hits']} | {item['wide_camera']['correct_unique']} | {item['old_ring_camera']['correct_unique']} |")
    if (OUTPUT / 'stress-results.json').exists():
        pressure = load_json(OUTPUT / 'stress-results.json')
        lines += ['', '| 压力协议 | 模型 | 联合 F1 | 正确方向/84 | 负例接受/84 |', '|---|---|---:|---:|---:|']
        for protocol, values in pressure['protocols'].items():
            for name, item in values['models'].items():
                metric = item['joint']
                lines.append(f"| {protocol} | {name} | {metric['f1']:.4f} | {metric['tp']} | {metric['accepted_negative']} |")
    lines += ['', '联合 TP 要求接受且方向误差严格小于 10°；接受错误方向计 FP+FN。未输出、拍不到和歧义均留在正例分母中。上表相机唯一结果仅为几何诊断，部署仍强制保留实际归属歧义。', '',
              '旧 Tiny 使用同样验证选择准则重新校准，但预定门限网格内没有满足负例接受率≤10%的点，freeze.json 明确记录 constraint_met=false；旧继承门限的验证表现也保留，未覆盖旧文件。PANNs 为冻结 PC 教师/对照，未接入板端。', '',
              '方案判断：推荐继续验证 BandMLP 的板端研究路线，不提升为已验收默认值；本次 Tiny CNN 重训未成功，不替换旧模型。BandMLP 保留频带位置、共 7826 参数，用 NumPy 推理；其更低误接受伴随较低召回，不能宣称全面超过 PANNs。', '',
              '可见协议将目标可见数从 46/84 提升到 84/84；PANNs 的单广角正确几何唯一数从 17/84 提高到 26/84，原环形相机在两协议均为 26/84。每个协议内两种相机使用相同声学分数；协议之间旋转音频。该诊断支持覆盖差异是旧方案优势的一部分；分类拒绝、角度误差和宽峰仍会阻止唯一关联。', '',
              'PANNs 在原协议 HornBase 负例上接受 15/36，而 ESC50 为 3/48，说明教师存在明显来源差异。BandMLP 原协议只正确接受 35/84 个鸣笛方向，仍有大量拒绝。不能把低误接受当作高召回。', '',
              '训练保持 1834 参数 RTL 架构，比较原始 Log-Mel 和去除每个完成上下文全局均值的 Log-Mel；使用弱场景多实例损失、教师波束软标签和仅在教师赞同且候选命中时启用的方向排序。错误波束仍可能含鸣笛，未硬标为干净负例。', '',
              '一半训练声场加入侧向单次反射、通道增益及微小时延误差；这是域随机化代理，没有模拟完整道路混响。验证/测试主组不含该扰动。', '',
              '压力组在测试前预定 ±12 dB 目标数字级别、反射增益 0.25、每通道 ±1 dB 与 ±0.15 样本残差，PCM 峰值余量仍有效；不继续改模型或门限。压力组可能改善部分波束分数，不能据此保证任意噪声/混响下稳健。', '',
              '4 个跨来源/类别验证向量已做整数/RTL 对拍；详细概率量化差、饱和统计、各来源指标、相机歧义/未匹配数和同向漏检压力诊断见 results.json。', '',
              '同向漏框压力诊断只在正确接受的方向上追加一辆更远的同向车并隐藏目标框，分母是该条件子组；不是独立无偏测试。主表相机比较保持同一音频与声学分数。', '',
              '停止继续调参：下一步应采集真实同步阵列、独立道路背景及实测镜头/曝光/地面控制点。仍缺连续事件标签、移动声源、FP/hour、ARM/PL 时延、真实视觉模型和上板证据。', '',
              '重放：在仓库根目录依次运行 prototype/improve_detector.py 的 plan、prepare --split train、prepare --split validation、baseline、train、train-spectral、export、freeze、prepare --split test、evaluate、prepare --split test --stress、evaluate --stress、benchmark、verify。使用既有 Python 环境；旧实验目录只读。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:17]), flush=True)


def benchmark():
    from ps.configuration import load_config
    from ps.detector import Detector
    from ps.pipeline import Pipeline
    cases, _ = load_cache('test-original')
    indices = [next(index for index, case in enumerate(cases) if case['dataset'] == domain and case['label'] == label)
               for domain in ('HornBase', 'ESC50') for label in (1, 0)]
    plan = check_plan()
    names = {cases[index][name] for index in indices for name in ('target', 'interferer')}
    sources = {record['file']: waveform(record) for record in plan['sources'] if record['file'] in names}
    audio = [synthesize(cases[index], sources, plan['profile'])[0] for index in indices]
    result = {}
    for name, detector in [('spectral_numpy', SpectralDetector(OUTPUT / 'spectral_model')),
                           ('recovered_integer', Detector(OUTPUT / 'fpga_model'))]:
        config = load_config(OUTPUT / (name + '-config.json'))
        pipeline = Pipeline(config, detector)
        durations = []
        for repeat in range(4):
            for signal in audio:
                started = time.perf_counter()
                event, _ = pipeline.analyze(dict(audio=signal, first=0, end=48000, epoch=repeat, clipped=False))
                if repeat:
                    durations.append(time.perf_counter() - started)
        result[name] = dict(median_seconds=float(np.median(durations)), p90_seconds=float(np.percentile(durations, 90)),
                            measured_windows=len(durations), warmup_windows=len(audio))
    write_json(OUTPUT / 'benchmark.json', dict(results=result, numpy=np.__version__, python=sys.version,
                 scope='PC only; completed 48000-sample window; full NumPy SRP + beams + features + classifier + event, excludes capture/conditioning/camera/vision/I-O; not ARM or PL timing'))
    print(json.dumps(result, indent=2), flush=True)


def verify():
    plan = check_plan()
    frozen = load_json(OUTPUT / 'freeze.json')
    if frozen['selection_sha256'] != digest(OUTPUT / 'selection.json'):
        raise ValueError('Frozen selection changed')
    count = 0
    for split in ('train', 'validation', 'test', 'stress'):
        for protocol in plan['profile']['protocols']:
            if split == 'train' and protocol != 'original':
                continue
            rows, data = load_cache(split + '-' + protocol)
            count += len(rows)
            if any(row['pcm']['clipped_samples'] or row['pcm']['peak'] > 0.9001 for row in rows):
                raise ValueError('A scene violates the PCM headroom contract')
            if protocol == 'visible' and any(not row['visible'] for row in rows):
                raise ValueError('Visible protocol contains an out-of-view target')
    for filename in ('results.json', 'stress-results.json'):
        result = load_json(OUTPUT / filename)
        if result['freeze_sha256'] != digest(OUTPUT / 'freeze.json'):
            raise ValueError('Result refers to a different frozen selection')
        for protocol, value in result['protocols'].items():
            if value['count'] != 168 or value['positives'] != 84:
                raise ValueError('Test denominator changed')
            for name, item in value['models'].items():
                for geometry in ('wide_camera', 'old_ring_camera'):
                    counts = item[geometry]
                    if sum(counts[key] for key in ('correct_unique', 'incorrect_unique', 'ambiguous', 'unmatched')) != 84:
                        raise ValueError('Association outcomes do not partition positives')
    rtl = load_json(OUTPUT / 'rtl.json')
    if rtl['manifest_sha256'] != frozen['models']['recovered_integer']['manifest_sha256'] or 'PASS CNN' not in rtl['output']:
        raise ValueError('RTL proof does not match frozen weights')
    SpectralDetector(OUTPUT / 'spectral_model')
    for name, directory in [('old_integer', BASE / 'fpga_model'), ('recovered_integer', OUTPUT / 'fpga_model'),
                             ('spectral_numpy', OUTPUT / 'spectral_model')]:
        if digest(directory / 'manifest.json') != frozen['models'][name]['manifest_sha256']:
            raise ValueError('Frozen model manifest changed')
        if name != 'spectral_numpy':
            fixed_module().check_export(directory)
    checksums = {path.relative_to(OUTPUT).as_posix(): digest(path) for path in OUTPUT.rglob('*')
                 if path.is_file() and path.name != 'verification.json'}
    write_json(OUTPUT / 'verification.json', dict(result='passed', cached_scenes=count, artifacts=checksums,
                   frozen_inputs_unchanged=True, denominators_preserved=True, no_board_or_road_claim=True))
    print(json.dumps(dict(result='passed', cached_scenes=count, artifacts=len(checksums))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'prepare', 'baseline', 'train', 'train-spectral', 'export', 'freeze', 'evaluate', 'report', 'benchmark', 'verify'])
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    args = parser.parse_args()
    if args.command == 'plan':
        freeze_plan()
    elif args.command == 'prepare':
        if args.stress and args.split != 'test':
            raise ValueError('Stress is a frozen test diagnostic, not a training option')
        if args.split == 'test' and not (OUTPUT / 'freeze.json').exists():
            raise ValueError('Freeze integer gates before scoring either test protocol')
        prepare([('stress' if args.stress else args.split) + '-' + protocol for protocol in check_plan()['profile']['protocols']
                 if args.split != 'train' or protocol == 'original'])
    elif args.command == 'train':
        train()
    elif args.command == 'baseline':
        baseline()
    elif args.command == 'train-spectral':
        train_spectral()
    elif args.command == 'export':
        export()
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'evaluate':
        evaluate(args.stress)
    elif args.command == 'report':
        report()
    elif args.command == 'benchmark':
        benchmark()
    elif args.command == 'verify':
        verify()


if __name__ == '__main__':
    main()
