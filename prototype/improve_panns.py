"""Compare training-only hard negatives and category-aware frozen PANNs heads."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))
import improve_detector as reference
from horn.recovery import digest, write_json, camera_rig
from horn.calibration import remove_dc
from horn.surround import FrozenPanns, wrap_angle
from ps.dsp import beamform

BASE = reference.BASE
PREVIOUS = reference.OUTPUT
OUTPUT = PROTO / 'outputs/panns_rejection_v1'
CLASSES = ('car_horn', 'background', 'church_bells', 'crying_baby', 'dog', 'engine',
           'helicopter', 'rain', 'siren', 'thunderstorm', 'train', 'wind')
SETTINGS = dict(seeds=[0, 17], epochs=60, patience=10, learning_rate=0.001, weight_decay=0.001,
                batch_scenes=8, hard_fraction=0.25, hard_weight=2.0, replay_weight=0.5,
                category_weight=0.5, ranking_weight=0.1,
                thresholds=[float(value) for value in np.linspace(0.05, 1.0, 20)],
                margins=[0.0, 0.05, 0.1, 0.2], maximum_domain_acceptance=0.1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure():
    return reference.configure()


def plan():
    reference.check_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [BASE / 'panns-horn.pt', BASE / 'panns-embeddings.npz', BASE / 'clips.json',
             PREVIOUS / 'plan.json', PREVIOUS / 'freeze.json', PROTO / 'improve_detector.py',
             PROTO / 'src/horn/pretrained.py', PROTO / 'src/horn/surround.py']
    value = dict(format=1, settings=SETTINGS, classes=list(CLASSES),
                 frozen={str(path.relative_to(PROTO.parent)): digest(path) for path in paths},
                 training='frozen encoder; binary hard-negative head versus category-aware head; both use scene MIL and original-clip replay',
                 selection='validation shared gate across original/visible; negative horn-presence acceptance <= 10% in each domain, never hidden by direction margin',
                 evidence='reused explored public recordings and synthetic arrays, not a new blind road test; inherited parent-recording overlap',
                 licenses='research/noncommercial; original HornBase/ESC50 restrictions retained')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('This experiment plan is already frozen')
    write_json(path, value)
    print('Frozen head experiment plan', flush=True)


def check_plan():
    value = read_json(OUTPUT / 'plan.json')
    if value['settings'] != SETTINGS or value['classes'] != list(CLASSES):
        raise ValueError('Settings differ from the frozen plan')
    for name, checksum in value['frozen'].items():
        if digest(PROTO.parent / name) != checksum:
            raise ValueError('Frozen input changed: ' + name)
    return value


def network():
    configure()
    model = FrozenPanns(BASE / 'panns-horn.pt')
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model


def initial_head(original, count):
    """Preserve the original horn probability when splitting the nonhorn class."""
    torch = configure()
    if count < 2:
        raise ValueError('Need horn and at least one nonhorn class')
    head = torch.nn.Linear(512, count)
    with torch.no_grad():
        head.weight[0].copy_(original.weight[1])
        head.bias[0].copy_(original.bias[1])
        for index in range(1, count):
            head.weight[index].copy_(original.weight[0])
            head.bias[index].copy_(original.bias[0] - np.log(count - 1))
    return head


def horn_margin(logits):
    torch = configure()
    return logits[..., 0] - torch.logsumexp(logits[..., 1:], dim=-1)


def scores(head, values):
    torch = configure()
    with torch.inference_mode():
        margin = horn_margin(head(torch.as_tensor(values, dtype=torch.float32)))
        probabilities = torch.sigmoid(margin.double()).numpy()
    return np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)


def outcomes(cases, probabilities, threshold, margin):
    """Count negative horn events even when their direction remains ambiguous."""
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(cases), 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Each scene needs two finite candidate probabilities')
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError('Operating gates must lie in [0, 1]')
    labels = np.asarray([bool(case['label']) for case in cases])
    chosen = values.argmax(axis=1)
    maximum = values.max(axis=1)
    difference = np.abs(values[:, 0] - values[:, 1])
    audible = maximum >= threshold
    valid = audible & (difference >= margin) & (difference > 0)
    correct = np.asarray([[item['azimuth'] is not None and abs(wrap_angle(item['azimuth'] - case['angle'])) < 10
                           for item in case['bearings']] for case in cases])
    hits = correct[np.arange(len(cases)), chosen]
    tp = int(np.sum(labels & valid & hits))
    wrong = int(np.sum(labels & valid & ~hits))
    negative = int(np.sum(~labels & audible))
    fn = int(np.sum(labels)) - tp
    fp = wrong + negative
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    detection_tp = int(np.sum(labels & audible))
    return dict(tp=tp, fp=fp, fn=fn, tn=int(np.sum(~labels & ~audible)), wrong_direction=wrong,
                positives=int(labels.sum()), negatives=int((~labels).sum()), accepted_negative=negative,
                negative_acceptance=negative / max(1, int((~labels).sum())),
                accepted_positive=detection_tp, ambiguous_positive=int(np.sum(labels & audible & ~valid)),
                ambiguous_negative=int(np.sum(~labels & audible & ~valid)), candidate_hits=int(np.sum(labels & correct.any(axis=1))),
                precision=precision, recall=recall, f1=2 * precision * recall / max(1e-12, precision + recall),
                detection_precision=detection_tp / max(1, detection_tp + negative),
                detection_recall=detection_tp / max(1, int(labels.sum())),
                detection_f1=2 * detection_tp / max(1, detection_tp + negative + int(labels.sum())))


def choose_gate(cases, probabilities):
    if not cases or any(case['split'] != 'validation' for case in cases):
        raise ValueError('Gate selection requires validation sources only')
    domains = {name: np.asarray([case['dataset'] == name for case in cases]) for name in ('HornBase', 'ESC50')}
    if any(not mask.any() for mask in domains.values()):
        raise ValueError('Both validation domains are required')
    if any({case['label'] for case in cases if case['dataset'] == name} != {0, 1} for name in domains):
        raise ValueError('Each validation domain needs positive and negative examples')
    values = np.asarray(probabilities)
    options = []
    for threshold in SETTINGS['thresholds']:
        for margin in SETTINGS['margins']:
            domain_metrics = {name: outcomes([case for case in cases if case['dataset'] == name], values[mask], threshold, margin)
                              for name, mask in domains.items()}
            macro = float(np.mean([item['f1'] for item in domain_metrics.values()]))
            worst = max(item['negative_acceptance'] for item in domain_metrics.values())
            enabled = worst <= SETTINGS['maximum_domain_acceptance']
            metric = outcomes(cases, values, threshold, margin)
            rank = (enabled, macro, -worst, metric['f1'], threshold, margin)
            options.append((rank, dict(threshold=threshold, margin=margin, constraint_met=enabled,
                                      macro_f1=macro, worst_domain_acceptance=worst, metrics=metric, domains=domain_metrics)))
    return max(options, key=lambda item: item[0])[1]


def hard_weights(cases, probabilities, split):
    """Mine within each training domain without consulting held-out scenes."""
    if split != 'train' or any(case['split'] != 'train' for case in cases):
        raise ValueError('Hard negatives may only come from train sources')
    values = np.asarray(probabilities)
    if len(values) != len(cases) or not np.isfinite(values).all():
        raise ValueError('Mining scores do not match training cases')
    maximum = values.max(axis=1) if values.ndim == 2 else values
    weights = np.ones(len(cases), dtype=np.float32)
    for domain in ('HornBase', 'ESC50'):
        indices = [index for index, case in enumerate(cases) if case['dataset'] == domain and not case['label']]
        count = int(np.ceil(len(indices) * SETTINGS['hard_fraction']))
        chosen = sorted(indices, key=lambda index: (-maximum[index], index))[:count]
        weights[chosen] = SETTINGS['hard_weight']
    return weights


def prepare(keys):
    torch = configure()
    check_plan()
    teacher = network()
    plan = reference.check_plan()
    records = {record['file']: record for record in plan['sources']}
    for key in keys:
        path = OUTPUT / (key + '.npz')
        if path.exists():
            load_cache(key)
            print('Reusing', key, flush=True)
            continue
        rows, cached = reference.load_cache(key)
        names = {row[name] for row in rows for name in ('target', 'interferer')}
        waveforms = {name: reference.waveform(records[name]) for name in names}
        embeddings = []
        began = time.monotonic()
        error = 0.0
        for index, row in enumerate(rows):
            audio, pcm, _, _ = reference.synthesize(row, waveforms, plan['profile'], stress=key.startswith('stress-'))
            if pcm['pcm_sha256'] != row['pcm']['pcm_sha256']:
                raise ValueError('Resynthesis no longer matches the frozen PCM scene')
            bearings = [reference.Bearing(**item) for item in row['bearings']]
            beams = [beamform(audio, bearing, dict(channels=14, diameter=0.24, sample_rate=48000, speed=343.0))[0]
                     for bearing in bearings]
            with torch.inference_mode():
                frontend = torch.cat([teacher.encoder.frontend(beam) for beam in beams])
                features = teacher.encoder.embedding(frontend)
                replay = torch.softmax(teacher.head(features), dim=1)[:, 1].numpy()
            error = max(error, float(np.abs(replay - cached['teacher'][index]).max()))
            embeddings.append(features.numpy().copy())
            if (index + 1) % 40 == 0 or index + 1 == len(rows):
                print('PANNs embeddings', key, index + 1, '/', len(rows), 'seconds', round(time.monotonic() - began), flush=True)
        if error > 1e-5:
            raise ValueError('Frozen encoder replay differs from the previous classifier')
        np.savez_compressed(path, features=np.asarray(embeddings), baseline=cached['teacher'])
        write_json(OUTPUT / (key + '.json'), dict(plan_sha256=digest(OUTPUT / 'plan.json'), cache_sha256=digest(path),
                   previous_cache_sha256=digest(PREVIOUS / (key + '.npz')), baseline_maximum_error=error))


def load_cache(key):
    check_plan()
    rows, previous = reference.load_cache(key)
    metadata = read_json(OUTPUT / (key + '.json'))
    if metadata['plan_sha256'] != digest(OUTPUT / 'plan.json') or metadata['cache_sha256'] != digest(OUTPUT / (key + '.npz')):
        raise ValueError('Embedding cache provenance mismatch')
    if metadata['previous_cache_sha256'] != digest(PREVIOUS / (key + '.npz')):
        raise ValueError('Prior scene cache changed')
    with np.load(OUTPUT / (key + '.npz'), allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if values['features'].shape != (len(rows), 2, 512) or not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Invalid frozen embeddings')
    return rows, values, previous


def clips():
    check_plan()
    records = read_json(BASE / 'clips.json')
    with np.load(BASE / 'panns-embeddings.npz', allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if values['features'].shape[1] != 512 or len(values['offsets']) != len(records) + 1:
        raise ValueError('Original clip embeddings have incompatible dimensions')
    if not np.array_equal(values['clip_splits'], [record['split'] for record in records]):
        raise ValueError('Original clip source splits changed')
    return records, values


def clip_scores(head, data, indices):
    return np.asarray([scores(head, data['features'][data['offsets'][index]:data['offsets'][index + 1]]).max()
                       for index in indices])


def audit():
    """Replay cached encodings and inspect failures without relabeling any source."""
    torch = configure()
    check_plan()
    records, original = clips()
    teacher = network()
    groups = [(split, domain, label) for split in ('train', 'validation') for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    checked = []
    for split, domain, label in groups:
        index = next(index for index, record in enumerate(records) if record['split'] == split and record['dataset'] == domain and record['label'] == label)
        record = records[index]
        if digest(record['path']) != record['sha256']:
            raise ValueError('Original clip checksum changed')
        audio = reference.read_audio(record['path'], 48000, channel=0)
        audio, _ = remove_dc(audio)
        with torch.inference_mode():
            actual = teacher.encoder.embedding(teacher.encoder.frontend(audio[:48000, 0])).numpy()[0]
        cached = original['features'][original['offsets'][index]]
        if not np.allclose(actual, cached, atol=1e-4, rtol=1e-5):
            raise ValueError('Original clip cache does not match the frozen encoder')
        checked.append(dict(file=record['file'], split=split, maximum_embedding_error=float(abs(actual - cached).max())))
    lookup = {record['file']: record for record in records}
    profile = reference.check_plan()['profile']
    rig = camera_rig()
    config = reference.Config(microphones=14, diameter=0.24)
    array = reference.circular_array(config)
    settings = dict(channels=14, diameter=0.24, sample_rate=48000, speed=343.0)
    threshold = read_json(PREVIOUS / 'freeze.json')['gates']['panns']['threshold']
    result = dict(clip_cache_replay=checked, splits={}, examples=[], labels_changed=False, listened_by_human=False)
    for key in ('train-original', 'validation-original'):
        cases, data, _ = load_cache(key)
        bad = [(index, case) for index, case in enumerate(cases) if not case['label'] and max(data['baseline'][index]) >= threshold]
        result['splits'][key] = dict(false_positive_scenes=len(bad),
             target_categories={category: sum(lookup[case['target']]['category'] == category for _, case in bad)
                                for category in sorted({lookup[case['target']]['category'] for _, case in bad})},
             interference_categories={category: sum(lookup[case['interferer']]['category'] == category for _, case in bad)
                                      for category in sorted({lookup[case['interferer']]['category'] for _, case in bad})})
        examples = sorted([(index, case) for index, case in bad if not case['augmented']],
                          key=lambda item: -max(data['baseline'][item[0]]))[:4]
        for index, case in examples:
            sources = {name: reference.waveform(lookup[name]) for name in (case['target'], case['interferer'])}
            audio, pcm, points, _ = reference.synthesize(case, sources, profile)
            chosen = int(np.argmax(data['baseline'][index]))
            bearing = reference.Bearing(**case['bearings'][chosen])
            beam = beamform(audio, bearing, settings)[0]
            path = OUTPUT / f'audit-{key}-{case["id"]}.wav'
            codes = np.rint(np.clip(beam, -1, 32767 / 32768) * 32768).astype('<i2')
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(48000)
                stream.writeframes(codes.tobytes())
            components = []
            for name, point, level in zip((case['target'], case['interferer']), points, (1, 10 ** (-case['sir_db'] / 20))):
                component = reference.propagate_audio(sources[name][0], np.r_[point, rig.horn_height - rig.height], array, config)[0]
                component *= pcm['gain'] * level
                conditioned = reference.Conditioning(14).feed(component * 32768)
                components.append(beamform(conditioned, bearing, settings)[0])
            probabilities = teacher.score(components)
            spectrum = np.abs(np.fft.rfft(beam * np.hanning(len(beam)))) ** 2
            frequency = np.fft.rfftfreq(len(beam), 1 / 48000)
            result['examples'].append(dict(key=key, id=case['id'], target=case['target'], interferer=case['interferer'],
               mixture_score=float(data['baseline'][index, chosen]), target_component_score=probabilities[0],
               interference_component_score=probabilities[1], component_peaks=[float(np.max(abs(value))) for value in components],
               peak_frequency_hz=float(frequency[np.argmax(spectrum)]), waveform=path.name,
               scope='recording-component counterfactual at the same full-scene gain and measured candidate; no added noise or ADC rounding, not clean semantic stems'))
    write_json(OUTPUT / 'audit.json', result)
    print('PASS original clip cache replay; exported', len(result['examples']), 'training/validation review clips', flush=True)


def validation():
    rows, features, baseline = [], [], []
    for protocol in ('original', 'visible'):
        cases, data, _ = load_cache('validation-' + protocol)
        rows += cases
        features.append(data['features'])
        baseline.append(data['baseline'])
    return rows, np.concatenate(features), np.concatenate(baseline)


def train():
    torch = configure()
    check_plan()
    if (OUTPUT / 'freeze.json').exists():
        raise ValueError('Models are frozen; use a separate experiment for further training')
    cases, data, _ = load_cache('train-original')
    held, held_features, _ = validation()
    records, original = clips()
    teacher = network()
    initial = initial_head(teacher.head, 2)
    selected = np.asarray([index for index, record in enumerate(records) if record['split'] == 'train'])
    clip_cases = [records[index] for index in selected]
    replay_scores = clip_scores(initial, original, selected)
    replay_weights = np.ones(len(records), dtype=np.float32)
    replay_weights[selected] = hard_weights(clip_cases, replay_scores, 'train')
    scene_weights = hard_weights(cases, data['baseline'], 'train')
    write_json(OUTPUT / 'mining.json', dict(split='train', scene_ids=np.flatnonzero(scene_weights > 1).tolist(),
               clip_files=[records[index]['file'] for index in selected if replay_weights[index] > 1],
               rule='top quarter negative old-model scores within each source domain, weight 2; all other training examples retained'))
    scene_groups = [np.flatnonzero([case['dataset'] == domain and case['label'] == label for case in cases])
                    for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    clip_groups = [np.asarray([index for index in selected if records[index]['dataset'] == domain and records[index]['label'] == label])
                   for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    features = torch.from_numpy(data['features'])
    labels = torch.tensor([case['label'] for case in cases], dtype=torch.float32)
    candidates = []
    for method in ('hard', 'category'):
        for seed in SETTINGS['seeds']:
            path = OUTPUT / f'{method}-{seed}.pt'
            metadata_path = path.with_suffix('.json')
            if path.exists() and metadata_path.exists():
                metadata = read_json(metadata_path)
                if metadata['sha256'] != digest(path):
                    raise ValueError('Candidate checkpoint changed')
                candidates.append(metadata)
                continue
            torch.manual_seed(seed)
            generator = np.random.default_rng(seed)
            head = initial_head(teacher.head, len(CLASSES) if method == 'category' else 2)
            optimizer = torch.optim.Adam(head.parameters(), lr=SETTINGS['learning_rate'], weight_decay=SETTINGS['weight_decay'])
            best, state, stale, chosen, history = None, None, 0, None, []
            for epoch in range(SETTINGS['epochs'] + 1):
                losses = []
                if epoch:
                    for batch in range(len(cases) // SETTINGS['batch_scenes']):
                        indices = np.concatenate([generator.choice(group, 2) for group in scene_groups])
                        beam_logits = horn_margin(head(features[indices]))
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(beam_logits.max(dim=1).values, labels[indices],
                                       weight=torch.from_numpy(scene_weights[indices]))
                        ranking, count = beam_logits.sum() * 0, 0
                        for row, index in enumerate(indices):
                            errors = [180 if item['azimuth'] is None else abs(wrap_angle(item['azimuth'] - cases[index]['angle']))
                                      for item in cases[index]['bearings']]
                            target = int(np.argmin(errors))
                            if cases[index]['label'] and errors[target] < 10 and data['baseline'][index, target] > data['baseline'][index, 1 - target]:
                                ranking = ranking + torch.nn.functional.softplus(beam_logits[row, 1 - target] - beam_logits[row, target])
                                count += 1
                        chosen_clips = np.concatenate([generator.choice(group, 2) for group in clip_groups])
                        parts = [original['features'][original['offsets'][index]:original['offsets'][index + 1]] for index in chosen_clips]
                        logits = head(torch.from_numpy(np.concatenate(parts)))
                        margins = horn_margin(logits).split([len(part) for part in parts])
                        bag = torch.stack([part.max() for part in margins])
                        targets = torch.tensor([records[index]['label'] for index in chosen_clips], dtype=torch.float32)
                        replay_loss = torch.nn.functional.binary_cross_entropy_with_logits(bag, targets, weight=torch.from_numpy(replay_weights[chosen_clips]))
                        loss = loss + SETTINGS['replay_weight'] * replay_loss + SETTINGS['ranking_weight'] * ranking / max(1, count)
                        if method == 'category':
                            pooled = torch.stack([part.max(dim=0).values for part in logits.split([len(part) for part in parts])])
                            categories = torch.tensor([CLASSES.index(records[index]['category']) for index in chosen_clips])
                            loss = loss + SETTINGS['category_weight'] * torch.nn.functional.cross_entropy(pooled, categories)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                head.eval()
                gate = choose_gate(held, scores(head, held_features))
                rank = (gate['constraint_met'], gate['macro_f1'], -gate['worst_domain_acceptance'])
                history.append(dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, gate=gate))
                if best is None or rank > best:
                    best, state, stale, chosen = rank, deepcopy(head.state_dict()), 0, dict(epoch=epoch, gate=gate)
                else:
                    stale += 1
                if epoch % 5 == 0:
                    print(method, seed, epoch, 'validation macro F1', round(gate['macro_f1'], 4), 'worst-domain FP', round(gate['worst_domain_acceptance'], 4), flush=True)
                if stale >= SETTINGS['patience']:
                    break
            payload = dict(format='horn-panns-rejection-head-v1', state=state, method=method,
                           classes=list(CLASSES) if method == 'category' else ['car_horn', 'non_horn'],
                           encoder_sha256=digest(BASE / 'panns-horn.pt'), selection=chosen)
            torch.save(payload, path)
            metadata = dict(method=method, seed=seed, checkpoint=path.name, sha256=digest(path), **chosen)
            write_json(metadata_path, metadata)
            write_json(OUTPUT / f'{method}-{seed}-history.json', history)
            candidates.append(metadata)
    selected = {method: max([candidate for candidate in candidates if candidate['method'] == method],
                            key=lambda candidate: (candidate['gate']['constraint_met'], candidate['gate']['macro_f1'],
                                                   -candidate['gate']['worst_domain_acceptance'])) for method in ('hard', 'category')}
    gate = choose_gate(held, validation()[2])
    value = dict(plan_sha256=digest(OUTPUT / 'plan.json'), selected=selected, candidates=candidates, baseline_gate=gate,
                 original_gate=read_json(PREVIOUS / 'freeze.json')['gates']['panns'],
                 policy='heads and shared gates frozen using validation only; presence negatives include ambiguous directions; test is reused diagnostic')
    write_json(OUTPUT / 'freeze.json', value)
    print('Frozen selected heads', {name: item['epoch'] for name, item in selected.items()}, flush=True)


def load_head(metadata):
    torch = configure()
    path = OUTPUT / metadata['checkpoint']
    if digest(path) != metadata['sha256']:
        raise ValueError('Selected head changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['encoder_sha256'] != digest(BASE / 'panns-horn.pt') or checkpoint['format'] != 'horn-panns-rejection-head-v1':
        raise ValueError('Unsupported or mismatched head checkpoint')
    expected = list(CLASSES) if checkpoint['method'] == 'category' else ['car_horn', 'non_horn']
    if checkpoint['method'] not in ('hard', 'category') or checkpoint['classes'] != expected:
        raise ValueError('Unsupported head class order')
    head = torch.nn.Linear(512, len(checkpoint['classes'])).eval()
    head.load_state_dict(checkpoint['state'])
    return head


class RefinedPanns:
    """Score caller-conditioned one-second beams with a validation-selected head."""

    def __init__(self, method='category'):
        check_plan()
        if method not in ('hard', 'category'):
            raise ValueError('Unknown head method')
        frozen = read_json(OUTPUT / 'freeze.json')
        self.encoder = network().encoder
        self.head = load_head(frozen['selected'][method])
        self.gate = frozen['selected'][method]['gate']

    def score(self, waveforms, sample_rate=48000):
        torch = configure()
        if not waveforms:
            return []
        if sample_rate != 48000 or any(np.asarray(value).shape != (48000,) for value in waveforms):
            raise ValueError('Each input must be one completed mono second at 48 kHz')
        with torch.inference_mode():
            frontend = torch.cat([self.encoder.frontend(value, sample_rate) for value in waveforms])
            return scores(self.head, self.encoder.embedding(frontend)).tolist()


def predict(path, method, conditioned=False):
    model = RefinedPanns(method)
    audio = reference.read_audio(path, 48000, channel=0)
    if not conditioned:
        audio, _ = remove_dc(audio)
    if len(audio) < 48000:
        raise ValueError('Input must contain a complete second')
    starts = sorted(set(range(0, len(audio) - 48000 + 1, 24000)) | {len(audio) - 48000})
    probabilities = []
    started = time.perf_counter()
    for begin in range(0, len(starts), 16):
        probabilities += model.score([audio[start:start + 48000, 0] for start in starts[begin:begin + 16]])
    print(json.dumps(dict(method=method, threshold=model.gate['threshold'],
              horn_candidate=bool(max(probabilities) >= model.gate['threshold']), processing_seconds=time.perf_counter() - started,
              windows=[dict(start_seconds=start / 48000, end_seconds=(start + 48000) / 48000, score=score)
                       for start, score in zip(starts, probabilities)],
              scope='single-channel file diagnostic at a beam-calibrated threshold; no direction, event-onset, camera or real-time claim'), indent=2), flush=True)


def evaluate():
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Frozen experiment plan changed')
    heads = {name: load_head(metadata) for name, metadata in frozen['selected'].items()}
    reports = {}
    for phase in ('test', 'stress'):
        reports[phase] = {}
        for protocol in ('original', 'visible'):
            key = phase + '-' + protocol
            cases, data, previous = load_cache(key)
            predictions = dict(baseline_original=data['baseline'], baseline_recalibrated=data['baseline'])
            predictions.update({name: scores(head, data['features']) for name, head in heads.items()})
            gates = dict(baseline_original=frozen['original_gate'], baseline_recalibrated=frozen['baseline_gate'])
            gates.update({name: value['gate'] for name, value in frozen['selected'].items()})
            metrics = {}
            for name, probabilities in predictions.items():
                gate = gates[name]
                domains = {domain: outcomes([case for case in cases if case['dataset'] == domain],
                           probabilities[[case['dataset'] == domain for case in cases]], gate['threshold'], gate['margin'])
                           for domain in ('HornBase', 'ESC50')}
                metrics[name] = dict(all=outcomes(cases, probabilities, gate['threshold'], gate['margin']), domains=domains,
                                     gate=dict(threshold=gate['threshold'], margin=gate['margin']),
                                     wide_camera=reference.association_metrics(cases, previous, probabilities,
                                       dict(gate, margin=max(gate['margin'], np.finfo(float).eps)), camera_rig()))
            reports[phase][protocol] = metrics
            np.savez_compressed(OUTPUT / (key + '-predictions.npz'), **predictions)
    write_json(OUTPUT / 'results.json', dict(freeze_sha256=digest(OUTPUT / 'freeze.json'), protocols=reports,
               scope='reused public weak labels and synthetic arrays/camera; no new blind test, event FP/hour, road or board claim'))
    report()


def report():
    value = read_json(OUTPUT / 'results.json')
    selected = read_json(OUTPUT / 'freeze.json')['selected']
    lines = ['# PANNs 误接受改进对照', '', '日期：2026-10-21。仅科研/非商业，复用来源的诊断实验。', '',
             '冻结原Cnn6编码器；困难负例与多类别头均只用train来源训练、validation来源选择epoch/种子/共享门限。没有新道路录音，测试不能称为新盲测。', '',
             '负例只要有鸣笛存在分数越过门限即计误接受，即使方向分差不足仍计入；避免通过隐藏方向歧义降低误报数字。联合TP要求方向误差<10°且分差通过，错误方向计FP+FN。', '',
             '| 组 | 协议 | 方法 | 联合F1 | 正确方向/84 | 负例鸣笛接受/84 | 正例鸣笛接受/84 |',
             '|---|---|---|---:|---:|---:|---:|']
    for phase, protocols in value['protocols'].items():
        for protocol, methods in protocols.items():
            for name, item in methods.items():
                metric = item['all']
                lines.append(f"| {phase} | {protocol} | {name} | {metric['f1']:.4f} | {metric['tp']} | {metric['accepted_negative']} | {metric['accepted_positive']} |")
    lines += ['', 'baseline_original使用上一轮门限；baseline_recalibrated使用本轮相同的逐来源误接受约束，区分模型收益与单纯重选门限。', '',
              '训练每个来源中旧模型分数最高的四分之一负例权重为2，其余样本保留，原录音多实例回放防止仅适应合成波束。多类别方案把非鸣笛分为原始数据已有类别；类别交叉熵只用于原始弱标签片段，混合波束仍采用鸣笛存在MIL，不硬标为单一干净类别。', '',
              '初始多类别头分摊旧非鸣笛logit，初始鸣笛概率与原二分类头相同；不同方案不靠初始化分数偏移占优。', '',
              '两方法固定种子0/17、最多60轮、耐心10；编码器不微调，没有改默认部署模型或TinyCNN Verilog，PANNs仍为PC研究模型。', '',
              '选定epoch：' + ', '.join(f"{name}={item['epoch']} (seed {item['seed']})" for name, item in selected.items()) + '。', '',
              '结果判断：多类别头在主组改善了正确方向与误接受的折中，但原协议压力组误接受16/84，高于旧模型13/84；没有跨扰动稳定降低误接受，不提升为默认。只重选旧门限的对照为10/84误接受，但损失正确方向；需要按召回代价理解。', '',
              '最差来源问题仍存在：多类别头原协议HornBase负例10/36（旧15/36），可见组9/36；验证集逐来源≤10%的约束未能迁移到该测试来源。需要新的道路录音和人工听审，而非继续用当前测试调门限。', '',
              '候选几何唯一车辆：多类别头原/可见为19/84、30/84，旧模型17/84、26/84，仅为模拟镜头和框的诊断，不能证明真实鸣笛车辆归属。', '',
              '复现顺序：plan；prepare --split train；prepare --split validation；audit；train；prepare --split test；prepare --split test --stress；evaluate；verify。输入缓存来自旧horn_detector_v1与horn_recovery_v1；这些大文件和新权重都被忽略，不随Git发布。', '',
              '原始WAV诊断入口：python prototype/improve_panns.py predict --method category --input PATH.wav；困难负例头选hard。已做DC处理的波束WAV（包括本轮audit音频）加--conditioned，避免重复处理。RefinedPanns.score接受调用方已处理的48kHz单声道1秒波束。依赖原PANNs编码器及新头，不能直接交给TinyCNN RTL或PYNQ部署入口。', '',
              '音频核查若仅由元数据/频谱或分量重放完成，不等于人工听审；标签未据模型分数修改。详细来源误接受和几何关联见results.json。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUTPUT / 'README.md', flush=True)


def verify():
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    value = read_json(OUTPUT / 'results.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json') or value['freeze_sha256'] != digest(OUTPUT / 'freeze.json'):
        raise ValueError('Results use a different frozen operating point')
    count = 0
    maximum = 0.0
    for key in ('train-original', 'validation-original', 'validation-visible', 'test-original', 'test-visible', 'stress-original', 'stress-visible'):
        rows, _, _ = load_cache(key)
        count += len(rows)
        maximum = max(maximum, read_json(OUTPUT / (key + '.json'))['baseline_maximum_error'])
    cases, cached, _ = load_cache('test-original')
    indices = [next(index for index, case in enumerate(cases) if case['dataset'] == domain and case['label'] == label)
               for domain in ('HornBase', 'ESC50') for label in (0, 1)]
    names = {cases[index][name] for index in indices for name in ('target', 'interferer')}
    plan = reference.check_plan()
    sources = {record['file']: reference.waveform(record) for record in plan['sources'] if record['file'] in names}
    beams = []
    for index in indices:
        audio = reference.synthesize(cases[index], sources, plan['profile'])[0]
        beams += [beamform(audio, reference.Bearing(**item), dict(channels=14, diameter=0.24, sample_rate=48000, speed=343.0))[0]
                  for item in cases[index]['bearings']]
    replay = {}
    for method, metadata in frozen['selected'].items():
        model = RefinedPanns(method)
        actual = np.asarray(model.score(beams)).reshape(len(indices), 2)
        expected = scores(model.head, cached['features'][indices])
        error = float(abs(actual - expected).max())
        if error > 1e-5:
            raise ValueError('New head raw-wave inference disagrees with frozen embeddings')
        replay[method] = dict(maximum_score_error=error, beam_count=len(beams), checkpoint_sha256=metadata['sha256'])
    for phase in value['protocols'].values():
        for protocol in phase.values():
            for metric in protocol.values():
                if metric['all']['positives'] != 84 or metric['all']['negatives'] != 84:
                    raise ValueError('Evaluation denominators changed')
                counts = metric['wide_camera']
                if sum(counts[key] for key in ('correct_unique', 'incorrect_unique', 'ambiguous', 'unmatched')) != 84:
                    raise ValueError('Association counts do not partition positives')
    artifacts = {path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file() and path.name != 'verification.json'}
    write_json(OUTPUT / 'verification.json', dict(result='passed', cached_scenes=count, baseline_maximum_error=maximum,
                    head_replay=replay, artifacts=artifacts, old_inputs_unchanged=True, board_validated=False))
    print(json.dumps(dict(result='passed', cached_scenes=count, baseline_maximum_error=maximum, head_replay=replay)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'audit', 'train', 'evaluate', 'report', 'verify', 'predict'))
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--method', choices=('hard', 'category'), default='category')
    parser.add_argument('--conditioned', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        plan()
    elif args.command == 'prepare':
        if args.split == 'test' and not (OUTPUT / 'freeze.json').exists():
            raise ValueError('Freeze models and gates before test inference')
        if args.stress and args.split != 'test':
            raise ValueError('Stress is a frozen test diagnostic')
        prepare([('stress' if args.stress else args.split) + '-' + protocol for protocol in ('original', 'visible')
                 if args.split != 'train' or protocol == 'original'])
    elif args.command == 'train':
        train()
    elif args.command == 'audit':
        audit()
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'report':
        report()
    elif args.command == 'verify':
        verify()
    elif args.command == 'predict':
        if args.input is None:
            raise ValueError('predict requires --input WAV')
        predict(args.input, args.method, args.conditioned)


if __name__ == '__main__':
    main()
