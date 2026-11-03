"""Compare one/two-window decisions on fixed-duration real-recording array simulations."""

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.conditioning import create_conditioning
from ps.configuration import load_config
from ps.detector import Detector
from ps.dsp import coordinates
from ps.evaluation import verify_assets
from ps.geometry import direction_vectors
from ps.pipeline import Events, Pipeline
from scripts.build import build
from scripts.simulate_downward import propagate


THRESHOLDS = (0.95, 0.90, 0.80)
DURATIONS = (0.2, 0.5, 0.8, 1.5, 2.0, 3.0)
SPANS = [('4-175855-A-43.wav', 0.20, 0.65), ('5-179860-A-43.wav', 0.04, 0.30),
         ('5-179863-A-43.wav', 0.52, 0.82), ('5-179868-A-43.wav', 0.05, 0.70),
         ('5-180156-A-43.wav', 3.72, 4.15), ('5-180156-B-43.wav', 0.90, 1.60),
         ('5-180156-C-43.wav', 1.00, 3.00)]
POSITIONS = ((1.8, 4.0), (-1.8, -4.0))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_audio(path, checksum):
    """Verify original PCM16 bytes and resample channel zero without repeating samples."""
    if digest(path) != checksum:
        raise ValueError('Source checksum mismatch: ' + str(path))
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('Uncompressed PCM16 sources are required')
        rate, channels = stream.getframerate(), stream.getnchannels()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').reshape(-1, channels)[:, 0] / 32768.0
    if rate != 48000:
        data = np.interp(np.arange(int(len(data) * 48000 / rate)) * rate / 48000, np.arange(len(data)), data)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError('Empty or nonfinite source')
    return data


def segment(values, start, end, duration):
    """Crop the center of a screened span; reject unsupported lengths instead of looping."""
    if not 0 <= start < end <= len(values) / 48000 or duration <= 0:
        raise ValueError('Invalid screened source span')
    count = int(round(duration * 48000))
    lower, upper = int(round(start * 48000)), int(round(end * 48000))
    if count > upper - lower:
        return None
    first = lower + (upper - lower - count) // 2
    result = values[first:first + count].copy()
    fade = min(240, count // 2)
    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        result[:fade] *= ramp
        result[-fade:] *= ramp[::-1]
    rms = float(np.sqrt(np.mean(result ** 2)))
    if rms < 1e-8:
        raise ValueError('Screened segment is silent')
    return result / rms, dict(first_48k=first, samples=count, duration_seconds=count / 48000,
                              fade_samples=fade, original_rms=rms,
                              scope='Controlled crop of a visually screened tonal candidate, not listening-verified natural horn duration')


def make_scene(config, target, background, position, sir, onset, seed, foreground=None):
    """Keep active target RMS and the background fixed across lengths and window policies."""
    count = config['audio']['window_samples'] + 8 * config['audio']['update_samples']
    array = coordinates(config['audio'])
    other = -2 * np.asarray(position)
    source = np.zeros(count)
    if target is not None:
        first = int(round(onset * 48000))
        if first < 0 or first + len(target) > count:
            raise ValueError('Target falls outside the scene')
        source[first:first + len(target)] = target
    if foreground is not None:
        source[:min(count, len(foreground))] = foreground[:count]
    back = np.zeros(count)
    back[:min(count, len(background))] = background[:count]
    mixture = 0.025 * propagate(source, position, array, config['camera']['height'])
    mixture += 0.025 * 10 ** (-sir / 20) * propagate(back, other, array, config['camera']['height'])
    mixture += np.random.RandomState(seed).normal(0, 0.001, mixture.shape)
    gain = min(1.0, 0.95 / max(1e-12, float(np.max(abs(mixture)))))
    pcm = np.rint(mixture * gain * 32768).astype('<i2')
    return pcm, dict(sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), headroom_gain=gain,
                     active_target_rms=0.025 if target is not None else None, background_rms=0.025 * 10 ** (-sir / 20),
                     scope='Digital center RMS control, synthetic spherical delays; not calibrated SPL')


def infer(config, detector, pcm, position, epoch):
    """Run the unchanged deployed acoustic chain once, retaining ungated candidate scores."""
    pipeline, conditioner = Pipeline(config, detector), create_conditioning(config)
    audio = np.vstack([conditioner.feed(pcm[first:first + 512]) for first in range(0, len(pcm), 512)])
    truth = np.r_[position, 0.8 - config['camera']['height']]
    truth /= np.linalg.norm(truth)
    frames = []
    for first in range(0, len(audio) - 48128 + 1, 24064):
        event, _ = pipeline.analyze(dict(audio=audio[first:first + 48128], first=first,
                                       end=first + 48128, epoch=epoch, clipped=False))
        bearings = list(pipeline.processing.beams.outputs)
        if len(bearings) != len(event['scores']):
            raise ValueError('Beam/candidate order mismatch')
        order = np.argsort(-np.asarray(event['scores']), kind='stable')
        chosen = int(order[0]) if len(order) else None
        bearing = bearings[chosen] if chosen is not None else None
        error = float(np.rad2deg(np.arccos(np.clip(direction_vectors(*bearing) @ truth, -1, 1)))) if bearing is not None else None
        valid = not any(event[key] for key in ('input_clipped', 'activation_clipped', 'frontend_clipped'))
        distinct = bool(len(order) == 1 or (len(order) > 1 and event['scores'][chosen] > event['scores'][int(order[1])]))
        frames.append(dict(first=first / 48000, end=(first + 48128) / 48000, epoch=epoch,
                           score=event['score'], scores=event['scores'], reliable=valid,
                           bearings=[list(item) for item in bearings], selected_error_degrees=error,
                           direction_correct=bool(distinct and error is not None and error < 10),
                           processing_seconds=event['processing_seconds']))
    return frames, conditioner.latency / 48000


def decide(frames, threshold, windows, interval=None):
    """Require consecutive valid gates; deduplicate alarms using the deployed release-two event identity."""
    events = Events(dict(confirm=windows, release=2), 24064)
    seen, alarms = set(), []
    for frame in frames:
        positive = bool(frame['reliable'] and frame['score'] >= threshold)
        event = events.feed(positive, dict(epoch=frame.get('epoch', 0), end=int(round(frame['end'] * 48000))))
        if positive and events.on >= windows and event['event_id'] not in seen:
            seen.add(event['event_id'])
            matched = bool(interval is not None and frame['first'] < interval[1] and frame['end'] > interval[0])
            alarms.append(dict(time=frame['end'], matched=matched,
                               direction_correct=bool(matched and frame['direction_correct']),
                               latency_seconds=frame['end'] - interval[0] if matched else None))
    hits = [item for item in alarms if item['matched']]
    return dict(detected=bool(hits), direction_correct=bool(hits and hits[0]['direction_correct']),
                latency_seconds=hits[0]['latency_seconds'] if hits else None,
                false_alarms=sum(not item['matched'] for item in alarms), alarms=alarms)


def summarize(cases):
    """Report event recall and negative exposure separately from conditional latency/direction."""
    results = []
    negatives = [case for case in cases if case['cohort'].startswith('negative')]
    groups = [('paired-short', [case for case in cases if case['cohort'] == 'paired' and case['duration'] <= 0.8]),
              ('paired-long', [case for case in cases if case['cohort'] == 'paired' and case['duration'] >= 1.5]),
              ('natural-short-candidates', [case for case in cases if case['cohort'] == 'natural'])]
    groups += [('paired-' + str(duration) + 's', [case for case in cases if case['cohort'] == 'paired' and case['duration'] == duration])
               for duration in DURATIONS]
    for group, positives in groups:
        if not positives:
            continue
        for threshold in THRESHOLDS:
            for windows in (1, 2):
                key = '{:.2f}/{}'.format(threshold, windows)
                accepted = [case['decisions'][key] for case in positives]
                rejected = [case['decisions'][key] for case in negatives]
                hits = sum(item['detected'] for item in accepted)
                joint = sum(item['direction_correct'] for item in accepted)
                false = sum(item['false_alarms'] for item in accepted + rejected)
                latency = [item['latency_seconds'] for item in accepted if item['detected']]
                results.append(dict(group=group, threshold=threshold, windows=windows,
                                    positives=len(positives), source_groups=len(set(case['source_group'] for case in positives)),
                                    detected=hits, recall=hits / len(positives), joint_correct=joint,
                                    joint_recall=joint / len(positives), negative_scenes=len(negatives),
                                    negative_triggered=sum(bool(item['false_alarms']) for item in rejected),
                                    negative_alarms=sum(item['false_alarms'] for item in rejected),
                                    negative_seconds=sum(case['frames'][-1]['end'] for case in negatives),
                                    positive_false_alarms=sum(item['false_alarms'] for item in accepted),
                                    event_precision=hits / max(1, hits + false),
                                    event_f1=2 * hits / max(1, len(positives) + hits + false),
                                    conditional_latency_median=float(np.median(latency)) if latency else None,
                                    conditional_latency_p90=float(np.percentile(latency, 90)) if latency else None))
    return results


def run(output, source_directory, plan_only=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    verify_assets(ROOT)
    config = load_config()
    records = json.loads((ROOT / 'evaluation/sources.json').read_text(encoding='utf-8'))
    negatives = {}
    for row in records:
        if row['dataset'] == 'ESC50' and row['label'] == 0 and row['split'] == 'test':
            negatives.setdefault(row['category'], row['file'])
    selected = sorted(set([span[0] for span in SPANS] + list(negatives.values())))
    provenance, audio = {}, {}
    for name in selected:
        row = next(row for row in records if row['file'] == name)
        audio[name] = load_audio(Path(source_directory) / name, row['sha256'])
        provenance[name] = row
    output.mkdir(parents=True)
    (output / 'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    audit = []
    for name, start, end in SPANS:
        values = audio[name]
        blocks = values[:len(values) // 960 * 960].reshape(-1, 960)
        levels = np.sqrt(np.mean(blocks ** 2, axis=1))
        audit.append(dict(file=name, screened_span=[start, end], rms_20ms=levels.tolist(),
                          status='Waveform/spectrogram-screened tonal candidate; no listening-verified temporal labels',
                          supported_durations=[duration for duration in DURATIONS if segment(values, start, end, duration) is not None]))
    backgrounds = [negatives['siren'], negatives['engine']]
    plans, skipped = [], []
    options = [('paired', SPANS[-1], duration) for duration in DURATIONS]
    options += [('natural', span, round(span[2] - span[1], 6)) for span in SPANS[:-1]]
    for cohort, span, duration in options:
        if segment(audio[span[0]], span[1], span[2], duration) is None:
            skipped.append(dict(cohort=cohort, source=span[0], duration=duration, reason='Screened continuous candidate interval too short; no looping'))
            continue
        for geometry, position in enumerate(POSITIONS):
            for background in backgrounds:
                for sir in (-5, 5):
                    for phase in range(4):
                        plans.append(dict(cohort=cohort, source=span[0], span=list(span[1:]), duration=duration,
                                          source_group=provenance[span[0]]['group'], position=position,
                                          background=background, sir_db=sir, onset=1.1 + phase * 24064 / 48000 / 4,
                                          seed=20261103 + geometry * 100 + backgrounds.index(background) * 10 + int(sir == 5)))
    for geometry, position in enumerate(POSITIONS):
        for category, source in sorted(negatives.items()):
            for sir in (-5, 5):
                for seed in (202611031, 202611032):
                    plans.append(dict(cohort='negative-hard', source=source, category=category,
                                      source_group=provenance[source]['group'], duration=None, position=position,
                                      background=backgrounds[0] if category != 'siren' else backgrounds[1],
                                      sir_db=sir, onset=None, seed=seed + geometry))
    for geometry, position in enumerate(POSITIONS):
        for index, background in enumerate(backgrounds):
            for sir in (-5, 5):
                plans.append(dict(cohort='negative-matched', source=None, source_group='background-only', duration=None,
                                  position=position, background=background, sir_db=sir, onset=None,
                                  seed=20261103 + geometry * 100 + index * 10 + int(sir == 5)))
    plan = dict(format=1, configuration=config, cases=plans, thresholds=THRESHOLDS, window_counts=[1, 2],
                skipped=skipped, source_audit=audit, sources=provenance, source_sha256=digest(__file__),
                model_sha256=digest(ROOT / 'model/weights.npz'),
                caveats='Only one long-horn parent source (180156); related crops and mixtures are not independent recordings. Screened spans are provisional, not exact natural duration ground truth. No classifier-score selection. No camera visibility filtering.')
    write_json(output / 'plan.json', plan)
    if plan_only:
        return plan
    builds, detector = build(), Detector(ROOT / 'model')
    normalized = {name: values / max(1e-12, float(np.sqrt(np.mean(values ** 2)))) for name, values in audio.items()}
    cases = []
    with (output / 'cases.jsonl').open('w', encoding='utf-8') as stream:
        for index, case in enumerate(plans):
            target = None
            if not case['cohort'].startswith('negative'):
                target, crop = segment(audio[case['source']], *case['span'], case['duration'])
                case = dict(case, crop=crop)
            foreground = normalized[case['source']] if case['cohort'] == 'negative-hard' else None
            pcm, meta = make_scene(config, target, normalized[case['background']], case['position'],
                                   case['sir_db'], case['onset'], case['seed'], foreground)
            frames, latency = infer(config, detector, pcm, case['position'], index)
            distance = float(np.linalg.norm(np.r_[case['position'], 0.8 - config['camera']['height']]))
            onset = case['onset'] + distance / 343 + latency if target is not None else None
            interval = [onset, onset + len(target) / 48000] if target is not None else None
            decisions = {'{:.2f}/{}'.format(threshold, windows): decide(frames, threshold, windows, interval)
                         for threshold in THRESHOLDS for windows in (1, 2)}
            result = dict(case, id=index, pcm=meta, frames=frames, arrival_interval=interval, decisions=decisions)
            cases.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(plans):
                print('completed {} / {}'.format(index + 1, len(plans)), flush=True)
    report = dict(format=1, result='completed', cases=len(cases), windows=sum(len(case['frames']) for case in cases),
                  summaries=summarize(cases), skipped=skipped, caveats=plan['caveats'],
                  plan_sha256=digest(output / 'plan.json'), source_sha256=digest(__file__), native_builds=builds,
                  model_sha256=plan['model_sha256'], default_config_sha256=digest(ROOT / 'config/default.json'),
                  board_validated=False, interval_semantics='Known injected source support plus center propagation and conditioning latency; provisional source-label validity',
                  timing='Algorithmic decision-window end minus injected arrival onset; excludes collection hardware, computation, camera and scheduling',
                  alarm_semantics='First strict consecutive acceptance per release-two event identity; matching requires decision window overlap with injected arrival support. Off-event alarms count as false.')
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    result = run(args.output, args.source_directory, args.plan_only)
    print(json.dumps(dict(result='plan' if args.plan_only else result['result'], cases=len(result['cases']) if args.plan_only else result['cases'])))


if __name__ == '__main__':
    main()
