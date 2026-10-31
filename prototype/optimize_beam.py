"""Select an exact beam backend on frozen validation scenes, then check both test protocols."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))
import optimize_srp as previous
from horn.recovery import digest, write_json
from ps.band import load_spectral
from ps.beamforming import Beamformer
from ps.configuration import load_config
from ps.dsp import features
from ps.geometry import Bearing

OUTPUT = PROTO / 'outputs/beam_optimization_v1_20261031'
ROOT = PROTO.parent / 'deployment/pynq_z2'
REFERENCE = previous.REFERENCE


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(native_library=None):
    previous.check_plan()
    previous.frozen.check_plan()
    paths = [Path(__file__), ROOT / 'ps/beamforming.py', ROOT / 'ps/beam_kernel.c',
             ROOT / 'ps/dsp.py', ROOT / 'ps/geometry.py', ROOT / 'ps/band.py',
             PROTO / 'optimize_srp.py', previous.OUTPUT / 'freeze.json',
             previous.frozen.OUTPUT / 'model-combined/manifest.json',
             previous.frozen.OUTPUT / 'model-combined/weights.npz']
    variants = dict(reference=dict(implementation='reference'), cached=dict(implementation='cached'),
                    fft512=dict(implementation='fft', fft_size=512), fft1024=dict(implementation='fft', fft_size=1024))
    if native_library:
        path = Path(native_library).resolve()
        paths += [path]
        variants['native'] = dict(implementation='native', native_library=str(path))
    value = dict(format=1, variants=variants, frozen={str(path): digest(path) for path in paths},
                 selection='minimum pooled paired validation beam time among exact backends; both protocols must retain all scores within 2e-10 and all decisions; no detector retuning',
                 scope='reused real recordings in synthetic fourteen-channel scenes; not new blind road, continuous event or ARM evidence')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Beam plan already frozen; preserve earlier artifacts')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return value


def check_plan():
    previous.check_plan()
    previous.frozen.check_plan()
    value = read_json(OUTPUT / 'plan.json')
    for name, checksum in value['frozen'].items():
        if digest(Path(name)) != checksum:
            raise ValueError('Frozen beam input changed: ' + name)
    return value


def phase(split):
    value = check_plan()
    if split not in ('validation', 'test', 'stress'):
        raise ValueError('Unknown beam protocol split')
    if split != 'validation' and not (OUTPUT / 'freeze.json').exists():
        raise ValueError('Freeze validation selection before held-out evaluation')
    variants = value['variants'] if split == 'validation' else {
        name: value['variants'][name] for name in dict.fromkeys(['reference', read_json(OUTPUT / 'freeze.json')['selected']])}
    settings = load_config(ROOT / 'config/band_temporal.json')['audio']
    detector = load_spectral(previous.frozen.OUTPUT / 'model-combined')
    source_plan = REFERENCE.check_plan()
    for protocol in ('original', 'visible'):
        key = split + '-' + protocol
        path = OUTPUT / (key + '.json')
        if path.exists():
            raise ValueError('Beam result already exists: ' + key)
        cases, models = previous.load_result(key)
        cached = models['single4']
        names = {case[name] for case in cases for name in ('target', 'interferer')}
        sources = {record['file']: REFERENCE.waveform(record) for record in source_plan['sources'] if record['file'] in names}
        engines = {name: Beamformer(settings, options) for name, options in variants.items()}
        records = {name: [] for name in variants}
        began = time.perf_counter()
        for index, (case, frozen) in enumerate(zip(cases, cached)):
            audio, pcm, _, _ = REFERENCE.synthesize(case, sources, source_plan['profile'], stress=split == 'stress')
            if pcm['pcm_sha256'] != case['pcm']['pcm_sha256']:
                raise ValueError('Beam PCM differs from the frozen protocol')
            bearings = [Bearing(**row) for row in frozen['bearings'][:frozen['candidate_count']]]
            if index == 0:
                for engine in engines.values():
                    engine.run(audio, bearings)
            names = list(engines)
            rotation = index % len(names)
            results = {}
            for name in names[rotation:] + names[:rotation]:
                started = time.perf_counter()
                beams, diagnostic = engines[name].run(audio, bearings)
                elapsed = time.perf_counter() - started
                contexts = [features(waveform) for waveform, _ in beams]
                predictions = [detector.score(context) for context in contexts]
                scores = [row['score'] for row in predictions]
                scores += [0] * (2 - len(scores))
                results[name] = (beams, contexts, scores, [row['clipped'] for row in predictions])
                records[name].append(dict(scores=scores, beam_seconds=elapsed, pcm_sha256=pcm['pcm_sha256'],
                                          diagnostic=diagnostic, maximum_waveform_error=0.0, maximum_feature_error=0.0))
            expected, context, baseline, clipped = results['reference']
            np.testing.assert_allclose(baseline, frozen['scores'], atol=2e-10, rtol=0)
            for name, (beams, contexts, scores, actual_clipped) in results.items():
                np.testing.assert_allclose(scores, baseline, atol=2e-10, rtol=0)
                if clipped != actual_clipped or np.argmax(scores) != np.argmax(baseline):
                    raise ValueError('Beam backend changed reliability or selected direction')
                for gate in (0.9, 0.95):
                    if not np.array_equal(np.asarray(scores) >= gate, np.asarray(baseline) >= gate):
                        raise ValueError('Beam backend changed an operating-gate decision')
                for (actual, latency), (reference, common), actual_context, expected_context in zip(beams, expected, contexts, context):
                    np.testing.assert_allclose(actual, reference, atol=3e-10, rtol=3e-13)
                    np.testing.assert_allclose(actual_context, expected_context, atol=2e-9, rtol=1e-10)
                    if common != latency:
                        raise ValueError('Beam backend changed common latency')
                    row = records[name][-1]
                    row['maximum_waveform_error'] = max(row['maximum_waveform_error'], float(np.max(np.abs(actual - reference))))
                    row['maximum_feature_error'] = max(row['maximum_feature_error'], float(np.max(np.abs(actual_context - expected_context))))
            if (index + 1) % 24 == 0 or index + 1 == len(cases):
                print(key, index + 1, '/', len(cases), 'seconds', round(time.perf_counter() - began), flush=True)
        write_json(path, dict(plan_sha256=digest(OUTPUT / 'plan.json'), source_sha256=digest(previous.OUTPUT / (key + '.json')),
                             rows_sha256=digest(REFERENCE.OUTPUT / (key + '.json')), models=records))


def result(key):
    value = read_json(OUTPUT / (key + '.json'))
    if value['plan_sha256'] != digest(OUTPUT / 'plan.json') or value['source_sha256'] != digest(previous.OUTPUT / (key + '.json')) or value['rows_sha256'] != digest(REFERENCE.OUTPUT / (key + '.json')):
        raise ValueError('Beam result provenance mismatch')
    cases, old = previous.load_result(key)
    paired = [dict(case, bearings=row['bearings']) for case, row in zip(cases, old['single4'])]
    return paired, value['models']


def select():
    value = check_plan()
    path = OUTPUT / 'freeze.json'
    if path.exists():
        raise ValueError('Beam selection already frozen')
    ratios = {name: [] for name in value['variants']}
    files = {}
    for protocol in ('original', 'visible'):
        key = 'validation-' + protocol
        cases, models = result(key)
        if any(case['split'] != 'validation' for case in cases):
            raise ValueError('Beam selection must use validation scenes only')
        for name, rows in models.items():
            ratios[name] += [row['beam_seconds'] / baseline['beam_seconds'] for row, baseline in zip(rows, models['reference'])]
        files[key] = digest(OUTPUT / (key + '.json'))
    times = {name: float(np.median(values)) for name, values in ratios.items()}
    selected = min(times, key=lambda name: (times[name], name))
    frozen = dict(selected=selected, options=value['variants'][selected], validation_relative_time=times,
                  validation_sha256=files, plan_sha256=digest(OUTPUT / 'plan.json'))
    write_json(path, frozen)
    print(json.dumps(frozen), flush=True)


def verify():
    check_plan()
    frozen = read_json(OUTPUT / 'freeze.json')
    if frozen['plan_sha256'] != digest(OUTPUT / 'plan.json'):
        raise ValueError('Beam freeze provenance mismatch')
    for key, checksum in frozen['validation_sha256'].items():
        if digest(OUTPUT / (key + '.json')) != checksum:
            raise ValueError('Beam validation result changed')
    report = {}
    scenes = 0
    for split in ('validation', 'test', 'stress'):
        for protocol in ('original', 'visible'):
            key = split + '-' + protocol
            cases, models = result(key)
            scenes += len(cases)
            report[key] = {}
            baseline = np.asarray([row['scores'] for row in models['reference']])
            for name, rows in models.items():
                scores = np.asarray([row['scores'] for row in rows])
                np.testing.assert_allclose(scores, baseline, atol=2e-10, rtol=0)
                metric = previous.frozen.evaluation.outcomes(cases, scores, 0.95, 0)
                expected = previous.frozen.evaluation.outcomes(cases, baseline, 0.95, 0)
                if metric != expected:
                    raise ValueError('Beam backend changed joint metrics')
                report[key][name] = dict(joint=metric,
                    capture_positive=int(sum(bool(case['label']) and max(row['scores']) >= 0.9 for case, row in zip(cases, rows))),
                    capture_negative=int(sum(not case['label'] and max(row['scores']) >= 0.9 for case, row in zip(cases, rows))),
                    maximum_score_error=float(np.max(np.abs(scores - baseline))),
                    maximum_waveform_error=max(row['maximum_waveform_error'] for row in rows),
                    maximum_feature_error=max(row['maximum_feature_error'] for row in rows),
                    median_relative_time=float(np.median([row['beam_seconds'] / base['beam_seconds'] for row, base in zip(rows, models['reference'])])))
    value = dict(result='passed', scenes=scenes, selected=frozen['selected'], protocols=report,
                 freeze_sha256=digest(OUTPUT / 'freeze.json'), scope=read_json(OUTPUT / 'plan.json')['scope'])
    write_json(OUTPUT / 'verification.json', value)
    print(json.dumps(dict(result='passed', scenes=scenes, selected=frozen['selected'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'phase', 'select', 'verify'))
    parser.add_argument('--split', choices=('validation', 'test', 'stress'), default='validation')
    parser.add_argument('--native-library')
    args = parser.parse_args()
    if args.action == 'plan':
        plan(args.native_library)
    elif args.action == 'phase':
        phase(args.split)
    elif args.action == 'select':
        select()
    else:
        verify()


if __name__ == '__main__':
    main()
