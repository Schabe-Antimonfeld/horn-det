"""Freeze streaming equivalence checks on both inherited scene protocols and contiguous mixtures."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from copy import deepcopy
import importlib.util
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
from ps.configuration import load_config
from ps.pipeline import Pipeline

ROOT = PROTO.parent / 'deployment/pynq_z2'
OUTPUT = PROTO / 'outputs/stream_optimization_v2_20261101'
REFERENCE = previous.REFERENCE
spec = importlib.util.spec_from_file_location('stream_benchmark', ROOT / 'scripts/benchmark_stream.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(configuration):
    previous.check_plan()
    previous.frozen.check_plan()
    config = load_config(configuration)
    paths = [Path(__file__), ROOT / 'ps/streaming.py', ROOT / 'ps/stream_kernel.c', ROOT / 'ps/pipeline.py',
             ROOT / 'ps/dsp.py', ROOT / 'ps/search.py', ROOT / 'ps/search_kernel.c', ROOT / 'ps/beamforming.py',
             ROOT / 'scripts/build_stream.py', ROOT / 'scripts/benchmark_stream.py',
             previous.frozen.OUTPUT / 'model-combined/weights.npz', previous.frozen.OUTPUT / 'model-combined/manifest.json',
             Path(config['streaming']['native_library']), Path(config['search']['native_library'])]
    value = dict(format=1, configuration=config, frozen={str(path): digest(path) for path in paths},
                 design='fixed complete-grid search, original model/gates, identical twelve-frame PHAT sums, direction-independent input FFT reuse; no test-driven architecture or gate selection',
                 evidence='inherited public recordings with synthetic arrays; contiguous mixtures test implementation consistency, not new road accuracy or ARM timing')
    path = OUTPUT / 'plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Streaming experiment already frozen')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    print('Frozen streaming equivalence plan', flush=True)


def check_plan():
    previous.check_plan()
    previous.frozen.check_plan()
    value = read_json(OUTPUT / 'plan.json')
    for name, checksum in value['frozen'].items():
        if digest(Path(name)) != checksum:
            raise ValueError('Frozen streaming input changed: ' + name)
    return value


def phase(split):
    value = check_plan()
    if split not in ('validation', 'test', 'stress'):
        raise ValueError('Unknown streaming split')
    config = value['configuration']
    baseline = deepcopy(config)
    baseline.pop('streaming')
    source_plan = REFERENCE.check_plan()
    model = previous.frozen.OUTPUT / 'model-combined'
    for protocol in ('original', 'visible'):
        key = split + '-' + protocol
        path = OUTPUT / (key + '.json')
        if path.exists():
            raise ValueError('Preserve existing streaming scene results')
        cases, old = previous.load_result(key)
        names = {case[name] for case in cases for name in ('target', 'interferer')}
        sources = {record['file']: REFERENCE.waveform(record) for record in source_plan['sources'] if record['file'] in names}
        engines = {'reference': Pipeline(baseline, load_spectral(model)), 'optimized': Pipeline(config, load_spectral(model))}
        records, raw = {name: [] for name in engines}, []
        began = time.perf_counter()
        for index, (case, frozen) in enumerate(zip(cases, old['single4'])):
            audio, pcm, _, _ = REFERENCE.synthesize(case, sources, source_plan['profile'], stress=split == 'stress')
            if pcm['pcm_sha256'] != case['pcm']['pcm_sha256']:
                raise ValueError('Streaming scene PCM differs from the frozen protocol')
            if index < 8:
                raw.append(audio)
            events = {}
            snapshot = dict(audio=audio, first=0, end=len(audio), epoch=index, clipped=False)
            order = ('reference', 'optimized') if index % 2 == 0 else ('optimized', 'reference')
            for name in order:
                event, _ = engines[name].analyze(snapshot)
                events[name] = event
                records[name].append(dict(scores=event['scores'], seconds=event['processing_seconds'],
                                          state=event['acoustic_state'], pcm_sha256=pcm['pcm_sha256']))
            benchmark.compare(events['reference'], events['optimized'])
            np.testing.assert_allclose(events['reference']['scores'], frozen['scores'][:len(events['reference']['scores'])], atol=2e-10, rtol=0)
            if (index + 1) % 24 == 0 or index + 1 == len(cases):
                print(key, index + 1, '/', len(cases), 'seconds', round(time.perf_counter() - began), flush=True)
        joined = np.concatenate(raw)
        continuous = []
        engines = {'reference': Pipeline(baseline, load_spectral(model)), 'optimized': Pipeline(config, load_spectral(model))}
        capacity, update = config['audio']['window_samples'], config['audio']['update_samples']
        for index, start in enumerate(range(0, len(joined) - capacity + 1, update)):
            epoch = 0 if index < 6 else 1
            snapshot = dict(audio=joined[start:start + capacity], first=start, end=start + capacity, epoch=epoch,
                            clipped=index == 4)
            events = {name: engine.analyze(snapshot)[0] for name, engine in engines.items()}
            error = benchmark.compare(events['reference'], events['optimized'])
            continuous.append(dict(maximum_score_error=error, diagnostics=events['optimized']['stream_diagnostics'],
                                   fft_frames=events['optimized']['srp_diagnostics']['fft_frames_computed'],
                                   beam=events['optimized']['beamforming_diagnostics'],
                                   state=events['optimized']['acoustic_state']))
        write_json(path, dict(plan_sha256=digest(OUTPUT / 'plan.json'), source_sha256=digest(previous.OUTPUT / (key + '.json')),
                             rows_sha256=digest(REFERENCE.OUTPUT / (key + '.json')), models=records, continuous=continuous))


def verify():
    check_plan()
    report, scenes, windows = {}, 0, 0
    for split in ('validation', 'test', 'stress'):
        for protocol in ('original', 'visible'):
            key = split + '-' + protocol
            path = OUTPUT / (key + '.json')
            value = read_json(path)
            if value['plan_sha256'] != digest(OUTPUT / 'plan.json') or value['source_sha256'] != digest(previous.OUTPUT / (key + '.json')) or value['rows_sha256'] != digest(REFERENCE.OUTPUT / (key + '.json')):
                raise ValueError('Streaming scene provenance mismatch')
            cases, old = previous.load_result(key)
            paired = [dict(case, bearings=row['bearings']) for case, row in zip(cases, old['single4'])]
            scores = {name: np.asarray([row['scores'] for row in rows]) for name, rows in value['models'].items()}
            np.testing.assert_allclose(scores['reference'], scores['optimized'], atol=2e-10, rtol=0)
            metrics = {name: previous.frozen.evaluation.outcomes(paired, prediction, 0.95, 0) for name, prediction in scores.items()}
            if metrics['reference'] != metrics['optimized']:
                raise ValueError('Streaming changed inherited joint metrics')
            positive = np.asarray([bool(case['label']) for case in cases])
            capture = {}
            for name, prediction in scores.items():
                accepted = prediction.max(axis=1) >= 0.9
                capture[name] = dict(positive=int((accepted & positive).sum()), negative=int((accepted & ~positive).sum()))
            if capture['reference'] != capture['optimized']:
                raise ValueError('Streaming changed inherited capture metrics')
            for index, row in enumerate(value['continuous']):
                expected = 93 if index in (0, 6) else 47
                if row['fft_frames'] != expected or row['maximum_score_error'] > 2e-10:
                    raise ValueError('Continuous reuse did not follow its epoch/counter contract')
            report[key] = dict(joint=metrics['optimized'], capture=capture['optimized'],
                               maximum_score_error=float(np.max(abs(scores['optimized'] - scores['reference']))),
                               continuous_windows=len(value['continuous']),
                               continuous_maximum_score_error=max(row['maximum_score_error'] for row in value['continuous']),
                               output_reuse_windows=sum(row['beam']['output_reused'] for row in value['continuous']))
            scenes += len(cases)
            windows += len(value['continuous'])
    result = dict(result='passed', scenes=scenes, continuous_windows=windows, protocols=report,
                  plan_sha256=digest(OUTPUT / 'plan.json'), scope=read_json(OUTPUT / 'plan.json')['evidence'])
    write_json(OUTPUT / 'verification.json', result)
    print(json.dumps(dict(result='passed', scenes=scenes, continuous_windows=windows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'phase', 'verify'))
    parser.add_argument('--config')
    parser.add_argument('--split', choices=('validation', 'test', 'stress'), default='validation')
    args = parser.parse_args()
    if args.action == 'plan':
        plan(args.config)
    elif args.action == 'phase':
        phase(args.split)
    else:
        verify()


if __name__ == '__main__':
    main()
