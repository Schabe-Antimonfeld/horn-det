"""Measure contiguous-window reuse and fallback latency against the frozen fast pipeline."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.band import load_spectral, validate_backend, validate_model
from ps.configuration import load_config
from ps.dsp import Conditioning, features
from ps.beamforming import Beamformer
from ps.geometry import Bearing
from ps.pipeline import Pipeline
from ps.streaming import Beams, Continuity, Frontend, StreamConditioning
from run import replay


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), maximum=float(max(values)))


def compare(reference, actual):
    """Check all nonnumeric decisions and candidate order, ignoring implementation diagnostics."""
    ignored = {'processing_seconds', 'phase_seconds', 'srp_diagnostics', 'beamforming_diagnostics', 'stream_diagnostics',
               'score', 'score_margin', 'scores', 'direction_candidates', 'capture_candidates'}
    first = {key: value for key, value in reference.items() if key not in ignored}
    second = {key: value for key, value in actual.items() if key not in ignored}
    if first != second:
        raise ValueError('Streaming changed an event or selected direction')
    for name in ('direction_candidates', 'capture_candidates'):
        a = [{key: value for key, value in row.items() if key != 'score'} for row in reference[name]]
        b = [{key: value for key, value in row.items() if key != 'score'} for row in actual[name]]
        if a != b:
            raise ValueError('Streaming changed candidate routing')
    np.testing.assert_allclose(reference['scores'], actual['scores'], atol=2e-10, rtol=0)
    return float(np.max(abs(np.asarray(reference['scores']) - actual['scores']), initial=0))


def beam_scenarios(snapshots, config, repeats):
    """Measure changing directions separately so output-cache hits cannot hide a slow path."""
    result = {}
    for scenario in ('stable', 'changing'):
        baseline = Beamformer(config['audio'], config['beamforming'])
        optimized = Beams(config['audio'], config['beamforming'])
        state, frontend = Continuity(config['audio']), Frontend()
        records, error = {'reference': [], 'optimized': []}, 0.0
        for repeat in range(repeats):
            for index, row in enumerate(snapshots):
                angle = index * 14 if scenario == 'changing' else 0
                bearings = [Bearing(23 + angle, 33, 1), Bearing(-113 - angle, 42, 1)]
                snapshot = dict(row, epoch=repeat)
                began = time.perf_counter()
                context = state.prepare(snapshot)
                continuity = time.perf_counter() - began
                outputs, contexts = {}, {}
                for name in (('reference', 'optimized') if index % 2 == 0 else ('optimized', 'reference')):
                    began = time.perf_counter()
                    if name == 'reference':
                        outputs[name], diagnostic = baseline.run(snapshot['audio'], bearings)
                    else:
                        outputs[name], diagnostic = optimized.run_snapshot(snapshot, bearings, context)
                    beam_seconds = time.perf_counter() - began
                    began = time.perf_counter()
                    contexts[name] = [(features(waveform) if name == 'reference' else
                                       frontend.run(waveform, bearing, context, diagnostic['support_samples'])[0])
                                      for bearing, (waveform, _) in zip(bearings, outputs[name])]
                    records[name].append(dict(beamforming=beam_seconds, features=time.perf_counter() - began,
                                               steady=index > 0, diagnostic=diagnostic,
                                               continuity_seconds=continuity if name == 'optimized' else 0))
                for (first, _), (second, _), a, b in zip(outputs['reference'], outputs['optimized'], contexts['reference'], contexts['optimized']):
                    np.testing.assert_allclose(first, second, atol=3e-14, rtol=3e-12)
                    np.testing.assert_allclose(a, b, atol=3e-10, rtol=2e-10)
                    error = max(error, float(np.max(abs(first - second))))
        result[scenario] = dict(maximum_waveform_error=error, rows=records,
             steady_seconds={name: {phase: summary([row[phase] for row in rows if row['steady']])
                                   for phase in ('beamforming', 'features')} for name, rows in records.items()})
    return result


def benchmark(source, destination, configuration=None, model=None, windows=8, repeats=5):
    output = Path(destination)
    if output.exists():
        raise ValueError('Choose a new streaming benchmark JSON path')
    if not 2 <= windows <= 32 or not 1 <= repeats <= 20:
        raise ValueError('Streaming benchmark needs 2..32 windows and 1..20 repetitions')
    config = load_config(configuration or ROOT / 'config/band_stream.json')
    if not config.get('streaming', {}).get('enabled') or config['vision']['enabled']:
        raise ValueError('Use enabled streaming with live vision disabled for historical WAV')
    directory = Path(model).resolve() if model else ROOT / config['detector']['model_directory']
    if not validate_backend(config, validate_model(directory)):
        raise ValueError('Use the trained band model')
    detector = load_spectral(directory)
    baseline = deepcopy(config)
    baseline.pop('streaming')
    snapshots = []
    for snapshot in replay(source, config['audio'], Conditioning(14, config['audio'].get('gains'), config['audio'].get('delays'))):
        snapshots.append(snapshot)
        if len(snapshots) == windows:
            break
    if len(snapshots) < 2:
        raise ValueError('Need at least two complete overlapping windows')
    engines, setup, cold = {}, {}, {}
    for name, settings in [('reference', baseline), ('optimized', config)]:
        began = time.perf_counter()
        engines[name] = Pipeline(settings, detector)
        setup[name] = time.perf_counter() - began
        event, _ = engines[name].analyze(dict(snapshots[0], epoch=-1))
        cold[name] = dict(event['phase_seconds'], total=event['processing_seconds'])
    rows = {name: [] for name in engines}
    score_error = 0.0
    for repeat in range(repeats):
        for index, snapshot in enumerate(snapshots):
            events = {}
            order = ('reference', 'optimized') if (repeat + index) % 2 == 0 else ('optimized', 'reference')
            for name in order:
                event, _ = engines[name].analyze(dict(snapshot, epoch=repeat))
                events[name] = event
                row = dict(event['phase_seconds'], total=event['processing_seconds'], steady=index > 0,
                           srp=event['phase_seconds']['srp'], srp_diagnostics=event['srp_diagnostics'],
                           beam_diagnostics=event['beamforming_diagnostics'],
                           stream_diagnostics=event.get('stream_diagnostics'))
                rows[name].append(row)
            score_error = max(score_error, compare(events['reference'], events['optimized']))
    phases = ('srp', 'beamforming', 'features', 'detector', 'total')
    summaries = {name: {group: {key: summary([row[key] for row in records if row['steady'] == (group == 'steady')])
                               for key in phases} for group in ('reset', 'steady')} for name, records in rows.items()}
    ratios = {key: float(np.median([actual[key] / reference[key] for reference, actual in
                       zip(rows['reference'], rows['optimized']) if actual['steady']])) for key in phases}
    fallback = {}
    for name, settings in [('reference', baseline), ('optimized', config)]:
        engine = Pipeline(settings, detector)
        engine.analyze(dict(snapshots[0], epoch=100))
        event, _ = engine.analyze(dict(snapshots[-1], epoch=101))
        fallback[name] = event
    score_error = max(score_error, compare(fallback['reference'], fallback['optimized']))
    with wave.open(str(source), 'rb') as stream:
        pcm = np.frombuffer(stream.readframes(48128), dtype='<i2').reshape(-1, 14)
    conditioning_rows, conditioned = {}, {}
    for name in ('reference', 'optimized'):
        engine = (Conditioning(14, config['audio'].get('gains'), config['audio'].get('delays')) if name == 'reference'
                  else StreamConditioning(14, engines['optimized'].processing.runtime, config['audio'].get('gains'), config['audio'].get('delays')))
        times, blocks = [], []
        for start in range(0, len(pcm), 512):
            began = time.perf_counter()
            blocks.append(engine.feed(pcm[start:start + 512]))
            times.append(time.perf_counter() - began)
        conditioning_rows[name] = dict(packet_seconds=summary(times), total_seconds=float(sum(times)))
        conditioned[name] = np.concatenate(blocks)
    np.testing.assert_allclose(conditioned['reference'], conditioned['optimized'], atol=3e-14, rtol=1e-12)
    scenarios = beam_scenarios(snapshots, config, repeats)
    files = ['ps/streaming.py', 'ps/stream_kernel.c', 'ps/search.py', 'ps/search_kernel.c',
             'ps/beamforming.py', 'ps/dsp.py', 'ps/pipeline.py', 'scripts/benchmark_stream.py']
    value = dict(result='passed', machine=platform.machine(), python=sys.version, numpy=np.__version__,
                 windows=len(snapshots), repeats=repeats, model_tag=detector.metadata['model_tag'],
                 setup_seconds=setup, first_window_seconds=cold, phase_seconds=summaries,
                 steady_relative_time=ratios, maximum_score_error=score_error,
                 conditioning=conditioning_rows, beam_scenarios=scenarios,
                 fallback_seconds={name: event['processing_seconds'] for name, event in fallback.items()},
                 fallback_reset_reason=fallback['optimized']['stream_diagnostics']['reset_reason'],
                 target_seconds=0.5, all_steady_windows_within_target=summaries['optimized']['steady']['total']['maximum'] <= 0.5,
                 source_sha256=digest(source), weights_sha256=digest(directory / 'weights.npz'),
                 implementation_sha256={name: digest(ROOT / name) for name in files}, configuration=config,
                 rows=rows, thread_environment={name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                 scope='paired contiguous PCM replay; reset and steady windows separated; excludes collection wait, camera, vehicle detection and logging; conditioning measured separately, not added to parallel analysis')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    result = benchmark(args.input, args.output, args.config, args.model, args.windows, args.repeats)
    print(json.dumps({name: result[name] for name in ('result', 'windows', 'steady_relative_time', 'maximum_score_error', 'all_steady_windows_within_target')}))


if __name__ == '__main__':
    main()
