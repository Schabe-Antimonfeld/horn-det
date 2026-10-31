"""Compare exact beam backends with identical SRP, model and completed PCM windows."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.band import load_spectral, validate_backend, validate_model
from ps.beamforming import Beamformer
from ps.configuration import load_config
from ps.dsp import Conditioning, features
from ps.geometry import Bearing
from ps.pipeline import Pipeline
from run import replay


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), maximum=float(max(values)))


def decisions(event):
    """Exclude timings and scores when comparing the actual event decisions."""
    result = {key: value for key, value in event.items() if key not in
            {'processing_seconds', 'phase_seconds', 'srp_diagnostics', 'beamforming_diagnostics', 'score',
             'score_margin', 'scores', 'direction_candidates', 'capture_candidates'}}
    for key in ('direction_candidates', 'capture_candidates'):
        result[key] = [{name: value for name, value in row.items() if name != 'score'} for row in event[key]]
    return result


def benchmark(source, destination, configuration=None, model=None, windows=6, repeats=5, sweep=False, native_library=None):
    output = Path(destination)
    if output.exists():
        raise ValueError('Choose a new beam benchmark JSON path')
    if not 1 <= windows <= 32 or not 1 <= repeats <= 20:
        raise ValueError('Invalid benchmark window or repetition count')
    config = load_config(configuration or ROOT / 'config/band_fast.json')
    if config['search'].get('backend', 'numpy') != 'numpy' or config['vision']['enabled']:
        raise ValueError('Beam WAV benchmark requires PS search and disabled live vision')
    directory = Path(model).resolve() if model else ROOT / config['detector']['model_directory']
    if not validate_backend(config, validate_model(directory)):
        raise ValueError('Beam benchmark requires a band model')
    detector = load_spectral(directory)
    options = dict(reference=dict(implementation='reference'))
    if sweep:
        options.update(cached=dict(implementation='cached'), fft512=dict(implementation='fft', fft_size=512),
                       fft1024=dict(implementation='fft', fft_size=1024))
        if native_library:
            options['native'] = dict(implementation='native', native_library=str(Path(native_library).resolve()))
    else:
        options['optimized'] = dict(config.get('beamforming', {}))
    snapshots = []
    conditioning = Conditioning(14, config['audio'].get('gains'), config['audio'].get('delays'))
    for snapshot in replay(source, config['audio'], conditioning):
        snapshots.append(snapshot)
        if len(snapshots) == windows:
            break
    if not snapshots:
        raise ValueError('Need one complete aligned PCM window')
    engines, setup, cold = {}, {}, {}
    for name, settings in options.items():
        current = deepcopy(config)
        current['beamforming'] = settings
        began = time.perf_counter()
        engines[name] = Pipeline(current, detector)
        setup[name] = time.perf_counter() - began
        event, _ = engines[name].analyze(dict(snapshots[0], epoch=-1))
        cold[name] = dict(event['phase_seconds'], total=event['processing_seconds'])
    rows, parity = {name: [] for name in engines}, {}
    names = list(engines)
    for repeat in range(repeats):
        for index, snapshot in enumerate(snapshots):
            rotation = (repeat + index) % len(names)
            events = {}
            for name in names[rotation:] + names[:rotation]:
                event, _ = engines[name].analyze(dict(snapshot, epoch=repeat * len(snapshots) + index))
                rows[name].append(dict(event['phase_seconds'], total=event['processing_seconds']))
                events[name] = event
            reference = events['reference']
            for name, event in events.items():
                if decisions(reference) != decisions(event):
                    raise ValueError('Beam backend changed an event decision: ' + name)
                np.testing.assert_allclose(event['scores'], reference['scores'], atol=2e-10, rtol=0)
                parity[name] = max(parity.get(name, 0), float(np.max(np.abs(np.asarray(event['scores']) - reference['scores']), initial=0)))
    summaries = {name: {key: summary([row[key] for row in records]) for key in records[0]} for name, records in rows.items()}
    relative = {name: {key: float(np.median([row[key] / baseline[key] for row, baseline in zip(records, rows['reference'])]))
                       for key in ('beamforming', 'total')} for name, records in rows.items()}
    micro = {}
    bearings = [Bearing(-179, 0, 1), Bearing(23, 33, 1), Bearing(-85, 51, 1), Bearing(None, 90, 1)]
    audio = snapshots[0]['audio']
    for count in (1, 2, 4):
        runs = {name: Beamformer(config['audio'], setting) for name, setting in options.items()}
        measurements = {name: [] for name in runs}
        reference, _ = runs['reference'].run(audio, bearings[:count])
        errors = {}
        for name, engine in runs.items():
            actual, _ = engine.run(audio, bearings[:count])
            errors[name] = max(float(np.max(np.abs(first[0] - second[0]))) for first, second in zip(actual, reference))
            for (waveform, latency), (expected, common) in zip(actual, reference):
                np.testing.assert_allclose(waveform, expected, atol=3e-14, rtol=3e-13)
                np.testing.assert_allclose(features(waveform), features(expected), atol=2e-10, rtol=1e-10)
                if latency != common:
                    raise ValueError('Beam backend changed latency')
        for repeat in range(repeats):
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                began = time.perf_counter()
                runs[name].run(audio, bearings[:count])
                measurements[name].append(time.perf_counter() - began)
        micro[str(count)] = {name: dict(seconds=summary(values), maximum_waveform_error=errors[name]) for name, values in measurements.items()}
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in [ROOT / 'ps/dsp.py', ROOT / 'ps/beamforming.py',
              ROOT / 'ps/beam_kernel.c', ROOT / 'ps/search.py', ROOT / 'ps/search_kernel.c', Path(__file__)]}
    value = dict(result='passed', machine=platform.machine(), python=sys.version, numpy=np.__version__,
                 model_tag=detector.metadata['model_tag'], weights_sha256=digest(directory / 'weights.npz'),
                 source_sha256=digest(source), implementation_sha256=inputs,
                 native_beam_sha256=digest(native_library) if native_library else None,
                 search_configuration=config['search'], beam_options=options, windows=len(snapshots), repeats=repeats,
                 window_samples=config['audio']['window_samples'], setup_seconds=setup, first_window_seconds=cold,
                 phase_seconds=summaries, relative_time=relative, maximum_score_error=parity, beam_counts=micro,
                 scope='paired steady PCM replay, unchanged SRP; excludes acquisition wait, vision, logging and startup; repeated windows are not road/event-quality evidence',
                 thread_environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--windows', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--native-library')
    args = parser.parse_args()
    result = benchmark(args.input, args.output, args.config, args.model, args.windows, args.repeats, args.sweep, args.native_library)
    print(json.dumps({key: result[key] for key in ('result', 'windows', 'relative_time', 'maximum_score_error')}))


if __name__ == '__main__':
    main()
