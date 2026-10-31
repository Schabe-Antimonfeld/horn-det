"""Measure matched steady acoustic windows on the actual host, excluding live vision."""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.band import load_spectral, validate_backend, validate_model
from ps.configuration import load_config
from ps.dsp import Conditioning
from ps.pipeline import Pipeline
from run import replay


def benchmark(source, destination, configuration=None, model=None, windows=6, repeats=3):
    output = Path(destination)
    if output.exists():
        raise ValueError('Choose a new benchmark JSON path')
    if not 1 <= windows <= 32 or not 1 <= repeats <= 20:
        raise ValueError('Invalid benchmark window or repetition count')
    config = load_config(configuration or ROOT / 'config/band_fast.json')
    if config['search'].get('backend', 'numpy') != 'numpy' or config['vision']['enabled']:
        raise ValueError('WAV benchmark requires PS search and disabled live vision')
    directory = Path(model).resolve() if model else ROOT / config['detector']['model_directory']
    if not validate_backend(config, validate_model(directory)):
        raise ValueError('This benchmark supports NumPy band models')
    detector = load_spectral(directory)
    baseline = deepcopy(config)
    for key in ('algorithm', 'implementation', 'pair_mode', 'fft_fold', 'native_precision', 'native_library',
                'coarse_azimuth', 'coarse_depression', 'regions', 'bounds'):
        baseline['search'].pop(key, None)
    snapshots = []
    for snapshot in replay(source, config['audio'], Conditioning(14, config['audio'].get('gains'), config['audio'].get('delays'))):
        snapshots.append(snapshot)
        if len(snapshots) == windows:
            break
    if not snapshots:
        raise ValueError('Need at least one complete aligned audio window')
    engines, setup = {}, {}
    for name, settings in (('reference', baseline), ('optimized', config)):
        began = time.perf_counter()
        engines[name] = Pipeline(settings, detector)
        setup[name] = time.perf_counter() - began
        engines[name].analyze(dict(snapshots[0], epoch=-1))
    rows = {name: [] for name in engines}
    for repeat in range(repeats):
        for index, snapshot in enumerate(snapshots):
            names = ('reference', 'optimized') if (repeat + index) % 2 == 0 else ('optimized', 'reference')
            for name in names:
                event, _ = engines[name].analyze(dict(snapshot, epoch=repeat * len(snapshots) + index))
                rows[name].append(dict(event['phase_seconds'], total=event['processing_seconds']))
    summaries = {name: {key: dict(median=float(np.median([row[key] for row in values])),
                                 p90=float(np.percentile([row[key] for row in values], 90)),
                                 maximum=float(max(row[key] for row in values))) for key in values[0]}
                 for name, values in rows.items()}
    relative = {key: float(np.median([current[key] / previous[key] for current, previous in
                                     zip(rows['optimized'], rows['reference']) if previous[key] > 0]))
                for key in ('srp', 'total')}
    value = dict(result='measured', machine=platform.machine(), python=sys.version, numpy=np.__version__,
                 model_tag=detector.metadata['model_tag'], windows=len(snapshots), repeats=repeats, setup_seconds=setup,
                 search_configuration=config['search'], window_samples=config['audio']['window_samples'],
                 update_samples=config['audio']['update_samples'],
                 phase_seconds=summaries, relative_time=relative,
                 srp_relative_target_met=relative['srp'] <= 0.1,
                 acoustic_steady_650ms_met=summaries['optimized']['total']['median'] <= 0.65,
                 scope='steady PCM replay; includes SRP/beam/features/model/event, excludes acquisition wait, camera, vehicle detection, logging and startup; repeated windows are not event-quality evidence',
                 thread_environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--windows', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.input, args.output, args.config, args.model, args.windows, args.repeats)))


if __name__ == '__main__':
    main()
