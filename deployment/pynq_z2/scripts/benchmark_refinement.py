"""Compare two independent refinements against identical conditioned streaming PCM."""

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
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_stream import compare, summary
from ps.band import load_spectral, validate_backend, validate_model
from ps.conditioning import create_conditioning
from ps.configuration import load_config
from ps.geometry import Bearing
from ps.pipeline import Pipeline
from ps.refinement import Beams, Processing
from ps.streaming import Beams as StreamBeams, Continuity
from ps.dsp import features
from run import replay


def beam_scenarios(snapshots, config, repeats):
    result = {}
    for scenario in ('stable', 'partial_change', 'all_change', 'reordered'):
        engines = dict(reference=StreamBeams(config['audio'], config['beamforming']),
                       refined=Beams(config['audio'], config['beamforming']))
        state, rows, error, feature_error = Continuity(config['audio']), {name: [] for name in engines}, 0.0, 0.0
        for repeat in range(repeats):
            for index, snapshot in enumerate(snapshots):
                context = state.prepare(dict(snapshot, epoch=repeat))
                bearings = [Bearing(23 + (index * 14 if scenario == 'all_change' else 0), 33, 1),
                            Bearing(-113 - (index * 14 if scenario in ('partial_change', 'all_change') else 0), 42, 1)]
                if scenario == 'reordered' and index % 2:
                    bearings.reverse()
                outputs = {}
                for name in (('reference', 'refined') if (repeat + index) % 2 == 0 else ('refined', 'reference')):
                    began = time.perf_counter()
                    outputs[name], metric = engines[name].run_snapshot(snapshot, bearings, context)
                    rows[name].append(dict(seconds=time.perf_counter() - began, steady=index > 0, diagnostic=metric))
                for (first, common), (second, actual_common) in zip(outputs['reference'], outputs['refined']):
                    np.testing.assert_allclose(first, second, atol=3e-14, rtol=3e-12)
                    a, b = features(first), features(second)
                    np.testing.assert_allclose(a, b, atol=3e-10, rtol=2e-10)
                    if common != actual_common:
                        raise ValueError('Beam latency changed')
                    error = max(error, float(np.max(abs(first - second))))
                    feature_error = max(feature_error, float(np.max(abs(a - b))))
        result[scenario] = dict(rows=rows, maximum_waveform_error=error, maximum_feature_error=feature_error,
                               steady_seconds={name: summary([row['seconds'] for row in values if row['steady']])
                                               for name, values in rows.items()})
    return result


def benchmark(source, destination, configuration=None, model=None, windows=12, repeats=5):
    output = Path(destination)
    if output.exists():
        raise ValueError('Choose a new refinement benchmark JSON path')
    if not 2 <= windows <= 32 or not 1 <= repeats <= 20:
        raise ValueError('Use 2..32 windows and 1..20 repetitions')
    config = load_config(configuration or ROOT / 'config/band_refined.json')
    if 'refinement' not in config or config['vision']['enabled']:
        raise ValueError('Use a refinement configuration with live vision disabled for historical WAV')
    directory = Path(model).resolve() if model else ROOT / config['detector']['model_directory']
    metadata = validate_model(directory)
    if not validate_backend(config, metadata):
        raise ValueError('Use the trained spectral detector')
    snapshots = []
    conditioner = create_conditioning(config)
    for snapshot in replay(source, config['audio'], conditioner):
        snapshots.append(snapshot)
        if len(snapshots) == windows:
            break
    if len(snapshots) < 2:
        raise ValueError('Need at least two complete overlapping windows')
    engines, setup, first = {}, {}, {}
    for name, fused, beams in [('reference', False, False), ('fusion_only', True, False),
                                ('beams_only', False, True), ('combined', True, True)]:
        settings = deepcopy(config)
        settings['refinement'].update(fused_input=fused, independent_beams=beams)
        if not fused:
            settings['refinement'].pop('native_library', None)
        began = time.perf_counter()
        detector = load_spectral(directory)
        engines[name] = Pipeline(settings, detector, processing=Processing(settings, detector.frontend))
        setup[name] = time.perf_counter() - began
        event, _ = engines[name].analyze(dict(snapshots[0], epoch=-1))
        first[name] = event['processing_seconds']
    rows = {name: [] for name in engines}
    score_error = 0.0
    decisions = {name: [] for name in engines}
    for repeat in range(repeats):
        for index, snapshot in enumerate(snapshots):
            events, maps, correlations = {}, {}, {}
            names = list(engines)
            rotation = (repeat + index) % len(names)
            for name in names[rotation:] + names[:rotation]:
                engine = engines[name]
                event, _ = engine.analyze(dict(snapshot, epoch=repeat))
                events[name] = event
                correlations[name] = engine.searcher.last_correlations.copy()
                maps[name] = engine.searcher.score(correlations[name], np.arange(engine.searcher.vectors.shape[0]))
                rows[name].append(dict(event['phase_seconds'], total=event['processing_seconds'], steady=index > 0,
                                      srp_diagnostics=event['srp_diagnostics'],
                                      beam_diagnostics=event['beamforming_diagnostics']))
                decisions[name].append(dict(state=event['state'], scores=event['scores']))
            for name in names[1:]:
                score_error = max(score_error, compare(events['reference'], events[name]))
                np.testing.assert_array_equal(correlations['reference'].view(np.uint64), correlations[name].view(np.uint64))
                np.testing.assert_array_equal(maps['reference'].view(np.uint64), maps[name].view(np.uint64))
    phases = ('srp', 'beamforming', 'features', 'detector', 'total')
    summaries = {name: {group: {phase: summary([row[phase] for row in values if row['steady'] == (group == 'steady')])
                                for phase in phases} for group in ('reset', 'steady')} for name, values in rows.items()}
    fallback, events = {}, {}
    for name, engine in engines.items():
        engine.analyze(dict(snapshots[0], epoch=100))
        event, _ = engine.analyze(dict(snapshots[-1], epoch=101))
        events[name] = event
        fallback[name] = dict(seconds=event['processing_seconds'], reason=event['stream_diagnostics']['reset_reason'],
                              diagnostic=event['beamforming_diagnostics'])
    for name in list(engines)[1:]:
        score_error = max(score_error, compare(events['reference'], events[name]))
    files = ['ps/refinement.py', 'ps/refinement_kernel.c', 'scripts/build_refinement.py', 'scripts/benchmark_refinement.py',
             'ps/streaming.py', 'ps/stream_kernel.c', 'ps/search.py', 'ps/search_kernel.c',
             'ps/beamforming.py', 'ps/dsp.py', 'ps/pipeline.py', 'ps/conditioning.py', 'ps/conditioning_kernel.c']
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    libraries = {name: str(value.library._name) for name, value in
                 [('search', engines['combined'].searcher.native), ('streaming', engines['combined'].processing.runtime),
                  ('refinement', engines['combined'].searcher.preparation), ('conditioning', getattr(conditioner, '_runtime', None))]
                 if value is not None}
    value = dict(result='passed', machine=platform.machine(), python=sys.version, numpy=np.__version__,
                 configuration=config, windows=len(snapshots), repeats=repeats, model_tag=metadata['model_tag'],
                 source_sha256=digest(source), weights_sha256=digest(directory / 'weights.npz'),
                 implementation_sha256={name: digest(ROOT / name) for name in files},
                 library_paths=libraries, library_sha256={name: digest(path) for name, path in libraries.items()},
                 setup_seconds=setup, first_window_seconds=first, phase_seconds=summaries, rows=rows,
                 maximum_score_error=score_error, correlations_bitwise_equal=True, complete_grid_bitwise_equal=True,
                 decisions=decisions, fallback=fallback, beam_scenarios=beam_scenarios(snapshots, config, repeats),
                 target_seconds=0.5,
                 overruns={name: sum(row['total'] > 0.5 for row in values if row['steady']) for name, values in rows.items()},
                 thread_environment={name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                 scope='paired conditioned PCM replay; four implementations with unchanged model/gates; no new road accuracy or ARM acceptance; collection, conditioning, camera and logging excluded from analysis time')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--windows', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    value = benchmark(args.input, args.output, args.config, args.model, args.windows, args.repeats)
    print(json.dumps(dict(result=value['result'], windows=value['windows'], maximum_score_error=value['maximum_score_error'],
                         steady_median_seconds={name: result['steady']['total']['median'] for name, result in value['phase_seconds'].items()})))


if __name__ == '__main__':
    main()
