"""Compare owned and caller-buffer conditioning across calibration and model cases."""

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
from ps.conditioning import Conditioner, create_conditioning
from ps.configuration import load_config
from ps.dsp import Conditioning
from ps.streaming import Runtime, StreamConditioning


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), maximum=float(max(values)))


def compare_events(first, second):
    keys = ('state', 'event_id', 'ended_event', 'history_reset', 'horn_candidate', 'doa_valid', 'capture_candidate',
            'azimuth', 'depression', 'input_clipped', 'activation_clipped', 'frontend_clipped', 'acoustic_state')
    if any(first[key] != second[key] for key in keys):
        raise ValueError('Conditioning changed an event or selected direction')
    for name in ('direction_candidates', 'capture_candidates'):
        a = [(row['azimuth'], row['depression']) for row in first[name]]
        b = [(row['azimuth'], row['depression']) for row in second[name]]
        if a != b:
            raise ValueError('Conditioning changed camera candidate routing')
    np.testing.assert_allclose(second['scores'], first['scores'], atol=2e-10, rtol=0)
    return float(np.max(abs(np.asarray(second['scores']) - first['scores']), initial=0))


def model_check(source, config, model, library, stream_library, cases):
    from ps.band import load_spectral
    from ps.pipeline import Pipeline
    from run import replay
    if not config['streaming']['enabled'] or config['vision']['enabled']:
        raise ValueError('Model comparison requires streaming and disabled live vision')
    result = {}
    for case, (gains, delays) in cases.items():
        settings = deepcopy(config)
        settings['audio']['gains'], settings['audio']['delays'] = gains, delays
        settings['streaming'].update(conditioning='native', native_library=str(Path(stream_library).resolve()))
        settings.pop('conditioning', None)
        records = {}
        for name in ('stream', 'buffered_numpy', 'buffered_native'):
            current = deepcopy(settings)
            if name != 'stream':
                current['conditioning'] = dict(implementation='buffered', fir=name.split('_')[-1],
                                                native_library=str(Path(library).resolve()))
            pipeline = Pipeline(current, load_spectral(model))
            engine = create_conditioning(current, pipeline.processing.runtime)
            events = []
            for snapshot in replay(source, current['audio'], engine):
                events.append(pipeline.analyze(snapshot)[0])
                if len(events) == 4:
                    break
            if not events:
                raise ValueError('Model comparison needs a complete window')
            records[name] = events
        errors = {}
        for name, events in records.items():
            if len(events) != len(records['stream']):
                raise ValueError('Conditioning changed replay window count')
            errors[name] = max(compare_events(a, b) for a, b in zip(records['stream'], events))
        result[case] = dict(windows=len(records['stream']), maximum_score_error=errors,
                           states=[event['acoustic_state'] for event in records['stream']])
    return result


def benchmark(destination, library=None, stream_library=None, source=None, configuration=None, model=None, repeats=5):
    output = Path(destination)
    if output.exists():
        raise ValueError('Choose a fresh conditioning benchmark JSON path')
    if not 1 <= repeats <= 20:
        raise ValueError('Repetitions must be in 1..20')
    config = load_config(configuration or ROOT / 'config/band_stream.json')
    stream_path = stream_library or config.get('streaming', {}).get('native_library')
    runtime = Runtime(stream_path)
    path = library or config.get('conditioning', {}).get('native_library') or ROOT / 'ps' / (
        'conditioning.dll' if sys.platform == 'win32' else 'libconditioning.so')
    if source:
        with wave.open(str(source), 'rb') as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (14, 2, 48000):
                raise ValueError('Use synchronized PCM16/14-channel/48k WAV')
            pcm = np.frombuffer(stream.readframes(48128), dtype='<i2').reshape(-1, 14)
    else:
        pcm = np.random.RandomState(3).randint(-12000, 12000, size=(48128, 14)).astype(np.int16)
    if not len(pcm):
        raise ValueError('Need PCM samples')
    cases = dict(default=(None, None), gain=(np.linspace(0.9, 1.1, 14).tolist(), None),
                 fractional=(np.linspace(0.9, 1.1, 14).tolist(), np.linspace(-0.3, 0.5, 14).tolist()),
                 long_delay=(np.linspace(0.9, 1.1, 14).tolist(), np.linspace(-520, 1, 14).tolist()),
                 configured=(config['audio'].get('gains'), config['audio'].get('delays')))
    reports = {}
    for case, (gains, delays) in cases.items():
        engines = dict(numpy=Conditioning(14, gains, delays), stream=StreamConditioning(14, runtime, gains, delays))
        for fir in ('numpy', 'native'):
            for api in ('owned', 'into'):
                engines['buffered_' + fir + '_' + api] = Conditioner(14, gains, delays, fir, path)
        storage = {name: np.empty((512, 14), dtype=np.float64) for name in engines}
        times, errors = {name: [] for name in engines}, {name: 0.0 for name in engines}
        names = list(engines)
        for repeat in range(repeats):
            for engine in engines.values():
                engine.reset()
            for index, start in enumerate(range(0, len(pcm), 512)):
                block = pcm[start:start + 512]
                rotation = (repeat + index) % len(names)
                results = {}
                for name in names[rotation:] + names[:rotation]:
                    began = time.perf_counter()
                    results[name] = (engines[name].feed_into(block, storage[name][:len(block)]) if name.endswith('_into')
                                     else engines[name].feed(block))
                    times[name].append(time.perf_counter() - began)
                for name, actual in results.items():
                    np.testing.assert_allclose(actual, results['numpy'], atol=3e-14, rtol=1e-11)
                    errors[name] = max(errors[name], float(np.max(abs(actual - results['numpy']))))
        reports[case] = dict(latency=engines['stream'].latency, gains=gains, delays=delays,
                             packet_seconds={name: summary(values) for name, values in times.items()},
                             relative_to_stream={name: float(np.median(np.asarray(values) / times['stream'])) for name, values in times.items()},
                             maximum_waveform_error=errors)
    inference = {}
    if model:
        if not source:
            raise ValueError('Real PCM input is required for model comparison')
        inference = model_check(source, config, model, path, stream_path or runtime.library._name, cases)
    paths = [ROOT / name for name in ('ps/conditioning.py', 'ps/conditioning_kernel.c', 'ps/acquisition.py',
                                      'ps/configuration.py', 'run.py', 'scripts/build_conditioning.py')]
    paths.append(Path(__file__))
    result = dict(result='passed', machine=platform.machine(), python=sys.version, numpy=np.__version__, repeats=repeats,
                  samples=len(pcm), packet_samples=512, cases=reports, model_comparison=inference,
                  model_tag=json.loads((Path(model) / 'manifest.json').read_text())['model_tag'] if model else None,
                  weights_sha256=digest(Path(model) / 'weights.npz') if model else None,
                  pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), source_sha256=digest(source) if source else None,
                  library=str(Path(path).resolve()), library_sha256=digest(path),
                  stream_library=str(Path(runtime.library._name).resolve()), stream_library_sha256=digest(runtime.library._name),
                  packet_period_seconds=512 / 48000,
                  buffered_packets_within_period={name: row['packet_seconds']['buffered_native_into']['maximum'] <= 512 / 48000
                                                 for name, row in reports.items()},
                  implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                  scope='rotated paired per-packet processing; independent model/event replay when requested; not ARM or full-pipeline speed proof')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--library')
    parser.add_argument('--stream-library')
    parser.add_argument('--input')
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    result = benchmark(args.output, args.library, args.stream_library, args.input, args.config, args.model, args.repeats)
    print(json.dumps(dict(result=result['result'], relative_to_stream={name: row['relative_to_stream'] for name, row in result['cases'].items()})))


if __name__ == '__main__':
    main()
