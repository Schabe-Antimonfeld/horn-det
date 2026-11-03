"""Standalone board service or deterministic PCM replay; no PC prototype imports."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np

from ps.acquisition import CaptureDevice, CapturePump, WindowBuffer
from ps.configuration import ROOT, load_config
from ps.detector import Detector, validate_backend, validate_model
from ps.pipeline import Pipeline
from ps.vision import CameraCache, Vision


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def replay(path, settings, conditioning):
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (14, 2, 48000):
            raise ValueError('Replay must be synchronized 14-channel PCM16/48k WAV')
        history = WindowBuffer(settings, np.float64)
        scratch = (np.empty((512, 14), dtype=np.float64) if callable(getattr(conditioning, 'feed_into', None)) else None)
        first = 0
        while True:
            data = source.readframes(512)
            audio = np.frombuffer(data, dtype='<i2').reshape(-1, 14)
            if len(audio) < 512:
                break
            processed = conditioning.feed_into(audio, scratch) if scratch is not None else conditioning.feed(audio)
            snapshot = history.feed({'audio': processed, 'first': first, 'gap': False, 'clipped': False})
            first += 512
            if snapshot is not None:
                yield snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--replay')
    parser.add_argument('--output')
    parser.add_argument('--output-root')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.replay and config['vision']['enabled']:
        raise ValueError('Historical audio cannot be matched to live camera images; disable vision for WAV replay')
    if args.replay and config['search'].get('backend', 'numpy') == 'fpga':
        raise ValueError('WAV replay requires the NumPy SRP backend')
    directory = Path(args.model).resolve() if args.model else resolve(config['detector']['model_directory'])
    metadata = validate_model(directory)
    validate_backend(config, metadata)
    from ps.streaming import Processing
    from ps.conditioning import create_conditioning
    processing = Processing(config)
    prepared = create_conditioning(config)
    if args.check:
        print(json.dumps({'config': 'valid', 'model_tag': metadata['model_tag'],
                          'classifier': metadata['architecture'], 'detector_backend': config['detector']['backend'],
                          'direction_threshold': config['detector']['threshold'],
                          'capture_threshold': config['detector'].get('capture_threshold', config['detector']['threshold']),
                          'board_runtime': 'requires PYNQ/NumPy; hardware not checked',
                          'srp_backend': config['search'].get('backend', 'numpy'),
                          'search_algorithm': config['search'].get('algorithm', 'reference'),
                          'search_implementation': config['search'].get('implementation', 'numpy'),
                          'beamforming_implementation': config.get('beamforming', {}).get('implementation', 'reference'),
                          'streaming_enabled': processing is not None,
                          'refinement': config.get('refinement'),
                          'conditioning_implementation': (config['conditioning'].get('implementation', 'buffered')
                                                           if prepared is not None else config.get('streaming', {}).get('conditioning', 'numpy')),
                          'conditioning_fir': getattr(prepared, 'fir', 'numpy'),
                          'thread_environment': {name: os.environ.get(name) for name in
                                                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                          'vision_enabled': config['vision']['enabled']}))
        return 0
    if args.output and args.output_root:
        raise ValueError('Choose --output or --output-root, not both')
    if args.output_root:
        args.output = str(Path(args.output_root) / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + str(os.getpid())))
    if not args.output:
        raise ValueError('Choose a fresh --output directory')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous event evidence')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    pump = device = camera_cache = capture = scorer = None
    processed, status = 0, 'starting'
    try:
        conditioning = prepared
        if args.replay:
            detector = Detector(directory)
            snapshots = replay(args.replay, config['audio'], conditioning)
        else:
            bitstream = resolve(config['hardware']['overlay'])
            if not bitstream.is_file() or not bitstream.with_suffix('.hwh').is_file():
                raise ValueError('Matching bitstream and HWH are missing; complete the hardware handoff')
            from pynq import Overlay, allocate
            overlay = Overlay(str(bitstream))
            detector = Detector(directory)
            capture = CaptureDevice(overlay, allocate, config['audio'], config['hardware']['timeout'])
            pump = CapturePump(capture, config['audio'], conditioning)
            pump.thread.start()
            def live_snapshots():
                while True:
                    yield pump.receive()
            snapshots = live_snapshots()
        vision = None
        if config['vision']['enabled']:
            settings = dict(config['vision'])
            settings['model'] = str(resolve(settings['model']))
            if settings['configuration']:
                settings['configuration'] = str(resolve(settings['configuration']))
            vision = Vision(settings)
            camera_cache = CameraCache(config['camera']['sources'], vision, **config.get('camera_cache', {}))
            camera_cache.start()
        pipeline = Pipeline(config, detector, processing=processing)
        clock_origin, clock_epoch = None, None
        status = 'running'
        with (output / 'events.jsonl').open('w', encoding='utf-8') as stream:
            for snapshot in snapshots:
                if clock_origin is None or clock_epoch != snapshot['epoch']:
                    clock_origin = time.monotonic() - snapshot['end'] / float(config['audio']['sample_rate'])
                    clock_epoch = snapshot['epoch']
                representative = clock_origin + ((snapshot['first'] + snapshot['end']) / 2 - conditioning.latency) / config['audio']['sample_rate']
                event, evidence = pipeline.analyze(snapshot)
                if 'conditioning' in snapshot:
                    event['conditioning'] = snapshot['conditioning']
                event['clock_kind'] = 'replay_sample_clock' if args.replay else 'unverified_host_clock'
                if vision is not None:
                    event['capture'] = pipeline.capture(event, evidence, camera_cache, vision, output, representative)
                    event['camera_health'] = camera_cache.health()
                event['skipped_windows'] = pump.skipped if pump else 0
                event['capture_gaps'] = pump.gaps if pump else 0
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
                processed += 1
                (output / 'health.json').write_text(json.dumps({'status': status, 'windows': processed,
                                                               'skipped_windows': event['skipped_windows'],
                                                               'capture_gaps': event['capture_gaps']}), encoding='utf-8')
                if args.limit and processed >= args.limit:
                    break
        status = 'completed'
        return 0
    except KeyboardInterrupt:
        status = 'stopped'
        return 0
    except Exception as error:
        status = 'failed: ' + str(error)
        raise
    finally:
        failures = []
        for resource in (camera_cache, pump if pump is not None else capture, scorer, device):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    failures.append(str(error))
        (output / 'health.json').write_text(json.dumps({'status': status, 'windows': processed,
                                                       'cleanup_errors': failures,
                                                       'cameras': camera_cache.health() if camera_cache is not None else None,
                                                       'board_validated': False}), encoding='utf-8')
        if failures:
            raise RuntimeError('Cleanup incomplete: ' + '; '.join(failures))


if __name__ == '__main__':
    main()
