"""Standalone board service or deterministic PCM replay; no PC prototype imports."""

import argparse
import importlib.util
import json
from pathlib import Path
import time
import wave

import numpy as np

from ps.acquisition import CaptureDevice, CapturePump, WindowBuffer
from ps.configuration import ROOT, load_config
from ps.detector import Detector
from ps.dsp import Conditioning
from ps.integer_kernel import check_export
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
        first = 0
        while True:
            data = source.readframes(512)
            audio = np.frombuffer(data, dtype='<i2').reshape(-1, 14)
            if len(audio) < 512:
                break
            snapshot = history.feed({'audio': conditioning.feed(audio), 'first': first, 'gap': False, 'clipped': False})
            first += 512
            if snapshot is not None:
                yield snapshot


def cnn_client():
    path = ROOT / 'vendor/fpga/cnn_host.py'
    if not path.is_file():
        path = ROOT.parents[1] / 'fpga/cnn_host.py'
    specification = importlib.util.spec_from_file_location('board_cnn_host', str(path))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.CnnDevice


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--replay')
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.replay and config['vision']['enabled']:
        raise ValueError('Historical audio cannot be matched to live camera images; disable vision for WAV replay')
    directory = resolve(args.model or config['detector']['model_directory'])
    metadata = check_export(directory)
    if args.check:
        print(json.dumps({'config': 'valid', 'model_tag': metadata['model_tag'],
                          'board_runtime': 'requires PYNQ/NumPy; hardware not checked',
                          'vision_enabled': config['vision']['enabled']}))
        return 0
    if not args.output:
        raise ValueError('Choose a fresh --output directory')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous event evidence')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    pump = device = camera_cache = capture = None
    processed, status = 0, 'starting'
    try:
        conditioning = Conditioning(config['audio']['channels'], config['audio'].get('gains'), config['audio'].get('delays'))
        if args.replay:
            detector = Detector(directory)
            snapshots = replay(args.replay, config['audio'], conditioning)
        else:
            if config['detector']['backend'] != 'fpga':
                raise ValueError('Live mode requires FPGA inference; integer backend is replay only')
            bitstream = resolve(config['hardware']['overlay'])
            if not bitstream.is_file() or not bitstream.with_suffix('.hwh').is_file():
                raise ValueError('Matching bitstream and HWH are missing; complete the hardware handoff')
            from pynq import Overlay, allocate
            overlay = Overlay(str(bitstream))
            device = cnn_client()(overlay, metadata, allocate, config['hardware']['timeout'])
            detector = Detector(directory, device)
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
            camera_cache = CameraCache(config['camera']['sources'], vision)
            camera_cache.start()
        pipeline = Pipeline(config, detector)
        clock_origin, clock_epoch = None, None
        status = 'running'
        with (output / 'events.jsonl').open('w', encoding='utf-8') as stream:
            for snapshot in snapshots:
                if clock_origin is None or clock_epoch != snapshot['epoch']:
                    clock_origin = time.monotonic() - snapshot['end'] / float(config['audio']['sample_rate'])
                    clock_epoch = snapshot['epoch']
                representative = clock_origin + ((snapshot['first'] + snapshot['end']) / 2 - conditioning.latency) / config['audio']['sample_rate']
                event, evidence = pipeline.analyze(snapshot)
                event['clock_kind'] = 'replay_sample_clock' if args.replay else 'unverified_host_clock'
                if vision is not None:
                    event['capture'] = pipeline.capture(event, evidence, camera_cache, vision, output, representative)
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
        for resource in (camera_cache, pump if pump is not None else capture, device):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    failures.append(str(error))
        (output / 'health.json').write_text(json.dumps({'status': status, 'windows': processed,
                                                       'cleanup_errors': failures,
                                                       'board_validated': False}), encoding='utf-8')
        if failures:
            raise RuntimeError('Cleanup incomplete: ' + '; '.join(failures))


if __name__ == '__main__':
    main()
