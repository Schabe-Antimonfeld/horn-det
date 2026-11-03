"""Measure paired acoustic runtimes on the same PCM without claiming board acceptance."""

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--reset-every', type=int, default=0)
    parser.add_argument('--skip-every', type=int, default=0)
    parser.add_argument('--numpy-dsp', action='store_true')
    parser.add_argument('--numpy-mel', action='store_true')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--native-prepare', action='store_true')
    parser.add_argument('--native-beam', action='store_true')
    parser.add_argument('--native-mel', action='store_true')
    args = parser.parse_args()
    root = Path(args.runtime).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from ps.configuration import load_config
    from ps.detector import Detector
    from ps.pipeline import Pipeline
    from ps.conditioning import create_conditioning
    from run import replay
    from scripts.build import build
    output = Path(args.output).resolve()
    if output.exists() or args.repeats < 1 or args.limit < 0 or args.reset_every < 0 or args.skip_every < 0:
        raise ValueError('Choose a fresh output and valid repetition/window counts')
    config = load_config(args.config or root / 'config/default.json')
    if args.numpy_dsp:
        config['acceleration'] = dict(enabled=False)
    if args.numpy_mel:
        config.setdefault('acceleration', {})['mel'] = False
    if args.native_beam:
        config.setdefault('acceleration', {})['beam'] = True
    if args.native_mel:
        config.setdefault('acceleration', {})['mel'] = True
    if args.native_prepare:
        config.setdefault('acceleration', {})['prepare'] = True
    builds = build()
    source = Path(args.input).resolve()
    with wave.open(str(source), 'rb') as stream:
        count = max(0, 1 + (stream.getnframes() - config['audio']['window_samples']) // config['audio']['update_samples'])
    count = min(count, args.limit) if args.limit else count
    if not count:
        raise ValueError('Input contains no completed analysis windows')
    rows = []
    for repeat in range(args.repeats):
        pipelines = {'native': Pipeline(config, Detector(root / 'model'))}
        if args.paired:
            reference = json.loads(json.dumps(config))
            reference['acceleration'] = dict(enabled=False)
            pipelines['numpy'] = Pipeline(reference, Detector(root / 'model'))
        snapshots = replay(source, config['audio'], create_conditioning(config))
        for index, snapshot in enumerate(snapshots):
            if index >= count:
                break
            if args.skip_every and (index + 1) % args.skip_every == 0:
                continue
            packet = dict(snapshot, epoch=index // args.reset_every if args.reset_every else snapshot['epoch'])
            order = list(pipelines)
            if (index + repeat) % 2:
                order.reverse()
            for variant in order:
                cpu_clock = getattr(time, 'thread_time', None)
                began = cpu_clock() if cpu_clock else None
                event, _ = pipelines[variant].analyze(packet)
                event['benchmark_cpu_seconds'] = cpu_clock() - began if cpu_clock else None
                event['repeat'], event['index'] = repeat, index
                event['variant'] = variant
                rows.append(event)
        print('repeat {} / {}: {} windows'.format(repeat + 1, args.repeats, count), flush=True)
    groups = {}
    for row in rows:
        key = '{}-{}-{}'.format(row.get('variant', 'native'), row['beamforming_diagnostics']['beams'],
                            'continuous' if row.get('stream_diagnostics', {}).get('reusable') else 'reset')
        groups.setdefault(key, []).append(row)
    summaries = {}
    for key, group in groups.items():
        values = [row['processing_seconds'] for row in group]
        summaries[key] = dict(windows=len(group), median_ms=float(np.median(values) * 1000),
                              p90_ms=float(np.percentile(values, 90) * 1000), maximum_ms=max(values) * 1000,
                              over_500ms=sum(value > .5 for value in values),
                              phase_median_ms={name: float(np.median([row['phase_seconds'][name] for row in group]) * 1000)
                                               for name in ('srp', 'beamforming', 'features', 'detector')})
    result = dict(format=1, machine=platform.machine(), python=sys.version, numpy=np.__version__,
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256((root / 'model/weights.npz').read_bytes()).hexdigest(),
                  configuration=config, builds=builds, windows=rows, summaries=summaries,
                  board_validated=False, scope='Unpaced same-PCM acoustic timing; excludes collection/camera/storage/queue pressure; no direction or horn ground truth in the handoff recording')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
