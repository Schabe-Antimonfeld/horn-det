"""Verify assets, build native libraries, replay PCM and recompute fixed-weight metrics."""

import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config
from ps.evaluation import compare, evaluate, verify_assets
from scripts.build import build, configure
from run import main as run


def reproduce(output, compiler=None, libraries=None):
    verify_assets(ROOT)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh reproduction directory')
    reports = build(compiler, libraries)
    config = load_config()
    if libraries:
        configure(config, libraries)
    output.mkdir(parents=True)
    configuration = output / 'configuration.json'
    configuration.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    expected = json.loads((ROOT / 'examples/expected.json').read_text(encoding='utf-8'))
    replays = {}
    for name, reference in expected.items():
        directory = output / Path(name).stem
        run(['--config', str(configuration), '--replay', str(ROOT / 'examples' / name), '--output', str(directory)])
        rows = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        frozen = [{key: value for key, value in row.items() if key not in ('state', 'direction_threshold')}
                  for row in reference]
        compare(frozen, rows, name)
        for row in rows:
            if row['direction_threshold'] != config['detector']['threshold']:
                raise ValueError('Replay direction threshold disagrees with configuration')
            if row['capture_candidate'] and row['state'] != 'CONFIRMED':
                raise ValueError('Single-window candidate did not confirm immediately')
            expected_horn = bool(not any(row[key] for key in ('input_clipped', 'activation_clipped', 'frontend_clipped'))
                                 and row['score'] >= config['detector']['threshold'])
            if row['horn_confirmed'] != expected_horn:
                raise ValueError('Single-window horn confirmation mismatch')
        elapsed = [row['processing_seconds'] for row in rows]
        replays[name] = dict(windows=len(rows), states=[row['state'] for row in rows],
                             median_seconds=float(np.median(elapsed)), maximum_seconds=float(max(elapsed)),
                             overruns_500ms=sum(value > 0.5 for value in elapsed))
    result = dict(result='passed', machine=platform.machine(), python=sys.version, numpy=np.__version__,
                  model_tag=405187002, assets_sha256=hashlib.sha256((ROOT / 'assets.json').read_bytes()).hexdigest(),
                  builds=reports, replay=replays, evaluation=evaluate(ROOT), board_validated=False,
                  operating_point=dict(threshold=config['detector']['threshold'], confirm=config['event']['confirm']),
                  replay_comparison='Frozen numerical and direction evidence; former state/threshold fields replaced by explicit single-window checks',
                  scope='Fixed-weight reproduction only; timing belongs to the current host, excludes collection wait and vision')
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiler')
    parser.add_argument('--libraries')
    args = parser.parse_args()
    result = reproduce(args.output, args.compiler, args.libraries)
    print(json.dumps(dict(result=result['result'], scenes=result['evaluation']['scenes'], replay=result['replay'])))


if __name__ == '__main__':
    main()
