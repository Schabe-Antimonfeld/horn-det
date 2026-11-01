"""Build the optional fused Hann/fold preparation library and verify exact FFT input."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(destination):
    import numpy as np
    from ps.refinement import Runtime
    from ps.dsp import frames
    runtime = Runtime(destination)
    window = np.hanning(1024)
    for audio in (np.random.RandomState(18).normal(0, 0.1, (48128, 14)), np.full((1024, 14), -0.0)):
        for fold in (1, 2, 4, 8):
            for first in ((0, 46, 92) if len(audio) > 1024 else (0,)):
                weighted = frames(audio)[first:] * window
                expected = (weighted if fold == 1 else
                            weighted.reshape(weighted.shape[:-1] + (fold, 1024 // fold)).sum(axis=-2))
                np.testing.assert_array_equal(runtime.prepare(audio, window, fold, first).view(np.uint64),
                                              expected.view(np.uint64))
    path = Path(destination).resolve()
    return dict(result='passed', abi=1, machine=platform.machine(), library=str(path),
                fft_input_bitwise_equal=True,
                source_sha256=hashlib.sha256((ROOT / 'ps/refinement_kernel.c').read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='host numerical check; no ARM speed or FPGA acceptance')


def build(output=None, compiler=None):
    path = Path(output or ROOT / 'ps' / ('refinement.dll' if sys.platform == 'win32' else 'librefinement.so')).resolve()
    if path.exists():
        raise ValueError('Choose a new refinement library path; existing builds are not overwritten')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('Existing GCC required; no compiler is installed automatically')
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math', '-ffp-contract=off']
    command += ['-static-libgcc'] if sys.platform == 'win32' else ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(ROOT / 'ps/refinement_kernel.c'), '-o', str(path), '-lm']
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    result = check(path)
    result['command'] = command
    path.with_name(path.name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--compiler')
    parser.add_argument('--check')
    args = parser.parse_args()
    print(json.dumps(check(args.check) if args.check else build(args.output, args.compiler)))


if __name__ == '__main__':
    main()
