"""Build the optional PCM16/buffered conditioning library and test both FIR paths."""

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
    from ps.conditioning import Conditioner
    from ps.dsp import Conditioning
    pcm = np.random.RandomState(18).randint(-12000, 12000, size=(2049, 14)).astype(np.int16)
    errors = {}
    for name, delays in [('default', np.zeros(14)), ('fractional', np.linspace(-0.3, 0.5, 14)),
                         ('long_delay', np.linspace(-520, 1, 14))]:
        for fir in ('numpy', 'native'):
            for dtype in (np.int16, np.float64):
                reference = Conditioning(14, delays=delays)
                current = Conditioner(14, delays=delays, fir=fir, library=destination)
                maximum = 0.0
                for block in np.array_split(pcm.astype(dtype), [1, 73, 512, 1025]):
                    expected = reference.feed(block)
                    output = np.empty(block.shape, dtype=np.float64)
                    actual = current.feed_into(block, output)
                    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=1e-11)
                    maximum = max(maximum, float(np.max(abs(actual - expected))))
                if reference.latency != current.latency:
                    raise ValueError('Conditioning latency mismatch')
                errors[name + '-' + fir + '-' + np.dtype(dtype).name] = maximum
    path = Path(destination).resolve()
    return dict(result='passed', abi=1, machine=platform.machine(), library=str(path), maximum_errors=errors,
                source_sha256=hashlib.sha256((ROOT / 'ps/conditioning_kernel.c').read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='host numerical check; no ARM speed or FPGA acceptance')


def build(output=None, compiler=None):
    path = Path(output or ROOT / 'ps' / ('conditioning.dll' if sys.platform == 'win32' else 'libconditioning.so')).resolve()
    if path.exists():
        raise ValueError('Choose a new conditioning library path; existing builds are not overwritten')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('Existing GCC required; no compiler is installed automatically')
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math']
    command += ['-static-libgcc'] if sys.platform == 'win32' else ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(ROOT / 'ps/conditioning_kernel.c'), '-o', str(path), '-lm']
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
