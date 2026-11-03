"""Compile and independently self-check the optional streaming primitives."""

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
    from ps.streaming import Runtime
    from ps.dsp import Conditioning
    runtime = Runtime(destination)
    rng = np.random.RandomState(37)
    spectrum = rng.normal(size=(29, 14, 129)) + 1j * rng.normal(size=(29, 14, 129))
    pairs = np.asarray([(a, b) for a in range(14) for b in range(a + 1, 14)], dtype=np.int32)
    errors = {}
    for precision, dtype, tolerance in [('single', np.complex64, 2e-6), ('double', np.complex128, 1e-12)]:
        values = spectrum.astype(dtype)
        first, second = values[:, pairs[:, 0]], values[:, pairs[:, 1]]
        cross = first * second.conj()
        expected = cross / (abs(cross) + 1e-12)
        actual = runtime.phat(values, pairs, precision)
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
        reduced = runtime.reduce(actual[:11], actual[11:], precision)
        total = np.zeros(actual.shape[1:], dtype=np.complex128)
        for start in range(0, len(actual), 12):
            block = np.zeros(actual.shape[1:], dtype=dtype)
            for frame in actual[start:start + 12]:
                block += frame
            total += block
        np.testing.assert_array_equal(reduced, total)
        first, cached = runtime.fused(values[:17], pairs, precision, None, 0, 12)
        expected_first = runtime.reduce(actual[:0], actual[:17], precision)
        np.testing.assert_array_equal(first, expected_first)
        shifted, _ = runtime.fused(values[17:], pairs, precision, cached, 12, 12)
        np.testing.assert_array_equal(shifted, runtime.reduce(actual[:0], actual[5:], precision))
        errors[precision] = float(np.max(abs(actual - expected)))
    source = ROOT / 'ps/stream_kernel.c'
    return dict(result='passed', abi=1, machine=platform.machine(), library=str(Path(destination).resolve()),
                phat_maximum_error=errors,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest(),
                scope='host numerical self-check; ARM throughput and the 500ms target require board measurement')


def build(output=None, compiler=None):
    destination = Path(output or ROOT / 'ps' / ('streaming.dll' if sys.platform == 'win32' else 'libstream.so')).resolve()
    if destination.exists():
        raise ValueError('Choose a new stream library path; existing builds are not overwritten')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('Existing GCC required; no compiler is installed automatically')
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math']
    command += ['-static-libgcc'] if sys.platform == 'win32' else ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(ROOT / 'ps/stream_kernel.c'), '-o', str(destination), '-lm']
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    result = check(destination)
    result['command'] = command
    destination.with_name(destination.name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
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
