"""Build the optional double-precision DSP kernels and verify numerical agreement."""

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
    from ps.dsp_native import NativeDSP
    from ps.detector import temporal_features
    from ps.streaming import Frontend
    runtime = NativeDSP(destination)
    rng = np.random.RandomState(20261104)
    spectrum = rng.normal(size=(33, 14, 513)) + 1j * rng.normal(size=(33, 14, 513))
    weights = rng.normal(size=(14, 513)) + 1j * rng.normal(size=(14, 513))
    np.testing.assert_allclose(runtime.beam(spectrum, weights), np.einsum('bcf,cf->bf', spectrum, weights, optimize=False), rtol=2e-13, atol=2e-12)
    values = spectrum[:, 0].copy()
    bank = Frontend().bank
    np.testing.assert_allclose(runtime.mel(values, bank), np.log(abs(values) ** 2 @ bank.T + 1e-12), atol=2e-12, rtol=2e-13)
    for contexts in (rng.normal(-10, 8, (7,40,24)), np.full((7,40,24), -20.0)):
        np.testing.assert_allclose(runtime.temporal(contexts), temporal_features(contexts), atol=2e-12, rtol=2e-13)
    from ps.dsp import frames
    for audio in (rng.normal(size=(48128, 14)), np.full((1024,14), -0.0)):
        for fold in (1,2,4,8):
            for first in ((0,46,92) if len(audio)>1024 else (0,)):
                weighted = frames(audio)[first:] * np.hanning(1024)
                expected = weighted if fold == 1 else weighted.reshape(weighted.shape[:-1] + (fold,1024//fold)).sum(axis=-2)
                np.testing.assert_array_equal(runtime.prepare(audio,np.hanning(1024),fold,first).view(np.uint64), expected.view(np.uint64))
    path = Path(destination).resolve()
    return dict(result='passed', abi=3, machine=platform.machine(), library=str(path),
                source_sha256=hashlib.sha256((ROOT / 'ps/dsp_kernel.c').read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Double precision numerical checks; no ARM performance acceptance')


def build(output=None, compiler=None):
    path = Path(output or ROOT / 'ps' / ('dsp_v3.dll' if sys.platform == 'win32' else 'libdsp_v3.so')).resolve()
    if path.exists():
        raise ValueError('Choose a new dsp library path; existing builds are not overwritten')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('Existing GCC required; no compiler is installed automatically')
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math', '-ffp-contract=off']
    command += ['-static-libgcc'] if sys.platform == 'win32' else ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(ROOT / 'ps/dsp_kernel.c'), '-o', str(path), '-lm']
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
