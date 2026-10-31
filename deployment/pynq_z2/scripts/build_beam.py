"""Build and self-check the optional beam FIR library with the existing GCC."""

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


def build(output=None, compiler=None):
    destination = Path(output or ROOT / 'ps' / ('beamforming.dll' if sys.platform == 'win32' else 'libbeam.so')).resolve()
    if destination.exists():
        raise ValueError('Choose a new beam library path; existing libraries are not overwritten')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('GCC is required; no compiler is installed automatically')
    source = ROOT / 'ps/beam_kernel.c'
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math']
    command += ['-static-libgcc'] if sys.platform == 'win32' else ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(source), '-o', str(destination)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    result = check(destination)
    result['command'] = command
    destination.with_name(destination.name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def check(destination):
    """Numerically check a built library; also useful after host quarantine restoration."""
    import numpy as np
    from ps.beamforming import Beamformer
    from ps.configuration import load_config
    from ps.dsp import beamform
    from ps.geometry import Bearing
    destination = Path(destination).resolve()
    source = ROOT / 'ps/beam_kernel.c'
    settings = load_config()['audio']
    native = Beamformer(settings, dict(implementation='native', native_library=str(destination)))
    bearings = [Bearing(-179, 0, 1), Bearing(47, 33, 1), Bearing(None, 90, 1)]
    maximum = 0.0
    for length in (31, 257, 4096):
        audio = np.random.RandomState(length).normal(size=(length, 14))
        actual, _ = native.run(audio, bearings)
        for bearing, (waveform, latency) in zip(bearings, actual):
            expected, common = beamform(audio, bearing, settings)
            np.testing.assert_allclose(waveform, expected, atol=3e-14, rtol=3e-13)
            if latency != common:
                raise ValueError('Beam latency mismatch')
            maximum = max(maximum, float(np.max(np.abs(waveform - expected))))
    result = dict(result='passed', abi=1, machine=platform.machine(), library=str(destination),
                  maximum_waveform_error=maximum, reference_implementation='frozen_numpy_65tap',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  scope='local numerical self-check; not ARM speed or FPGA acceptance')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--compiler')
    parser.add_argument('--check', help='Check an existing local library without compiling or overwriting it')
    args = parser.parse_args()
    print(json.dumps(check(args.check) if args.check else build(args.output, args.compiler)))


if __name__ == '__main__':
    main()
