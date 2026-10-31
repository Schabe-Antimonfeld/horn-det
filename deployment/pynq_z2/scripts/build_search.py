"""Build and numerically check the optional local SRP library without downloads."""

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
    import numpy as np
    from ps.configuration import load_config
    from ps.search import Search
    destination = Path(output or ROOT / 'ps' / ('srp_search.dll' if sys.platform == 'win32' else 'libsrp.so')).resolve()
    if destination.exists():
        raise ValueError('Choose a new library path; an earlier library already exists')
    executable = compiler or shutil.which('gcc')
    if not executable:
        raise ValueError('GCC is required to build locally; no compiler is installed automatically')
    source = ROOT / 'ps/search_kernel.c'
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [str(executable), '-O3', '-std=c99', '-shared', '-fno-math-errno', '-fno-trapping-math']
    if sys.platform == 'win32':
        command += ['-static-libgcc']
    else:
        command += ['-fPIC']
    if platform.machine().lower().startswith('armv7'):
        command += ['-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard']
    command += [str(source), '-o', str(destination), '-lm']
    subprocess.run(command, check=True)
    config = load_config()
    settings = dict(config['search'], algorithm='cached', fft_fold=4, implementation='numpy')
    settings.pop('native_precision', None)
    settings.pop('native_library', None)
    reference = Search(config['audio'], settings)
    native = Search(config['audio'], dict(settings, implementation='native', native_library=str(destination)))
    audio = np.random.default_rng(20261031).normal(size=(4096, 14))
    expected, actual = reference.correlations(audio), native.correlations(audio)
    error = float(np.abs(actual - expected).max())
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    indices = np.arange(0, 5580, 37)
    np.testing.assert_allclose(native.score(actual, indices), reference.score(expected, indices), atol=1e-11, rtol=1e-12)
    single = Search(config['audio'], dict(settings, implementation='native', native_precision='single', native_library=str(destination)))
    single_error = float(np.abs(single.correlations(audio) - expected).max())
    if single_error > 1e-6:
        raise ValueError('Single-precision PHAT failed its numerical self-check')
    result = dict(result='passed', library=str(destination), machine=platform.machine(), abi=2,
                  reference_implementation='native' if reference.native is not None else 'numpy',
                  double_check_precision=native.native.precision,
                  gcc_maximum_error=error, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  gcc_single_maximum_error=single_error,
                  library_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  scope='host-only native numerical test; not ARM or FPGA hardware acceptance')
    destination.with_name(destination.name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--compiler')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.compiler)))


if __name__ == '__main__':
    main()
