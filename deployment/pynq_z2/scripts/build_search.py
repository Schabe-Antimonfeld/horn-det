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
    result = check(destination)
    result['command'] = command
    destination.with_name(destination.name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def check(destination):
    import numpy as np
    from types import SimpleNamespace
    from ps.search import Native
    rng = np.random.RandomState(20261103)
    first = rng.randint(0, 91 * 256, (91, 23)).astype(np.int32)
    second = rng.randint(0, 91 * 256, (91, 23)).astype(np.int32)
    fraction = rng.uniform(size=(91, 23))
    search = SimpleNamespace(first=first, second=second, fraction=fraction, settings={'native_precision': 'single'})
    native = Native(search, destination)
    correlations = rng.normal(size=(91, 256))
    values = correlations.ravel()
    expected = (values[first] + fraction * (values[second] - values[first])).sum(axis=0)
    actual = native.score(correlations, np.arange(23))
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)
    return dict(result='passed', abi=2, library=str(Path(destination).resolve()), machine=platform.machine(),
                maximum_error=float(np.max(abs(actual - expected))),
                source_sha256=hashlib.sha256((ROOT / 'ps/search_kernel.c').read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--compiler')
    parser.add_argument('--check')
    args = parser.parse_args()
    print(json.dumps(check(args.check) if args.check else build(args.output, args.compiler)))


if __name__ == '__main__':
    main()
