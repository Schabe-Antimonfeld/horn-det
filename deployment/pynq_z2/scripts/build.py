"""Build or verify the five native libraries used by the single current pipeline."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'search': ('srp_search.dll', 'libsrp.so', 'search_kernel.c'),
         'stream': ('streaming.dll', 'libstream.so', 'stream_kernel.c'),
         'conditioning': ('conditioning.dll', 'libconditioning.so', 'conditioning_kernel.c'),
         'refinement': ('refinement.dll', 'librefinement.so', 'refinement_kernel.c'),
         'dsp': ('dsp_v3.dll', 'libdsp_v3.so', 'dsp_kernel.c')}


def build(compiler=None, directory=None):
    directory = Path(directory or ROOT / 'ps').resolve()
    reports = {}
    for name, (windows, linux, source) in NAMES.items():
        spec = importlib.util.spec_from_file_location('build_' + name, str(ROOT / ('scripts/build_' + name + '.py')))
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        destination = directory / (windows if sys.platform == 'win32' else linux)
        if destination.exists():
            stamp = destination.with_name(destination.name + '.json')
            if not stamp.is_file():
                raise ValueError('Library has no build record; choose a fresh --directory: ' + str(destination))
            record = json.loads(stamp.read_text(encoding='utf-8'))
            source_hash = hashlib.sha256((ROOT / 'ps' / source).read_bytes()).hexdigest()
            library_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            if record['source_sha256'] != source_hash or record['library_sha256'] != library_hash:
                raise ValueError('Library source or binary changed; choose a fresh --directory: ' + str(destination))
            reports[name] = builder.check(destination)
        else:
            reports[name] = builder.build(destination, compiler)
    return reports


def configure(config, directory):
    """Bind a verified library directory without introducing a second algorithm configuration."""
    directory = Path(directory).resolve()
    for name, (windows, linux, _) in NAMES.items():
        section = 'streaming' if name == 'stream' else 'acceleration' if name == 'dsp' else name
        config[section]['native_library'] = str(directory / (windows if sys.platform == 'win32' else linux))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler')
    parser.add_argument('--directory')
    args = parser.parse_args()
    print(json.dumps(build(args.compiler, args.directory)))


if __name__ == '__main__':
    main()
