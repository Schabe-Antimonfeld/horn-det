"""Package one current algorithm, fixed weights, evidence and locked environment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.evaluation import verify_assets


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'bundle.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['files'].items():
        path = (directory / name).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            raise ValueError('Bundle path escapes its directory')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('Bundle integrity failure: ' + name)
    verify_assets(directory)
    return manifest


def package(output):
    verify_assets(ROOT)
    output = Path(output).resolve()
    if output.exists() or output.suffix != '.zip':
        raise ValueError('Choose a fresh ZIP path')
    names = ['README.md', 'HANDOFF.md', 'BOARD_PERFORMANCE.md', 'CAMERA.md', 'DATA.md', 'FULLFLOW_PERFORMANCE.md',
             'run.py', 'web.py', 'reproduce.py', 'assets.json', 'pyproject.toml', 'uv.lock', '.python-version', '.gitattributes']
    for folder, pattern in [('ps', '*.py'), ('ps', '*.c'), ('scripts', '*.py'),
                            ('config', '*.json'), ('model', '*'), ('examples', '*'), ('evaluation', '*'), ('tests', '*.py')]:
        names.extend(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / folder).glob(pattern)) if path.is_file())
    names = sorted(set(name for name in names if name not in ('tests/test_rtl.py', 'tests/test_srp.py')))
    content = {name: (ROOT / name).read_bytes() for name in names}
    frontend = ROOT / 'frontend' if (ROOT / 'frontend').is_dir() else ROOT.parents[1] / 'frontend'
    for name in ('index.html', 'app.js', 'README.md'):
        content['frontend/' + name] = (frontend / name).read_bytes()
    names = sorted(content)
    for name in names:
        if name.endswith('.md'):
            content[name] = content[name].decode('utf-8').replace('](../../docs/', '](https://github.com/Schabe-Antimonfeld/horn-det/blob/main/docs/').encode('utf-8')
    manifest = dict(format=1, version='1.0.0', archive_branch='codex/archive-2026-11-03', model_tag=405187002,
                    files={name: hashlib.sha256(value).hexdigest() for name, value in content.items()},
                    native_build_components=['search','stream','conditioning','refinement','dsp'],
                    native_library_included=False, firmware_included=False, board_validated=False,
                    scope='Single current algorithm; fixed-weight replay and frozen-feature evaluation; research/noncommercial')
    content['bundle.json'] = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(output), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(content.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 3, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, value)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--verify')
    args = parser.parse_args()
    result = verify(args.verify) if args.verify else package(args.output)
    print(json.dumps(dict(model_tag=result['model_tag'], files=len(result['files']), board_validated=False)))


if __name__ == '__main__':
    main()
