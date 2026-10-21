"""Create a self-contained, hashed source/runtime bundle without installing tools."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.detector import validate_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'bundle.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['files'].items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(path) != checksum:
            raise ValueError('Bundle integrity failure: ' + name)
    validate_model(directory / 'model')
    return manifest


def package(model, output, bitstream=None, handoff=None, configuration=None):
    model, output = Path(model).resolve(), Path(output).resolve()
    metadata = validate_model(model)
    config = None
    if configuration is not None:
        from ps.configuration import load_config
        config = load_config(configuration)
        if (config['detector']['backend'] == 'spectral') != (metadata['architecture'] == 'BandMLP-7826'):
            raise ValueError('Package backend and model architecture disagree')
        config['detector']['model_directory'] = 'model'
    elif metadata['architecture'] == 'BandMLP-7826' or metadata.get('provenance', {}).get('plan_sha256'):
        raise ValueError('Recovery models require their frozen operating configuration')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty bundle directory')
    if bool(bitstream) != bool(handoff):
        raise ValueError('Supply matching bitstream and HWH together')
    if bitstream and Path(bitstream).stem != Path(handoff).stem:
        raise ValueError('Bitstream/HWH must share a build basename')
    output.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / 'run.py', ROOT / 'README.md', ROOT / 'HANDOFF.md', ROOT / 'CAMERA.md']:
        shutil.copy2(str(path), str(output / path.name))
    for folder, pattern in [('ps', '*.py'), ('config', '*.json'), ('scripts', '*.py'), ('scripts', '*.tcl'),
                             ('scripts', '*.service'), ('rtl', '*.v')]:
        (output / folder).mkdir(exist_ok=True)
        for path in (ROOT / folder).glob(pattern):
            shutil.copy2(str(path), str(output / folder / path.name))
    if config is not None:
        (output / 'config/default.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    vendor = ROOT / 'vendor/fpga'
    if not vendor.is_dir():
        vendor = ROOT.parents[1] / 'fpga'
    for folder, pattern in [('', '*.py'), ('rtl', '*.v'), ('sim', '*.sv'), ('scripts', '*.tcl'), ('constraints', '*.xdc')]:
        destination = output / 'vendor/fpga' / folder
        destination.mkdir(parents=True, exist_ok=True)
        for path in (vendor / folder).glob(pattern):
            shutil.copy2(str(path), str(destination / path.name))
    destination = output / 'model'
    destination.mkdir()
    for name in list(metadata['sha256']) + ['manifest.json']:
        if Path(name).name != name:
            raise ValueError('Model memory paths must be local basenames')
        shutil.copy2(str(model / name), str(destination / name))
    if bitstream:
        (output / 'firmware').mkdir()
        shutil.copy2(str(bitstream), str(output / 'firmware/horn_cnn.bit'))
        shutil.copy2(str(handoff), str(output / 'firmware/horn_cnn.hwh'))
    files = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob('*') if path.is_file()}
    manifest = {'format': 1, 'model_tag': metadata['model_tag'], 'files': files,
                'firmware_included': bool(bitstream), 'board_validated': False,
                'vision_weights_included': False, 'license_scope': 'research/noncommercial; model provenance in model/manifest.json'}
    (output / 'bundle.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify(output)
    archive = output.with_name(output.name + '.zip')
    if archive.exists():
        raise ValueError('Archive already exists; choose a new bundle name')
    with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in output.rglob('*'):
            if path.is_file():
                stream.write(str(path), arcname=path.relative_to(output).as_posix())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--output')
    parser.add_argument('--bitstream')
    parser.add_argument('--handoff')
    parser.add_argument('--config')
    parser.add_argument('--verify')
    args = parser.parse_args()
    result = verify(args.verify) if args.verify else package(args.model, args.output, args.bitstream, args.handoff, args.config)
    print(json.dumps({'model_tag': result['model_tag'], 'files': len(result['files']),
                      'firmware_included': result['firmware_included'], 'board_validated': False}))


if __name__ == '__main__':
    main()
