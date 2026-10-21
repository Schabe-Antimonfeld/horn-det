import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_bundle_runs_without_source_checkout_and_detects_tampering(tmp_path):
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained model absent')
    specification = importlib.util.spec_from_file_location('board_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    bundle = tmp_path / 'standalone'
    manifest = module.package(model, bundle)
    assert not manifest['firmware_included'] and not manifest['board_validated']
    assert (tmp_path / 'standalone.zip').is_file()
    for name in ('ps/srp.py', 'rtl/srp_batch.v', 'rtl/srp_accumulator.v', 'scripts/connect_srp.tcl',
                 'CAMERA.md', 'scripts/camera.py'):
        assert name in manifest['files']
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['model_tag'] == manifest['model_tag']
    module.verify(bundle)
    with (bundle / 'run.py').open('a', encoding='utf-8') as output:
        output.write('\npass\n')
    with pytest.raises(ValueError, match='integrity'):
        module.verify(bundle)


def test_every_board_entry_uses_python36_syntax_and_no_pc_imports():
    paths = list((ROOT / 'ps').glob('*.py')) + [ROOT / 'run.py', ROOT / 'scripts/camera.py']
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 6))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(value.name.split('.')[0] not in {'torch', 'scipy', 'horn'} for value in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'torch', 'scipy', 'horn'}


def test_model_validator_can_ignore_vivado_python2_environment():
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained weights absent')
    environment = dict(os.environ, PYTHONHOME='nonexistent-vivado-python2', PYTHONPATH='nonexistent-vivado-python2')
    result = subprocess.run([sys.executable, '-E', '-B', str(ROOT.parents[1] / 'fpga/cnn.py'),
                             'check', '--model-dir', str(model)], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
