import ast
import importlib.util
import json
import os
import re
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


def test_spectral_bundle_preserves_backend_and_frozen_gates(tmp_path):
    base = ROOT.parents[1] / 'prototype/outputs/horn_recovery_v1'
    configuration = base / 'spectral_numpy-config.json'
    if not configuration.exists():
        pytest.skip('Optional local spectral experiment absent')
    specification = importlib.util.spec_from_file_location('spectral_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    output = tmp_path / 'spectral'
    manifest = module.package(base / 'spectral_model', output, configuration=configuration)
    config = json.loads((output / 'config/default.json').read_text())
    assert config['detector']['backend'] == 'spectral' and config['detector']['model_directory'] == 'model'
    assert config['detector']['threshold'] == json.loads(configuration.read_text())['detector']['threshold']
    result = subprocess.run([sys.executable, '-B', str(output / 'run.py'), '--check'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['model_tag'] == manifest['model_tag']
    module.verify(output)


def test_model_validator_can_ignore_vivado_python2_environment():
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained weights absent')
    environment = dict(os.environ, PYTHONHOME='nonexistent-vivado-python2', PYTHONPATH='nonexistent-vivado-python2')
    result = subprocess.run([sys.executable, '-E', '-B', str(ROOT.parents[1] / 'fpga/cnn.py'),
                             'check', '--model-dir', str(model)], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_temporal_bundle_runs_outside_checkout_without_training_imports(tmp_path):
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal weights absent')
    specification = importlib.util.spec_from_file_location('temporal_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    bundle = tmp_path / 'temporal'
    manifest = module.package(model, bundle, configuration=ROOT / 'config/band_temporal.json')
    config = json.loads((bundle / 'config/default.json').read_text())
    assert config['detector']['model_directory'] == 'model' and config['detector']['backend'] == 'spectral'
    assert config['detector']['threshold'] == 0.95 and config['detector']['capture_threshold'] == 0.9
    assert config['hardware']['overlay'] == 'firmware/horn_capture.bit'
    assert manifest['classifier'] == 'BandTemporalMLP-30866' and manifest['model_tag'] == 405187002
    assert not manifest['firmware_included'] and not manifest['board_validated']
    for document in bundle.glob('*.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if not target.startswith(('https:', 'http:', '#')):
                assert (document.parent / target.split('#', 1)[0]).exists(), target
    for name in ('ps/band.py', 'scripts/build_ps.tcl', 'vendor/fpga/rtl/pdm_capture.v',
                 'vendor/fpga/scripts/build.tcl', 'vendor/fpga/constraints/pynq_z2.xdc'):
        assert name in manifest['files']
    code = 'import sys; sys.path.insert(0, ' + repr(str(bundle)) + '); import run; run.main(["--check"]); '
    code += 'assert not {"torch", "scipy", "horn"}.intersection(sys.modules)'
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['classifier'] == 'BandTemporalMLP-30866'
    module.verify(bundle)
    with pytest.raises(ValueError, match='frozen operating configuration'):
        module.package(model, tmp_path / 'missing-config')
    assert not (tmp_path / 'missing-config').exists()


@pytest.mark.parametrize('key,value', [('backend', 'fpga'), ('model_tag', 405187003)])
def test_temporal_package_rejects_mismatched_contract_before_copying(tmp_path, key, value):
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal weights absent')
    specification = importlib.util.spec_from_file_location('temporal_invalid_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    config = json.loads((ROOT / 'config/band_temporal.json').read_text())
    config['detector'][key] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='architecture|model tag'):
        module.package(model, tmp_path / 'bundle', configuration=path)
    assert not (tmp_path / 'bundle').exists()


def test_temporal_package_keeps_firmware_paths_consistent(tmp_path):
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal weights absent')
    specification = importlib.util.spec_from_file_location('temporal_firmware_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    bitstream, handoff = tmp_path / 'fixture.bit', tmp_path / 'fixture.hwh'
    bitstream.write_bytes(b'test-only-not-real-firmware')
    handoff.write_bytes(b'test-only-not-real-handoff')
    bundle = tmp_path / 'packaged'
    manifest = module.package(model, bundle, bitstream=bitstream, handoff=handoff,
                              configuration=ROOT / 'config/band_temporal.json')
    config = json.loads((bundle / 'config/default.json').read_text())
    overlay = bundle / config['hardware']['overlay']
    assert overlay.read_bytes() == bitstream.read_bytes()
    assert overlay.with_suffix('.hwh').read_bytes() == handoff.read_bytes()
    assert manifest['firmware_included'] and not manifest['board_validated']
