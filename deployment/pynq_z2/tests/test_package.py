import ast
import importlib.util
import json
import os
import re
import shutil
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


def test_fast_bundle_requires_target_build_then_replays_and_benchmarks(tmp_path):
    import numpy as np
    import wave
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal weights absent')
    spec = importlib.util.spec_from_file_location('fast_search_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / 'fast'
    manifest = module.package(model, bundle, configuration=ROOT / 'config/band_fast.json')
    assert manifest['native_build_required'] and not manifest['native_library_included']
    exact = json.loads((bundle / 'config/native_exact.json').read_text())
    assert exact['search']['fft_fold'] == 1 and exact['search']['native_precision'] == 'double'
    assert exact['detector']['model_tag'] == 405187002 and exact['detector']['threshold'] == 0.95
    assert 'ps/search_kernel.c' in manifest['files'] and 'scripts/build_search.py' in manifest['files']
    assert not list((bundle / 'ps').glob('*.dll')) and not list((bundle / 'ps').glob('*.so'))
    checked = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode != 0 and 'build_search.py' in checked.stderr
    if not shutil.which('gcc'):
        pytest.skip('Optional compiler absent; portable native build boundary already checked')
    built = subprocess.run([sys.executable, '-B', str(bundle / 'scripts/build_search.py')], cwd=tmp_path,
                           check=True, capture_output=True, text=True, timeout=60)
    build_result = json.loads(built.stdout)
    assert build_result['reference_implementation'] == 'numpy' and build_result['double_check_precision'] == 'double'
    checked = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                             check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(checked.stdout)['search_implementation'] == 'native'
    samples = np.arange(48128)[:, None] / 48000 + 0.12 * np.cos(np.arange(14)[None] * 2 * np.pi / 14) / 343
    audio = (5000 * np.sin(2 * np.pi * 700 * samples)).astype('<i2')
    source = tmp_path / 'array14.wav'
    with wave.open(str(source), 'wb') as stream:
        stream.setnchannels(14)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(audio.tobytes())
    subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--replay', str(source), '--output', str(tmp_path / 'run')],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    event = json.loads((tmp_path / 'run/events.jsonl').read_text())
    assert event['scores'] and event['srp_diagnostics']['pair_count'] == 91
    assert event['srp_diagnostics']['fft_length'] == 256 and event['srp_diagnostics']['native_precision'] == 'single'
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_search.py'), '--input', str(source),
                    '--output', str(tmp_path / 'benchmark.json'), '--windows', '1', '--repeats', '1'],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    measured = json.loads((tmp_path / 'benchmark.json').read_text())
    assert measured['windows'] == 1 and measured['relative_time']['srp'] > 0
    assert measured['search_configuration']['fft_fold'] == 4 and measured['window_samples'] == 48128
    assert measured['phase_seconds']['optimized']['total']['median'] > 0
    module.verify(bundle)
