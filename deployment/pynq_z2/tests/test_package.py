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
    assert json.loads(checked.stdout)['beamforming_implementation'] == 'fft'
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
    assert event['beamforming_diagnostics']['implementation'] == 'fft'
    assert event['beamforming_diagnostics']['taps'] == 65
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_search.py'), '--input', str(source),
                    '--output', str(tmp_path / 'benchmark.json'), '--windows', '1', '--repeats', '1'],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    measured = json.loads((tmp_path / 'benchmark.json').read_text())
    assert measured['windows'] == 1 and measured['relative_time']['srp'] > 0
    assert measured['search_configuration']['fft_fold'] == 4 and measured['window_samples'] == 48128
    assert measured['phase_seconds']['optimized']['total']['median'] > 0
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_beam.py'), '--input', str(source),
                    '--output', str(tmp_path / 'beam.json'), '--windows', '1', '--repeats', '1'],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    beam = json.loads((tmp_path / 'beam.json').read_text())
    assert beam['relative_time']['optimized']['beamforming'] > 0
    assert beam['search_configuration'] == measured['search_configuration']
    assert beam['maximum_score_error']['optimized'] < 2e-10
    assert manifest['native_build_components'] == ['search']
    assert 'ps/beam_kernel.c' in manifest['files'] and 'scripts/build_beam.py' in manifest['files']
    module.verify(bundle)


def test_native_beam_package_rejects_host_override_and_missing_library_before_output(tmp_path):
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal weights absent')
    specification = importlib.util.spec_from_file_location('native_beam_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    config = json.loads((ROOT / 'config/band_temporal.json').read_text())
    config['beamforming'] = dict(implementation='native', native_library=str(tmp_path / 'host.dll'))
    path = tmp_path / 'configuration.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='host-specific'):
        module.package(model, tmp_path / 'invalid', configuration=path)
    assert not (tmp_path / 'invalid').exists()
    config['beamforming'].pop('native_library')
    path.write_text(json.dumps(config))
    bundle = tmp_path / 'beam-native'
    manifest = module.package(model, bundle, configuration=path)
    assert manifest['native_build_required'] and manifest['native_build_components'] == ['beamforming']
    output = tmp_path / 'not-created'
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--output', str(output)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'build_beam.py' in result.stderr
    assert not output.exists()


def test_stream_bundle_builds_both_libraries_and_replays_contiguous_pcm(tmp_path):
    import numpy as np
    import wave
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir() or not shutil.which('gcc'):
        pytest.skip('Optional local model or compiler absent')
    spec = importlib.util.spec_from_file_location('stream_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / 'streaming'
    manifest = module.package(model, bundle, configuration=ROOT / 'config/band_stream.json')
    assert manifest['native_build_components'] == ['search', 'streaming']
    assert manifest['native_build_required'] and not manifest['native_library_included']
    assert not list(bundle.rglob('*.dll')) and not list(bundle.rglob('*.so'))
    for name in ('ps/streaming.py', 'ps/stream_kernel.c', 'scripts/build_stream.py', 'scripts/benchmark_stream.py'):
        assert name in manifest['files']
    checked = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode != 0 and 'build_stream.py' in checked.stderr
    for name in ('build_search.py', 'build_stream.py'):
        subprocess.run([sys.executable, '-B', str(bundle / 'scripts' / name)], cwd=tmp_path,
                       capture_output=True, text=True, check=True, timeout=60)
    checked = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                             capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(checked.stdout)['streaming_enabled']
    samples = np.arange(72192)[:, None] / 48000 + 0.12 * np.cos(np.arange(14)[None] * 2 * np.pi / 14) / 343
    audio = (5000 * np.sin(2 * np.pi * 700 * samples)).astype('<i2')
    source = tmp_path / 'array14.wav'
    with wave.open(str(source), 'wb') as stream:
        stream.setnchannels(14)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(audio.tobytes())
    subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--replay', str(source), '--output', str(tmp_path / 'replay')],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    events = [json.loads(line) for line in (tmp_path / 'replay/events.jsonl').read_text().splitlines()]
    assert len(events) == 2 and events[0]['srp_diagnostics']['fft_frames_computed'] == 93
    assert events[1]['srp_diagnostics']['fft_frames_computed'] == 47
    assert events[1]['beamforming_diagnostics']['fft_blocks_computed'] == 33
    assert events[1]['stream_diagnostics']['reusable']
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_stream.py'), '--input', str(source),
                    '--output', str(tmp_path / 'stream-benchmark.json'), '--windows', '2', '--repeats', '1'],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    measured = json.loads((tmp_path / 'stream-benchmark.json').read_text())
    assert measured['maximum_score_error'] < 2e-10 and measured['target_seconds'] == 0.5
    assert measured['fallback_reset_reason'] == 'epoch_changed'
    assert set(measured['beam_scenarios']) == {'stable', 'changing'}
    module.verify(bundle)


def test_conditioned_bundle_builds_and_replays_without_source_checkout(tmp_path):
    import numpy as np
    import wave
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir() or not shutil.which('gcc'):
        pytest.skip('Optional model or compiler absent')
    spec = importlib.util.spec_from_file_location('conditioned_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / 'conditioned'
    manifest = module.package(model, bundle, configuration=ROOT / 'config/band_conditioned.json')
    assert manifest['native_build_components'] == ['search', 'streaming', 'conditioning']
    assert not manifest['native_library_included'] and not list(bundle.rglob('*.dll')) and not list(bundle.rglob('*.so'))
    for name in ('build_search.py', 'build_stream.py'):
        subprocess.run([sys.executable, '-B', str(bundle / 'scripts' / name)], cwd=tmp_path,
                       capture_output=True, text=True, check=True, timeout=60)
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--output', str(tmp_path / 'not-created')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'build_conditioning.py' in result.stderr
    assert not (tmp_path / 'not-created').exists()
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/build_conditioning.py')], cwd=tmp_path,
                   capture_output=True, text=True, check=True, timeout=60)
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check'], cwd=tmp_path,
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout)['conditioning_implementation'] == 'buffered'
    assert json.loads(result.stdout)['conditioning_fir'] == 'native'
    source = tmp_path / 'array.wav'
    samples = np.arange(72192)[:, None] / 48000 + np.cos(np.arange(14)[None] * 2 * np.pi / 14) * 0.12 / 343
    data = (5000 * np.sin(2 * np.pi * 700 * samples)).astype('<i2')
    with wave.open(str(source), 'wb') as stream:
        stream.setnchannels(14)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(data.tobytes())
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_conditioning.py'), '--input', str(source),
                    '--output', str(tmp_path / 'benchmark.json'), '--config', str(bundle / 'config/default.json'),
                    '--model', str(bundle / 'model'), '--repeats', '1'], cwd=tmp_path,
                   capture_output=True, text=True, check=True, timeout=120)
    report = json.loads((tmp_path / 'benchmark.json').read_text())
    assert report['result'] == 'passed' and report['model_comparison']['fractional']['windows'] == 2
    config = json.loads((bundle / 'config/default.json').read_text())
    config['conditioning']['native_library'] = str(tmp_path / 'host.so')
    path = tmp_path / 'host.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='host-specific'):
        module.package(model, tmp_path / 'invalid', configuration=path)
    assert not (tmp_path / 'invalid').exists()
    module.verify(bundle)


def test_refined_bundle_checks_native_boundary_and_replays_four_backends(tmp_path):
    import hashlib
    import numpy as np
    import wave
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.is_dir() or not shutil.which('gcc'):
        pytest.skip('Optional model or compiler absent')
    spec = importlib.util.spec_from_file_location('refined_package', ROOT / 'scripts/package.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / 'refined'
    manifest = module.package(model, bundle, configuration=ROOT / 'config/band_refined.json')
    assert manifest['native_build_components'] == ['search', 'streaming', 'conditioning', 'refinement']
    assert not manifest['native_library_included'] and not list(bundle.rglob('*.dll')) and not list(bundle.rglob('*.so'))
    for name in ('ps/refinement.py', 'ps/refinement_kernel.c', 'scripts/build_refinement.py', 'scripts/benchmark_refinement.py'):
        assert name in manifest['files']
    for name in ('search', 'stream', 'conditioning'):
        subprocess.run([sys.executable, '-B', str(bundle / ('scripts/build_' + name + '.py'))], cwd=tmp_path,
                       capture_output=True, text=True, check=True, timeout=60)
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--output', str(tmp_path / 'not-created')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'build_refinement.py' in result.stderr and not (tmp_path / 'not-created').exists()
    existing = os.environ.get('HORN_REFINEMENT_LIBRARY')
    if existing:
        report = json.loads(Path(existing + '.json').read_text())
        assert report['source_sha256'] == hashlib.sha256((bundle / 'ps/refinement_kernel.c').read_bytes()).hexdigest()
        assert report['library_sha256'] == hashlib.sha256(Path(existing).read_bytes()).hexdigest()
        arguments = ['--check', existing]
    else:
        arguments = []
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/build_refinement.py')] + arguments,
                   cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    config = json.loads((bundle / 'config/default.json').read_text())
    if existing:
        config['refinement']['native_library'] = existing
    runtime_config = tmp_path / 'runtime.json'
    runtime_config.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--check', '--config', str(runtime_config)], cwd=tmp_path,
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout)['refinement']['independent_beams'] is True
    source = tmp_path / 'array.wav'
    samples = np.arange(72192)[:, None] / 48000 + np.cos(np.arange(14)[None] * 2 * np.pi / 14) * 0.12 / 343
    with wave.open(str(source), 'wb') as stream:
        stream.setparams((14, 2, 48000, 0, 'NONE', 'not compressed'))
        stream.writeframes((5000 * np.sin(2 * np.pi * 700 * samples)).astype('<i2').tobytes())
    subprocess.run([sys.executable, '-B', str(bundle / 'scripts/benchmark_refinement.py'), '--input', str(source),
                    '--output', str(tmp_path / 'benchmark.json'), '--config', str(runtime_config), '--repeats', '1'], cwd=tmp_path,
                   capture_output=True, text=True, check=True, timeout=120)
    value = json.loads((tmp_path / 'benchmark.json').read_text())
    assert value['correlations_bitwise_equal'] and value['complete_grid_bitwise_equal']
    assert value['maximum_score_error'] < 2e-10 and len(value['phase_seconds']) == 4
    assert value['beam_scenarios']['partial_change']['rows']['refined'][1]['diagnostic']['output_reused_beams'] == 1
    exact = json.loads((bundle / 'config/native_exact.json').read_text())
    if existing:
        exact['refinement']['native_library'] = existing
    exact_config = tmp_path / 'exact.json'
    exact_config.write_text(json.dumps(exact))
    subprocess.run([sys.executable, '-B', str(bundle / 'run.py'), '--config', str(exact_config),
                    '--replay', str(source), '--output', str(tmp_path / 'exact')], cwd=tmp_path,
                   capture_output=True, text=True, check=True, timeout=60)
    events = [json.loads(line) for line in (tmp_path / 'exact/events.jsonl').read_text().splitlines()]
    assert events[1]['srp_diagnostics']['fused_input'] and events[1]['srp_diagnostics']['fft_length'] == 1024
    assert events[1]['srp_diagnostics']['fft_frames_reused'] == 46
    assert events[1]['beamforming_diagnostics']['implementation'] == 'refined_stream_fft'
    config = json.loads((bundle / 'config/default.json').read_text())
    config['refinement']['native_library'] = str(tmp_path / 'host.so')
    path = tmp_path / 'host.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='host-specific'):
        module.package(model, tmp_path / 'invalid', configuration=path)
    config['refinement'] = dict(fused_input=False, independent_beams=True)
    path.write_text(json.dumps(config))
    beams = module.package(model, tmp_path / 'beams-only', configuration=path)
    assert beams['native_build_components'] == ['search', 'streaming', 'conditioning']
    module.verify(bundle)
