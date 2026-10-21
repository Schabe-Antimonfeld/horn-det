import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import wave

import numpy as np
import pytest


def experiment():
    path = Path(__file__).resolve().parents[1] / 'improve_band.py'
    spec = importlib.util.spec_from_file_location('band_improvement_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_features_preserve_order_and_relative_envelope_without_gain_dependence():
    module = experiment()
    values = np.arange(7)[:, None, None] + np.arange(40)[None, :, None] + np.zeros((7, 40, 24))
    original = values.copy()
    actual = module.sequence_features(values)
    assert actual.shape == (640,)
    for index, offset in enumerate((-2, 0.5, 2.5, 3)):
        np.testing.assert_allclose(actual[index * 160:index * 160 + 40], np.arange(40) - 19.5 + offset)
        np.testing.assert_allclose(actual[index * 160 + 40:(index + 1) * 160], 0)
    np.testing.assert_allclose(module.sequence_features(values + 250), actual, atol=1e-12)
    np.testing.assert_array_equal(values, original)
    assert not np.array_equal(module.sequence_features(values[::-1]), actual)
    np.testing.assert_array_equal(module.band_features(values[::-1]), module.band_features(values))
    with pytest.raises(ValueError, match='seven finite'):
        module.sequence_features(values[:6])
    with pytest.raises(ValueError, match='seven finite'):
        module.sequence_features(values * np.nan)


@pytest.mark.parametrize('mode,incoming,architecture', [('static', 160, 'BandMLP-7826'), ('temporal', 640, 'BandTemporalMLP-30866')])
def test_exported_numpy_runtime_matches_analytic_logits_and_rejects_tampering(monkeypatch, tmp_path, mode, incoming, architecture):
    module = experiment()
    monkeypatch.setattr(module, 'OUTPUT', tmp_path)
    runtime = module.export_runtime()
    model = tmp_path / 'model'
    model.mkdir()
    first = np.zeros((48, incoming))
    first[0, 39] = 0.1
    final = np.zeros((2, 48))
    final[1, 0] = 2
    np.savez(model / 'weights.npz', weight0=first, bias0=np.zeros(48), weight1=final, bias1=np.zeros(2))
    checksum = hashlib.sha256((model / 'weights.npz').read_bytes()).hexdigest()
    (model / 'manifest.json').write_text(json.dumps(dict(format=1, architecture=architecture, feature_mode=mode,
                model_tag=int(checksum[:8], 16), sha256={'weights.npz': checksum})), encoding='utf-8')
    detector = runtime.BandDetector(model)
    values = np.arange(40)[None, :, None] + np.zeros((7, 40, 24))
    assert detector.score(values)['score'] == pytest.approx(1 / (1 + np.exp(-3.9)))
    assert detector.score([])['score'] == 0
    for samples in (48000, 48128):
        waveform = np.sin(2 * np.pi * 700 * np.arange(samples) / 48000) * 0.1
        assert detector.score_waveform(waveform) == module.BandDetector(model).score(module.features(waveform))
    with pytest.raises(ValueError, match='completed mono beam'):
        detector.score_waveform(np.zeros(47999))
    with pytest.raises(ValueError, match='completed mono beam'):
        detector.score_waveform(np.full(48000, np.nan))
    tree = ast.parse((tmp_path / 'numpy_runtime.py').read_text(encoding='utf-8'), feature_version=(3, 6))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(value.name.split('.')[0] not in {'torch', 'scipy', 'horn', 'ps'} for value in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in {'torch', 'scipy', 'horn', 'ps'}
    (model / 'weights.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        runtime.BandDetector(model)


def test_replay_preserves_recording_bags_and_never_reads_held_out_sources(monkeypatch, tmp_path):
    module = experiment()
    monkeypatch.setattr(module, 'BASE', tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    monkeypatch.setattr(module, 'OUTPUT', output)
    monkeypatch.setattr(module, 'check_plan', lambda: {})
    (output / 'plan.json').write_text('{}', encoding='utf-8')
    time = np.arange(96000) / 48000
    path = tmp_path / 'recording.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((np.sin(2 * np.pi * 700 * time) * 1000).astype('<i2').tobytes())
    records = [dict(path=str(path), file=path.name, label=1, split='train', dataset='HornBase', sha256=module.digest(path)),
               dict(path='must-not-be-read.wav', file='missing.wav', label=0, split='test', dataset='ESC50', sha256='unknown')]
    (tmp_path / 'clips.json').write_text(json.dumps(records), encoding='utf-8')
    module.prepare_replay('train')
    selected, data = module.load_replay('train')
    assert selected == records[:1]
    np.testing.assert_array_equal(data['offsets'], [0, 3])
    assert data['static'].shape == (3, 7, 160) and data['temporal'].shape == (3, 640)
    metadata = json.loads((output / 'replay-train.json').read_text())
    assert metadata['starts'] == [[0, 24000, 48000]]
    with pytest.raises(ValueError, match='Freeze models'):
        module.prepare_replay('test')


def test_gate_budgets_do_not_mutate_other_experiments():
    module = experiment()
    original = dict(module.evaluation.SETTINGS)
    cases, probabilities = [], []
    for domain in ('HornBase', 'ESC50'):
        for index in range(21):
            cases.append(dict(label=int(index == 0), angle=0, dataset=domain, split='validation',
                              bearings=[dict(azimuth=0), dict(azimuth=90)]))
            probabilities.append([0.7 if domain == 'HornBase' else 0.95, 0.1] if index == 0
                                 else [0.8, 0.79] if domain == 'HornBase' and index == 1 else [0.1, 0.1])
    ordinary = module.choose_gate(cases, probabilities, 0.1)
    strict = module.choose_gate(cases, probabilities, 0.01)
    assert ordinary['metrics']['tp'] == 2 and strict['metrics']['tp'] == 1
    assert module.evaluation.SETTINGS == original
    with pytest.raises(ValueError, match='validation sources'):
        module.choose_gate([dict(case, split='test') for case in cases], probabilities, 0.1)
