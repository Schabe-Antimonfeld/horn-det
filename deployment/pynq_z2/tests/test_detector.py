import ast
import importlib.util
from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_integer_export_is_reproducible_and_python36_compatible():
    spec = importlib.util.spec_from_file_location('integer_export', ROOT / 'scripts/export_integer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = (ROOT / 'ps/integer_kernel.py').read_text(encoding='utf-8')
    assert generated == module.generate()
    ast.parse(generated, feature_version=(3, 6))


def test_board_integer_matches_real_trained_golden_vectors():
    from ps.detector import Detector
    from ps.integer_kernel import fixed_forward, read_memory
    directory = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not directory.is_dir():
        pytest.skip('Existing locally trained integer weights are absent')
    detector = Detector(directory)
    codes = read_memory(directory / 'inputs.mem', 32, True).reshape(-1, 40, 24)
    expected = read_memory(directory / 'expected.mem', 32).reshape(-1, 8)
    for code, words in zip(codes, expected):
        result = fixed_forward(detector.model, code)
        np.testing.assert_array_equal(result['logits'].astype(np.int64) & 0xffffffff, words[3:5])
        assert result['probability'] == words[5] & 65535
        inference = detector.infer(code / float(1 << detector.metadata['input_fraction']))
        assert inference['score'] == (words[5] & 65535) / 65536.0
    with pytest.raises(ValueError, match='calibrated'):
        detector.infer(np.full((40, 24), 100000.0))


def test_numpy_spectral_model_matches_analytic_logits_and_checks_integrity(tmp_path):
    from ps.detector import SpectralDetector
    weights = np.zeros((48, 160), dtype=np.float32)
    weights[0, 39] = 0.1
    head = np.zeros((2, 48), dtype=np.float32)
    head[1, 0] = 2
    np.savez(tmp_path / 'weights.npz', weight0=weights, bias0=np.zeros(48), weight1=head, bias1=np.zeros(2))
    checksum = hashlib.sha256((tmp_path / 'weights.npz').read_bytes()).hexdigest()
    metadata = dict(format=1, architecture='BandMLP-7826', model_tag=int(checksum[:8], 16), sha256={'weights.npz': checksum})
    (tmp_path / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
    model = SpectralDetector(tmp_path)
    values = (np.arange(40)[:, None] + np.zeros((40, 24)))[None]
    assert model.score(values)['score'] == pytest.approx(1 / (1 + np.exp(-3.9)), abs=1e-8)
    assert model.score(values[:, ::-1])['score'] == pytest.approx(0.5)
    assert model.score([])['score'] == 0
    with pytest.raises(ValueError, match='N x'):
        model.score(values[0])
    with pytest.raises(ValueError, match='finite'):
        model.score(values * np.nan)
    (tmp_path / 'weights.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        SpectralDetector(tmp_path)


def test_temporal_features_preserve_analytic_envelope_order_and_gain():
    from ps.band import temporal_features
    values = np.arange(7)[:, None, None] + np.arange(40)[None, :, None] + np.zeros((7, 40, 24))
    original = values.copy()
    actual = temporal_features(values)
    for index, offset in enumerate((-2, 0.5, 2.5, 3)):
        np.testing.assert_allclose(actual[index * 160:index * 160 + 40], np.arange(40) - 19.5 + offset)
        np.testing.assert_allclose(actual[index * 160 + 40:(index + 1) * 160], 0)
    np.testing.assert_allclose(temporal_features(values + 100), actual, atol=1e-12)
    np.testing.assert_array_equal(original, values)
    assert not np.array_equal(temporal_features(values[::-1]), actual)
    with pytest.raises(ValueError, match='seven finite'):
        temporal_features(values[:6])
    with pytest.raises(ValueError, match='seven finite'):
        temporal_features(values * np.nan)


def test_temporal_model_checks_frontend_identity_and_analytic_logits(tmp_path):
    from ps.band import TemporalDetector, load_spectral, validate_model
    first = np.zeros((48, 640))
    first[0, 39] = 0.1
    final = np.zeros((2, 48))
    final[1, 0] = 2
    np.savez(tmp_path / 'weights.npz', weight0=first, bias0=np.zeros(48), weight1=final, bias1=np.zeros(2))
    checksum = hashlib.sha256((tmp_path / 'weights.npz').read_bytes()).hexdigest()
    metadata = dict(format=1, architecture='BandTemporalMLP-30866', feature_mode='temporal', contexts=7,
                    input_shape=[40, 24], provenance=dict(frontend='logmel'),
                    model_tag=int(checksum[:8], 16), sha256={'weights.npz': checksum})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(metadata), encoding='utf-8')
    model = load_spectral(tmp_path)
    assert isinstance(model, TemporalDetector) and validate_model(tmp_path) == metadata
    values = np.arange(40)[None, :, None] + np.zeros((7, 40, 24))
    assert model.score(values)['score'] == pytest.approx(1 / (1 + np.exp(-3.9)))
    assert model.score(values[:, ::-1])['score'] == 0.5
    assert model.score([])['score'] == 0
    with pytest.raises(ValueError, match='seven finite'):
        model.score(values[:6])
    for key, value in [('feature_mode', 'static'), ('contexts', 6), ('input_shape', [24, 40]),
                       ('provenance', dict(frontend='centered_logmel'))]:
        path.write_text(json.dumps(dict(metadata, **{key: value})), encoding='utf-8')
        with pytest.raises(ValueError, match='contract'):
            validate_model(tmp_path)
    path.write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / 'weights.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        validate_model(tmp_path)


def test_temporal_deployment_matches_all_frozen_test_beam_scores():
    from ps.band import TemporalDetector
    base = ROOT.parents[1] / 'prototype/outputs'
    model = base / 'band_improvements_v1/model-combined'
    if not model.is_dir():
        pytest.skip('Optional local temporal training artifacts absent')
    detector = TemporalDetector(model)
    maximum = 0
    for phase in ('test', 'stress'):
        for protocol in ('original', 'visible'):
            key = phase + '-' + protocol
            with np.load(base / ('horn_recovery_v1/' + key + '.npz'), allow_pickle=False) as archive:
                contexts = archive['contexts']
            with np.load(base / ('band_improvements_v1/' + key + '-predictions.npz'), allow_pickle=False) as archive:
                expected = archive['combined']
            actual = np.asarray([[detector.score(beam)['score'] for beam in scene] for scene in contexts])
            maximum = max(maximum, float(np.abs(actual - expected).max()))
    assert maximum < 1e-12
