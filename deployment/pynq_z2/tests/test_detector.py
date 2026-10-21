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
