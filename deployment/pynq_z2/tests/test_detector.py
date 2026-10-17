import ast
import importlib.util
from pathlib import Path
import sys

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
