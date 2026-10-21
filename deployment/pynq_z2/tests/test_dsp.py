import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config
from ps.dsp import Conditioning, beamform, candidates, features, spatial_map
from ps.geometry import Bearing

from horn.calibration import StreamCalibration, remove_dc
from horn.config import Config
from horn.features import log_mel, mel_energy
from horn.frontend import stft
from horn.geometry import circular_array
from horn.surround import form_beam, spatial_response


def test_conditioning_matches_causal_reference_across_partitions():
    audio = np.random.default_rng(3).normal(1000, 3000, (7000, 14))
    expected, _ = remove_dc(audio / 32768)
    filter = Conditioning(14)
    actual = np.vstack([filter.feed(block) for block in np.array_split(audio, [73, 800, 2048])])
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    gains = np.linspace(0.9, 1.1, 14)
    delays = np.linspace(-0.3, 0.5, 14)
    expected = StreamCalibration(gains, delays).feed(expected)
    filter = Conditioning(14, gains, delays)
    actual = np.vstack([filter.feed(block) for block in np.array_split(audio, [73, 800, 2048])])
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_srp_map_and_beam_match_pc_reference():
    config = load_config()
    config['search'].update(azimuth_step=6, depression_step=6)
    audio = np.random.default_rng(4).normal(size=(4096, 14))
    parameters = Config(microphones=14, diameter=0.24)
    expected = spatial_response(audio, circular_array(parameters), parameters, 6, 6)
    actual = spatial_map(audio, config['audio'], config['search'])
    for first, second in zip(actual, expected):
        np.testing.assert_allclose(first, second, atol=1e-12)
    bearing = Bearing(30, 42, 1)
    actual, latency = beamform(audio, bearing, config['audio'])
    expected, expected_latency = form_beam(audio, circular_array(parameters), bearing, parameters)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert latency == expected_latency


def test_feature_order_and_scale_match_trained_frontend():
    audio = np.random.default_rng(8).normal(0, 0.1, 48128)
    actual = features(audio)
    reference = np.asarray([log_mel(mel_energy(abs(frame.spectrum[:, 0]) ** 2)) for frame in stft(audio[:, None])])
    np.testing.assert_allclose(actual[0], reference[:24].T, atol=1e-12)
    assert actual.shape[1:] == (40, 24)
    silence = spatial_map(np.zeros((2048, 14)), load_config()['audio'], load_config()['search'])
    assert candidates(silence) == []


def test_centered_frontend_removes_common_gain_without_altering_spectral_contrast():
    audio = np.random.default_rng(42).normal(0, 0.01, 48000)
    expected = features(audio, 'centered_logmel')
    for gain in (0.01, 0.1, 10):
        np.testing.assert_allclose(features(audio * gain, 'centered_logmel'), expected, atol=1e-5)
    raw = features(audio)
    np.testing.assert_allclose(expected, raw - raw.mean(axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(expected.mean(axis=(1, 2)), 0, atol=1e-12)
    assert np.isfinite(features(np.zeros(48000), 'centered_logmel')).all()
    with pytest.raises(ValueError, match='frontend'):
        features(audio, 'unknown')


def test_geometry_generation_is_reproducible_and_board_syntax_is_compatible():
    spec = importlib.util.spec_from_file_location('geometry_export', ROOT / 'scripts/export_geometry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (ROOT / 'ps/geometry.py').read_text(encoding='utf-8') == module.generate()
    for path in (ROOT / 'ps').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 6))
