import numpy as np
import pytest

from horn.config import Config
from horn.frontend import stft
from horn.geometry import circular_array
from horn.localization import gcc, gcc_direction, interpolate, locate, Search, Correlations
from horn.simulation import far_field


def test_gcc_sign_against_delayed_impulse():
    config = Config(microphones=2)
    audio = np.zeros((1024, 2))
    audio[300, 0] = 1
    audio[307, 1] = 1
    result = gcc(np.fft.rfft(audio, axis=0), config)
    assert np.argmax(result.values[0]) == 1024 - 7


def test_negative_fractional_interpolation():
    values = np.array([[0, 10, 20, 30, 40, 50, 60, 70]])
    assert interpolate(values, np.array([[-2.3]]))[0, 0] == pytest.approx(57)
    assert interpolate(values, np.array([[-0.5]]), linear=False)[0, 0] == 70


@pytest.mark.parametrize("angle", [-80, -35, 0, 25, 80])
@pytest.mark.parametrize("mode", ["full", "coarse", "sector"])
def test_broadband_angle_recovery(angle, mode):
    scene = far_field(np.random.default_rng(16).normal(size=4096), angle)
    spectrum = stft(scene.audio)[3].spectrum
    result = locate(gcc(spectrum), scene.coordinates, mode=mode)
    assert result.valid
    assert abs(result.angle - angle) <= 1
    assert 0 <= result.confidence <= 1
    baseline = gcc_direction(gcc(spectrum), scene.coordinates)
    assert baseline.valid and abs(baseline.angle - angle) <= 2


def test_zero_weights_silence_faults_and_flat_response_are_invalid():
    config = Config()
    coordinates = circular_array(config)
    zero = np.zeros((513, 8), dtype=complex)
    assert not locate(gcc(zero), coordinates).valid
    signal = np.random.default_rng(1).normal(size=(1024, 8))
    spectrum = np.fft.rfft(signal, axis=0)
    assert not locate(gcc(spectrum, weight=np.zeros(513)), coordinates).valid
    assert not locate(gcc(spectrum, channels=[True, True] + [False] * 6), coordinates).valid
    flat = Correlations(np.ones((28, 1024)), np.ones(28, dtype=bool), np.zeros((28, 513)))
    assert not locate(flat, coordinates).valid


def test_untrusted_gcc_falls_back_to_configured_full_range():
    scene = far_field(np.random.default_rng(12).normal(size=4096), 40)
    correlations = gcc(stft(scene.audio)[2].spectrum)
    settings = Search(residual_limit=0.00001)
    result = locate(correlations, scene.coordinates, settings=settings)
    assert result.valid and result.fallback
    assert result.mode in ("coarse_fine", "full_fallback")
    assert abs(result.angle - 40) <= 1
