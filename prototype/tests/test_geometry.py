import numpy as np
import pytest

from horn.calibration import calibrate, fractional_delay, remove_dc
from horn.config import Config
from horn.geometry import circular_array, delay_table, exact_delays, pairs, round_away, source_delays


def test_document_parameters():
    config = Config()
    assert config.context_seconds == pytest.approx(0.2666666667)
    assert len(config.angles()) == 181
    assert len(config.angles(coarse=True)) == 37
    assert pairs(8).shape == (28, 2)
    assert np.max(np.abs(delay_table(circular_array(config), config.angles()))) == pytest.approx(9600 / 343)


@pytest.mark.parametrize("angle", [-80, -20, 0, 35, 80])
def test_signed_far_field_matches_exact_geometry(angle):
    config = Config()
    coordinates = circular_array(config)
    source = 1000 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0])
    arrivals = exact_delays(coordinates, source) * config.sample_rate
    selected = pairs(8)
    exact = arrivals[selected[:, 0]] - arrivals[selected[:, 1]]
    predicted = delay_table(coordinates, [angle])[:, 0]
    assert np.max(np.abs(exact - predicted)) < 0.002
    relative = source_delays(coordinates, angle)
    np.testing.assert_allclose(relative[selected[:, 0]] - relative[selected[:, 1]], predicted, atol=1e-12)


def test_calibration_aligns_impulses():
    audio = np.zeros((32, 2))
    audio[5, 0] = 2
    audio[8, 1] = 0.5
    aligned, delay = calibrate(audio, [2, 0.5], [0, 3])
    assert delay == 3
    np.testing.assert_allclose(aligned[:, 0], aligned[:, 1], atol=1e-14)
    assert np.argmax(aligned[:, 0]) == 8


def test_shift_does_not_wrap_and_rounding_is_explicit():
    signal = np.zeros(64)
    signal[-1] = 1
    assert np.max(np.abs(fractional_delay(signal, 4)[:32])) < 1e-14
    np.testing.assert_array_equal(round_away([-1.5, -0.5, 0.5, 1.5]), [-2, -1, 1, 2])


def test_dc_stream_matches_batch():
    signal = np.random.default_rng(4).normal(size=(200, 3)) + 4
    expected, _ = remove_dc(signal)
    first, state = remove_dc(signal[:73])
    second, _ = remove_dc(signal[73:], state=state)
    np.testing.assert_allclose(np.vstack([first, second]), expected)


@pytest.mark.parametrize("settings", [{"hop": 2048}, {"frame": 1000}, {"sound_speed": 0}, {"context": 0}])
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        Config(**settings)
