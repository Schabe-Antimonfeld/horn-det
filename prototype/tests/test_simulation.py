import numpy as np
import pytest

from horn.config import Config
from horn.simulation import far_field, mix_noise, randomized, room_audio, room_rir


def test_far_field_impulse_has_correct_signed_delays_and_guard():
    source = np.zeros(256)
    source[100] = 1
    scene = far_field(source, 0)
    peaks = np.argmax(scene.audio, axis=0)
    assert peaks[0] < peaks[4]
    assert peaks[4] - peaks[0] == 28
    assert np.max(np.abs(scene.audio[:30])) < 1e-12


def test_noise_scaling_measures_requested_snr():
    source = np.ones((1000, 2))
    noise = np.random.default_rng(2).normal(size=source.shape)
    mixed = mix_noise(source, noise, -5)
    assert 10 * np.log10(np.mean(source ** 2) / np.mean((mixed - source) ** 2)) == pytest.approx(-5)


def test_randomization_repeats_and_preserves_group():
    source = np.random.default_rng(4).normal(size=500)
    first = randomized(source, 35, seed=19, source_group="recording-a")
    second = randomized(source, 35, seed=19, source_group="recording-a")
    np.testing.assert_array_equal(first.audio, second.audio)
    assert first.metadata["source_group"] == "recording-a"
    assert 330 <= first.speed <= 355
    assert first.valid.sum() >= 6


def test_rir_direct_path_and_linear_tail():
    coordinates = np.array([[1, 1, 1], [1.1, 1, 1]])
    rir = room_rir(coordinates, [2, 1, 1], [4, 4, 3], reflection=0, rt60=0.02)
    assert np.argmax(rir[:, 0]) == round(48000 / 343)
    source = np.array([1.0, 0, 0])
    output = room_audio(source, rir)
    assert len(output) == len(rir) + 2
    np.testing.assert_allclose(output[:len(rir)], rir, atol=1e-14)
