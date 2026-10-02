import numpy as np
import pytest

from horn.config import Config
from horn.fixed import Format, integer_fft, integer_phat, multiply, PHAT, PCM, Reciprocal, shift_round, TRANSFORM, golden_frame
from horn.geometry import circular_array


def test_saturation_half_ties_and_full_product():
    codes, count = PHAT.encode([-2, -1, -0.5 / 32768, 0.5 / 32768, 1, 2])
    np.testing.assert_array_equal(codes, [-32768, -32768, -1, 1, 32767, 32767])
    assert count == 3
    np.testing.assert_array_equal(shift_round([-3, -1, 1, 3], 1), [-2, -1, 1, 2])
    wide = Format(48, 30)
    actual, _ = multiply([8388607], [8388607], Format(24, 15), Format(24, 15), wide)
    assert actual[0] == 8388607 ** 2


@pytest.mark.parametrize("inverse", [False, True])
def test_integer_transform_matches_analytic_scaled_fft(inverse):
    generator = np.random.default_rng(5)
    audio = generator.uniform(-0.4, 0.4, (64, 2))
    codes, _ = PCM.encode(audio)
    real, imaginary, stats = integer_fft(codes, inverse=inverse)
    expected = np.fft.ifft(audio, axis=0) if inverse else np.fft.fft(audio, axis=0) / 64
    actual = TRANSFORM.decode(real) + 1j * TRANSFORM.decode(imaginary)
    assert np.max(np.abs(actual - expected)) < 0.0002
    assert stats["saturations"] == 0


def test_reciprocal_exponent_and_minimum_negative_are_widened():
    lut = Reciprocal()
    for magnitude in [1, 3, 256, 1025, 2 ** 47, 2 ** 48 - 1]:
        assert abs(lut.approximate(magnitude) * magnitude - 1) < 0.001
    real, imaginary, _ = integer_phat(np.array([-2 ** 47, 0]), np.array([0, 0]))
    np.testing.assert_array_equal(real, [-32768, 0])
    np.testing.assert_array_equal(imaginary, [0, 0])


def test_golden_files_and_signed_rom(tmp_path):
    config = Config(frame=64, hop=32, mel_bands=4)
    audio = np.zeros((64, 8))
    audio[20] = 0.5
    metadata = golden_frame(audio, circular_array(config), tmp_path, config)
    archive = np.load(tmp_path / "golden.npz", allow_pickle=False)
    assert archive["pcm"].shape == (64, 8)
    assert archive["integer_delay"].min() < 0
    assert len((tmp_path / "pcm.mem").read_text().splitlines()) == 512
    assert metadata["fft_max_error"] < 0.0002


def test_integer_weight_noise_and_log_match_float():
    from horn.fixed import POWER, WEIGHT, integer_weights, noise_update, log_codes
    from horn.weighting import frequency_weights
    power, noise, prior = np.array([1., 2., 4.]), np.ones(3), np.array([0.5, 1., 0.25])
    power_codes, _ = POWER.encode(power)
    noise_codes, _ = POWER.encode(noise)
    prior_codes, _ = WEIGHT.encode(prior)
    codes, _ = integer_weights(power_codes, noise_codes, prior_codes)
    np.testing.assert_allclose(WEIGHT.decode(codes), frequency_weights(power, noise, prior), atol=4 / 65536)
    updated, _ = noise_update(noise_codes, power_codes, alpha=0.5)
    np.testing.assert_allclose(POWER.decode(updated), 0.5 * (noise + power))
    frozen, _ = noise_update(noise_codes, power_codes, enabled=False)
    np.testing.assert_array_equal(frozen, noise_codes)
    logarithms, _ = log_codes(power_codes)
    np.testing.assert_allclose(Format(24, 15).decode(logarithms), np.log(power), atol=0.0011)
