import numpy as np
import pytest

from horn.weighting import frequency_weights, learned_prior, NoisePSD, Weighting


def test_exact_weights_and_independent_ablations():
    power, noise, prior = np.array([1., 2., 4.]), np.ones(3), np.array([0.5, 1., 0.25])
    expected = prior * np.array([0., 0.5, 0.75]) * np.array([0., 1 / 3, 1.])
    np.testing.assert_allclose(frequency_weights(power, noise, prior), expected)
    np.testing.assert_allclose(frequency_weights(power, noise, prior, use_subtraction=False, use_snr=False), prior)
    np.testing.assert_allclose(frequency_weights(power, noise, prior, use_prior=False,
                                               use_subtraction=False, use_snr=False), 1)


def test_previous_noise_and_trigger_frame_freeze():
    estimator = NoisePSD(alpha=0.5)
    assert not estimator.weights(np.ones(3), np.ones(3)).any()
    with pytest.raises(ValueError):
        estimator.initialize(np.ones((3, 3)))
    estimator.initialize(np.ones((3, 3)), confirmed=True)
    before = estimator.weights(np.full(3, 4.), np.ones(3))
    assert estimator.update(np.full(3, 2.), "IDLE")
    np.testing.assert_allclose(estimator.value, 1.5)
    for state in ("IDLE", "CANDIDATE", "CONFIRMED", "END"):
        assert not estimator.update(np.full(3, 20.), state, triggered=True)
    np.testing.assert_allclose(estimator.value, 1.5)
    np.testing.assert_allclose(before, 0.75)


def test_training_only_and_degenerate_prior():
    powers = np.array([[1., 1., 1.], [1., 2., 4.]])
    prior, metadata = learned_prior(powers, [0, 1], ["train", "train"], [0, 1, 0])
    np.testing.assert_allclose(prior, [0, 0.5, 1])
    assert not metadata["degenerate"]
    with pytest.raises(ValueError):
        learned_prior(powers, [0, 1], ["train", "test"], [0, 1, 0])
    prior, metadata = learned_prior(np.ones((2, 3)), [0, 1], ["train", "train"], [0, 1, 0])
    assert metadata["degenerate"]
    np.testing.assert_array_equal(prior, [0, 1, 0])
