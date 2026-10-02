"""Training-only priors, previous-frame noise PSD and ablations, chapter 08."""

from dataclasses import dataclass

import numpy as np


def nonnegative(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Power must be a finite nonnegative vector")
    return values


def fixed_band(frequencies, lower=200.0, upper=3000.0):
    if not 0 <= lower < upper:
        raise ValueError("Invalid frequency band")
    frequencies = np.asarray(frequencies)
    return ((frequencies >= lower) & (frequencies <= upper)).astype(float)


def learned_prior(powers, labels, splits, fallback, epsilon=1e-12):
    powers = np.asarray(powers, dtype=float)
    labels, splits = np.asarray(labels), np.asarray(splits)
    fallback = nonnegative(fallback)
    if powers.ndim != 2 or powers.shape[1] != len(fallback) or len(powers) != len(labels) or labels.shape != splits.shape:
        raise ValueError("Prior inputs have incompatible shapes")
    if not np.isfinite(powers).all() or np.any(powers < 0) or np.any(fallback > 1):
        raise ValueError("Invalid power or fallback weight")
    if not np.isin(labels, [0, 1]).all() or not np.all(splits == "train"):
        raise ValueError("Prior estimation requires binary training samples only")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Both classes are required")
    logarithm = np.log(powers + epsilon)
    difference = logarithm[labels == 1].mean(axis=0) - logarithm[labels == 0].mean(axis=0)
    span = float(np.ptp(difference))
    if span <= epsilon:
        return fallback.copy(), {"degenerate": True, "minimum": float(difference.min()), "maximum": float(difference.max())}
    return np.clip((difference - difference.min()) / span, 0, 1), {
        "degenerate": False, "minimum": float(difference.min()), "maximum": float(difference.max())}


@dataclass(frozen=True)
class Weighting:
    subtraction: float = 1.0
    lower: float = 1.0
    upper: float = 4.0
    epsilon: float = 1e-12

    def __post_init__(self):
        if not np.isfinite([self.subtraction, self.lower, self.upper, self.epsilon]).all():
            raise ValueError("Weight parameters must be finite")
        if self.subtraction < 0 or self.lower < 0 or self.upper <= self.lower or self.epsilon <= 0:
            raise ValueError("Invalid subtraction or SNR thresholds")


def frequency_weights(power, noise, prior, settings=Weighting(),
                      use_prior=True, use_subtraction=True, use_snr=True):
    power, noise, prior = map(nonnegative, (power, noise, prior))
    if power.shape != noise.shape or prior.shape != power.shape or np.any(prior > 1):
        raise ValueError("Weight vectors must agree and prior must be in [0, 1]")
    retention = np.maximum(power - settings.subtraction * noise, 0) / (power + settings.epsilon)
    ratio = power / (noise + settings.epsilon)
    snr = np.clip((ratio - settings.lower) / (settings.upper - settings.lower), 0, 1)
    return ((prior if use_prior else np.ones_like(power))
            * (retention if use_subtraction else 1)
            * (snr if use_snr else 1))


class NoisePSD:
    def __init__(self, alpha=0.98):
        if not 0 <= alpha < 1:
            raise ValueError("Noise smoothing must be in [0, 1)")
        self.alpha = alpha
        self.value = None

    def initialize(self, backgrounds, confirmed=False):
        """Require explicit background designation instead of learning arbitrary startup audio."""
        backgrounds = np.asarray(backgrounds, dtype=float)
        if not confirmed or backgrounds.ndim != 2 or not len(backgrounds):
            raise ValueError("Initialization requires confirmed background frames")
        if not np.isfinite(backgrounds).all() or np.any(backgrounds < 0):
            raise ValueError("Invalid background PSD")
        self.value = backgrounds.mean(axis=0)

    def weights(self, power, prior, settings=Weighting(), **ablations):
        if self.value is None:
            return np.zeros_like(nonnegative(power))
        return frequency_weights(power, self.value, prior, settings, **ablations)

    def update(self, power, state, triggered=False, channels_valid=True):
        power = nonnegative(power)
        if self.value is None:
            return False
        if power.shape != self.value.shape:
            raise ValueError("Noise PSD shape changed")
        name = getattr(state, "name", state)
        if name == "IDLE" and not triggered and channels_valid:
            self.value = self.alpha * self.value + (1 - self.alpha) * power
            return True
        return False
