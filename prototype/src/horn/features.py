"""HTK Mel, MFCC, causal PCEN and spectral candidate rules, chapter 07."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct

from .config import Config


def mel_bank(config=Config(), lower=0.0, upper=None):
    upper = config.sample_rate / 2 if upper is None else upper
    if not 0 <= lower < upper <= config.sample_rate / 2:
        raise ValueError("Mel band must lie inside Nyquist")
    edges = 700 * (10 ** (np.linspace(2595 * np.log10(1 + lower / 700),
                                       2595 * np.log10(1 + upper / 700), config.mel_bands + 2) / 2595) - 1)
    frequency = np.fft.rfftfreq(config.frame, 1 / config.sample_rate)
    bank = np.maximum(0, np.minimum((frequency[None, :] - edges[:-2, None])
                                     / (edges[1:-1, None] - edges[:-2, None]),
                                     (edges[2:, None] - frequency[None, :])
                                     / (edges[2:, None] - edges[1:-1, None])))
    if np.any(bank.sum(axis=1) == 0):
        raise ValueError("FFT resolution is too low for the requested Mel bands")
    return bank


def mel_energy(powers, config=Config()):
    powers = np.asarray(powers, dtype=float)
    if powers.shape[-1] != config.frame // 2 + 1 or not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("Invalid power spectrum")
    return powers @ mel_bank(config).T


def log_mel(energies, epsilon=1e-12):
    return np.log(np.asarray(energies) + epsilon)


def mfcc(logarithms, coefficients=13):
    logarithms = np.asarray(logarithms)
    if not 1 <= coefficients <= logarithms.shape[-1]:
        raise ValueError("Invalid MFCC coefficient count")
    return dct(logarithms, type=2, norm="ortho", axis=-1)[..., :coefficients]


def statistics(coefficients):
    coefficients = np.asarray(coefficients)
    return np.concatenate((coefficients.mean(axis=-2), coefficients.std(axis=-2)), axis=-1)


class PCEN:
    def __init__(self, smoothing=0.025, gain=0.98, bias=2.0, exponent=0.5, epsilon=1e-6):
        if not 0 < smoothing <= 1 or not 0 <= gain <= 1 or bias <= 0 or not 0 < exponent <= 1 or epsilon <= 0:
            raise ValueError("Invalid PCEN settings")
        self.smoothing, self.gain, self.bias = smoothing, gain, bias
        self.exponent, self.epsilon = exponent, epsilon
        self.state = None

    def feed(self, energies):
        energies = np.asarray(energies, dtype=float)
        if energies.ndim != 2 or not np.isfinite(energies).all() or np.any(energies < 0):
            raise ValueError("PCEN expects nonnegative (frames, bands) energy")
        if self.state is None and len(energies):
            self.state = energies[0].copy()
        output = []
        for energy in energies:
            if self.state.shape != energy.shape:
                raise ValueError("PCEN band count changed")
            self.state = (1 - self.smoothing) * self.state + self.smoothing * energy
            output.append((energy / (self.epsilon + self.state) ** self.gain + self.bias)
                          ** self.exponent - self.bias ** self.exponent)
        return np.asarray(output).reshape(energies.shape)


@dataclass(frozen=True)
class Gate:
    lower: float = 200.0
    upper: float = 3000.0
    energy: float = 1e-5
    band_ratio: float = 0.4
    prominence: float = 8.0
    consecutive: int = 2


class FFTGate:
    """Configurable candidate rule; its score is not a horn probability."""

    def __init__(self, config=Config(), settings=Gate()):
        if not 0 <= settings.lower < settings.upper <= config.sample_rate / 2:
            raise ValueError("Invalid gate frequency range")
        if not 0 <= settings.band_ratio <= 1 or settings.energy < 0 or settings.prominence < 0 or settings.consecutive < 1:
            raise ValueError("Invalid candidate thresholds")
        self.config, self.settings = config, settings
        frequency = np.fft.rfftfreq(config.frame, 1 / config.sample_rate)
        self.band = (frequency >= settings.lower) & (frequency <= settings.upper)
        if not self.band.any():
            raise ValueError("Gate frequency range has no FFT bins")
        self.count = 0

    def feed(self, power):
        power = np.asarray(power)
        if power.shape != self.band.shape or not np.isfinite(power).all() or np.any(power < 0):
            raise ValueError("Invalid gate power spectrum")
        selected = power[self.band]
        energy = float(power.sum() / self.config.frame ** 2)
        ratio = float(selected.sum() / (power.sum() + self.config.epsilon))
        prominence = float(selected.max() / (np.median(selected) + self.config.epsilon))
        matched = energy >= self.settings.energy and ratio >= self.settings.band_ratio and prominence >= self.settings.prominence
        self.count = self.count + 1 if matched else 0
        return self.count >= self.settings.consecutive, {"energy": energy, "band_ratio": ratio, "prominence": prominence}
