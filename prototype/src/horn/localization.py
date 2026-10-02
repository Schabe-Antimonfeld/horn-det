"""Signed GCC, interpolated SRP and guarded sector search, chapters 04 to 06."""

from dataclasses import dataclass

import numpy as np

from .config import Config
from .geometry import baselines, delay_table, pairs, reliability, round_away


@dataclass(frozen=True)
class Correlations:
    values: np.ndarray
    valid: np.ndarray
    spectrum: np.ndarray


@dataclass(frozen=True)
class Estimate:
    angle: float = 0.0
    valid: bool = False
    confidence: float = 0.0
    psr: float = 0.0
    mode: str = "invalid"
    pair_count: int = 0
    evaluations: int = 0
    fallback: bool = False


@dataclass(frozen=True)
class Search:
    interpolation: bool = True
    exclusion: float = 2.0
    psr_minimum: float = 1.0
    confidence_low: float = 1.0
    confidence_high: float = 8.0
    minimum_pairs: int = 3
    sector: float = 8.0
    coarse_pairs: int = 4
    selected_pairs: int | None = None
    residual_limit: float = 2.0

    def __post_init__(self):
        values = (self.exclusion, self.psr_minimum, self.confidence_low,
                  self.confidence_high, self.sector, self.residual_limit)
        if not np.isfinite(values).all() or self.exclusion < 0 or self.sector <= 0 or self.residual_limit <= 0:
            raise ValueError("Search settings must be finite with a positive sector")
        if self.confidence_high <= self.confidence_low or self.minimum_pairs < 2 or self.coarse_pairs < 2:
            raise ValueError("Invalid confidence or pair settings")
        if self.selected_pairs is not None and self.selected_pairs < self.minimum_pairs:
            raise ValueError("Pair selection cannot undercut the minimum")


def gcc(spectrum, config=Config(), weight=None, channels=None):
    spectrum = np.asarray(spectrum)
    if spectrum.shape != (config.frame // 2 + 1, config.microphones) or not np.isfinite(spectrum).all():
        raise ValueError("Spectrum does not match the configured half spectrum")
    selected = pairs(config.microphones)
    cross = spectrum[:, selected[:, 0]].T * spectrum[:, selected[:, 1]].T.conj()
    normalized = cross / (np.abs(cross) + config.epsilon)
    if weight is None:
        weight = np.ones(spectrum.shape[0])
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (spectrum.shape[0],) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Frequency weight must be a finite nonnegative half spectrum")
    if channels is None:
        channels = np.ones(config.microphones, dtype=bool)
    channels = np.asarray(channels, dtype=bool)
    if channels.shape != (config.microphones,):
        raise ValueError("Channel mask does not match the array")
    active = np.sum((np.abs(cross) > config.epsilon) & (weight[None, :] > config.epsilon), axis=1) >= 2
    valid = active & channels[selected[:, 0]] & channels[selected[:, 1]]
    normalized *= weight[None, :]
    normalized[~valid] = 0
    correlations = np.fft.irfft(normalized, n=config.frame, axis=1)
    return Correlations(correlations, valid, normalized)


def interpolate(values, delays, linear=True):
    values, delays = np.asarray(values), np.asarray(delays)
    if values.ndim != 2 or delays.ndim != 2 or values.shape[0] != delays.shape[0]:
        raise ValueError("Correlation rows and delay rows must agree")
    if not np.isfinite(delays).all() or np.max(np.abs(delays), initial=0) >= values.shape[1] / 2 - 1:
        raise ValueError("Delay exceeds unambiguous FFT support")
    row = np.arange(len(values))[:, None]
    if not linear:
        return values[row, round_away(delays) % values.shape[1]]
    lower = np.floor(delays).astype(int)
    fraction = delays - lower
    return ((1 - fraction) * values[row, lower % values.shape[1]]
            + fraction * values[row, (lower + 1) % values.shape[1]])


def srp(correlations, coordinates, angles, config=Config(), mask=None, weights=None, linear=True):
    mask = correlations.valid.copy() if mask is None else np.asarray(mask, dtype=bool) & correlations.valid
    if mask.shape != correlations.valid.shape:
        raise ValueError("Pair mask has an invalid shape")
    if weights is None:
        weights = np.ones(len(mask))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != mask.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Pair weights must be finite and nonnegative")
    delays = delay_table(coordinates, angles, config.sample_rate, config.sound_speed)
    sampled = interpolate(correlations.values, delays, linear)
    return (sampled * (weights * mask)[:, None]).sum(axis=0)


def peak_quality(angles, response, settings=Search(), epsilon=1e-12):
    angles, response = np.asarray(angles), np.asarray(response)
    if len(response) == 0 or not np.isfinite(response).all() or np.ptp(response) <= epsilon:
        return 0, 0.0, 0.0, False
    peak = int(np.argmax(response))
    sides = response[np.abs(angles - angles[peak]) > settings.exclusion]
    if len(sides) < 4:
        return peak, 0.0, 0.0, False
    psr = float((response[peak] - sides.mean()) / (sides.std() + epsilon))
    confidence = float(np.clip((psr - settings.confidence_low)
                               / (settings.confidence_high - settings.confidence_low), 0, 1))
    return peak, psr, confidence, psr >= settings.psr_minimum


def diverse_pairs(coordinates, valid, count=4):
    """Prefer long baselines while avoiding parallel choices when alternatives exist."""
    vectors = baselines(coordinates)[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    available = list(np.flatnonzero(valid & (lengths > 0)))
    selected = []
    while available and len(selected) < count:
        scores = []
        for index in available:
            score = lengths[index]
            if selected:
                alignment = np.max(np.abs(vectors[selected] @ vectors[index])
                                   / (lengths[selected] * lengths[index]))
                score *= 1 - 0.65 * alignment
            scores.append(score)
        selected.append(available.pop(int(np.argmax(scores))))
    mask = np.zeros(len(valid), dtype=bool)
    mask[selected] = True
    return mask


def gcc_direction(correlations, coordinates, config=Config(), settings=Search()):
    mask = diverse_pairs(coordinates, correlations.valid, settings.coarse_pairs)
    vectors = baselines(coordinates)[mask, :2]
    if mask.sum() < settings.minimum_pairs or np.linalg.matrix_rank(vectors) < 2:
        return Estimate()
    bounds = np.ceil(np.linalg.norm(vectors, axis=1) * config.sample_rate / config.sound_speed).astype(int)
    delays, significance = [], []
    for values, bound in zip(correlations.values[mask], bounds):
        lag = np.arange(-bound, bound + 1)
        response = values[lag % config.frame]
        peak = int(np.argmax(response))
        sides = response[np.abs(lag - lag[peak]) > 2]
        quality = ((response[peak] - sides.mean()) / (sides.std() + config.epsilon)) if len(sides) >= 4 else 0
        delays.append(lag[peak])
        significance.append(quality)
    angles = config.angles()
    theory = delay_table(coordinates, angles, config.sample_rate, config.sound_speed)[mask]
    residual = np.sqrt(np.mean((theory - np.asarray(delays)[:, None]) ** 2, axis=0))
    peak = int(np.argmin(residual))
    valid = min(significance) >= settings.psr_minimum and residual[peak] <= settings.residual_limit
    confidence = float(np.clip(1 - residual[peak] / settings.residual_limit, 0, 1)) if valid else 0
    return Estimate(float(angles[peak]) if valid else 0, bool(valid), confidence,
                    float(min(significance)), "gcc", int(mask.sum()), len(angles) * int(mask.sum()))


def locate(correlations, coordinates, config=Config(), settings=Search(), mode="sector"):
    if mode not in ("full", "coarse", "sector"):
        raise ValueError("Search mode must be full, coarse or sector")
    valid = correlations.valid.copy()
    if valid.sum() < settings.minimum_pairs or np.linalg.matrix_rank(baselines(coordinates)[valid, :2]) < 2:
        return Estimate()
    evaluations = 0
    fallback = False

    def evaluate(angles, mask, label):
        nonlocal evaluations
        response = srp(correlations, coordinates, angles, config, mask, linear=settings.interpolation)
        evaluations += len(angles) * int(mask.sum())
        peak, psr, confidence, good = peak_quality(angles, response, settings, config.epsilon)
        return Estimate(float(angles[peak]) if good else 0, bool(good), confidence if good else 0,
                        psr, label, int(mask.sum()), evaluations, fallback), peak

    if mode == "full":
        return evaluate(config.angles(), valid, "full")[0]
    anchor = gcc_direction(correlations, coordinates, config, settings) if mode == "sector" else Estimate()
    if mode == "sector":
        evaluations += anchor.evaluations
    if anchor.valid:
        mask = valid.copy()
        if settings.selected_pairs is not None:
            score = reliability(coordinates, anchor.angle)
            mask[:] = False
            indices = np.flatnonzero(valid)
            mask[indices[np.argsort(score[indices])[-settings.selected_pairs:]]] = True
        for radius in (settings.sector, 2 * settings.sector):
            angles = config.angles()
            angles = angles[np.abs(angles - anchor.angle) <= radius]
            result, peak = evaluate(angles, mask, "sector")
            boundary = peak in (0, len(angles) - 1)
            physical_edge = len(angles) and angles[peak] in (config.lower, config.upper)
            if result.valid and (not boundary or physical_edge):
                return result
            fallback = True
    elif mode == "sector":
        fallback = True
    coarse, peak = evaluate(config.angles(coarse=True), valid, "coarse")
    if not coarse.valid:
        return Estimate(mode="coarse", pair_count=int(valid.sum()), evaluations=evaluations, fallback=fallback)
    angles = config.angles()
    angles = angles[np.abs(angles - coarse.angle) <= max(settings.sector, config.coarse_step)]
    result, peak = evaluate(angles, valid, "coarse_fine")
    if result.valid and peak not in (0, len(angles) - 1):
        return result
    fallback = True
    return evaluate(config.angles(), valid, "full_fallback")[0]
