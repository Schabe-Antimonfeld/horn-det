"""Array geometry, ordered pairs and signed delays from chapter 02."""

from itertools import combinations

import numpy as np

from .config import Config


def sound_speed(temperature):
    if not np.isfinite(temperature):
        raise ValueError("Temperature must be finite")
    value = 331.3 + 0.606 * temperature
    if value <= 0:
        raise ValueError("Sound speed must be positive")
    return value


def circular_array(config: Config):
    phase = np.arange(config.microphones) * 2 * np.pi / config.microphones
    return config.diameter / 2 * np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros(config.microphones)))


def positions(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 2:
        raise ValueError("Positions must have shape (microphones, 3)")
    if not np.isfinite(value).all():
        raise ValueError("Positions must be finite")
    return value


def pairs(microphones):
    if microphones < 2:
        raise ValueError("Require at least two microphones")
    return np.asarray(list(combinations(range(microphones), 2)), dtype=int)


def baselines(coordinates):
    coordinates = positions(coordinates)
    selected = pairs(len(coordinates))
    return coordinates[selected[:, 1]] - coordinates[selected[:, 0]]


def directions(angles):
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("Angles must be a finite vector")
    radians = np.deg2rad(angles)
    return np.column_stack((np.cos(radians), np.sin(radians), np.zeros(len(angles))))


def delay_table(coordinates, angles, sample_rate=48000, speed=343.0):
    if sample_rate <= 0 or speed <= 0 or not np.isfinite(speed):
        raise ValueError("Sampling rate and sound speed must be positive")
    return baselines(coordinates) @ directions(angles).T * sample_rate / speed


def source_delays(coordinates, angle, sample_rate=48000, speed=343.0):
    return -(positions(coordinates) @ directions([angle])[0]) * sample_rate / speed


def exact_delays(coordinates, source, speed=343.0):
    source = np.asarray(source, dtype=float)
    if source.shape != (3,) or not np.isfinite(source).all() or speed <= 0:
        raise ValueError("Require a finite 3D source and positive sound speed")
    return np.linalg.norm(source - positions(coordinates), axis=1) / speed


def reliability(coordinates, angle):
    baseline = baselines(coordinates)
    perpendicular = directions([angle + 90])[0]
    return np.abs(baseline @ perpendicular)


def round_away(values):
    values = np.asarray(values, dtype=float)
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)
