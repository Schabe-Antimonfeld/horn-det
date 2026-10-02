"""Finite-record far-field and shoebox image-source simulation, chapter 11."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt, firwin2

from .calibration import fractional_delay
from .config import Config
from .geometry import circular_array, directions, source_delays, positions


@dataclass(frozen=True)
class Scene:
    audio: np.ndarray
    coordinates: np.ndarray
    angle: float
    speed: float
    common_delay: float
    valid: np.ndarray
    metadata: dict


def noise_signal(kind, samples, sample_rate, generator):
    """Generate named synthetic proxies, not measured traffic recordings."""
    time = np.arange(samples) / sample_rate
    white = generator.normal(size=samples)
    if kind == "white":
        return white
    if kind == "wind":
        return sosfilt(butter(3, 180, fs=sample_rate, output="sos"), white)
    if kind == "engine":
        return sum(np.sin(2 * np.pi * frequency * time + generator.uniform(0, 2 * np.pi))
                   / (index + 1) for index, frequency in enumerate([70, 140, 210, 280]))
    if kind == "traffic":
        return sosfilt(butter(2, [100, 3000], btype="bandpass", fs=sample_rate, output="sos"), white)
    if kind == "siren":
        frequency = 1000 + 350 * np.sin(2 * np.pi * 1.5 * time)
        return np.sin(2 * np.pi * np.cumsum(frequency) / sample_rate)
    raise ValueError("Unknown noise kind")


def mix_noise(signal, noise, snr, region=None):
    signal, noise = np.asarray(signal), np.asarray(noise)
    if signal.shape != noise.shape or not np.isfinite(snr):
        raise ValueError("Noise shape or SNR is invalid")
    selected = slice(None) if region is None else region
    energy = np.mean(signal[selected] ** 2)
    background = np.mean(noise[selected] ** 2)
    if energy <= 0 or background <= 0:
        raise ValueError("SNR requires nonzero source and background energy")
    return signal + noise * np.sqrt(energy / background / 10 ** (snr / 10))


def far_field(source, angle, config=Config(), coordinates=None):
    source = np.asarray(source, dtype=float)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ValueError("Source must be a finite nonempty mono waveform")
    coordinates = circular_array(config) if coordinates is None else positions(coordinates)
    delays = source_delays(coordinates, angle, config.sample_rate, config.sound_speed)
    padding = int(np.ceil(np.max(np.abs(delays)))) + 65
    padded = np.pad(source, (padding, padding))
    audio = np.column_stack([fractional_delay(padded, delay) for delay in delays])
    return Scene(audio, coordinates, angle, config.sound_speed, float(padding),
                 np.ones(len(coordinates), dtype=bool), {"model": "far_field", "source_samples": len(source)})


def room_rir(coordinates, source, room, sample_rate=48000, speed=343.0,
             reflection=0.6, rt60=0.3, generator=None):
    """First-order image sources plus a stochastic decay tail; not a full room solver."""
    coordinates = positions(coordinates)
    source, room = np.asarray(source, dtype=float), np.asarray(room, dtype=float)
    if source.shape != (3,) or room.shape != (3,) or not np.isfinite(source).all():
        raise ValueError("Source and room must be 3D")
    if not np.isfinite(room).all() or np.any(room <= 0) or speed <= 0 or rt60 <= 0:
        raise ValueError("Room, RT60 and sound speed must be positive")
    if np.any(source <= 0) or np.any(source >= room) or np.any(coordinates <= 0) or np.any(coordinates >= room):
        raise ValueError("Source and microphones must be inside the room")
    if not 0 <= reflection < 1:
        raise ValueError("Reflection coefficient must be in [0, 1)")
    generator = np.random.default_rng(0) if generator is None else generator
    images = [(source, 1.0)]
    for axis in range(3):
        for plane in (0, room[axis]):
            image = source.copy()
            image[axis] = 2 * plane - source[axis]
            images.append((image, reflection))
    distances = [np.linalg.norm(coordinates - image, axis=1) for image, _ in images]
    length = int(np.ceil(max(distance.max() for distance in distances) / speed * sample_rate
                         + rt60 * sample_rate)) + 65
    impulses = np.zeros((length, len(coordinates)))
    unit = np.zeros(length)
    unit[0] = 1
    for (_, amplitude), distance in zip(images, distances):
        for channel in range(len(coordinates)):
            impulses[:, channel] += amplitude / max(distance[channel], 0.01) * fractional_delay(
                unit, distance[channel] / speed * sample_rate)
    for channel in range(len(coordinates)):
        start = int(np.ceil(distances[0][channel] / speed * sample_rate)) + 65
        time = np.arange(length - start) / sample_rate
        tail = generator.normal(size=len(time)) * np.exp(-np.log(1000) * time / rt60)
        impulses[start:, channel] += reflection * 0.015 / max(distances[0][channel], 0.01) * tail
    return impulses


def room_audio(source, rir):
    source, rir = np.asarray(source), np.asarray(rir)
    if source.ndim != 1 or rir.ndim != 2:
        raise ValueError("Require mono source and multichannel RIR")
    return np.column_stack([fftconvolve(source, rir[:, channel]) for channel in range(rir.shape[1])])


def randomized(source, angle, config=Config(), seed=0, kind="white", source_group="synthetic"):
    generator = np.random.default_rng(seed)
    deviation = generator.uniform(0.001, 0.003)
    coordinates = circular_array(config) + generator.uniform(-deviation, deviation, (config.microphones, 3))
    speed = generator.uniform(330, 355)
    from dataclasses import replace
    scene = far_field(source, angle, replace(config, sound_speed=speed), coordinates)
    gains = 10 ** (generator.uniform(-2, 2, config.microphones) / 20)
    delays = generator.uniform(-0.2, 0.2, config.microphones)
    audio = np.column_stack([fractional_delay(scene.audio[:, channel], delays[channel]) * gains[channel]
                             for channel in range(config.microphones)])
    eq = firwin2(65, np.linspace(0, config.sample_rate / 2, 8),
                 10 ** (generator.uniform(-3, 3, 8) / 20), fs=config.sample_rate)
    audio = np.column_stack([fftconvolve(audio[:, channel], eq, mode="same")
                             for channel in range(config.microphones)])
    snr = generator.uniform(-10, 30)
    independent = np.column_stack([noise_signal(kind, len(audio), config.sample_rate, generator)
                                   for _ in range(config.microphones)])
    common = noise_signal(kind, len(audio), config.sample_rate, generator)[:, None]
    audio = mix_noise(audio, 0.7 * independent + 0.3 * common, snr)
    valid = np.ones(config.microphones, dtype=bool)
    failed = generator.choice(config.microphones, size=int(generator.integers(0, 3)), replace=False)
    valid[failed] = False
    audio[:, failed] = 0
    metadata = {"model": "randomized_far_field", "seed": seed, "source_group": source_group,
                "position_bound_m": deviation, "gains": gains.tolist(), "delays": delays.tolist(),
                "snr_db": snr, "noise_proxy": kind, "eq_db_bound": 3}
    return Scene(audio, coordinates, angle, speed, scene.common_delay, valid, metadata)
