"""Shared causal STFT and sound-file input from chapter 03."""

from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .calibration import audio_array
from .config import Config


def read_audio(path, sample_rate=48000, channel=None):
    audio, original = sf.read(path, always_2d=True, dtype="float64")
    if channel is not None:
        if not 0 <= channel < audio.shape[1]:
            raise ValueError("Channel is out of range")
        audio = audio[:, channel:channel + 1]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Sampling rate must be a positive integer")
    if original != sample_rate:
        divisor = gcd(original, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original // divisor, axis=0)
    return audio_array(audio)


@dataclass(frozen=True)
class Frame:
    identity: int
    start: int
    end: int
    spectrum: np.ndarray

    def timestamp(self, sample_rate, origin=0):
        return origin + self.end * 1000000 // sample_rate


class STFT:
    """Emit only complete acquired frames; retain overlap across arbitrary input blocks."""

    def __init__(self, config=Config()):
        self.config = config
        self.window = np.hanning(config.frame)
        self.buffer = None
        self.start = 0
        self.identity = 0

    def feed(self, audio):
        audio = audio_array(audio)
        if self.buffer is None:
            self.buffer = np.empty((0, audio.shape[1]))
        if self.buffer.shape[1] != audio.shape[1]:
            raise ValueError("Channel count changed during stream")
        self.buffer = np.concatenate((self.buffer, audio))
        frames = []
        while len(self.buffer) >= self.config.frame:
            spectrum = np.fft.rfft(self.buffer[:self.config.frame] * self.window[:, None], axis=0)
            frames.append(Frame(self.identity, self.start,
                                self.start + self.config.frame - 1, spectrum))
            self.buffer = self.buffer[self.config.hop:]
            self.start += self.config.hop
            self.identity += 1
        return frames


def stft(audio, config=Config()):
    return STFT(config).feed(audio)


def power(spectrum, valid=None):
    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 2 or not np.isfinite(spectrum).all():
        raise ValueError("Spectrum must have shape (bins, channels) and be finite")
    if valid is None:
        valid = np.ones(spectrum.shape[1], dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (spectrum.shape[1],) or not valid.any():
        raise ValueError("Require at least one valid channel")
    return np.mean(np.abs(spectrum[:, valid]) ** 2, axis=1)
