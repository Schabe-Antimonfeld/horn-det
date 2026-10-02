"""Validated reference parameters from chapters 01 through 03."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    sample_rate: int = 48000
    microphones: int = 8
    diameter: float = 0.20
    sound_speed: float = 343.0
    frame: int = 1024
    hop: int = 512
    mel_bands: int = 40
    context: int = 24
    lower: float = -90.0
    upper: float = 90.0
    coarse_step: float = 5.0
    fine_step: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self):
        integers = (self.sample_rate, self.microphones, self.frame, self.hop,
                    self.mel_bands, self.context)
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("Sampling and frame parameters must be positive integers")
        if self.microphones < 2 or self.hop > self.frame:
            raise ValueError("Require two microphones and hop <= frame")
        if self.frame & (self.frame - 1):
            raise ValueError("FFT size must be a power of two")
        values = (self.diameter, self.sound_speed, self.lower, self.upper,
                  self.coarse_step, self.fine_step, self.epsilon)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Configuration must be finite")
        if min(self.diameter, self.sound_speed, self.coarse_step,
               self.fine_step, self.epsilon) <= 0:
            raise ValueError("Geometry, steps and epsilon must be positive")
        if not 0 < self.upper - self.lower <= 360:
            raise ValueError("Angular range must be within 360 degrees")
        for step in (self.coarse_step, self.fine_step):
            count = (self.upper - self.lower) / step
            if not math.isclose(count, round(count), abs_tol=1e-9):
                raise ValueError("Search endpoints must align with the angular step")

    def angles(self, coarse=False):
        step = self.coarse_step if coarse else self.fine_step
        return np.linspace(self.lower, self.upper,
                           round((self.upper - self.lower) / step) + 1)

    @property
    def context_seconds(self):
        return (self.frame + (self.context - 1) * self.hop) / self.sample_rate
