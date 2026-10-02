"""Frame-aligned hysteresis, event IDs and configured acoustic wire protocol."""

from dataclasses import dataclass
from enum import IntEnum
import struct

import numpy as np

from .geometry import round_away
from .localization import Estimate


class State(IntEnum):
    IDLE = 0
    CANDIDATE = 1
    CONFIRMED = 2
    END = 3


@dataclass(frozen=True)
class Fusion:
    beta: float = 0.8
    on: float = 0.7
    off: float = 0.3
    confirm: int = 3
    release: int = 5
    timeout: int = 100

    def __post_init__(self):
        if not 0 <= self.beta < 1 or not 0 <= self.off < self.on <= 1:
            raise ValueError("Invalid smoothing or hysteresis thresholds")
        if min(self.confirm, self.release, self.timeout) < 1:
            raise ValueError("Event durations must be positive frame counts")


@dataclass(frozen=True)
class Probability:
    frame: int
    value: float


@dataclass(frozen=True)
class Direction:
    frame: int
    estimate: Estimate


@dataclass(frozen=True)
class Decision:
    frame: int
    state: State
    probability: float
    direction: Estimate
    timestamp: int
    event_id: int

    @property
    def horn_valid(self):
        return self.state == State.CONFIRMED


class EventFusion:
    def __init__(self, settings=Fusion()):
        self.settings = settings
        self.state = State.IDLE
        self.event_id = 0
        self.counter = 0
        self.age = self.high = self.low = 0
        self.smoothed = 0.0
        self.last_frame = -1
        self.last_time = -1

    def feed(self, frame, timestamp, triggered, probability=None, direction=None):
        if frame <= self.last_frame or timestamp < self.last_time or not 0 <= timestamp < 2 ** 64:
            raise ValueError("Decision frames must be unique, ordered and timestamped")
        value = 0.0
        if direction is not None and direction.frame != frame:
            raise ValueError("Direction must refer to this exact decision frame")
        estimate = Estimate() if direction is None else direction.estimate
        if probability is not None:
            if probability.frame != frame or not np.isfinite(probability.value) or not 0 <= probability.value <= 1:
                raise ValueError("Classifier result must refer to this exact decision frame")
            value = probability.value
            self.smoothed = self.settings.beta * self.smoothed + (1 - self.settings.beta) * value
        self.last_frame, self.last_time = frame, timestamp
        if self.state == State.END:
            self.state, self.event_id = State.IDLE, 0
            self.age = self.high = self.low = 0
        elif self.state == State.IDLE:
            if triggered:
                self.counter = self.counter % (2 ** 32 - 1) + 1
                self.event_id = self.counter
                self.state = State.CANDIDATE
                self.age = self.high = self.low = 0
        else:
            self.age += 1
            self.high = self.high + 1 if probability is not None and self.smoothed >= self.settings.on else 0
            self.low = self.low + 1 if probability is not None and self.smoothed <= self.settings.off else 0
            if self.state == State.CANDIDATE:
                if self.high >= self.settings.confirm:
                    self.state = State.CONFIRMED
                elif self.age >= self.settings.timeout or self.low >= self.settings.release:
                    self.state = State.END
            elif self.state == State.CONFIRMED and self.low >= self.settings.release:
                self.state = State.END
        return Decision(frame, self.state, value, estimate, timestamp, self.event_id)


WIRE = struct.Struct("<BHhHQI")


def probability_code(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be in [0, 1]")
    return min(65535, max(0, int(round_away(value * 65536))))


def encode(decision):
    """Prototype convention: little endian, 19 bytes, flags in bits 0 through 3."""
    if not isinstance(decision.state, State) or not 0 <= decision.event_id < 2 ** 32 or not 0 <= decision.timestamp < 2 ** 64:
        raise ValueError("Decision fields exceed protocol bounds")
    if (decision.state == State.IDLE) != (decision.event_id == 0):
        raise ValueError("Idle and active event IDs are inconsistent")
    valid = decision.direction.valid
    angle = int(round_away(decision.direction.angle * 128)) if valid else 0
    if not -32768 <= angle <= 32767:
        raise ValueError("Angle exceeds Q9.7 range")
    flags = int(decision.state) | int(decision.horn_valid) << 2 | int(valid) << 3
    return WIRE.pack(flags, probability_code(decision.probability), angle,
                     probability_code(decision.direction.confidence) if valid else 0,
                     decision.timestamp, decision.event_id)


def decode(message, frame=-1):
    flags, probability, angle, confidence, timestamp, event_id = WIRE.unpack(message)
    state = State(flags & 3)
    if flags & 0xF0 or bool(flags & 4) != (state == State.CONFIRMED):
        raise ValueError("Reserved flags or horn_valid are inconsistent")
    if (state == State.IDLE) != (event_id == 0):
        raise ValueError("Invalid event ID")
    valid = bool(flags & 8)
    if not valid and (angle != 0 or confidence != 0):
        raise ValueError("Invalid DOA must encode zero angle and confidence")
    return Decision(frame, state, probability / 65536,
                    Estimate(angle / 128, valid, confidence / 65536, mode="wire"), timestamp, event_id)
