"""Camera policy and clock-based mechanical simulation; no actual camera I/O."""

from dataclasses import dataclass

import numpy as np

from .events import State


@dataclass(frozen=True)
class Camera:
    offset: float = 0.0
    sign: int = 1
    lower: float = -90.0
    upper: float = 90.0
    turn_threshold: float = 0.1
    capture_threshold: float = 0.4

    def __post_init__(self):
        if self.sign not in (-1, 1) or not -180 <= self.lower < self.upper <= 180:
            raise ValueError("Invalid camera coordinates or mechanical range")
        if not np.isfinite([self.offset, self.turn_threshold, self.capture_threshold]).all():
            raise ValueError("Camera settings must be finite")
        if not 0 <= self.turn_threshold <= 1 or not 0 <= self.capture_threshold <= 1:
            raise ValueError("Invalid camera confidence thresholds")


@dataclass(frozen=True)
class Action:
    kind: str
    event_id: int
    timestamp: int
    pan: float | None = None


class CameraPolicy:
    def __init__(self, settings=Camera()):
        self.settings = settings
        self.captured = set()
        self.saved = set()
        self.last_time = -1

    def feed(self, decision):
        if decision.timestamp < self.last_time:
            raise ValueError("Camera received an out-of-order acoustic message")
        self.last_time = decision.timestamp
        if decision.state == State.END:
            if decision.event_id in self.saved:
                return []
            self.saved.add(decision.event_id)
            return [Action("save", decision.event_id, decision.timestamp)]
        if decision.state not in (State.CANDIDATE, State.CONFIRMED) or not decision.direction.valid:
            return []
        pan = (self.settings.sign * decision.direction.angle + self.settings.offset + 180) % 360 - 180
        if not self.settings.lower <= pan <= self.settings.upper:
            return [Action("unreachable", decision.event_id, decision.timestamp, pan)]
        actions = []
        if decision.direction.confidence >= self.settings.turn_threshold:
            actions.append(Action("turn", decision.event_id, decision.timestamp, pan))
        if (decision.horn_valid and decision.direction.confidence >= self.settings.capture_threshold
                and decision.event_id not in self.captured):
            self.captured.add(decision.event_id)
            actions.append(Action("capture", decision.event_id, decision.timestamp, pan))
        return actions


class CameraSimulation:
    def __init__(self, speed=90.0, exposure_delay=0.02):
        if speed <= 0 or exposure_delay < 0:
            raise ValueError("Invalid camera simulation timing")
        self.speed, self.exposure_delay = speed, exposure_delay
        self.pan = 0.0
        self.target = 0.0
        self.time = 0.0
        self.pending = []
        self.exposures = []
        self.arrival = 0.0

    def advance(self, time):
        if not np.isfinite(time) or time < self.time:
            raise ValueError("Camera time must be finite and monotonic")
        travel = self.speed * (time - self.time)
        self.pan += np.clip(self.target - self.pan, -travel, travel)
        completed = []
        for event_id, ready in self.pending:
            if time >= ready and self.pan == self.target:
                self.exposures.append({"event_id": event_id, "exposure_s": ready, "pan": self.pan})
                completed.append(event_id)
        self.pending = [(event_id, ready) for event_id, ready in self.pending if event_id not in completed]
        self.time = time

    def feed(self, action, received_s=None):
        received_s = action.timestamp / 1000000 if received_s is None else received_s
        self.advance(received_s)
        if action.kind in ("turn", "capture"):
            if action.pan != self.target:
                self.target = action.pan
                self.arrival = received_s + abs(self.target - self.pan) / self.speed
                self.pending = [(event_id, self.arrival + self.exposure_delay) for event_id, _ in self.pending]
        if action.kind == "capture" and action.event_id not in [event_id for event_id, _ in self.pending]:
            self.pending.append((action.event_id, max(self.arrival, received_s) + self.exposure_delay))
