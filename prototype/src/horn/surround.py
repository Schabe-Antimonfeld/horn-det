"""Parameterized fixed-camera coverage and simulated audiovisual association.

Camera coordinates use image right, image down and optical forward. World z
points upward; depression is positive below the horizontal plane. Detection
inputs must come from an external detector or an explicitly labeled simulator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .calibration import fractional_delay
from .config import Config
from .geometry import pairs, positions
from .localization import interpolate


def wrap_angle(value):
    return (np.asarray(value) + 180) % 360 - 180


@dataclass(frozen=True)
class CameraModel:
    identity: str
    center: tuple[float, float, float]
    yaw: float
    pitch: float
    hfov: float
    vfov: float
    width: int = 1280
    height: int = 720

    def __post_init__(self):
        if len(self.center) != 3 or not np.isfinite(self.center).all() or self.center[2] <= 0:
            raise ValueError("Camera center must be finite and above ground")
        if not self.identity or not np.isfinite([self.yaw, self.pitch, self.hfov, self.vfov]).all():
            raise ValueError("Camera identity and angles are required")
        if not 0 <= self.pitch <= 90 or not 0 < self.hfov < 179 or not 0 < self.vfov < 179:
            raise ValueError("Invalid camera pitch or field of view")
        if any(type(value) is not int or value <= 0 for value in (self.width, self.height)):
            raise ValueError("Image dimensions must be positive integers")

    @property
    def basis(self):
        yaw, pitch = np.deg2rad([self.yaw, self.pitch])
        return np.array([[np.sin(yaw), -np.cos(yaw), 0],
                         [-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), -np.cos(pitch)],
                         [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)]])

    @property
    def focal(self):
        return np.array([self.width, self.height]) / (2 * np.tan(np.deg2rad([self.hfov, self.vfov]) / 2))

    def project(self, points):
        """Project world points; invalid depths produce NaNs and a false visibility mask."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("World points must be finite with shape (count, 3)")
        local = (points - self.center) @ self.basis.T
        forward = local[:, 2] > 1e-9
        pixels = np.full((len(points), 2), np.nan)
        pixels[forward] = local[forward, :2] / local[forward, 2, None] * self.focal
        pixels += np.array([self.width, self.height]) / 2
        visible = forward & np.all((pixels >= 0) & (pixels <= [self.width, self.height]), axis=1)
        return pixels, visible

    def ground(self, pixels):
        """Intersect image rays with z=0; out-of-image and upward rays are invalid."""
        pixels = np.asarray(pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Image points must be finite with shape (count, 2)")
        normalized = (pixels - np.array([self.width, self.height]) / 2) / self.focal
        rays = np.column_stack((normalized, np.ones(len(pixels)))) @ self.basis
        valid = (rays[:, 2] < -1e-9) & np.all((pixels >= 0) & (pixels <= [self.width, self.height]), axis=1)
        points = np.full((len(pixels), 3), np.nan)
        points[valid] = np.asarray(self.center) - self.center[2] / rays[valid, 2, None] * rays[valid]
        return points, valid


@dataclass(frozen=True)
class Rig:
    count: int = 6
    height: float = 6.0
    pitch: float = 40.0
    hfov: float = 90.0
    vfov: float = 70.0
    down_hfov: float = 120.0
    down_vfov: float = 100.0
    yaw: float = 0.0
    width: int = 1280
    image_height: int = 720
    minimum_range: float = 1.0
    maximum_range: float = 40.0
    horn_height: float = 0.8

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1:
            raise ValueError("The ring needs a positive integer camera count")
        values = [self.height, self.minimum_range, self.maximum_range, self.horn_height]
        if not np.isfinite(values).all() or not 0 <= self.minimum_range < self.maximum_range:
            raise ValueError("Invalid monitoring range")
        if not 0 <= self.horn_height < self.height:
            raise ValueError("Horn height must be below the array reference height")
        self.cameras()

    def cameras(self):
        center = (0.0, 0.0, self.height)
        result = [CameraModel(f"ring-{index}", center, self.yaw + 360 * index / self.count,
                              self.pitch, self.hfov, self.vfov, self.width, self.image_height)
                  for index in range(self.count)]
        result.append(CameraModel("down", center, self.yaw, 90, self.down_hfov,
                                  self.down_vfov, self.width, self.image_height))
        return result


def coverage_grid(rig, step=1.0, cameras=None):
    """Sample equal-area ground cells; this is geometric coverage, not detector recall."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Coverage step must be positive")
    axis = np.arange(-rig.maximum_range + step / 2, rig.maximum_range, step)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x.ravel(), y.ravel())
    mask = (radius >= rig.minimum_range) & (radius <= rig.maximum_range)
    points = np.column_stack((x.ravel()[mask], y.ravel()[mask], np.zeros(mask.sum())))
    if not len(points):
        raise ValueError("Coverage sampling produced no monitoring points")
    cameras = rig.cameras() if cameras is None else list(cameras)
    if not cameras or len({camera.identity for camera in cameras}) != len(cameras):
        raise ValueError("Coverage needs distinct camera identities")
    visible = np.column_stack([camera.project(points)[1] for camera in cameras])
    return points, visible


def route_cameras(rig, azimuth, tolerance=6.0, cameras=None):
    """Route a sampled direction wedge over the configured range and vehicle heights."""
    cameras = rig.cameras() if cameras is None else list(cameras)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 180:
        raise ValueError("Direction tolerance must be within [0, 180]")
    if azimuth is None:
        return [camera.identity for camera in cameras]
    if not np.isfinite(azimuth):
        raise ValueError("Azimuth must be finite or None for an undefined overhead bearing")
    angles = np.deg2rad(azimuth + np.linspace(-tolerance, tolerance, max(3, int(tolerance * 2) + 1)))
    radii, angles, heights = np.meshgrid(np.linspace(rig.minimum_range, rig.maximum_range, 128),
                                       angles, [0, rig.horn_height, 1.5], indexing="ij")
    points = np.column_stack(((radii * np.cos(angles)).ravel(),
                              (radii * np.sin(angles)).ravel(), heights.ravel()))
    return [camera.identity for camera in cameras if camera.project(points)[1].any()]


def vehicle_box(camera, center, yaw=0.0, minimum_pixels=12.0):
    """Project an ideal 4.5 x 1.8 x 1.5 m cuboid; no image detector is executed."""
    if not np.isfinite([*center, yaw, minimum_pixels]).all() or len(center) != 2 or minimum_pixels < 0:
        raise ValueError("Invalid simulated vehicle")
    corners = np.array([[x, y, z] for x in (-2.25, 2.25) for y in (-0.9, 0.9) for z in (0, 1.5)])
    angle = np.deg2rad(yaw)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners[:, :2] = corners[:, :2] @ rotation.T + center
    pixels, _ = camera.project(corners)
    foot, visible = camera.project([[*center, 0]])
    if not np.isfinite(pixels).all() or not visible[0]:
        return None
    lower = np.maximum(pixels.min(axis=0), 0)
    upper = np.minimum(pixels.max(axis=0), [camera.width, camera.height])
    if np.min(upper - lower) < minimum_pixels:
        return None
    return tuple(np.r_[lower, upper]), tuple(foot[0])


@dataclass(frozen=True)
class Detection:
    identity: str
    camera: str
    box: tuple[float, float, float, float]
    foot: tuple[float, float]
    exposure: float
    velocity: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if len(self.box) != 4 or len(self.foot) != 2 or len(self.velocity) != 2:
            raise ValueError("Invalid detection dimensions")
        if not self.identity or not self.camera or not np.isfinite([*self.box, *self.foot, self.exposure, *self.velocity]).all():
            raise ValueError("Detection data must be finite and identified")
        if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            raise ValueError("Detection box must have positive dimensions")


@dataclass(frozen=True)
class Association:
    method: str = "spatial"
    angle_gate: float = 6.0
    depression_gate: float = 3.0
    height_error: float = 0.5
    position_error: float = 0.75
    response_fraction: float = 0.85
    maximum_width: float = 12.0
    maximum_skew: float = 0.5
    merge_distance: float = 0.75
    speed: float = 343.0

    def __post_init__(self):
        if self.method not in ("azimuth", "depression", "spatial"):
            raise ValueError("Association method must be azimuth, depression or spatial")
        if not np.isfinite([self.angle_gate, self.depression_gate, self.height_error, self.position_error,
                            self.response_fraction, self.maximum_width,
                            self.maximum_skew, self.merge_distance, self.speed]).all():
            raise ValueError("Association settings must be finite")
        if not 0 < self.angle_gate <= 180 or min(self.maximum_skew, self.merge_distance, self.speed) <= 0:
            raise ValueError("Invalid association settings")
        if not 0 <= self.depression_gate <= 90 or min(self.height_error, self.position_error) < 0:
            raise ValueError("Invalid geometric uncertainty bounds")
        if not 0 < self.response_fraction <= 1 or not 0 < self.maximum_width <= 90:
            raise ValueError("Invalid spatial support settings")


def associate(rig, azimuth, arrival_time, detections, settings=Association(), cameras=None, evidence=None):
    """Match at an approximate emission time; preserve all angle-compatible vehicles.

Arrival time is a representative audio-window time, never decision wall time.
Velocity is an external visual estimate; the default assumes a stationary car.
Spatial clustering uses no vehicle truth identifiers and is only a prototype.
The default requires measured spatial evidence; azimuth-only behavior is an
explicit comparison mode and must not substitute for verified association.
"""
    if not np.isfinite(arrival_time) or (azimuth is not None and not np.isfinite(azimuth)):
        raise ValueError("Invalid acoustic reference")
    if evidence is not None:
        if (evidence.bearing.azimuth is None) != (azimuth is None) or (
                azimuth is not None and abs(wrap_angle(evidence.bearing.azimuth - azimuth)) > 1e-6):
            raise ValueError("Acoustic evidence and association azimuth disagree")
    cameras = rig.cameras() if cameras is None else cameras
    lookup = {camera.identity: camera for camera in cameras}
    accepted, rejected = [], []
    for detection in detections:
        if detection.camera not in lookup:
            raise ValueError("Detection references an unknown camera")
        point, valid = lookup[detection.camera].ground([detection.foot])
        if not valid[0]:
            rejected.append(dict(identity=detection.identity, reason="invalid_ground_ray"))
            continue
        original = point[0, :2]
        position = original.copy()
        for _ in range(4):
            distance = np.sqrt(np.dot(position, position) + (rig.height - rig.horn_height) ** 2)
            emission = arrival_time - distance / settings.speed
            position = original + np.asarray(detection.velocity) * (emission - detection.exposure)
        distance = np.sqrt(np.dot(position, position) + (rig.height - rig.horn_height) ** 2)
        emission = arrival_time - distance / settings.speed
        skew = detection.exposure - emission
        angle = float(np.rad2deg(np.arctan2(position[1], position[0])))
        radius = float(np.linalg.norm(position))
        error = None if azimuth is None else float(abs(wrap_angle(angle - azimuth)))
        reason = ("outside_monitoring_range" if not rig.minimum_range <= radius <= rig.maximum_range
                  else "stale_frame" if abs(skew) > settings.maximum_skew
                  else "angle_mismatch" if error is not None and error > settings.angle_gate else None)
        if reason:
            rejected.append(dict(identity=detection.identity, reason=reason))
            continue
        support = dict(reason=None, ambiguous=False)
        if settings.method != "azimuth":
            support = acoustic_support(rig, position, evidence, settings)
            if support["reason"]:
                rejected.append(dict(identity=detection.identity, **support))
                continue
        accepted.append(dict(identity=detection.identity, camera=detection.camera, position=position.tolist(),
                             angle=angle, error=error, emission=emission, skew=skew, box=list(detection.box),
                             acoustic=support))
    groups = []
    for candidate in sorted(accepted, key=lambda value: (value["position"], value["camera"], value["identity"])):
        for group in groups:
            if candidate["camera"] not in {item["camera"] for item in group} and all(
                    np.linalg.norm(np.asarray(candidate["position"]) - item["position"]) <= settings.merge_distance
                    for item in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    state = "unmatched" if not groups else "unique" if len(groups) == 1 else "ambiguous"
    if groups and (azimuth is None or any(item["acoustic"]["ambiguous"] for group in groups for item in group)):
        state = "ambiguous"
    return dict(state=state, groups=groups, rejected=rejected,
                method=settings.method, source_unexplained=not bool(groups),
                score_definition="acoustic compatibility, not proof of horn identity or calibrated probability")


def direction_vectors(azimuths, depressions):
    azimuths, depressions = np.broadcast_arrays(np.asarray(azimuths, dtype=float), np.asarray(depressions, dtype=float))
    if not np.isfinite(azimuths).all() or not np.isfinite(depressions).all() or np.any((depressions < 0) | (depressions > 90)):
        raise ValueError("Directions require finite azimuths and depressions within [0, 90]")
    azimuths, depressions = np.deg2rad(azimuths), np.deg2rad(depressions)
    return np.stack((np.cos(depressions) * np.cos(azimuths),
                     np.cos(depressions) * np.sin(azimuths), -np.sin(depressions)), axis=-1)


@dataclass(frozen=True)
class Bearing:
    azimuth: float | None
    depression: float
    response: float

    def __post_init__(self):
        if self.azimuth is not None and not np.isfinite(self.azimuth):
            raise ValueError("Azimuth must be finite or undefined")
        if not np.isfinite([self.depression, self.response]).all() or not 0 <= self.depression <= 90:
            raise ValueError("Invalid bearing depression or response")


@dataclass(frozen=True)
class AcousticEvidence:
    """Measured horn-selected bearing and optional SRP map from the same audio window."""

    bearing: Bearing
    grid: tuple | None = None

    def __post_init__(self):
        if self.grid is None:
            return
        if len(self.grid) != 3:
            raise ValueError("Spatial evidence needs azimuths, depressions and response")
        azimuths, depressions, response = [np.asarray(value, dtype=float) for value in self.grid]
        if azimuths.ndim != 1 or depressions.ndim != 1 or len(azimuths) < 3 or len(depressions) < 2:
            raise ValueError("Spatial evidence needs a nonempty angular grid")
        if response.shape != (len(azimuths), len(depressions)) or not all(
                np.isfinite(value).all() for value in (azimuths, depressions, response)):
            raise ValueError("Invalid spatial evidence dimensions or values")
        if np.any(np.diff(azimuths) <= 0) or np.any(np.diff(depressions) <= 0):
            raise ValueError("Spatial grid angles must be strictly increasing")
        if azimuths[-1] - azimuths[0] >= 360 or depressions[0] < 0 or depressions[-1] > 90:
            raise ValueError("Spatial evidence has duplicate azimuth endpoints or invalid depressions")


def acoustic_support(rig, position, evidence, settings):
    """Test height/position intervals against a horn-selected peak, never a visual-only winner."""
    if evidence is None:
        return dict(reason="missing_acoustic_evidence", ambiguous=False)
    radius = float(np.linalg.norm(position))
    lower_height = max(0, rig.horn_height - settings.height_error)
    upper_height = min(rig.height, rig.horn_height + settings.height_error)
    lower = float(np.rad2deg(np.arctan2(rig.height - upper_height, radius + settings.position_error)))
    upper = float(np.rad2deg(np.arctan2(rig.height - lower_height, max(0, radius - settings.position_error))))
    depression = evidence.bearing.depression
    bounds = dict(depression_bounds=[lower, upper], measured_depression=depression)
    if depression < lower - settings.depression_gate or depression > upper + settings.depression_gate:
        return dict(reason="depression_mismatch", ambiguous=False, **bounds)
    if settings.method == "depression":
        return dict(reason=None, ambiguous=False, **bounds)
    if evidence.grid is None:
        return dict(reason="missing_spatial_response", ambiguous=False, **bounds)
    azimuths, depressions, response = [np.asarray(value) for value in evidence.grid]
    if evidence.bearing.azimuth is None:
        return dict(reason=None, ambiguous=True, **bounds)
    row = int(np.argmin(abs(wrap_angle(azimuths - evidence.bearing.azimuth))))
    column = int(np.argmin(abs(depressions - depression)))
    reference = float(response[row, column])
    floor = float(np.median(response))
    contrast = reference - floor
    if contrast <= max(1e-9, np.ptp(response) * 1e-6):
        return dict(reason="uninformative_spatial_response", ambiguous=False, **bounds)
    threshold = floor + settings.response_fraction * contrast
    profile = response[row]
    first = final = column
    while first > 0 and profile[first - 1] >= threshold:
        first -= 1
    while final + 1 < len(profile) and profile[final + 1] >= threshold:
        final += 1
    step = float(np.max(np.diff(depressions)))
    width = float(depressions[final] - depressions[first] + step)
    angle = float(np.rad2deg(np.arctan2(position[1], position[0])))
    angular_error = float(np.rad2deg(np.arcsin(min(1, settings.position_error / max(radius, 1e-9)))))
    azimuth_step = float(np.max(np.diff(azimuths)))
    azimuth_mask = abs(wrap_angle(azimuths - angle)) <= angular_error + azimuth_step / 2 + 1e-9
    depression_mask = (depressions >= lower - step / 2) & (depressions <= upper + step / 2)
    if not azimuth_mask.any() or not depression_mask.any():
        return dict(reason="spatial_grid_miss", ambiguous=False, **bounds)
    best = float(response[np.ix_(azimuth_mask, depression_mask)].max())
    ratio = (best - floor) / contrast
    if best < threshold:
        return dict(reason="spatial_response_mismatch", ambiguous=False, response_ratio=ratio,
                    profile_width=width, **bounds)
    return dict(reason=None, ambiguous=width > settings.maximum_width,
                response_ratio=ratio, profile_width=width, **bounds)


def spatial_response(audio, coordinates, config=Config(microphones=14, diameter=0.24),
                     azimuth_step=2.0, depression_step=3.0):
    """Average framewise PHAT, then search periodic azimuth and downward depression."""
    audio, coordinates = np.asarray(audio, dtype=float), positions(coordinates)
    if audio.ndim != 2 or audio.shape[1] != len(coordinates) or len(coordinates) != config.microphones:
        raise ValueError("Audio and array channel dimensions must match configuration")
    if not np.isfinite(audio).all() or len(audio) < config.frame:
        raise ValueError("A complete finite audio frame is required")
    for span, step in ((360, azimuth_step), (90, depression_step)):
        if not np.isfinite(step) or step <= 0 or not np.isclose(span / step, round(span / step)):
            raise ValueError("Search steps must divide their angular spans")
    azimuths = np.linspace(-180, 180, round(360 / azimuth_step), endpoint=False)
    depressions = np.linspace(0, 90, round(90 / depression_step) + 1)
    selected = pairs(config.microphones)
    frames = np.lib.stride_tricks.sliding_window_view(audio, config.frame, axis=0)[::config.hop]
    total = np.zeros((len(selected), config.frame // 2 + 1), dtype=complex)
    for start in range(0, len(frames), 12):
        spectrum = np.fft.rfft(frames[start:start + 12] * np.hanning(config.frame), axis=-1)
        cross = spectrum[:, selected[:, 0]] * spectrum[:, selected[:, 1]].conj()
        total += (cross / (np.abs(cross) + config.epsilon)).sum(axis=0)
    correlations = np.fft.irfft(total / len(frames), n=config.frame, axis=-1)
    azimuth, depression = np.meshgrid(azimuths, depressions, indexing="ij")
    vectors = direction_vectors(azimuth.ravel(), depression.ravel())
    baselines = coordinates[selected[:, 1]] - coordinates[selected[:, 0]]
    delays = baselines @ vectors.T * config.sample_rate / config.sound_speed
    response = interpolate(correlations, delays).sum(axis=0).reshape(azimuth.shape)
    return azimuths, depressions, response


def bearing_candidates(azimuths, depressions, response, count=2, separation=12.0):
    """Find periodic spatial peaks, with no arbitrary azimuth for near-overhead sound."""
    azimuths, depressions, response = np.asarray(azimuths), np.asarray(depressions), np.asarray(response)
    if response.shape != (len(azimuths), len(depressions)) or len(azimuths) < 3 or not np.isfinite(response).all():
        raise ValueError("Spatial grid dimensions are invalid")
    if type(count) is not int or count <= 0 or not np.isfinite(separation) or not 0 < separation <= 180:
        raise ValueError("Invalid candidate count or separation")
    if np.ptp(response) <= 1e-12:
        return []
    curve = response.max(axis=1)
    if np.ptp(curve) <= 1e-12:
        index = np.unravel_index(np.argmax(response), response.shape)
        return [Bearing(None, float(depressions[index[1]]), float(response[index]))]
    peaks = find_peaks(np.tile(curve, 3))[0]
    peaks = peaks[(peaks >= len(curve)) & (peaks < 2 * len(curve))] % len(curve)
    chosen, vectors = [], []
    for index in sorted(peaks, key=lambda value: (-curve[value], value)):
        depression = float(depressions[np.argmax(response[index])])
        azimuth = None if depression >= 85 else float(azimuths[index])
        vector = direction_vectors(float(azimuths[index]), depression)
        if all(np.rad2deg(np.arccos(np.clip(np.dot(vector, previous), -1, 1))) >= separation
               for previous in vectors):
            chosen.append(Bearing(azimuth, depression, float(curve[index])))
            vectors.append(vector)
        if len(chosen) == count:
            break
    return chosen


def propagate_audio(signal, source, coordinates, config):
    """Synthesize exact relative spherical delays; absolute flight time is metadata."""
    signal, source, coordinates = np.asarray(signal, dtype=float), np.asarray(source, dtype=float), positions(coordinates)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("Source waveform must be finite nonempty mono audio")
    if source.shape != (3,) or not np.isfinite(source).all() or len(coordinates) != config.microphones:
        raise ValueError("Invalid source position or channel count")
    distance = np.linalg.norm(source)
    distances = np.linalg.norm(source - coordinates, axis=1)
    if distance <= 0 or distances.min() <= 0:
        raise ValueError("Source must be separated from the array")
    delays = (distances - distance) * config.sample_rate / config.sound_speed
    common = int(np.ceil(np.max(np.abs(delays)))) + 32
    audio = np.column_stack([fractional_delay(signal, common + delay) * distance / channel_distance
                             for delay, channel_distance in zip(delays, distances)])
    return audio, dict(buffer_samples=common, flight_seconds=distance / config.sound_speed,
                       relative_delays=delays.tolist(), model="direct spherical propagation; no reflections")


def form_beam(audio, coordinates, bearing, config):
    """Causal finite-window delay-and-sum; reset boundaries are not a live stream."""
    audio, coordinates = np.asarray(audio, dtype=float), positions(coordinates)
    if audio.ndim != 2 or audio.shape[1] != len(coordinates) or len(coordinates) != config.microphones or not np.isfinite(audio).all():
        raise ValueError("Invalid beam audio or coordinates")
    vector = direction_vectors(0 if bearing.azimuth is None else bearing.azimuth,
                               90 if bearing.azimuth is None else bearing.depression)
    delays = -coordinates @ vector * config.sample_rate / config.sound_speed
    common = int(np.ceil(delays.max())) + 32
    waveform = sum(fractional_delay(audio[:, channel], common - delay)
                   for channel, delay in enumerate(delays)) / len(coordinates)
    return waveform, common


class FrozenPanns:
    """Reuse a trained Cnn6 horn checkpoint; no download or training occurs here."""

    def __init__(self, path):
        import torch
        from .pretrained import Cnn6
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "horn-panns-clip-v1":
            raise ValueError("Unsupported horn checkpoint")
        self.encoder = Cnn6().eval()
        self.encoder.load_state_dict(payload["encoder"])
        self.encoder.pretrained = True
        self.head = torch.nn.Linear(512, 2).eval()
        self.head.load_state_dict(payload["head"])

    def score(self, waveforms, sample_rate=48000):
        """Score independent completed one-second beams in inference mode."""
        import torch
        if not waveforms:
            return []
        if any(len(waveform) != sample_rate for waveform in waveforms):
            raise ValueError("Each classifier input must be exactly one completed second")
        with torch.inference_mode():
            features = torch.cat([self.encoder.frontend(waveform, sample_rate) for waveform in waveforms])
            return torch.softmax(self.head(self.encoder.embedding(features)), dim=1)[:, 1].tolist()


def select_bearing(candidates, scores, threshold=0.2, margin=0.0):
    """Apply demonstration score gates; scores are not calibrated probabilities."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must match candidates and lie within [0, 1]")
    if not np.isfinite([threshold, margin]).all() or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Score gates must lie within [0, 1]")
    if not len(scores):
        return dict(valid=False, index=None, score=None)
    order = np.argsort(-scores, kind="stable")
    index = int(order[0])
    difference = float(scores[index] - scores[order[1]]) if len(order) > 1 else float(scores[index])
    return dict(valid=bool(scores[index] >= threshold and difference >= margin),
                index=index, score=float(scores[index]), difference=difference)
