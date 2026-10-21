"""Generated geometry; regenerate with scripts/export_geometry.py."""

import numpy as np

SOURCE_SHA256 = '69f58438b78082d55dc4a017495a0bccd7012b68cb1263bd4bba645eeccd0644'

def wrap_angle(value):
    return (np.asarray(value) + 180) % 360 - 180

class Lens:
    """Calibrated OpenCV Brown or fisheye projection on the original image grid."""

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = {'brown': 5, 'fisheye': 4}
        if self.model not in expected or coefficients.shape != (expected[self.model],):
            raise ValueError('Lens requires brown(5) or fisheye(4) coefficients')
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or (not np.isfinite(coefficients).all()):
            raise ValueError('Lens parameters must be finite')
        if not np.array_equal(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0 or (min(matrix[0, 0], matrix[1, 1]) <= 0):
            raise ValueError('Lens requires positive focal lengths and zero skew')
        if any((type(value) is not int or value <= 0 for value in (self.width, self.height))):
            raise ValueError('Calibration image dimensions must be positive integers')
        if not 0 <= matrix[0, 2] < self.width or not 0 <= matrix[1, 2] < self.height:
            raise ValueError('Principal point must be on the calibration image')
        if not np.isfinite(self.maximum_angle) or not 0 < self.maximum_angle < 89:
            raise ValueError('Calibrated ray angle must be in (0,89) degrees')
        if self.model == 'fisheye':
            theta = np.linspace(0, np.deg2rad(self.maximum_angle), 1025)
            derivative = np.ones_like(theta)
            for index, value in enumerate(coefficients):
                derivative += (2 * index + 3) * value * theta ** (2 * index + 2)
            if np.any(derivative <= 0):
                raise ValueError('Fisheye mapping folds inside the calibrated domain')
        else:
            first, second, tangent_x, tangent_y, third = coefficients
            radius = np.tan(np.deg2rad(self.maximum_angle))
            for polynomial in ([1, first, second, third], [1, 3 * first, 5 * second, 7 * third]):
                roots = np.roots([3 * polynomial[3], 2 * polynomial[2], polynomial[1]])
                locations = [0, radius ** 2] + [value.real for value in roots if abs(value.imag) < 1e-10 and 0 < value.real < radius ** 2]
                minimum = np.polynomial.polynomial.polyval(locations, polynomial).min()
                if minimum <= 12 * (abs(tangent_x) + abs(tangent_y)) * radius:
                    raise ValueError('Brown mapping lacks an invertible calibrated domain; reduce maximum_angle')

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or record.get('format') != 1:
            raise ValueError('Unsupported lens calibration format')
        return cls(**{name: record[name] for name in ('model', 'matrix', 'coefficients', 'width', 'height', 'maximum_angle')})

    def distort(self, points):
        points = np.asarray(points, dtype=float)
        radius = np.linalg.norm(points, axis=1)
        if self.model == 'fisheye':
            theta = np.arctan(radius)
            angle = theta.copy()
            for index, value in enumerate(self.coefficients):
                angle += value * theta ** (2 * index + 3)
            return points * np.divide(angle, radius, out=np.ones_like(radius), where=radius > 0)[:, None]
        x, y = points.T
        square = radius ** 2
        first, second, tangent_x, tangent_y, third = self.coefficients
        radial = 1 + first * square + second * square ** 2 + third * square ** 3
        return np.column_stack((x * radial + 2 * tangent_x * x * y + tangent_y * (square + 2 * x ** 2), y * radial + tangent_x * (square + 2 * y ** 2) + 2 * tangent_y * x * y))

    def project(self, points):
        matrix = np.asarray(self.matrix)
        valid = np.arctan(np.linalg.norm(points, axis=1)) <= np.deg2rad(self.maximum_angle)
        pixels = self.distort(points) * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]
        return (pixels, valid & np.isfinite(pixels).all(axis=1))

    def unproject(self, pixels):
        matrix = np.asarray(self.matrix)
        observed = (pixels - matrix[:2, 2]) / [matrix[0, 0], matrix[1, 1]]
        if self.model == 'fisheye':
            radius = np.linalg.norm(observed, axis=1)
            low, high = (np.zeros(len(pixels)), np.full(len(pixels), np.deg2rad(self.maximum_angle)))
            for _ in range(50):
                middle = (low + high) / 2
                mapped = middle.copy()
                for index, value in enumerate(self.coefficients):
                    mapped += value * middle ** (2 * index + 3)
                smaller = mapped < radius
                low = np.where(smaller, middle, low)
                high = np.where(smaller, high, middle)
            angle = (low + high) / 2
            points = observed * np.divide(np.tan(angle), radius, out=np.ones_like(radius), where=radius > 0)[:, None]
        else:
            points = observed.copy()
            for _ in range(40):
                mapped = self.distort(points)
                error = mapped - observed
                first = (self.distort(points + [1e-06, 0]) - mapped) / 1e-06
                second = (self.distort(points + [0, 1e-06]) - mapped) / 1e-06
                determinant = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
                safe = np.abs(determinant) > 1e-10
                denominator = np.where(safe, determinant, 1)
                step = np.column_stack(((second[:, 1] * error[:, 0] - second[:, 0] * error[:, 1]) / denominator, (first[:, 0] * error[:, 1] - first[:, 1] * error[:, 0]) / denominator))
                points -= np.clip(step, -0.5, 0.5) * safe[:, None]
                points = np.clip(points, -np.tan(np.deg2rad(self.maximum_angle)), np.tan(np.deg2rad(self.maximum_angle)))
        residual = np.linalg.norm(self.distort(points) - observed, axis=1)
        valid = np.isfinite(points).all(axis=1) & (residual < 1e-08)
        valid &= np.arctan(np.linalg.norm(points, axis=1)) <= np.deg2rad(self.maximum_angle) + 1e-12
        return (points, valid)

    def __init__(self, model, matrix, coefficients, width, height, maximum_angle=80.0):
        self.model = model
        self.matrix = matrix
        self.coefficients = coefficients
        self.width = width
        self.height = height
        self.maximum_angle = maximum_angle
        self.__post_init__()

class CameraModel:

    def __post_init__(self):
        if len(self.center) != 3 or not np.isfinite(self.center).all() or self.center[2] <= 0:
            raise ValueError('Camera center must be finite and above ground')
        if not self.identity or not np.isfinite([self.yaw, self.pitch, self.hfov, self.vfov]).all():
            raise ValueError('Camera identity and angles are required')
        if not 0 <= self.pitch <= 90 or not 0 < self.hfov < 179 or (not 0 < self.vfov < 179):
            raise ValueError('Invalid camera pitch or field of view')
        if any((type(value) is not int or value <= 0 for value in (self.width, self.height))):
            raise ValueError('Image dimensions must be positive integers')
        if not np.isfinite(self.roll):
            raise ValueError('Camera roll must be finite')
        if self.lens is not None and (self.width, self.height) != (self.lens.width, self.lens.height):
            raise ValueError('Image size must match lens calibration; recalibrate cropped modes')

    @property
    def basis(self):
        yaw, pitch = np.deg2rad([self.yaw, self.pitch])
        basis = np.array([[np.sin(yaw), -np.cos(yaw), 0], [-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), -np.cos(pitch)], [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)]])
        roll = np.deg2rad(self.roll)
        rotation = np.array([[np.cos(roll), np.sin(roll), 0], [-np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        return rotation @ basis

    @property
    def focal(self):
        return np.array([self.width, self.height]) / (2 * np.tan(np.deg2rad([self.hfov, self.vfov]) / 2))

    def project(self, points):
        """Project world points; invalid depths produce NaNs and a false visibility mask."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or (not np.isfinite(points).all()):
            raise ValueError('World points must be finite with shape (count, 3)')
        local = (points - self.center) @ self.basis.T
        forward = local[:, 2] > 1e-09
        pixels = np.full((len(points), 2), np.nan)
        normalized = local[forward, :2] / local[forward, 2, None]
        if self.lens is None:
            pixels[forward] = normalized * self.focal + np.array([self.width, self.height]) / 2
        else:
            pixels[forward], supported = self.lens.project(normalized)
            forward[np.flatnonzero(forward)] &= supported
        visible = forward & np.all((pixels >= 0) & (pixels <= [self.width, self.height]), axis=1)
        return (pixels, visible)

    def ground(self, pixels):
        """Intersect image rays with z=0; out-of-image and upward rays are invalid."""
        pixels = np.asarray(pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or (not np.isfinite(pixels).all()):
            raise ValueError('Image points must be finite with shape (count, 2)')
        normalized = (pixels - np.array([self.width, self.height]) / 2) / self.focal
        supported = np.ones(len(pixels), dtype=bool)
        if self.lens is not None:
            normalized, supported = self.lens.unproject(pixels)
        rays = np.column_stack((normalized, np.ones(len(pixels)))) @ self.basis
        valid = supported & (rays[:, 2] < -1e-09) & np.all((pixels >= 0) & (pixels <= [self.width, self.height]), axis=1)
        points = np.full((len(pixels), 3), np.nan)
        points[valid] = np.asarray(self.center) - self.center[2] / rays[valid, 2, None] * rays[valid]
        return (points, valid)

    def __init__(self, identity, center, yaw, pitch, hfov, vfov, width=1280, height=720, lens=None, roll=0.0):
        self.identity = identity
        self.center = center
        self.yaw = yaw
        self.pitch = pitch
        self.hfov = hfov
        self.vfov = vfov
        self.width = width
        self.height = height
        self.lens = lens
        self.roll = roll
        self.__post_init__()

class Rig:

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1:
            raise ValueError('The ring needs a positive integer camera count')
        values = [self.height, self.minimum_range, self.maximum_range, self.horn_height]
        if not np.isfinite(values).all() or not 0 <= self.minimum_range < self.maximum_range:
            raise ValueError('Invalid monitoring range')
        if not 0 <= self.horn_height < self.height:
            raise ValueError('Horn height must be below the array reference height')
        self.cameras()

    def cameras(self):
        if self.layout not in ('ring', 'single') or (self.layout == 'single' and self.count != 1):
            raise ValueError('Single layout needs exactly one camera')
        if len(self.offset) != 3 or not np.isfinite(self.offset).all():
            raise ValueError('Camera offset needs three finite values')
        center = tuple(np.asarray(self.offset) + [0, 0, self.height])
        lens = Lens.from_record(self.calibration) if self.calibration is not None else None
        if self.layout == 'single':
            return [CameraModel('wide-0', center, self.yaw, self.pitch, self.hfov, self.vfov, self.width, self.image_height, lens, self.roll)]
        if lens is not None:
            raise ValueError('Ring cameras require separate calibrations; use the legacy uncalibrated profile')
        result = [CameraModel(f'ring-{index}', center, self.yaw + 360 * index / self.count, self.pitch, self.hfov, self.vfov, self.width, self.image_height, None, self.roll) for index in range(self.count)]
        result.append(CameraModel('down', center, self.yaw, 90, self.down_hfov, self.down_vfov, self.width, self.image_height, None, self.roll))
        return result

    def __init__(self, count=6, height=6.0, pitch=40.0, hfov=90.0, vfov=70.0, down_hfov=120.0, down_vfov=100.0, yaw=0.0, width=1280, image_height=720, minimum_range=1.0, maximum_range=40.0, horn_height=0.8, layout='ring', roll=0.0, offset=(0.0, 0.0, 0.0), calibration=None):
        self.count = count
        self.height = height
        self.pitch = pitch
        self.hfov = hfov
        self.vfov = vfov
        self.down_hfov = down_hfov
        self.down_vfov = down_vfov
        self.yaw = yaw
        self.width = width
        self.image_height = image_height
        self.minimum_range = minimum_range
        self.maximum_range = maximum_range
        self.horn_height = horn_height
        self.layout = layout
        self.roll = roll
        self.offset = offset
        self.calibration = calibration
        self.__post_init__()

def route_cameras(rig, azimuth, tolerance=6.0, cameras=None):
    """Route a sampled direction wedge over the configured range and vehicle heights."""
    cameras = rig.cameras() if cameras is None else list(cameras)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 180:
        raise ValueError('Direction tolerance must be within [0, 180]')
    if azimuth is None:
        return [camera.identity for camera in cameras]
    if not np.isfinite(azimuth):
        raise ValueError('Azimuth must be finite or None for an undefined overhead bearing')
    angles = np.deg2rad(azimuth + np.linspace(-tolerance, tolerance, max(3, int(tolerance * 2) + 1)))
    radii, angles, heights = np.meshgrid(np.linspace(rig.minimum_range, rig.maximum_range, 128), angles, [0, rig.horn_height, 1.5], indexing='ij')
    points = np.column_stack(((radii * np.cos(angles)).ravel(), (radii * np.sin(angles)).ravel(), heights.ravel()))
    return [camera.identity for camera in cameras if camera.project(points)[1].any()]

class Detection:

    def __post_init__(self):
        if len(self.box) != 4 or len(self.foot) != 2 or len(self.velocity) != 2:
            raise ValueError('Invalid detection dimensions')
        if not self.identity or not self.camera or (not np.isfinite([*self.box, *self.foot, self.exposure, *self.velocity]).all()):
            raise ValueError('Detection data must be finite and identified')
        if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            raise ValueError('Detection box must have positive dimensions')

    def __init__(self, identity, camera, box, foot, exposure, velocity=(0.0, 0.0)):
        self.identity = identity
        self.camera = camera
        self.box = box
        self.foot = foot
        self.exposure = exposure
        self.velocity = velocity
        self.__post_init__()

class Association:

    def __post_init__(self):
        if self.method not in ('azimuth', 'depression', 'spatial'):
            raise ValueError('Association method must be azimuth, depression or spatial')
        if not np.isfinite([self.angle_gate, self.depression_gate, self.height_error, self.position_error, self.response_fraction, self.maximum_width, self.maximum_skew, self.merge_distance, self.speed]).all():
            raise ValueError('Association settings must be finite')
        if not 0 < self.angle_gate <= 180 or min(self.maximum_skew, self.merge_distance, self.speed) <= 0:
            raise ValueError('Invalid association settings')
        if not 0 <= self.depression_gate <= 90 or min(self.height_error, self.position_error) < 0:
            raise ValueError('Invalid geometric uncertainty bounds')
        if not 0 < self.response_fraction <= 1 or not 0 < self.maximum_width <= 90:
            raise ValueError('Invalid spatial support settings')

    def __init__(self, method='spatial', angle_gate=6.0, depression_gate=3.0, height_error=0.5, position_error=0.75, response_fraction=0.85, maximum_width=12.0, maximum_skew=0.5, merge_distance=0.75, speed=343.0):
        self.method = method
        self.angle_gate = angle_gate
        self.depression_gate = depression_gate
        self.height_error = height_error
        self.position_error = position_error
        self.response_fraction = response_fraction
        self.maximum_width = maximum_width
        self.maximum_skew = maximum_skew
        self.merge_distance = merge_distance
        self.speed = speed
        self.__post_init__()

def associate(rig, azimuth, arrival_time, detections, settings=Association(), cameras=None, evidence=None):
    """Match at an approximate emission time; preserve all angle-compatible vehicles.

Arrival time is a representative audio-window time, never decision wall time.
Velocity is an external visual estimate; the default assumes a stationary car.
Spatial clustering uses no vehicle truth identifiers and is only a prototype.
The default requires measured spatial evidence; azimuth-only behavior is an
explicit comparison mode and must not substitute for verified association.
"""
    if not np.isfinite(arrival_time) or (azimuth is not None and (not np.isfinite(azimuth))):
        raise ValueError('Invalid acoustic reference')
    if evidence is not None:
        if (evidence.bearing.azimuth is None) != (azimuth is None) or (azimuth is not None and abs(wrap_angle(evidence.bearing.azimuth - azimuth)) > 1e-06):
            raise ValueError('Acoustic evidence and association azimuth disagree')
    cameras = rig.cameras() if cameras is None else cameras
    lookup = {camera.identity: camera for camera in cameras}
    accepted, rejected = ([], [])
    for detection in detections:
        if detection.camera not in lookup:
            raise ValueError('Detection references an unknown camera')
        point, valid = lookup[detection.camera].ground([detection.foot])
        if not valid[0]:
            rejected.append(dict(identity=detection.identity, reason='invalid_ground_ray'))
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
        reason = 'outside_monitoring_range' if not rig.minimum_range <= radius <= rig.maximum_range else 'stale_frame' if abs(skew) > settings.maximum_skew else 'angle_mismatch' if error is not None and error > settings.angle_gate else None
        if reason:
            rejected.append(dict(identity=detection.identity, reason=reason))
            continue
        support = dict(reason=None, ambiguous=False)
        if settings.method != 'azimuth':
            support = acoustic_support(rig, position, evidence, settings)
            if support['reason']:
                rejected.append(dict(identity=detection.identity, **support))
                continue
        accepted.append(dict(identity=detection.identity, camera=detection.camera, position=position.tolist(), angle=angle, error=error, emission=emission, skew=skew, box=list(detection.box), acoustic=support))
    groups = []
    for candidate in sorted(accepted, key=lambda value: (value['position'], value['camera'], value['identity'])):
        for group in groups:
            if candidate['camera'] not in {item['camera'] for item in group} and all((np.linalg.norm(np.asarray(candidate['position']) - item['position']) <= settings.merge_distance for item in group)):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    state = 'unmatched' if not groups else 'unique' if len(groups) == 1 else 'ambiguous'
    if groups and (azimuth is None or any((item['acoustic']['ambiguous'] for group in groups for item in group))):
        state = 'ambiguous'
    return dict(state=state, groups=groups, rejected=rejected, method=settings.method, source_unexplained=not bool(groups), score_definition='acoustic compatibility, not proof of horn identity or calibrated probability')

def direction_vectors(azimuths, depressions):
    azimuths, depressions = np.broadcast_arrays(np.asarray(azimuths, dtype=float), np.asarray(depressions, dtype=float))
    if not np.isfinite(azimuths).all() or not np.isfinite(depressions).all() or np.any((depressions < 0) | (depressions > 90)):
        raise ValueError('Directions require finite azimuths and depressions within [0, 90]')
    azimuths, depressions = (np.deg2rad(azimuths), np.deg2rad(depressions))
    return np.stack((np.cos(depressions) * np.cos(azimuths), np.cos(depressions) * np.sin(azimuths), -np.sin(depressions)), axis=-1)

class Bearing:

    def __post_init__(self):
        if self.azimuth is not None and (not np.isfinite(self.azimuth)):
            raise ValueError('Azimuth must be finite or undefined')
        if not np.isfinite([self.depression, self.response]).all() or not 0 <= self.depression <= 90:
            raise ValueError('Invalid bearing depression or response')

    def __init__(self, azimuth, depression, response):
        self.azimuth = azimuth
        self.depression = depression
        self.response = response
        self.__post_init__()

class AcousticEvidence:
    """Measured horn-selected bearing and optional SRP map from the same audio window."""

    def __post_init__(self):
        if self.grid is None:
            return
        if len(self.grid) != 3:
            raise ValueError('Spatial evidence needs azimuths, depressions and response')
        azimuths, depressions, response = [np.asarray(value, dtype=float) for value in self.grid]
        if azimuths.ndim != 1 or depressions.ndim != 1 or len(azimuths) < 3 or (len(depressions) < 2):
            raise ValueError('Spatial evidence needs a nonempty angular grid')
        if response.shape != (len(azimuths), len(depressions)) or not all((np.isfinite(value).all() for value in (azimuths, depressions, response))):
            raise ValueError('Invalid spatial evidence dimensions or values')
        if np.any(np.diff(azimuths) <= 0) or np.any(np.diff(depressions) <= 0):
            raise ValueError('Spatial grid angles must be strictly increasing')
        if azimuths[-1] - azimuths[0] >= 360 or depressions[0] < 0 or depressions[-1] > 90:
            raise ValueError('Spatial evidence has duplicate azimuth endpoints or invalid depressions')

    def __init__(self, bearing, grid=None):
        self.bearing = bearing
        self.grid = grid
        self.__post_init__()

def acoustic_support(rig, position, evidence, settings):
    """Test height/position intervals against a horn-selected peak, never a visual-only winner."""
    if evidence is None:
        return dict(reason='missing_acoustic_evidence', ambiguous=False)
    radius = float(np.linalg.norm(position))
    lower_height = max(0, rig.horn_height - settings.height_error)
    upper_height = min(rig.height, rig.horn_height + settings.height_error)
    lower = float(np.rad2deg(np.arctan2(rig.height - upper_height, radius + settings.position_error)))
    upper = float(np.rad2deg(np.arctan2(rig.height - lower_height, max(0, radius - settings.position_error))))
    depression = evidence.bearing.depression
    bounds = dict(depression_bounds=[lower, upper], measured_depression=depression)
    if depression < lower - settings.depression_gate or depression > upper + settings.depression_gate:
        return dict(reason='depression_mismatch', ambiguous=False, **bounds)
    if settings.method == 'depression':
        return dict(reason=None, ambiguous=False, **bounds)
    if evidence.grid is None:
        return dict(reason='missing_spatial_response', ambiguous=False, **bounds)
    azimuths, depressions, response = [np.asarray(value) for value in evidence.grid]
    if evidence.bearing.azimuth is None:
        return dict(reason=None, ambiguous=True, **bounds)
    row = int(np.argmin(abs(wrap_angle(azimuths - evidence.bearing.azimuth))))
    column = int(np.argmin(abs(depressions - depression)))
    reference = float(response[row, column])
    floor = float(np.median(response))
    contrast = reference - floor
    if contrast <= max(1e-09, np.ptp(response) * 1e-06):
        return dict(reason='uninformative_spatial_response', ambiguous=False, **bounds)
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
    angular_error = float(np.rad2deg(np.arcsin(min(1, settings.position_error / max(radius, 1e-09)))))
    azimuth_step = float(np.max(np.diff(azimuths)))
    azimuth_mask = abs(wrap_angle(azimuths - angle)) <= angular_error + azimuth_step / 2 + 1e-09
    depression_mask = (depressions >= lower - step / 2) & (depressions <= upper + step / 2)
    if not azimuth_mask.any() or not depression_mask.any():
        return dict(reason='spatial_grid_miss', ambiguous=False, **bounds)
    best = float(response[np.ix_(azimuth_mask, depression_mask)].max())
    ratio = (best - floor) / contrast
    if best < threshold:
        return dict(reason='spatial_response_mismatch', ambiguous=False, response_ratio=ratio, profile_width=width, **bounds)
    return dict(reason=None, ambiguous=width > settings.maximum_width, response_ratio=ratio, profile_width=width, **bounds)
