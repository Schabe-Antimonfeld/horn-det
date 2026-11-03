"""Project acoustic directions into calibrated images without vehicle detection."""

import numpy as np

from ps.geometry import direction_vectors


def project_point(rig, camera, hypothesis, distance=None):
    """Use a bearing ray; translated centers require an independently measured range."""
    azimuth, depression = hypothesis.get('azimuth'), hypothesis.get('depression')
    result = dict(azimuth=azimuth, depression=depression, score=hypothesis.get('score'),
                  pixel=None, state='direction_incomplete', vehicle_attribution='unverified')
    if depression is None or not np.isfinite(depression) or not 0 <= depression <= 90:
        return result
    if azimuth is None and not np.isclose(depression, 90, rtol=0, atol=1e-9):
        result['state'] = 'azimuth_ambiguous'
        return result
    if azimuth is not None and not np.isfinite(azimuth):
        return result
    origin = np.array([0.0, 0.0, rig.height])
    if distance is None and np.linalg.norm(np.asarray(camera.center) - origin) > 1e-9:
        result['state'] = 'depth_required'
        return result
    if distance is not None and (not np.isfinite(distance) or distance <= 0):
        raise ValueError('Source distance must be finite and positive')
    vector = direction_vectors(azimuth if azimuth is not None else 0, depression)
    pixels, visible = camera.project([origin + (distance if distance is not None else 1.0) * vector])
    pixel = pixels[0]
    if not visible[0] or not np.all((pixel >= 0) & (pixel < [camera.width, camera.height])):
        result['state'] = 'outside_camera_view'
        return result
    result.update(state='projected', pixel=pixel.tolist(),
                  calibration='nominal' if camera.lens is None else 'calibrated')
    return result


def point_sources(rig, hypotheses):
    """Keep each hypothesis independently, including invisible and unresolved directions."""
    return {camera.identity: [dict(project_point(rig, camera, hypothesis), candidate=index)
                              for index, hypothesis in enumerate(hypotheses)]
            for camera in rig.cameras()}
