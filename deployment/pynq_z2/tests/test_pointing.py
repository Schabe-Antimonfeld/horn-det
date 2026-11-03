import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ps.geometry import Lens, Rig
from ps.pointing import point_sources, project_point
from ps.vision import ImageIO


def test_direct_ray_matches_independent_pinhole_formula_and_needs_no_height():
    for height in (3, 6, 12):
        rig = Rig(layout='single', count=1, height=height, pitch=90, width=1000, image_height=800)
        camera = rig.cameras()[0]
        result = project_point(rig, camera, dict(azimuth=30, depression=65))
        radius = 1 / np.tan(np.deg2rad(65))
        expected = np.array([500, 400]) - camera.focal * radius * np.array([0.5, np.sqrt(3) / 2])
        np.testing.assert_allclose(result['pixel'], expected, atol=1e-10)
        for distance in (1, 5, 50):
            np.testing.assert_allclose(project_point(rig, camera, dict(azimuth=30, depression=65), distance)['pixel'], expected)


def test_translated_camera_requires_range_and_uses_the_array_origin():
    rig = Rig(layout='single', count=1, pitch=90, offset=(0.5, 0, 0))
    camera = rig.cameras()[0]
    hypothesis = dict(azimuth=0, depression=90)
    assert project_point(rig, camera, hypothesis)['state'] == 'depth_required'
    expected, valid = camera.project([[0, 0, 1]])
    assert valid[0]
    np.testing.assert_allclose(project_point(rig, camera, hypothesis, 5)['pixel'], expected[0])
    for distance in (0, -1, float('nan')):
        with pytest.raises(ValueError, match='distance'):
            project_point(rig, camera, hypothesis, distance)


def test_vertical_ray_is_unique_but_near_vertical_undefined_azimuth_is_not():
    rig = Rig(layout='single', count=1, pitch=90)
    camera = rig.cameras()[0]
    np.testing.assert_allclose(project_point(rig, camera, dict(azimuth=None, depression=90))['pixel'], [640, 360])
    assert project_point(rig, camera, dict(azimuth=None, depression=87))['state'] == 'azimuth_ambiguous'
    for hypothesis in (dict(azimuth=0), dict(azimuth=float('nan'), depression=30), dict(azimuth=0, depression=100)):
        assert project_point(rig, camera, hypothesis)['pixel'] is None


def test_front_back_and_multiple_hypotheses_remain_independent():
    rig = Rig(layout='single', count=1)
    points = point_sources(rig, [dict(azimuth=0, depression=40, score=0.96), dict(azimuth=180, depression=0, score=0.92)])['wide-0']
    assert points[0]['state'] == 'projected' and points[0]['score'] == 0.96
    assert points[1]['state'] == 'outside_camera_view' and points[1]['pixel'] is None
    assert [point['candidate'] for point in points] == [0, 1]
    assert point_sources(rig, []) == {'wide-0': []}


def test_image_adapter_draws_points_without_a_network_or_rectangles(monkeypatch, tmp_path):
    calls = []
    fake = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0,
                           circle=lambda *args: calls.append(('circle', args[1])),
                           putText=lambda *args: calls.append(('text', args[1])),
                           imwrite=lambda *args: True)
    monkeypatch.setitem(sys.modules, 'cv2', fake)
    adapter = ImageIO()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    adapter.annotate(image, [dict(state='projected', pixel=[99.9, 10], candidate=0),
                             dict(state='azimuth_ambiguous', pixel=None, candidate=1)],
                     dict(horn_confirmed=False), tmp_path / 'points.jpg')
    assert calls[0] == ('circle', (99, 10))
    assert any('UNCONFIRMED' in value for name, value in calls if name == 'text')
    assert not image.any()
    fake.imwrite = lambda *args: False
    with pytest.raises(RuntimeError, match='save'):
        adapter.annotate(image, [], dict(horn_confirmed=True), tmp_path / 'failed.jpg')



def test_calibrated_domain_rejects_an_inside_image_pixel_outside_supported_rays():
    rig = Rig(layout='single', count=1, pitch=90, width=1000, image_height=800)
    camera = rig.cameras()[0]
    camera.lens = Lens('fisheye', [[100, 0, 500], [0, 100, 400], [0, 0, 1]], [0, 0, 0, 0], 1000, 800, 40)
    point = project_point(rig, camera, dict(azimuth=0, depression=30))
    assert point['state'] == 'outside_camera_view' and point['pixel'] is None
