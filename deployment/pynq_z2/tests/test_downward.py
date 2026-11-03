import json
from pathlib import Path
import wave
from xml.etree import ElementTree

import numpy as np
import pytest

from ps.detector import Detector
from ps.geometry import direction_vectors
from scripts.simulate_downward import (POSITIONS, analyze, configuration, digest, geometry_checks,
                                      make_rig, propagate, positive_control, recording, render, simulate, vehicles)

ROOT = Path(__file__).resolve().parents[1]



def test_nominal_axis_angles_match_an_independent_fisheye_formula():
    config = configuration()
    camera = make_rig(config).cameras()[0]
    horizontal, vertical = 6 * np.tan(np.deg2rad([144, 77.6]) / 2)
    pixels, _ = camera.project([[0, horizontal, 0], [vertical, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(pixels, [[0, 540], [960, 0], [960, 540]], atol=1e-10)
    assert config['camera']['pitch'] == 90 and config['camera']['calibration']['simulation_only']



def test_spherical_delays_are_causal_and_srp_recovers_the_source_without_truth_input():
    from ps.dsp import coordinates
    from ps.streaming import Processing
    config = configuration()
    random = np.random.RandomState(26).normal(0, 0.03, 96256)
    position = [1.8, 7]
    array = coordinates(config['audio'])
    audio = propagate(random, position, array, 6)
    distances = np.linalg.norm(np.r_[position, -5.2] - array, axis=1)
    assert not audio[:int(distances.min() * 48000 / 343)].any()
    processing = Processing(config)
    snapshot = dict(audio=audio[:48128], first=0, end=48128, epoch=0)
    _, bearings, _ = processing.search.run_snapshot(snapshot, processing.continuity.prepare(snapshot))
    truth = np.r_[position, -5.2]
    truth /= np.linalg.norm(truth)
    errors = [np.rad2deg(np.arccos(np.clip(direction_vectors(bearing.azimuth or 0, bearing.depression) @ truth, -1, 1))) for bearing in bearings]
    assert min(errors) < 4



def test_source_loader_records_resampling_and_rejects_silent_input(tmp_path):
    source = tmp_path / 'source.wav'
    samples = np.arange(44100) / 44100
    with wave.open(str(source), 'wb') as stream:
        stream.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        stream.writeframes((5000 * np.sin(2 * np.pi * 600 * samples)).astype('<i2').tobytes())
    output, meta = recording(source, 48000)
    assert len(output) == 48000 and meta['sha256'] == digest(source)
    assert np.sqrt(np.mean(output ** 2)) == pytest.approx(1)
    assert meta['original_rate'] == 44100
    with wave.open(str(source), 'wb') as stream:
        stream.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
        stream.writeframes(np.zeros(48000, dtype='<i2').tobytes())
    with pytest.raises(ValueError, match='energy'):
        recording(source, 48000)



def test_geometry_only_does_not_change_production_configuration_or_require_raw_sources(tmp_path):
    before = digest(ROOT / 'config/default.json')
    report = simulate(tmp_path / 'geometry', geometry_only=True)
    assert not report['acoustic'] and report['board_validated'] is False
    assert digest(ROOT / 'config/default.json') == before
    assert (tmp_path / 'geometry/coverage.svg').is_file()
    plan = json.loads((tmp_path / 'geometry/plan.json').read_text())
    assert plan['configuration']['camera']['pitch'] == 90
    with pytest.raises(ValueError, match='new simulation'):
        simulate(tmp_path / 'geometry', geometry_only=True)



def test_injected_ray_controls_detect_pose_and_lens_errors():
    result = geometry_checks(configuration())
    controls = result['independent_ground_controls']
    assert controls['nominal']['metres_rms'] < 1e-9
    for condition in ('height_error', 'tilt_error', 'lens_error', 'pixel_noise'):
        assert controls[condition]['metres_rms'] > 0.01
    assert len(result['rows']) == len(POSITIONS) * 6
    nominal = [row['pixel_error'] for row in result['rows'] if row['condition'] == 'nominal' and row['pixel_error'] is not None]
    assert max(nominal) < 1e-9
    height_error = [row['pixel_error'] for row in result['rows'] if row['condition'] == 'height_error' and row['pixel_error'] is not None]
    assert max(height_error) < 1e-9



@pytest.mark.parametrize('height', [0, 2, float('nan')])
def test_height_validation_rejects_camera_below_car_roofs(height):
    with pytest.raises(ValueError):
        configuration(height)



def test_silence_never_creates_candidate_points():
    config = configuration()
    event, evidence, frames, triggered = analyze(config, Detector(ROOT / 'model'), np.zeros((96256, 14), dtype='<i2'), 2)
    assert len(frames) == 3 and event['state'] == 'IDLE'
    assert evidence is None and not triggered and not event['capture_candidates']
    assert not any(row['horn_confirmed'] for row in frames)



def test_existing_positive_control_confirms_immediately_under_current_default(tmp_path):
    config = configuration()
    control = positive_control(config, Detector(ROOT / 'model'), tmp_path)
    assert control['triggered'] and control['event']['horn_candidate']
    assert control['event']['state'] == 'CONFIRMED' and control['event']['horn_confirmed']
    assert [row['state'] for row in control['events']] == ['CONFIRMED', 'CONFIRMED', 'CONFIRMED']
    assert [row['horn_confirmed'] for row in control['events']] == [True, True, True]
    assert control['horn_confirmed'] and any(point['state'] == 'projected' for point in control['points'])
    svg = (tmp_path / control['images'][0]).read_text(encoding='utf-8')
    rectangles = ElementTree.fromstring(svg).findall('.//{http://www.w3.org/2000/svg}rect')
    assert 'acoustic-point' in svg and all('stroke' not in item.attrib for item in rectangles)
    assert config['camera']['yaw'] == 0 and config['detector']['threshold'] == 0.9
    assert config['detector']['capture_threshold'] == 0.9



def test_camera_pitch_cannot_change_real_acoustic_scores_or_event_states():
    config = configuration()
    with wave.open(str(ROOT / 'examples/horn-array14.wav'), 'rb') as stream:
        pcm = np.frombuffer(stream.readframes(96256), dtype='<i2').reshape(-1, 14)
    detector = Detector(ROOT / 'model')
    _, _, down, _ = analyze(config, detector, pcm, 5)
    config['camera']['pitch'] = 40
    _, _, forward, _ = analyze(config, detector, pcm, 5)
    for first, second in zip(down, forward):
        for key in ('scores', 'state', 'horn_confirmed', 'azimuth', 'depression', 'capture_candidates'):
            assert first[key] == second[key]




def test_narrow_field_of_view_reports_missing_controls_instead_of_nan(tmp_path):
    result = simulate(tmp_path / 'narrow', geometry_only=True, hfov=1, vfov=1)
    assert result['geometry']['independent_ground_controls']['nominal']['metres_rms'] is None
    assert result['geometry']['independent_ground_controls']['nominal']['points'] == 0
