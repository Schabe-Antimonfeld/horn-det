from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps.configuration import load_config as load_current
from ps.pipeline import Events, Pipeline
from ps.geometry import Bearing


def load_config():
    """Construct injected event-gate scenarios independently of native processing."""
    config = load_current()
    for name in ('streaming', 'refinement', 'beamforming'):
        config.pop(name)
    config['search']['algorithm'] = 'reference'
    config['detector'].update(threshold=0.65, margin=0.05)
    config['detector'].pop('capture_threshold')
    config['event']['confirm'] = 2
    return config


def test_event_confirmation_and_skipped_windows_reset_evidence():
    events = Events({'confirm': 2, 'release': 2}, 100)
    assert events.feed(True, {'epoch': 0, 'end': 100})['state'] == 'CANDIDATE'
    assert events.feed(True, {'epoch': 0, 'end': 200})['state'] == 'CONFIRMED'
    record = events.feed(True, {'epoch': 0, 'end': 400})
    assert record['state'] == 'CANDIDATE' and record['history_reset']
    assert record['ended_event'] == 1 and record['event_id'] == 2
    assert events.feed(False, {'epoch': 1, 'end': 500})['state'] == 'IDLE'


def test_point_output_never_loads_a_vehicle_model():
    config = load_current()
    assert config['vision'] == dict(enabled=False, output='acoustic_points')


def test_silence_never_creates_an_event_or_invokes_classifier():
    class Detector:
        def score(self, contexts):
            raise AssertionError('Silence must not reach classification')
    config = load_config()
    pipeline = Pipeline(config, Detector())
    event, evidence = pipeline.analyze({'audio': np.zeros((2048, 14)), 'first': 0, 'end': 2048,
                                         'epoch': 0, 'clipped': False})
    assert not event['horn_candidate'] and event['state'] == 'IDLE' and evidence is None


def test_unavailable_camera_is_retried_and_successful_capture_is_deduplicated(tmp_path):
    pipeline = Pipeline(load_config(), None)
    event = dict(state='CANDIDATE', horn_candidate=True, capture_candidate=True, event_id=1,
                 capture_candidates=[dict(azimuth=0, depression=40)], horn_confirmed=False)
    assert pipeline.capture(event, None, None, None, tmp_path, 0)['photos'] == 0
    assert not pipeline.captured
    class Cache:
        def nearest(self, identity, timestamp):
            return dict(image=np.zeros((1080, 1920, 3), dtype=np.uint8), jpeg=b'test-only', time=0)
    class Imaging:
        def annotate(self, image, points, result, path):
            assert points[0]['state'] == 'projected' and result['stage'] == 'candidate'
    result = pipeline.capture(event, None, Cache(), Imaging(), tmp_path, 0)
    assert result['photos'] == 1 and result['frames'][0]['timing_verified'] is False
    assert pipeline.capture(event, None, Cache(), Imaging(), tmp_path, 0) is None


def test_outside_wide_camera_view_does_not_request_unrelated_images(tmp_path):
    pipeline = Pipeline(load_config(), None)
    class Cache:
        def nearest(self, identity, timestamp):
            raise AssertionError('Rear source must not request an unrelated camera frame')
    event = dict(state='CANDIDATE', capture_candidate=True, event_id=1,
                 capture_candidates=[dict(azimuth=180, depression=20)])
    result = pipeline.capture(event, None, Cache(), None, tmp_path, 0)
    assert result['visibility'] == 'outside_camera_view' and result['photos'] == 0
    assert result['vehicle_attribution'] == 'unverified'


def test_ambiguous_directions_keep_horn_event_and_both_hypotheses(monkeypatch, tmp_path):
    import ps.pipeline as module
    grid = (np.arange(-180, 180, 2), np.arange(0, 91, 3), np.ones((180, 31)))
    monkeypatch.setattr(module, 'spatial_map', lambda *args: grid)
    monkeypatch.setattr(module, 'candidates', lambda *args: [Bearing(0, 20, 2), Bearing(180, 20, 1)])
    monkeypatch.setattr(module, 'beamform', lambda *args: (np.zeros(48000), 32))
    class Detector:
        def __init__(self):
            self.index = 0
        def score(self, contexts):
            self.index += 1
            return dict(score=0.9 if self.index % 2 else 0.88, clipped=False)
    config = load_config()
    pipeline = Pipeline(config, Detector())
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    event, evidence = pipeline.analyze(snapshot)
    assert event['horn_candidate'] and not event['doa_valid'] and event['azimuth'] is None
    assert event['acoustic_state'] == 'ambiguous_direction' and len(event['direction_candidates']) == 2
    assert evidence is not None
    snapshot['end'] += config['audio']['update_samples']
    event, evidence = pipeline.analyze(snapshot)
    assert event['state'] == 'CONFIRMED'
    result = pipeline.capture(event, evidence, None, None, tmp_path, 0)
    assert result['selected_cameras'] == ['wide-0'] and result['photos'] == 0
    assert result['visibility'] == 'partly_outside_camera_view'
    assert len(result['projections']['wide-0']) == 2
    snapshot['clipped'] = True
    event, evidence = pipeline.analyze(snapshot)
    assert not event['horn_candidate'] and event['acoustic_state'] == 'invalid_audio' and evidence is None
    snapshot['clipped'] = False
    config['detector']['margin'] = 0
    monkeypatch.setattr(Detector, 'score', lambda *args: dict(score=0.9, clipped=False))
    event, _ = pipeline.analyze(snapshot)
    assert event['horn_candidate'] and not event['doa_valid'] and len(event['direction_candidates']) == 2


def test_frontend_overflow_abstains_without_terminating_service(monkeypatch):
    import ps.pipeline as module
    from ps.detector import FrontendOverflow
    monkeypatch.setattr(module, 'spatial_map', lambda *args: None)
    monkeypatch.setattr(module, 'candidates', lambda *args: [Bearing(0, 20, 1)])
    monkeypatch.setattr(module, 'beamform', lambda *args: (np.zeros(48000), 32))
    class Detector:
        def score(self, contexts):
            raise FrontendOverflow('Outside calibrated input range')
    pipeline = Pipeline(load_config(), Detector())
    event, evidence = pipeline.analyze(dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False))
    assert event['frontend_clipped'] and not event['activation_clipped']
    assert event['acoustic_state'] == 'invalid_audio' and not event['horn_candidate'] and evidence is None


def test_multiple_candidates_preserve_visible_and_outside_hypotheses(tmp_path):
    pipeline = Pipeline(load_config(), None)
    event = dict(capture_candidate=True, event_id=1, horn_confirmed=False,
                 capture_candidates=[dict(azimuth=0, depression=40), dict(azimuth=180, depression=20)])
    result = pipeline.capture(event, None, None, None, tmp_path, 0)
    assert result['selected_cameras'] == ['wide-0']
    assert [point['state'] for point in result['projections']['wide-0']] == ['projected', 'outside_camera_view']
    assert result['visibility'] == 'partly_outside_camera_view'


def capture_pipeline(monkeypatch, scores, capture=0.5):
    import ps.pipeline as module
    grid = (np.arange(-180, 180, 2), np.arange(0, 91, 3), np.ones((180, 31)))
    monkeypatch.setattr(module, 'spatial_map', lambda *args: grid)
    monkeypatch.setattr(module, 'candidates', lambda *args: [Bearing(180, 20, 2), Bearing(0, 20, 1)])
    monkeypatch.setattr(module, 'beamform', lambda *args: (np.zeros(48000), 32))
    monkeypatch.setattr(module, 'features', lambda *args: np.zeros((7, 40, 24)))
    class Detector:
        def __init__(self):
            self.index = 0
        def score(self, contexts):
            result = dict(score=scores[self.index % 2], clipped=False)
            self.index += 1
            return result
    config = load_config()
    if capture is not None:
        config['detector']['capture_threshold'] = capture
    return Pipeline(config, Detector())


def test_low_score_candidate_saves_early_without_confirming_horn(monkeypatch, tmp_path):
    pipeline = capture_pipeline(monkeypatch, [0.6, 0.55])
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    event, evidence = pipeline.analyze(snapshot)
    assert event['state'] == 'CANDIDATE' and event['capture_candidate']
    assert not event['horn_candidate'] and not event['horn_confirmed'] and not event['doa_valid']
    assert event['acoustic_state'] == 'capture_only' and event['direction_candidates'] == []
    class Cache:
        def nearest(self, identity, timestamp):
            return dict(image=np.zeros((1080, 1920, 3), dtype=np.uint8), jpeg=b'test-only', time=0)
    class Imaging:
        def annotate(self, image, points, result, path):
            assert not result['horn_confirmed'] and result['stage'] == 'candidate'
            assert result['vehicle_attribution'] == 'unverified'
    result = pipeline.capture(event, evidence, Cache(), Imaging(), tmp_path, 0)
    assert result['photos'] == 1 and result['stage'] == 'candidate'
    assert pipeline.capture(event, evidence, Cache(), Imaging(), tmp_path, 0) is None
    snapshot['end'] += pipeline.config['audio']['update_samples']
    event, _ = pipeline.analyze(snapshot)
    assert event['state'] == 'CONFIRMED' and event['horn_event_state'] == 'IDLE'
    assert not event['horn_confirmed']
    assert pipeline.capture(event, evidence, Cache(), Imaging(), tmp_path, 0) is None
    snapshot['end'] += 2 * pipeline.config['audio']['update_samples']
    event, _ = pipeline.analyze(snapshot)
    assert event['history_reset'] and event['state'] == 'CANDIDATE'


@pytest.mark.parametrize('capture', [None, 0.5])
def test_capture_gate_never_changes_strict_horn_or_direction_decisions(monkeypatch, capture):
    pipeline = capture_pipeline(monkeypatch, [0.8, 0.55], capture)
    event, _ = pipeline.analyze(dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False))
    assert event['horn_candidate'] and event['doa_valid'] and event['azimuth'] == 180
    assert len(event['direction_candidates']) == 1
    assert len(event['capture_candidates']) == (1 if capture is None else 2)
    assert event['event_basis'] == ('horn_candidate' if capture is None else 'capture_candidate')


@pytest.mark.parametrize('failure', ['input', 'activation', 'frontend'])
def test_low_capture_gate_still_rejects_invalid_audio(monkeypatch, failure):
    from ps.detector import FrontendOverflow
    pipeline = capture_pipeline(monkeypatch, [0.6, 0.55])
    if failure == 'activation':
        monkeypatch.setattr(pipeline.detector, 'score', lambda *args: dict(score=0.6, clipped=True))
    elif failure == 'frontend':
        def overflow(*args):
            raise FrontendOverflow('Outside calibrated frontend')
        monkeypatch.setattr(pipeline.detector, 'score', overflow)
    event, evidence = pipeline.analyze(dict(audio=np.zeros((48000, 14)), first=0, end=48000,
                                            epoch=0, clipped=failure == 'input'))
    assert not event['capture_candidate'] and not event['horn_candidate'] and not event['doa_valid']
    assert event['state'] == 'IDLE' and event['capture_candidates'] == [] and evidence is None


def test_capture_threshold_is_inclusive_but_default_keeps_low_scores_idle(monkeypatch):
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    event, _ = capture_pipeline(monkeypatch, [0.5, 0.49]).analyze(snapshot)
    assert event['capture_candidate'] and len(event['capture_candidates']) == 1
    assert not event['horn_candidate'] and event['acoustic_state'] == 'capture_only'
    event, _ = capture_pipeline(monkeypatch, [0.5, 0.49], None).analyze(snapshot)
    assert event['state'] == 'IDLE' and not event['capture_candidate'] and not event['horn_candidate']


def test_strict_horn_confirmation_needs_two_consecutive_strict_windows(monkeypatch):
    pipeline = capture_pipeline(monkeypatch, [0.8, 0.55])
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    event, _ = pipeline.analyze(snapshot)
    assert event['horn_candidate'] and not event['horn_confirmed']
    snapshot['end'] += pipeline.config['audio']['update_samples']
    event, _ = pipeline.analyze(snapshot)
    assert event['horn_confirmed'] and event['horn_event_state'] == 'CONFIRMED'
    snapshot['end'] += 2 * pipeline.config['audio']['update_samples']
    event, _ = pipeline.analyze(snapshot)
    assert not event['horn_confirmed'] and event['horn_event_state'] == 'CANDIDATE'


def test_stale_frames_retry_and_confirmation_can_save_a_second_stage(tmp_path):
    pipeline = Pipeline(load_config(), None)
    event = dict(capture_candidate=True, event_id=1, horn_confirmed=False,
                 capture_candidates=[dict(azimuth=0, depression=40)])
    class Cache:
        timestamp = 1.0
        def nearest(self, identity, timestamp):
            return dict(image=np.zeros((1080, 1920, 3), dtype=np.uint8), jpeg=b'test-only', time=self.timestamp)
    class Imaging:
        def annotate(self, image, points, result, path):
            path.write_bytes(b'test-point-render')
    cache, imaging = Cache(), Imaging()
    result = pipeline.capture(event, None, cache, imaging, tmp_path, 0)
    assert result['photos'] == 0 and result['frames'][0]['state'] == 'stale'
    cache.timestamp = 0
    assert pipeline.capture(event, None, cache, imaging, tmp_path, 0)['photos'] == 1
    event['horn_confirmed'] = True
    assert pipeline.capture(event, None, cache, imaging, tmp_path, 0)['photos'] == 1
    assert pipeline.capture(event, None, cache, imaging, tmp_path, 0) is None
    assert len(list(tmp_path.glob('*.jpg'))) == 4
    event['event_id'] = 2
    event['horn_confirmed'] = False
    assert pipeline.capture(event, None, cache, imaging, tmp_path, 0)['photos'] == 1


@pytest.mark.parametrize('failure', ['input_clipped', 'activation_clipped', 'frontend_clipped'])
def test_invalid_audio_cannot_save_even_with_a_fabricated_candidate(failure, tmp_path):
    event = dict(capture_candidate=True, event_id=1, capture_candidates=[dict(azimuth=0, depression=40)])
    event[failure] = True
    assert Pipeline(load_config(), None).capture(event, None, None, None, tmp_path, 0) is None



def test_strict_confirmation_does_not_bridge_a_low_score_window(monkeypatch):
    pipeline = capture_pipeline(monkeypatch, [0.8, 0.55])
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    def step(score):
        monkeypatch.setattr(pipeline.detector, 'score', lambda *args: dict(score=score, clipped=False))
        event, _ = pipeline.analyze(snapshot)
        snapshot['end'] += pipeline.config['audio']['update_samples']
        return event
    assert not step(0.8)['horn_confirmed']
    assert step(0.8)['horn_confirmed']
    assert not step(0.6)['horn_confirmed']
    assert not step(0.8)['horn_confirmed']
    assert step(0.8)['horn_confirmed']



def test_nonfinite_frame_time_is_rejected_with_serializable_status(tmp_path):
    import json
    pipeline = Pipeline(load_config(), None)
    event = dict(capture_candidate=True, event_id=1, capture_candidates=[dict(azimuth=0, depression=40)])
    class Cache:
        def nearest(self, *args):
            return dict(time=float('nan'))
    result = pipeline.capture(event, None, Cache(), None, tmp_path, 0)
    assert result['frames'][0]['state'] == 'stale' and result['frames'][0]['skew_seconds'] is None
    assert result['photos'] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('hypothesis,offset,state', [
    (dict(azimuth=None, depression=87), [0, 0, 0], 'azimuth_ambiguous'),
    (dict(azimuth=0, depression=40), [0.1, 0, 0], 'depth_required')])
def test_unresolved_direction_can_retain_an_image_without_inventing_a_point(tmp_path, hypothesis, offset, state):
    config = load_config()
    config['camera']['offset'] = offset
    pipeline = Pipeline(config, None)
    event = dict(capture_candidate=True, event_id=1, capture_candidates=[hypothesis])
    class Cache:
        def nearest(self, *args):
            return dict(image=np.zeros((1080, 1920, 3), dtype=np.uint8), jpeg=b'test-only', time=0)
    class Imaging:
        def annotate(self, image, points, result, path):
            assert points[0]['state'] == state and points[0]['pixel'] is None
    result = pipeline.capture(event, None, Cache(), Imaging(), tmp_path, 0)
    assert result['photos'] == 1 and result['visibility'] == 'projection_unresolved'



@pytest.mark.parametrize('score,confirmed', [(0.899999, False), (0.9, True), (0.92, True), (0.95, True)])
def test_current_default_confirms_a_single_reliable_window_at_point_nine(monkeypatch, tmp_path, score, confirmed):
    pipeline = capture_pipeline(monkeypatch, [score, 0.1])
    current = load_current()
    assert current['event']['confirm'] == 1
    assert current['detector']['threshold'] == current['detector']['capture_threshold'] == 0.9
    pipeline.config['detector'] = current['detector']
    pipeline.config['event'] = current['event']
    pipeline.events = Events(current['event'], current['audio']['update_samples'])
    pipeline.horn_events = Events(current['event'], current['audio']['update_samples'])
    event, evidence = pipeline.analyze(dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False))
    assert event['horn_confirmed'] is confirmed and event['horn_candidate'] is confirmed
    assert event['capture_candidate'] is confirmed
    assert event['state'] == ('CONFIRMED' if confirmed else 'IDLE')
    if confirmed:
        result = pipeline.capture(event, evidence, None, None, tmp_path, 0)
        assert result['stage'] == 'confirmed'
    else:
        assert evidence is None
