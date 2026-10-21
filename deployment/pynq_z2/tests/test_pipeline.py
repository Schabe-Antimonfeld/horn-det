from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps.configuration import load_config
from ps.pipeline import Events, Pipeline
from ps.vision import decode_boxes
from ps.geometry import Bearing


def test_event_confirmation_and_skipped_windows_reset_evidence():
    events = Events({'confirm': 2, 'release': 2}, 100)
    assert events.feed(True, {'epoch': 0, 'end': 100})['state'] == 'CANDIDATE'
    assert events.feed(True, {'epoch': 0, 'end': 200})['state'] == 'CONFIRMED'
    record = events.feed(True, {'epoch': 0, 'end': 400})
    assert record['state'] == 'CANDIDATE' and record['history_reset']
    assert record['ended_event'] == 1 and record['event_id'] == 2
    assert events.feed(False, {'epoch': 1, 'end': 500})['state'] == 'IDLE'


def test_decoded_vehicle_boxes_filter_classes_and_suppress_duplicates():
    config = load_config()['vision']
    rows = np.zeros((3, 85))
    rows[:, :5] = [0.5, 0.5, 0.2, 0.2, 0.9]
    rows[0, 7], rows[1, 7], rows[2, 5] = 0.9, 0.8, 0.99
    results = decode_boxes(rows, 1000, 500, config)
    assert len(results) == 1 and results[0]['class'] == 2
    np.testing.assert_allclose(results[0]['box'], [400, 200, 600, 300])


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
    event = {'state': 'CONFIRMED', 'horn_candidate': True, 'event_id': 1, 'azimuth': 0}
    assert pipeline.capture(event, None, None, None, tmp_path, 0)['photos'] == 0
    assert pipeline.captured == 0
    class Cache:
        def nearest(self, identity, timestamp):
            return {'image': np.zeros((1080, 1920, 3), dtype=np.uint8), 'jpeg': b'test-only-not-a-photo', 'time': 0}
    class Vision:
        def infer(self, image):
            return []
        def annotate(self, image, boxes, result, path):
            pass
    assert pipeline.capture(event, None, Cache(), Vision(), tmp_path, 0)['photos'] > 0
    assert pipeline.capture(event, None, Cache(), Vision(), tmp_path, 0) is None


def test_outside_wide_camera_view_does_not_assign_visible_vehicles(tmp_path):
    pipeline = Pipeline(load_config(), None)
    class Cache:
        def nearest(self, identity, timestamp):
            raise AssertionError('Rear source must not request an unrelated camera frame')
    event = {'state': 'CONFIRMED', 'horn_candidate': True, 'event_id': 1, 'azimuth': 180}
    result = pipeline.capture(event, None, Cache(), None, tmp_path, 0)
    assert result['visibility'] == 'outside_camera_view' and result['photos'] == 0
    assert result['association']['source_unexplained'] and not result['association']['groups']


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
    assert len(result['association']['acoustic_hypotheses']) == 2
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


def test_hypothesis_union_never_rejects_a_vehicle_supported_by_another(monkeypatch, tmp_path):
    import ps.pipeline as module
    comparisons = iter([
        dict(state='unmatched', groups=[], rejected=[dict(identity='car-a', reason='angle_mismatch')], source_unexplained=True),
        dict(state='unique', groups=[[dict(identity='car-a')]], rejected=[], source_unexplained=False)])
    monkeypatch.setattr(module, 'associate', lambda *args, **kwargs: next(comparisons))
    pipeline = Pipeline(load_config(), None)
    event = dict(state='CONFIRMED', horn_candidate=True, event_id=1, azimuth=None,
                 direction_candidates=[dict(azimuth=0), dict(azimuth=180)])
    result = pipeline.capture(event, None, None, None, tmp_path, 0)['association']
    assert result['state'] == 'ambiguous' and result['groups'] == [[dict(identity='car-a')]]
    assert result['rejected'] == [] and result['source_unexplained']


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


def test_low_score_capture_retains_both_directions_but_never_confirms_horn_or_vehicle(monkeypatch, tmp_path):
    pipeline = capture_pipeline(monkeypatch, [0.6, 0.55])
    snapshot = dict(audio=np.zeros((48000, 14)), first=0, end=48000, epoch=0, clipped=False)
    event, evidence = pipeline.analyze(snapshot)
    assert event['state'] == 'CANDIDATE' and event['capture_candidate']
    assert not event['horn_candidate'] and not event['doa_valid'] and event['azimuth'] is None
    assert event['acoustic_state'] == 'capture_only' and event['event_basis'] == 'capture_candidate'
    assert event['direction_candidates'] == [] and len(event['capture_candidates']) == 2
    assert pipeline.capture(event, evidence, None, None, tmp_path, 0) is None
    snapshot['end'] += pipeline.config['audio']['update_samples']
    event, evidence = pipeline.analyze(snapshot)
    assert event['state'] == 'CONFIRMED' and not event['horn_candidate']
    class Cache:
        def nearest(self, identity, timestamp):
            return dict(image=np.zeros((1080, 1920, 3), dtype=np.uint8), jpeg=b'test-only', time=0)
    class Vision:
        def infer(self, image):
            return []
        def annotate(self, image, boxes, result, path):
            assert result['state'] != 'unique' and result['evidence_level'] == 'capture_only'
    result = pipeline.capture(event, evidence, Cache(), Vision(), tmp_path, 0)
    assert result['selected_cameras'] == ['wide-0'] and result['photos'] == 1
    assert result['visibility'] == 'partly_outside_camera_view'
    assert result['association']['evidence_level'] == 'capture_only'
    assert 'horn presence unconfirmed' in result['association']['verification']
    assert pipeline.capture(event, evidence, Cache(), Vision(), tmp_path, 0) is None
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


def test_capture_only_cannot_promote_a_unique_geometric_match(monkeypatch, tmp_path):
    import ps.pipeline as module
    monkeypatch.setattr(module, 'associate', lambda *args, **kwargs:
                        dict(state='unique', groups=[[dict(identity='car-a')]], rejected=[], source_unexplained=False))
    pipeline = Pipeline(load_config(), None)
    event = dict(state='CONFIRMED', horn_candidate=False, capture_candidate=True, event_id=1,
                 azimuth=None, doa_valid=False, capture_candidates=[dict(azimuth=0)])
    result = pipeline.capture(event, None, None, None, tmp_path, 0)['association']
    assert result['state'] == 'ambiguous' and result['groups'] == [[dict(identity='car-a')]]
    assert result['evidence_level'] == 'capture_only' and 'horn presence unconfirmed' in result['verification']
