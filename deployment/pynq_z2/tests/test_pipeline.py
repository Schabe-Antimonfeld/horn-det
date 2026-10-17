from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps.configuration import load_config
from ps.pipeline import Events, Pipeline
from ps.vision import decode_boxes


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
            return {'image': np.zeros((10, 10, 3)), 'jpeg': b'test-only-not-a-photo', 'time': 0}
    class Vision:
        def infer(self, image):
            return []
        def annotate(self, image, boxes, result, path):
            pass
    assert pipeline.capture(event, None, Cache(), Vision(), tmp_path, 0)['photos'] > 0
    assert pipeline.capture(event, None, Cache(), Vision(), tmp_path, 0) is None
