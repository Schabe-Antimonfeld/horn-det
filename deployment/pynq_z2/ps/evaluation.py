"""Recompute fixed-weight metrics and verify versioned replay evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ps.detector import Detector


def verify_assets(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'assets.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 1 or manifest.get('model_tag') != 405187002:
        raise ValueError('Unsupported algorithm assets')
    for name, checksum in manifest['files'].items():
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError('Asset path escapes the package: ' + name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('Asset integrity failure: ' + name)
    return manifest


def compare(expected, actual, location='result'):
    """Compare frozen decisions exactly and numeric evidence within a fixed tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) - set(actual):
            raise ValueError('Missing evidence at ' + location)
        for key, value in expected.items():
            compare(value, actual[key], location + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError('Evidence length changed at ' + location)
        for index, (first, second) in enumerate(zip(expected, actual)):
            compare(first, second, location + '[' + str(index) + ']')
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not np.isclose(expected, actual, atol=2e-10, rtol=2e-10):
            raise ValueError('Numeric evidence changed at ' + location)
    elif expected != actual:
        raise ValueError('Decision changed at ' + location)


def outcomes(cases, probabilities, threshold, margin):
    """Keep horn detection and the joint horn/direction metric distinct."""
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(cases), 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Each scene needs two finite candidate probabilities')
    labels = np.asarray([bool(case['label']) for case in cases])
    chosen = values.argmax(axis=1)
    audible = values.max(axis=1) >= threshold
    difference = np.abs(values[:, 0] - values[:, 1])
    valid = audible & (difference >= margin) & (difference > 0)
    correct = np.asarray([[item['azimuth'] is not None and
                           abs((item['azimuth'] - case['angle'] + 180) % 360 - 180) < 10
                           for item in case['bearings']] for case in cases])
    hits = correct[np.arange(len(cases)), chosen]
    tp = int(np.sum(labels & valid & hits))
    wrong = int(np.sum(labels & valid & ~hits))
    negative = int(np.sum(~labels & audible))
    fn, fp = int(labels.sum()) - tp, wrong + negative
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    detected = int(np.sum(labels & audible))
    return dict(tp=tp, fp=fp, fn=fn, tn=int(np.sum(~labels & ~audible)), wrong_direction=wrong,
                positives=int(labels.sum()), negatives=int((~labels).sum()), accepted_negative=negative,
                negative_acceptance=negative / max(1, int((~labels).sum())), accepted_positive=detected,
                ambiguous_positive=int(np.sum(labels & audible & ~valid)),
                ambiguous_negative=int(np.sum(~labels & audible & ~valid)),
                candidate_hits=int(np.sum(labels & correct.any(axis=1))), precision=precision, recall=recall,
                f1=2 * precision * recall / max(1e-12, precision + recall),
                detection_precision=detected / max(1, detected + negative),
                detection_recall=detected / max(1, int(labels.sum())),
                detection_f1=2 * detected / max(1, detected + negative + int(labels.sum())))


def evaluate(root):
    root = Path(root)
    verify_assets(root)
    detector = Detector(root / 'model')
    expected = json.loads((root / 'evaluation/expected.json').read_text(encoding='utf-8'))
    from ps.configuration import load_config
    config = load_config(root / 'config/default.json')
    results, maximum = {}, 0.0
    for name in sorted(expected):
        cases = json.loads((root / ('evaluation/' + name + '.json')).read_text(encoding='utf-8'))
        with np.load(str(root / ('evaluation/' + name + '.npz')), allow_pickle=False) as archive:
            inputs, reference = archive['inputs'], archive['scores']
        if inputs.shape != (len(cases), 2, 640) or not np.isfinite(inputs).all():
            raise ValueError('Invalid evaluation feature dimensions')
        actual = np.asarray([[detector.score_inputs(vector)['score'] for vector in pair] for pair in inputs])
        np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=0)
        maximum = max(maximum, float(np.max(abs(actual - reference))))
        metric = outcomes(cases, actual, 0.95, 0.0)
        compare(expected[name], metric, name)
        historical = metric
        metric = outcomes(cases, actual, config['detector']['threshold'], config['detector']['margin'])
        capture = actual.max(axis=1) >= 0.9
        labels = np.asarray([bool(case['label']) for case in cases])
        results[name] = dict(scenes=len(cases), joint=metric, historical_joint=historical,
                             capture=dict(positive=int((capture & labels).sum()), negative=int((capture & ~labels).sum())))
    return dict(result='passed', model_tag=detector.metadata['model_tag'], protocols=results,
                scenes=sum(row['scenes'] for row in results.values()), maximum_score_error=maximum,
                operating_point=dict(threshold=config['detector']['threshold'], confirm=config['event']['confirm']),
                historical_threshold=0.95, temporal_confirmation_evaluated=False,
                scope='Current single-scene metrics and separately verified historical 0.95 metrics from frozen SRP/beam temporal features; replay examples separately cover PCM through events; no new road or hardware evidence')
