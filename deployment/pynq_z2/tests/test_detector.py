from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from ps.detector import Detector, temporal_features, validate_backend
from ps.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_model_scores_match_frozen_inputs_and_context_statistics():
    detector = Detector(ROOT / 'model')
    assert detector.metadata['model_tag'] == 405187002
    assert detector.metadata['architecture'] == 'BandTemporalMLP-30866'
    values = np.zeros((7, 40, 24))
    assert temporal_features(values).shape == (640,)
    assert detector.score(values) == detector.score_inputs(temporal_features(values))
    with np.load(ROOT / 'evaluation/test-original.npz', allow_pickle=False) as archive:
        for index in (0, 40, 100, 167):
            for vector, expected in zip(archive['inputs'][index], archive['scores'][index]):
                assert detector.score_inputs(vector)['score'] == pytest.approx(expected, abs=2e-10)


def test_temporal_order_and_gain_centering_are_preserved():
    contexts = np.random.RandomState(8).normal(size=(7, 40, 24))
    np.testing.assert_allclose(temporal_features(contexts), temporal_features(contexts + 8), atol=1e-13)
    assert not np.allclose(temporal_features(contexts), temporal_features(contexts[::-1]))
    with pytest.raises(ValueError):
        temporal_features(contexts[:6])


@pytest.mark.parametrize('change', ['weights', 'shape', 'architecture'])
def test_model_integrity_rejects_changed_weights_and_contract(tmp_path, change):
    shutil.copytree(ROOT / 'model', tmp_path / 'model')
    model = tmp_path / 'model'
    if change == 'weights':
        with (model / 'weights.npz').open('ab') as stream:
            stream.write(b'changed')
    else:
        path = model / 'manifest.json'
        value = json.loads(path.read_text())
        value['input_shape' if change == 'shape' else 'architecture'] = [24, 40] if change == 'shape' else 'OldModel'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        Detector(model)


def test_model_tag_mismatch_is_rejected():
    config = load_config()
    config['detector']['model_tag'] += 1
    with pytest.raises(ValueError, match='model tag'):
        validate_backend(config, Detector(ROOT / 'model').metadata)
