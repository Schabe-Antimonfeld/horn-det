import importlib.util
from pathlib import Path

import numpy as np
import pytest
import wave


def experiment():
    path = Path(__file__).resolve().parents[1] / 'improve_panns.py'
    spec = importlib.util.spec_from_file_location('panns_rejection_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(label, domain='HornBase', split='validation'):
    return dict(label=label, angle=0, dataset=domain, split=split,
                bearings=[dict(azimuth=0, depression=20, response=2), dict(azimuth=90, depression=20, response=1)])


def test_direction_ambiguity_never_hides_a_negative_horn_event():
    module = experiment()
    result = module.outcomes([case(0)], [[0.9, 0.89]], 0.5, 0.2)
    assert result['accepted_negative'] == result['fp'] == result['ambiguous_negative'] == 1
    assert result['tn'] == 0 and result['negative_acceptance'] == 1


def test_presence_and_correct_direction_have_different_denominators():
    module = experiment()
    result = module.outcomes([case(1), case(1), case(0)], [[0.9, 0.1], [0.1, 0.9], [0.1, 0.2]], 0.5, 0)
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 1)
    assert result['accepted_positive'] == 2 and result['detection_recall'] == 1
    assert result['recall'] == 0.5


def test_gate_selection_constrains_presence_in_each_source_domain():
    module = experiment()
    cases = [case(1), case(0), case(1, 'ESC50')] + [case(0, 'ESC50') for _ in range(20)]
    values = np.array([[0.7, 0.1], [0.8, 0.79], [0.96, 0.1]] + [[0.1, 0.1] for _ in range(20)])
    chosen = module.choose_gate(cases, values)
    assert chosen['constraint_met'] and chosen['threshold'] > 0.8
    assert chosen['metrics']['tp'] == 1 and chosen['worst_domain_acceptance'] == 0
    with pytest.raises(ValueError, match='validation'):
        module.choose_gate([dict(item, split='test') for item in cases], values)


def test_hard_mining_uses_only_training_negatives_and_preserves_positives():
    module = experiment()
    cases = [case(label, domain, 'train') for domain in ('HornBase', 'ESC50') for label in (0, 0, 0, 0, 1)]
    values = np.tile([0.1, 0.2, 0.3, 0.9, 0.99], 2)
    weights = module.hard_weights(cases, values, 'train')
    np.testing.assert_array_equal(np.flatnonzero(weights > 1), [3, 8])
    assert weights[4] == weights[9] == 1
    with pytest.raises(ValueError, match='train'):
        module.hard_weights([dict(item, split='validation') for item in cases], values, 'validation')


def test_multiclass_initialization_preserves_original_binary_probability():
    torch = pytest.importorskip('torch')
    module = experiment()
    torch.manual_seed(31)
    original = torch.nn.Linear(512, 2).eval()
    values = torch.randn(8, 512)
    with torch.inference_mode():
        expected = torch.softmax(original(values), dim=-1)[:, 1].numpy()
    for count in (2, 12):
        head = module.initial_head(original, count)
        np.testing.assert_allclose(module.scores(head, values), expected, atol=2e-7)


def test_invalid_candidate_scores_are_rejected_before_evaluation():
    module = experiment()
    for values in ([[0.3]], [[np.nan, 0.5]], [[1.1, 0.1]]):
        with pytest.raises(ValueError, match='finite candidate'):
            module.outcomes([case(1)], values, 0.5, 0)


def test_file_prediction_applies_dc_once_and_accepts_conditioned_beams(monkeypatch, tmp_path):
    module = experiment()
    received = []
    class Model:
        gate = dict(threshold=0.5)
        def score(self, values):
            received.extend(values)
            return [0.1] * len(values)
    monkeypatch.setattr(module, 'RefinedPanns', lambda method: Model())
    path = tmp_path / 'constant.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(np.full(48000, 3276, dtype='<i2').tobytes())
    module.predict(path, 'hard')
    module.predict(path, 'hard', conditioned=True)
    level = 3276 / 32768
    np.testing.assert_allclose(received[0], level * 0.995 ** np.arange(48000), atol=1e-13)
    np.testing.assert_array_equal(received[1], np.full(48000, level))
