import numpy as np
import pytest

from horn.calibration import StreamCalibration
from horn.config import Config
from horn.evaluation import Event, event_metrics, localization_metrics, window_metrics
from horn.events import EventFusion, Fusion, State
from horn.pipeline import Detector, Pipeline
from horn.simulation import far_field


def test_causal_calibration_aligns_and_matches_block_partitions():
    audio = np.zeros((512, 2))
    audio[100, 0] = 2
    audio[103, 1] = 0.5
    expected = StreamCalibration([2, 0.5], [0, 3]).feed(audio)
    stream = StreamCalibration([2, 0.5], [0, 3])
    actual = np.vstack([stream.feed(block) for block in np.array_split(audio, [77, 121, 360])])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual[:, 0], actual[:, 1], atol=1e-14)
    assert np.argmax(actual[:, 0]) == 135


def test_invalid_cases_remain_in_success_denominator():
    result = localization_metrics([0, 30, 40], [1, 35, 0], [True, True, False])
    assert result["mae_deg"] == 3
    assert result["sr10"] == pytest.approx(2 / 3)
    assert localization_metrics([], [], [])["mae_deg"] is None
    assert window_metrics([], [])["f1"] is None


def test_one_to_one_event_matching_counts_duplicates_and_misses():
    labels = [Event(1, 1.2), Event(2, 2.5), Event(4, 4.2)]
    predictions = [Event(1.05, 1.3, 1.1, 1), Event(1.07, 1.4, 1.15, 2), Event(2.1, 2.6, 2.2, 3),
                   Event(6, 6.2, None, 4)]
    metrics = event_metrics(labels, predictions, duration=10)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 1)
    assert metrics["false_positive_per_hour"] == 360
    assert metrics["nonhorn_duration_s"] == pytest.approx(9.1)
    assert metrics["short_horn_recall"] == 0.5


def test_matching_maximizes_cardinality_before_onset_distance():
    labels = [Event(1, 1.5), Event(1.4, 1.8)]
    predictions = [Event(1.3, 1.7, 1.3), Event(0.9, 1.3, 0.9)]
    assert event_metrics(labels, predictions, tolerance=0.4)["tp"] == 2


def test_pipeline_stream_partitioning_and_horn_confirmation():
    torch = pytest.importorskip("torch")
    from horn.models import TinyCNN
    torch.set_num_threads(2)
    model = TinyCNN().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[1] = 10
    scene = far_field(np.random.default_rng(8).normal(0, 0.1, 20000), 30)
    fusion = Fusion(beta=0, confirm=2)
    whole = Pipeline(detector=Detector(model), mode="srp", fusion=EventFusion(fusion), association_verified=True)
    expected = whole.feed(scene.audio)
    stream = Pipeline(detector=Detector(model), mode="srp", fusion=EventFusion(fusion), association_verified=True)
    actual = []
    for block in np.array_split(scene.audio, [73, 800, 2100, 9000, 15000]):
        actual.extend(stream.feed(block))
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        assert left.decision == right.decision
    confirmed = [output for output in actual if output.decision.horn_valid]
    assert confirmed
    assert confirmed[0].decision.frame >= 24
    assert all(abs(output.decision.direction.angle - 30) <= 1 for output in confirmed)
    assert sum(action.kind == "capture" for output in actual for action in output.actions) <= 1


def test_aware_pipeline_without_background_never_claims_valid_doa():
    scene = far_field(np.random.default_rng(3).normal(size=4096), -20)
    outputs = Pipeline(association_verified=True).feed(scene.audio)
    assert not any(output.decision.direction.valid for output in outputs)


def test_numerical_direction_does_not_automatically_prove_horn_association():
    scene = far_field(np.random.default_rng(3).normal(size=4096), -20)
    outputs = Pipeline(mode="srp").feed(scene.audio)
    assert any(output.numerical_direction.valid for output in outputs)
    assert not any(output.decision.direction.valid for output in outputs)
    assert not any(action.kind in ("turn", "capture") for output in outputs for action in output.actions)


@pytest.mark.parametrize("frontend", ["mfcc", "logmel", "pcen"])
def test_training_and_streaming_features_match_with_dc_offset(tmp_path, frontend):
    import soundfile as sf
    from horn.datasets import Recording, prepare

    class Collector:
        def __init__(self):
            self.features = []

        def probability(self, feature):
            self.features.append(feature[0])
            return np.zeros(1)

    config = Config(microphones=2)
    generator = np.random.default_rng(88)
    audio = generator.normal(0, 0.02, size=(20000, 2)) + [0.5, -0.3]
    path = tmp_path / "offset.wav"
    sf.write(path, audio, config.sample_rate, subtype="DOUBLE")
    dataset = prepare([Recording(str(path), "one", "train", 0)], config, frontend)
    model = Collector()
    Pipeline(config, Detector(model, config, frontend), mode="detect").feed(audio)
    np.testing.assert_allclose(np.asarray(model.features), dataset["features"], atol=2e-6, rtol=1e-6)


def test_matched_benchmark_contains_every_sample_for_every_route():
    from horn.benchmark import run_benchmark
    report = run_benchmark(quick=True, seed=89)
    assert len(report["samples"]) == 12
    for predictions in report["predictions"].values():
        assert [item["sample_id"] for item in predictions] == list(range(12))
    assert report["fpga_resources"] is None
