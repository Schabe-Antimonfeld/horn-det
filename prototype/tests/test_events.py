import numpy as np
import pytest

from horn.camera import Camera, CameraPolicy, CameraSimulation
from horn.events import Decision, Direction, EventFusion, Fusion, Probability, State, decode, encode
from horn.localization import Estimate


def test_complete_event_and_cancelled_candidate_keep_ids():
    fusion = EventFusion(Fusion(beta=0, confirm=2, release=2, timeout=4))
    decisions = []
    for frame, (triggered, probability) in enumerate([(True, None), (False, 0.9), (False, 0.9),
                                                     (False, 0.1), (False, 0.1), (False, None),
                                                     (True, None), (False, 0.1), (False, 0.1)]):
        result = fusion.feed(frame, frame * 10667, triggered,
                             None if probability is None else Probability(frame, probability))
        decisions.append(result)
    assert [decision.state for decision in decisions] == [State.CANDIDATE, State.CANDIDATE, State.CONFIRMED,
                                                          State.CONFIRMED, State.END, State.IDLE,
                                                          State.CANDIDATE, State.CANDIDATE, State.END]
    assert [decision.event_id for decision in decisions] == [1, 1, 1, 1, 1, 0, 2, 2, 2]
    assert [decision.horn_valid for decision in decisions] == [False, False, True, True, False, False, False, False, False]


def test_cold_start_times_out_and_mismatched_results_are_rejected():
    fusion = EventFusion(Fusion(timeout=2))
    assert fusion.feed(0, 0, True).state == State.CANDIDATE
    assert fusion.feed(1, 100, False).state == State.CANDIDATE
    assert fusion.feed(2, 200, False).state == State.END
    with pytest.raises(ValueError):
        fusion.feed(3, 300, False, Probability(1, 0.8))
    with pytest.raises(ValueError):
        fusion.feed(2, 400, False)
    with pytest.raises(ValueError):
        fusion.feed(3, 300, False, direction=Direction(2, Estimate(20, True, 1)))


def test_wire_bytes_signed_angle_and_saturated_probability():
    decision = Decision(5, State.CONFIRMED, 1.0, Estimate(-20.5, True, 0.5), 123456, 42)
    message = encode(decision)
    assert len(message) == 19
    assert message[:3] == bytes([14, 255, 255])
    restored = decode(message, frame=5)
    assert restored.direction.angle == -20.5
    assert restored.probability == 65535 / 65536
    assert restored.direction.confidence == 0.5
    assert restored.timestamp == 123456 and restored.event_id == 42
    invalid = encode(Decision(0, State.IDLE, 0, Estimate(50, False, 1), 0, 0))
    assert invalid == bytes(19)
    corrupted = bytearray(invalid)
    corrupted[3] = 1
    with pytest.raises(ValueError):
        decode(corrupted)


def test_camera_deduplicates_and_rejects_unreachable_angles():
    policy = CameraPolicy(Camera(sign=-1, offset=10))
    direction = Estimate(30, True, 0.9)
    candidate = Decision(0, State.CANDIDATE, 0.4, direction, 1000, 1)
    assert policy.feed(candidate)[0].pan == -20
    confirmed = Decision(1, State.CONFIRMED, 0.9, direction, 2000, 1)
    actions = policy.feed(confirmed)
    assert [action.kind for action in actions] == ["turn", "capture"]
    assert [action.kind for action in policy.feed(confirmed)] == ["turn"]
    unreachable = Decision(2, State.CONFIRMED, 0.9, Estimate(-120, True, 1), 3000, 2)
    assert policy.feed(unreachable)[0].kind == "unreachable"
    simulation = CameraSimulation(speed=100)
    for action in actions:
        simulation.feed(action)
    simulation.advance(1)
    assert len(simulation.exposures) == 1
    assert simulation.exposures[0]["pan"] == -20
    assert simulation.exposures[0]["exposure_s"] == pytest.approx(0.222)


def test_camera_polling_does_not_postpone_exposure():
    from horn.camera import Action
    simulation = CameraSimulation(speed=100, exposure_delay=0.02)
    simulation.feed(Action("capture", 1, 0, 1))
    for time in np.arange(0.001, 0.04, 0.001):
        simulation.advance(float(time))
    assert len(simulation.exposures) == 1
    assert simulation.exposures[0]["exposure_s"] == pytest.approx(0.03)
