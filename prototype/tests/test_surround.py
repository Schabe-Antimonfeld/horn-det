from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from horn.config import Config
from horn.geometry import circular_array
from horn.surround import (AcousticEvidence, Association, Bearing, CameraModel, Detection, Rig, associate,
                           FrozenPanns, bearing_candidates, coverage_grid, direction_vectors, form_beam,
                           propagate_audio, route_cameras, select_bearing, spatial_response,
                           vehicle_box, wrap_angle)


@pytest.mark.parametrize("yaw", [-179, -90, 0, 45, 179])
def test_optical_axis_and_ground_roundtrip(yaw):
    camera = CameraModel("test", (1, -2, 6), yaw, 40, 90, 70)
    distance = 6 / np.tan(np.deg2rad(40))
    target = np.array([[1 + distance * np.cos(np.deg2rad(yaw)),
                        -2 + distance * np.sin(np.deg2rad(yaw)), 0]])
    pixels, valid = camera.project(target)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(pixels, [[640, 360]], atol=1e-10)
    ground, valid = camera.ground(pixels)
    np.testing.assert_allclose(ground, target, atol=1e-10)
    np.testing.assert_allclose(camera.basis @ camera.basis.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(camera.basis) == pytest.approx(1)


def test_downward_footprint_has_analytic_edges():
    camera = CameraModel("down", (0, 0, 6), 0, 90, 90, 60)
    ground, valid = camera.ground([[0, 360], [1280, 360], [640, 0], [640, 720]])
    assert valid.all()
    np.testing.assert_allclose(ground[:, :2], [[0, 6], [0, -6], [6 / np.sqrt(3), 0], [-6 / np.sqrt(3), 0]], atol=1e-12)


def test_upward_and_outside_rays_are_rejected():
    camera = CameraModel("test", (0, 0, 6), 0, 10, 90, 60)
    _, valid = camera.ground([[640, 0], [-1, 360], [640, 720]])
    assert valid.tolist() == [False, False, True]
    pixels, valid = camera.project([[-10, 0, 6]])
    assert not valid[0] and np.isnan(pixels).all()


@pytest.mark.parametrize("changes", [{"count": 0}, {"count": 2.5}, {"height": -1},
                                     {"pitch": 91}, {"hfov": 180}, {"image_height": 0},
                                     {"minimum_range": 50}, {"horn_height": 9}])
def test_rig_rejects_invalid_parameters(changes):
    with pytest.raises(ValueError):
        Rig(**changes)


def test_camera_spacing_and_height_scale():
    rig = Rig(count=4)
    assert [camera.yaw for camera in rig.cameras()[:-1]] == [0, 90, 180, 270]
    camera = rig.cameras()[0]
    points = np.array([[7, 1, 0], [8, -2, 0]])
    first, _ = camera.project(points)
    second, _ = replace(camera, center=(0, 0, 12)).project(points * 2)
    np.testing.assert_allclose(first, second)


def test_ring_count_does_not_guarantee_radial_coverage():
    rig = Rig(count=12, pitch=80, hfov=90, vfov=10)
    points, visible = coverage_grid(rig, step=2)
    assert not visible[np.linalg.norm(points[:, :2], axis=1) > 20].any()


def test_routing_wraps_at_the_seam_and_handles_overhead():
    rig = Rig()
    assert route_cameras(rig, 180) == route_cameras(rig, -180)
    assert "ring-3" in route_cameras(rig, 179)
    assert "down" in route_cameras(rig, 20)
    assert len(route_cameras(rig, None)) == rig.count + 1
    assert wrap_angle(-179 - 179) == 2


def detection_at(rig, center, identity="object", exposure=0.0, velocity=(0, 0), camera=None):
    cameras = rig.cameras() if camera is None else [camera]
    for selected in cameras:
        result = vehicle_box(selected, center)
        if result:
            box, foot = result
            return Detection(identity, selected.identity, box, foot, exposure, velocity)
    raise AssertionError("Test vehicle is outside camera coverage")


def measured_evidence(azimuth=0.0, depression=23.4, width=2.0):
    angles = np.arange(-180, 180, 2.0)
    depressions = np.arange(0, 91, 3.0)
    response = np.exp(-0.5 * (wrap_angle(angles[:, None] - azimuth) / 2) ** 2
                      -0.5 * ((depressions[None, :] - depression) / width) ** 2)
    return AcousticEvidence(Bearing(azimuth, depression, 1), (angles, depressions, response))


def test_known_ground_vehicle_matches_without_truth_identifier():
    rig = Rig()
    detection = detection_at(rig, (12, 0))
    result = associate(rig, 0, np.hypot(12, 5.2) / 343, [detection], evidence=measured_evidence())
    assert result["state"] == "unique"
    np.testing.assert_allclose(result["groups"][0][0]["position"], [12, 0], atol=1e-12)


def test_same_bearing_vehicles_remain_ambiguous():
    rig = Rig()
    result = associate(rig, 0, 0.1, [detection_at(rig, (12, 0), "a"), detection_at(rig, (24, 0), "b")],
                       Association(method="azimuth"))
    assert result["state"] == "ambiguous"
    assert len(result["groups"]) == 2


def test_overlap_observations_merge_by_position_not_identity():
    rig = Rig()
    center = (12 * np.cos(np.pi / 6), 12 * np.sin(np.pi / 6))
    observations = [detection_at(rig, center, f"independent-{index}", camera=camera)
                    for index, camera in enumerate(rig.cameras()[:2])]
    result = associate(rig, 30, 0.1, observations, evidence=measured_evidence(30))
    assert result["state"] == "unique"
    assert len(result["groups"][0]) == 2


def test_stale_and_out_of_range_observations_are_rejected():
    rig = Rig()
    stale = detection_at(rig, (12, 0), exposure=2)
    result = associate(rig, 0, 0, [stale])
    assert result["state"] == "unmatched"
    assert result["rejected"][0]["reason"] == "stale_frame"
    beyond = detection_at(rig, (45, 0))
    result = associate(rig, 0, 0.1, [beyond])
    assert result["rejected"][0]["reason"] == "outside_monitoring_range"


def test_motion_compensation_uses_sound_flight_and_exposure_time():
    rig = Rig()
    flight = np.hypot(12, 5.2) / 343
    observation = detection_at(rig, (12, 3), exposure=0.25, velocity=(0, 12))
    result = associate(rig, 0, flight, [observation], Association(angle_gate=2), evidence=measured_evidence())
    assert result["state"] == "unique"
    np.testing.assert_allclose(result["groups"][0][0]["position"], [12, 0], atol=1e-5)
    assert associate(rig, 0, flight, [replace(observation, velocity=(0, 0))], Association(angle_gate=2))["state"] == "unmatched"


def test_undefined_azimuth_never_claims_unique_vehicle():
    rig = Rig()
    assert associate(rig, None, 0.1, [detection_at(rig, (2, 0))], Association(method="azimuth"))["state"] == "ambiguous"


def test_silence_has_no_spatial_candidates():
    config = Config(microphones=14, diameter=0.24)
    grid = spatial_response(np.zeros((2048, 14)), circular_array(config), config, 6, 6)
    assert bearing_candidates(*grid) == []


def test_periodic_peaks_are_not_duplicated_at_180_degrees():
    angles = np.arange(-180, 180, 2)
    response = np.exp(-wrap_angle(angles - 179) ** 2 / 8)[:, None]
    candidates = bearing_candidates(angles, [0], response, count=3)
    assert len(candidates) == 1
    assert abs(wrap_angle(candidates[0].azimuth - 179)) <= 1


@pytest.mark.parametrize("azimuth,depression", [(-178, 30), (90, 45), (0, 60)])
def test_spherical_simulation_recovers_signed_elevated_bearing(azimuth, depression):
    config = Config(microphones=14, diameter=0.24)
    coordinates = circular_array(config)
    signal = np.random.default_rng(9).normal(size=8192)
    source = 100 * direction_vectors(azimuth, depression)
    audio, metadata = propagate_audio(signal, source, coordinates, config)
    selected = bearing_candidates(*spatial_response(audio, coordinates, config, 2, 3), count=1)[0]
    assert abs(wrap_angle(selected.azimuth - azimuth)) <= 2
    assert abs(selected.depression - depression) <= 3
    assert metadata["flight_seconds"] == pytest.approx(100 / 343)


def test_causal_beam_ignores_future_input_changes():
    config = Config(microphones=14, diameter=0.24)
    coordinates = circular_array(config)
    audio = np.random.default_rng(4).normal(size=(4096, 14))
    altered = audio.copy()
    altered[2000:] += 100
    bearing = Bearing(20, 30, 1)
    first, latency = form_beam(audio, coordinates, bearing, config)
    second, _ = form_beam(altered, coordinates, bearing, config)
    assert latency >= 32
    np.testing.assert_allclose(first[:2000], second[:2000], atol=1e-12)


def test_correct_beam_preserves_broadband_more_than_opposite_beam():
    config = Config(microphones=14, diameter=0.24)
    coordinates = circular_array(config)
    signal = np.random.default_rng(9).normal(size=16000)
    audio, _ = propagate_audio(signal, 100 * direction_vectors(30, 30), coordinates, config)
    wanted, _ = form_beam(audio, coordinates, Bearing(30, 30, 1), config)
    other, _ = form_beam(audio, coordinates, Bearing(-150, 30, 1), config)
    assert np.mean(wanted[200:-200] ** 2) > 5 * np.mean(other[200:-200] ** 2)


def test_score_rejection_and_positive_margin_are_explicit():
    candidates = [Bearing(0, 0, 1), Bearing(30, 0, 0.8)]
    assert not select_bearing(candidates, [0.1, 0.15])["valid"]
    assert select_bearing(candidates, [0.78, 0.79])["valid"]
    assert not select_bearing(candidates, [0.78, 0.79], margin=0.1)["valid"]
    assert select_bearing([], [])["index"] is None


@pytest.mark.parametrize("settings", [{"angle_gate": 0}, {"speed": 0}, {"maximum_skew": -1}])
def test_invalid_association_settings(settings):
    with pytest.raises(ValueError):
        Association(**settings)


def test_a_missing_horn_vehicle_can_produce_a_wrong_unique_match():
    rig = Rig()
    remaining_car = detection_at(rig, (24, 0), "nonhorn-car")
    result = associate(rig, 0, np.hypot(12, 5.2) / 343, [remaining_car], Association(method="azimuth"))
    assert result["state"] == "unique"
    assert result["groups"][0][0]["identity"] == "nonhorn-car"


def test_overhead_and_lateral_candidates_can_coexist():
    angles = np.arange(-180, 180, 2)
    response = np.zeros((len(angles), 3))
    response[:, 0] = np.exp(-wrap_angle(angles - 90) ** 2 / 100)
    response[:, 1] = 2 * np.exp(-wrap_angle(angles) ** 2 / 100)
    selected = bearing_candidates(angles, [0, 87, 90], response)
    assert len(selected) == 2
    assert selected[0].azimuth is None
    assert selected[1].azimuth == 90


def test_frozen_panns_batch_matches_independent_completed_windows():
    torch = pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "outputs" / "horn_detector_v1" / "panns-horn.pt"
    if not path.is_file():
        pytest.skip("Existing trained horn PANNs checkpoint is not installed")
    torch.set_num_threads(2)
    model = FrozenPanns(path)
    generator = np.random.default_rng(13)
    windows = [generator.normal(0, 0.1, 48000), generator.normal(0, 0.03, 48000)]
    together = model.score(windows)
    separately = [model.score([window])[0] for window in windows]
    np.testing.assert_allclose(together, separately, atol=1e-6, rtol=1e-5)
    assert all(0 <= score <= 1 for score in together)


def test_cli_profile_precedence_and_output_isolation(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"rig": {"count": 3, "height": 8, "maximum_range": 12},
                                   "camera_overrides": {"ring-0": {"yaw": 1, "center": [0.05, 0, 8]}}}), encoding="utf-8")
    output = tmp_path / "simulation"
    script = Path(__file__).resolve().parents[1] / "simulate_surround.py"
    command = [sys.executable, "-B", str(script), "--geometry-only", "--profile", str(profile),
               "--cameras", "4", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    config = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    assert config["rig"]["count"] == 4
    assert config["rig"]["height"] == 8
    assert config["cameras"][0]["yaw"] == 1
    assert config["cameras"][0]["center"] == [0.05, 0, 8]
    assert len(config["cameras"]) == 5
    assert (output / "coverage-grid.npz").is_file()
    assert not (output / "acoustic-report.json").exists()
    result = subprocess.run(command + ["--height", "10"], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode != 0
    assert "Configuration changed" in result.stderr
    assert json.loads((output / "configuration.json").read_text(encoding="utf-8"))["rig"]["height"] == 8


def test_missing_horn_vehicle_is_rejected_by_depression_and_spatial_methods():
    rig = Rig()
    remaining = detection_at(rig, (24, 0), "nonhorn")
    for method in ("depression", "spatial"):
        result = associate(rig, 0, 0.04, [remaining], Association(method=method), evidence=measured_evidence())
        assert result["state"] == "unmatched"
        assert result["source_unexplained"]
        assert result["rejected"][0]["reason"] == "depression_mismatch"


def test_default_never_substitutes_visual_uniqueness_for_missing_acoustic_evidence():
    rig = Rig()
    detection = detection_at(rig, (12, 0))
    result = associate(rig, 0, 0.04, [detection])
    assert result["state"] == "unmatched"
    assert result["rejected"][0]["reason"] == "missing_acoustic_evidence"
    result = associate(rig, 0, 0.04, [detection], evidence=AcousticEvidence(Bearing(0, 23.4, 1)))
    assert result["rejected"][0]["reason"] == "missing_spatial_response"


def test_depression_selects_near_car_without_using_its_identity():
    rig = Rig()
    near = detection_at(rig, (12, 0), "random-a")
    far = detection_at(rig, (24, 0), "random-b")
    result = associate(rig, 0, 0.04, [far, near], evidence=measured_evidence())
    assert result["state"] == "unique"
    assert result["groups"][0][0]["identity"] == "random-a"


def test_raw_spatial_response_can_reject_an_angle_compatible_vehicle():
    rig = Rig()
    radians = np.deg2rad(5)
    detection = detection_at(rig, (12 * np.cos(radians), 12 * np.sin(radians)))
    evidence = measured_evidence()
    directional = associate(rig, 0, 0.04, [detection], Association(method="depression"), evidence=evidence)
    spatial = associate(rig, 0, 0.04, [detection], evidence=evidence)
    assert directional["state"] == "unique"
    assert spatial["state"] == "unmatched"
    assert spatial["rejected"][0]["reason"] == "spatial_response_mismatch"


def test_wide_spatial_peak_preserves_ambiguity_even_for_one_visible_vehicle():
    rig = Rig()
    detection = detection_at(rig, (12, 0))
    result = associate(rig, 0, 0.04, [detection], evidence=measured_evidence(width=12))
    assert result["state"] == "ambiguous"
    assert result["groups"][0][0]["acoustic"]["ambiguous"]


def test_nonhorn_global_peak_does_not_override_the_horn_selected_reference():
    rig = Rig()
    evidence = measured_evidence()
    angles, depressions, response = evidence.grid
    response = response * 0.4
    response += np.exp(-0.5 * (wrap_angle(angles[:, None] - 90) / 2) ** 2
                       -0.5 * ((depressions[None, :] - 12) / 2) ** 2)
    evidence = AcousticEvidence(evidence.bearing, (angles, depressions, response))
    result = associate(rig, 0, 0.04, [detection_at(rig, (12, 0))], evidence=evidence)
    assert result["state"] == "unique"


def test_flat_spatial_evidence_and_mismatched_bearings_are_rejected():
    rig = Rig()
    evidence = measured_evidence()
    angles, depressions, response = evidence.grid
    flat = AcousticEvidence(evidence.bearing, (angles, depressions, np.ones_like(response)))
    result = associate(rig, 0, 0.04, [detection_at(rig, (12, 0))], evidence=flat)
    assert result["rejected"][0]["reason"] == "uninformative_spatial_response"
    with pytest.raises(ValueError, match="disagree"):
        associate(rig, 30, 0.04, [], evidence=evidence)


@pytest.mark.parametrize("changes", [{"method": "unknown"}, {"depression_gate": -1},
                                     {"height_error": -1}, {"response_fraction": 0}, {"maximum_width": 0}])
def test_invalid_acoustic_verification_settings(changes):
    with pytest.raises(ValueError):
        Association(**changes)
