"""Frozen paired protocols, PCM contracts and validation-only horn gate selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .surround import Bearing, Rig, select_bearing, vehicle_box, wrap_angle


ROOT = Path(__file__).resolve().parents[3]
POSITIONS = ((-179, 60), (179, -60), (-90, 40), (0, 120), (90, -20), (45, -135))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def camera_rig():
    record = json.loads((ROOT / "prototype/profiles/gc2093_single.json").read_text(encoding="utf-8"))
    return Rig(**record["rig"])


def visible_target(rig, angle, radius):
    point = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    for camera in rig.cameras():
        _, visible = camera.project([[*point, 0]])
        if visible[0] and vehicle_box(camera, point) is not None:
            return True
    return False


def source_pool(records, split):
    positive = [item for item in records if item["split"] == split and item["label"] == 1]
    selected = []
    scenarios = sorted({item["scenario"] for item in positive if item["dataset"] == "HornBase"})
    for scenario in scenarios:
        for style in ("S", "L", "V"):
            choices = [item for item in positive if item["dataset"] == "HornBase" and item["scenario"] == scenario
                       and item["file"].endswith("_" + style + ".wav")]
            selected.append(min(choices, key=lambda item: item["file"]))
    selected += sorted([item for item in positive if item["dataset"] == "ESC50"], key=lambda item: item["file"])
    return selected


def scene_plan(records, split, protocol, seed=20261021):
    """Pair source/SIR/level across protocols; visibility never uses model scores."""
    if split not in ("train", "validation", "test") or protocol not in ("original", "visible"):
        raise ValueError("Unknown source split or protocol")
    rig, cases = camera_rig(), []
    for source_index, target in enumerate(source_pool(records, split)):
        for position, (angle, other) in enumerate(POSITIONS):
            radius = (8.0, 15.0, 30.0)[(source_index + position) % 3]
            rotation = 0.0
            if protocol == "visible":
                candidates = [-50, -30, -10, 10, 30, 50]
                chosen = [value for value in candidates if visible_target(rig, value, radius)]
                if not chosen:
                    raise ValueError("No valid camera-visible target position")
                rotation = chosen[position % len(chosen)] - angle
                angle, other = float(wrap_angle(angle + rotation)), float(wrap_angle(other + rotation))
            category = ("engine", "siren", "train", "church_bells")[(source_index + position) % 4]
            interferers = sorted([item for item in records if item["split"] == split and item["category"] == category
                                  and item["label"] == 0], key=lambda item: item["file"])
            interferer = interferers[(source_index * 3 + position) % len(interferers)]
            negatives = sorted([item for item in records if item["split"] == split and item["dataset"] == target["dataset"]
                                and item["label"] == 0 and item["file"] != interferer["file"]], key=lambda item: item["file"])
            for label in (1, 0):
                identifier = len(cases)
                selected = target if label else negatives[(source_index * 7 + position) % len(negatives)]
                cases.append(dict(id=identifier, pair=source_index * len(POSITIONS) + position, split=split,
                                  protocol=protocol, target=selected["file"], interferer=interferer["file"],
                                  dataset=selected["dataset"], label=label, angle=angle, other=other, radius=radius,
                                  sir_db=(-5.0, 0.0, 5.0)[(source_index + position) % 3],
                                  target_rms=(0.025, 0.075, 0.15)[(source_index // 3 + position) % 3],
                                  noise_seed=seed + identifier, visual_seed=seed + 100000 + identifier,
                                  rotation=rotation, occluded=False))
    return cases


def pcm_roundtrip(signal, interference, noise, target_rms, headroom=0.9):
    """Apply one scene gain to every channel, then encode/decode signed PCM16."""
    arrays = [np.asarray(value, dtype=float) for value in (signal, interference, noise)]
    if any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays) or arrays[0].ndim != 2:
        raise ValueError("PCM components must be equally shaped finite channel arrays")
    if not len(arrays[0]) or not 0 < headroom < 1 or not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("Invalid PCM target level or headroom")
    rms = float(np.sqrt(np.mean(arrays[0] ** 2)))
    mixture = sum(arrays)
    peak = float(np.max(np.abs(mixture)))
    if rms <= 0 or peak <= 0:
        raise ValueError("Cannot assign a physical level to a silent source")
    requested = target_rms / rms
    gain = min(requested, headroom / peak)
    scaled = mixture * gain
    codes = (np.sign(scaled) * np.floor(np.abs(scaled) * 32768 + 0.5)).astype(np.int64)
    clips = int(np.count_nonzero((codes < -32768) | (codes > 32767)))
    if clips:
        raise ValueError("PCM headroom contract failed")
    pcm = codes.astype(np.int16)
    metadata = dict(gain=gain, requested_gain=requested, headroom_limited=gain < requested,
                    target_rms_requested=target_rms, target_rms_actual=rms * gain,
                    peak=float(np.max(np.abs(pcm.astype(float) / 32768))),
                    mixture_rms=float(np.sqrt(np.mean((pcm.astype(float) / 32768) ** 2))),
                    clipped_samples=clips, pcm_sha256=hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest())
    return pcm.astype(float) / 32768, metadata


def joint_metrics(cases, scores, threshold, margin):
    if len(cases) != len(scores):
        raise ValueError("Each scene must have one candidate score list")
    counts = dict(tp=0, fp=0, fn=0, tn=0, positives=0, negatives=0,
                  accepted_positive=0, wrong_direction=0, accepted_negative=0, candidate_hits=0)
    for case, values in zip(cases, scores):
        bearings = [Bearing(**value) for value in case["bearings"]]
        decision = select_bearing(bearings, values, threshold, margin)
        positive = bool(case["label"])
        counts["positives" if positive else "negatives"] += 1
        if positive:
            counts["candidate_hits"] += int(any(item.azimuth is not None and abs(wrap_angle(item.azimuth - case["angle"])) < 10 for item in bearings))
        bearing = bearings[decision["index"]] if decision["index"] is not None else None
        correct = bool(positive and decision["valid"] and bearing.azimuth is not None
                       and abs(wrap_angle(bearing.azimuth - case["angle"])) < 10)
        if positive:
            counts["accepted_positive"] += int(decision["valid"])
            counts["tp"] += int(correct)
            counts["fn"] += int(not correct)
            counts["wrong_direction"] += int(decision["valid"] and not correct)
            counts["fp"] += int(decision["valid"] and not correct)
        else:
            counts["accepted_negative"] += int(decision["valid"])
            counts["fp"] += int(decision["valid"])
            counts["tn"] += int(not decision["valid"])
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return dict(counts, precision=precision, recall=recall, f1=2 * precision * recall / max(1e-12, precision + recall),
                negative_acceptance=counts["accepted_negative"] / max(1, counts["negatives"]))


def choose_gate(cases, scores, split):
    """Choose a shared operating point using validation cases only."""
    if split != "validation" or any(case["split"] != "validation" for case in cases):
        raise ValueError("Operating points may only use validation sources")
    if not cases or len(cases) != len(scores) or {case['dataset'] for case in cases} != {'HornBase', 'ESC50'}:
        raise ValueError("Gate selection needs paired scores from both validation domains")
    choices = []
    for threshold in np.linspace(0.05, 0.95, 19):
        for margin in (0.0, 0.05, 0.1, 0.2):
            domains = []
            for dataset in ("HornBase", "ESC50"):
                indices = [index for index, case in enumerate(cases) if case["dataset"] == dataset]
                domains.append(joint_metrics([cases[index] for index in indices], [scores[index] for index in indices], threshold, margin))
            metrics = joint_metrics(cases, scores, threshold, margin)
            rank = (metrics["negative_acceptance"] <= 0.1, np.mean([value["f1"] for value in domains]),
                    -metrics["negative_acceptance"], metrics["f1"], threshold, margin)
            choices.append((rank, dict(threshold=float(threshold), margin=margin, metrics=metrics,
                                       macro_f1=float(np.mean([value["f1"] for value in domains])),
                                       constraint_met=metrics["negative_acceptance"] <= 0.1)))
    return max(choices, key=lambda value: value[0])[1]
