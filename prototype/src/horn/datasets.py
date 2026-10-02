"""Explicit source grouping and interval-aware short-window supervision."""

from dataclasses import dataclass, replace, asdict
import json
from pathlib import Path

import numpy as np

from .config import Config
from .calibration import remove_dc
from .features import mel_energy, log_mel, mfcc, statistics, PCEN
from .frontend import read_audio, stft, power


@dataclass(frozen=True)
class Recording:
    path: str
    group: str
    split: str
    label: int
    intervals: tuple = ()
    channel: int | None = None
    event: str | None = None
    origin: str | None = None


def validate_records(records):
    assignments, files = {}, {}
    for record in records:
        if not record.group or record.split not in ("train", "validation", "test", "unassigned") or record.label not in (0, 1):
            raise ValueError("Records require group, valid split and binary label")
        for identity in ("group:" + record.group, "event:" + (record.event or record.group),
                         "origin:" + (record.origin or record.group)):
            previous = assignments.setdefault(identity, record.split)
            if previous != record.split:
                raise ValueError("Related sources crossed dataset partitions")
        path = str(Path(record.path).resolve())
        if path in files:
            raise ValueError("Duplicate recording path")
        files[path] = record
        for start, end in record.intervals:
            if not np.isfinite([start, end]).all() or not 0 <= start < end:
                raise ValueError("Intervals must be finite and increasing")
        if record.label == 0 and record.intervals:
            raise ValueError("Negative recordings cannot have horn intervals")
    return records


def load_manifest(path):
    path = Path(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = json.loads(line)
        fields["path"] = str((path.parent / fields["path"]).resolve())
        fields["intervals"] = tuple(tuple(interval) for interval in fields.get("intervals", []))
        records.append(Recording(**fields))
    return validate_records(records)


def split_groups(records, seed=0, validation=0.2, test=0.2):
    """Split connected source groups before any resampling, windows or augmentation."""
    if not 0 < validation < 1 or not 0 < test < 1 or validation + test >= 1:
        raise ValueError("Invalid partition fractions")
    if any(record.split != "unassigned" for record in records):
        raise ValueError("Refuse to overwrite an existing split")
    links = {}
    for record in records:
        identifiers = ["g:" + record.group, "e:" + (record.event or record.group),
                       "o:" + (record.origin or record.group)]
        for identity in identifiers:
            links.setdefault(identity, set()).update(identifiers)
    components = []
    remaining = set(links)
    while remaining:
        stack = [min(remaining)]
        component = set()
        while stack:
            identity = stack.pop()
            if identity not in component:
                component.add(identity)
                stack.extend(links[identity] - component)
        remaining -= component
        components.append(component)
    if len(components) < 3:
        raise ValueError("Require at least three independent source groups")
    generator = np.random.default_rng(seed)
    generator.shuffle(components)
    validation_count = max(1, round(len(components) * validation))
    test_count = max(1, round(len(components) * test))
    if validation_count + test_count >= len(components):
        raise ValueError("Partition sizes leave no training groups")
    assignments = {}
    for index, component in enumerate(components):
        split = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
        assignments.update({identity: split for identity in component})
    result = [replace(record, split=assignments["g:" + record.group]) for record in records]
    return validate_records(result)


def window_label(record, start, end, minimum=0.02):
    if record.label == 0:
        return 0
    if not record.intervals:
        raise ValueError("Positive clips require exhaustive horn intervals for short-window labels")
    intersections = [max(0, min(end, upper) - max(start, lower)) for lower, upper in record.intervals]
    overlap = max(intersections)
    if overlap >= minimum:
        return 1
    return None if overlap > 0 else 0


def prepare(records, config=Config(), frontend="logmel", stride=1, encoder=None):
    validate_records(records)
    if frontend not in ("logmel", "pcen", "mfcc", "panns") or stride < 1:
        raise ValueError("Invalid frontend or context stride")
    if frontend == "panns" and encoder is None:
        raise ValueError("PANNs extraction requires the verified encoder")
    features, labels, groups, splits, powers, identities = [], [], [], [], [], []
    for record in records:
        audio = read_audio(record.path, config.sample_rate, record.channel)
        audio, _ = remove_dc(audio)
        if any(end > len(audio) / config.sample_rate for _, end in record.intervals):
            raise ValueError("Horn interval extends beyond its recording")
        if frontend == "panns" and audio.shape[1] != 1:
            raise ValueError("Select a fixed mono channel explicitly for PANNs")
        frames = stft(audio, config)
        if len(frames) < config.context:
            continue
        spectra = np.asarray([power(frame.spectrum) for frame in frames])
        energies = mel_energy(spectra, config)
        transformed = PCEN().feed(energies) if frontend == "pcen" else log_mel(energies, config.epsilon)
        for index in range(0, len(frames) - config.context + 1, stride):
            end = index + config.context
            label = window_label(record, frames[index].start / config.sample_rate,
                                 (frames[end - 1].end + 1) / config.sample_rate)
            if label is None:
                continue
            if frontend == "panns":
                segment = audio[frames[index].start:frames[end - 1].end + 1, 0]
                feature = encoder.frontend(segment, config.sample_rate)[0].numpy()
            else:
                feature = transformed[index:end]
                feature = statistics(mfcc(feature)) if frontend == "mfcc" else feature.T[None]
            features.append(feature)
            labels.append(label)
            groups.append(record.group)
            splits.append(record.split)
            powers.append(spectra[index:end].mean(axis=0))
            identities.append(f"{record.path}:{frames[index].start}:{frames[end - 1].end}")
    if not features:
        raise ValueError("No complete labelled contexts were produced")
    return {"features": np.asarray(features, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int64),
            "groups": np.asarray(groups), "splits": np.asarray(splits), "powers": np.asarray(powers),
            "frontend": np.asarray(frontend), "identities": np.asarray(identities),
            "config": np.asarray(json.dumps(asdict(config)))}
