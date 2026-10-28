"""Load temporal band models without changing the frozen static reference."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ps.detector import SpectralDetector, validate_model as validate_legacy
from ps.dsp import band_features


def temporal_features(contexts):
    """Retain ordered thirds, relative log-energy envelopes and feature peaks."""
    values = np.asarray(contexts, dtype=float)
    if values.shape != (7, 40, 24) or not np.isfinite(values).all():
        raise ValueError('Temporal inference requires seven finite 40x24 contexts')
    statistics = band_features(values)
    energy = values.mean(axis=(-2, -1))
    statistics[..., :40] += (energy - energy.mean())[..., None]
    segments = np.array_split(np.arange(7), 3)
    pooled = [statistics[segment].mean(axis=0) for segment in segments]
    return np.concatenate(pooled + [statistics.max(axis=0)])


class TemporalDetector:
    """Score a completed one-second beam using the 30866-parameter NumPy MLP."""

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (self.metadata.get('format') != 1 or self.metadata.get('architecture') != 'BandTemporalMLP-30866'
                or self.metadata.get('feature_mode') != 'temporal' or self.metadata.get('contexts') != 7
                or self.metadata.get('input_shape') != [40, 24]
                or self.metadata.get('provenance', {}).get('frontend') != 'logmel'):
            raise ValueError('Unsupported temporal model or frontend contract')
        checksum = hashlib.sha256((directory / 'weights.npz').read_bytes()).hexdigest()
        if self.metadata.get('sha256') != {'weights.npz': checksum} or self.metadata.get('model_tag') != int(checksum[:8], 16):
            raise ValueError('Temporal model checksum mismatch')
        with np.load(str(directory / 'weights.npz'), allow_pickle=False) as archive:
            self.weights = {name: archive[name] for name in archive.files}
        expected = dict(weight0=(48, 640), bias0=(48,), weight1=(2, 48), bias1=(2,))
        if set(self.weights) != set(expected) or any(self.weights[name].shape != shape
                or self.weights[name].dtype.kind not in 'fiu' or not np.isfinite(self.weights[name]).all()
                for name, shape in expected.items()):
            raise ValueError('Invalid temporal weights')
        self.frontend = 'logmel'

    def score(self, contexts):
        if len(contexts) == 0:
            return dict(score=0.0, clipped=False, contexts=0, backend='spectral-temporal-numpy')
        inputs = temporal_features(contexts)
        hidden = np.maximum(0, inputs @ self.weights['weight0'].T + self.weights['bias0'])
        logits = hidden @ self.weights['weight1'].T + self.weights['bias1']
        probability = 1 / (1 + np.exp(-np.clip(logits[1] - logits[0], -80, 80)))
        return dict(score=float(probability), clipped=False, contexts=7, backend='spectral-temporal-numpy')


def validate_model(directory):
    metadata = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') == 'BandTemporalMLP-30866':
        return TemporalDetector(directory).metadata
    return validate_legacy(directory)


def validate_backend(config, metadata):
    """Reject architecture, exported model identity and window mismatches early."""
    spectral = metadata['architecture'] in ('BandMLP-7826', 'BandTemporalMLP-30866')
    if (config['detector']['backend'] == 'spectral') != spectral:
        raise ValueError('Configured backend and model architecture disagree')
    identity = config['detector'].get('model_tag')
    if identity is not None and (type(identity) is not int or identity != metadata['model_tag']):
        raise ValueError('Configured model tag and exported weights disagree')
    if metadata['architecture'] == 'BandTemporalMLP-30866' and config['audio']['window_samples'] != 48128:
        raise ValueError('Temporal deployment requires the 48128-sample seven-context window')
    return spectral


def load_spectral(directory):
    metadata = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') == 'BandTemporalMLP-30866':
        return TemporalDetector(directory)
    return SpectralDetector(directory)
