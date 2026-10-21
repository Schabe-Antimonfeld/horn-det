"""Exact trained-model inference, with FPGA runtime and slow integer replay backend."""

import numpy as np
import hashlib
import json
from pathlib import Path

from ps.integer_kernel import check_export, fixed_forward, input_codes, load_fixed
from ps.dsp import band_features


class FrontendOverflow(ValueError):
    """A physical input lies outside the exported feature calibration range."""


class Detector:
    def __init__(self, directory, device=None):
        self.metadata = check_export(directory)
        self.frontend = self.metadata.get('provenance', {}).get('frontend', 'logmel')
        if self.frontend not in ('logmel', 'centered_logmel'):
            raise ValueError('Unsupported model frontend')
        self.model = load_fixed(directory) if device is None else None
        self.device = device

    def infer(self, features):
        codes, clipped = input_codes(features, self.metadata['input_fraction'])
        if clipped:
            raise FrontendOverflow('Frontend exceeds calibrated input range; do not silently clip')
        if self.device is not None:
            result = self.device.infer(features)
            return {'score': result['horn_probability'], 'clipped': result['activation_clipped'],
                    'backend': 'fpga', 'model_tag': result['model_tag']}
        result = fixed_forward(self.model, codes)
        return {'score': result['probability'] / 65536.0, 'clipped': bool(result['flags']),
                'backend': 'integer-replay', 'model_tag': self.model['model_tag']}

    def score(self, contexts):
        if len(contexts) == 0:
            return {'score': 0.0, 'clipped': False, 'contexts': 0}
        results = [self.infer(value) for value in contexts]
        return {'score': max(item['score'] for item in results),
                'clipped': any(item['clipped'] for item in results), 'contexts': len(results)}


class SpectralDetector:
    """Run the separate band-statistic MLP on PS using NumPy only."""

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.metadata.get('format') != 1 or self.metadata.get('architecture') != 'BandMLP-7826':
            raise ValueError('Unsupported spectral model')
        checksum = hashlib.sha256((directory / 'weights.npz').read_bytes()).hexdigest()
        if self.metadata.get('sha256') != {'weights.npz': checksum} or self.metadata.get('model_tag') != int(checksum[:8], 16):
            raise ValueError('Spectral model checksum mismatch')
        with np.load(str(directory / 'weights.npz'), allow_pickle=False) as archive:
            self.weights = {key: archive[key] for key in archive.files}
        expected = dict(weight0=(48, 160), bias0=(48,), weight1=(2, 48), bias1=(2,))
        if set(self.weights) != set(expected) or any(self.weights[key].shape != shape or not np.isfinite(self.weights[key]).all()
                                                   for key, shape in expected.items()):
            raise ValueError('Invalid spectral model dimensions or values')
        self.frontend = 'logmel'

    def score(self, contexts):
        if len(contexts) == 0:
            return dict(score=0.0, clipped=False, contexts=0, backend='spectral-numpy')
        contexts = np.asarray(contexts)
        if contexts.ndim != 3 or contexts.shape[1:] != (40, 24):
            raise ValueError('Spectral scoring requires N x 40 x 24 contexts')
        weights = self.weights
        hidden = np.maximum(0, band_features(contexts) @ weights['weight0'].T + weights['bias0'])
        logits = hidden @ weights['weight1'].T + weights['bias1']
        probabilities = 1 / (1 + np.exp(-np.clip(logits[..., 1] - logits[..., 0], -80, 80)))
        return dict(score=float(probabilities.max()), clipped=False, contexts=len(contexts), backend='spectral-numpy')


def validate_model(directory):
    metadata = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') == 'BandMLP-7826':
        return SpectralDetector(directory).metadata
    return check_export(directory)
