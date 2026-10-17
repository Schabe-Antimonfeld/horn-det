"""Exact trained-model inference, with FPGA runtime and slow integer replay backend."""

import numpy as np

from ps.integer_kernel import check_export, fixed_forward, input_codes, load_fixed


class Detector:
    def __init__(self, directory, device=None):
        self.metadata = check_export(directory)
        self.model = load_fixed(directory) if device is None else None
        self.device = device

    def infer(self, features):
        codes, clipped = input_codes(features, self.metadata['input_fraction'])
        if clipped:
            raise ValueError('Frontend exceeds calibrated input range; do not silently clip')
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
