"""PYNQ TinyCNN DMA client; requires only the board's existing PYNQ and NumPy."""

import argparse
import json
import time

import numpy as np


def encode_features(features, fraction):
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (40, 24) or not np.isfinite(values).all() or not 0 <= fraction <= 15:
        raise ValueError('Require finite 40x24 features and a supported input scale')
    codes = np.sign(values) * np.floor(np.abs(values) * (1 << fraction) + 0.5)
    if np.any(codes < -32768) or np.any(codes > 32767):
        raise ValueError('Features exceed the calibrated input range; recalibrate the model')
    return codes.astype(np.int32).reshape(-1).view(np.uint32)


class CnnDevice:
    """Bind a freshly downloaded overlay; reload it after a DMA/protocol failure."""

    def __init__(self, overlay, metadata, allocate, timeout=2.0):
        if metadata['format'] != 1 or metadata['architecture'] != 'TinyCNN-1834' or metadata['input_shape'] != [40, 24]:
            raise ValueError('Unsupported FPGA CNN manifest')
        if not 0 <= metadata['input_fraction'] <= 15 or not 0 <= metadata['logit_fraction'] <= 15 or timeout <= 0:
            raise ValueError('Unsupported CNN scale or timeout')
        self.metadata = metadata
        self.dma = overlay.cnn_dma
        self.timeout = timeout
        self.sequence = 0
        self.faulted = False
        self.closed = False
        self.inputs = allocate(shape=(960,), dtype=np.uint32)
        try:
            self.outputs = allocate(shape=(8,), dtype=np.uint32)
        except Exception:
            self.inputs.freebuffer()
            raise

    def infer(self, features, frame=None):
        """Return an exact Q0.16 probability and preserve the caller's acoustic frame ID."""
        if self.closed or self.faulted:
            raise RuntimeError('Device unavailable; reload the overlay before another session')
        self.inputs[:] = encode_features(features, self.metadata['input_fraction'])
        started = time.monotonic()
        try:
            self.dma.recvchannel.transfer(self.outputs)
            self.dma.sendchannel.transfer(self.inputs)
            deadline = started + self.timeout
            while True:
                sending = self.dma.mmio.read(0x04)
                receiving = self.dma.mmio.read(0x34)
                if (sending | receiving) & 0x70:
                    raise RuntimeError('CNN AXI DMA error')
                if sending & 2 and receiving & 2:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('CNN DMA timed out')
                time.sleep(0.0001)
            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()
            words = [int(value) for value in self.outputs]
            if words[0] != 0x314e4e43 or words[1] != 0x00010008 or words[2] != self.sequence:
                raise RuntimeError('CNN output format/sequence mismatch')
            if words[7] != self.metadata['model_tag']:
                raise RuntimeError('Bitstream and model manifest do not match')
            flags = words[5] >> 16
            if flags & ~3 or flags & 2:
                raise RuntimeError('CNN reported an invalid input packet')
            logits = self.outputs[3:5].view(np.int32).astype(np.int64)
            if np.any(logits < -32768) or np.any(logits > 32767):
                raise RuntimeError('CNN logits are not sign-extended INT16')
            self.sequence = (self.sequence + 1) & 0xffffffff
            return {'frame': frame, 'sequence': words[2], 'probability_code': words[5] & 65535,
                    'horn_probability': (words[5] & 65535) / 65536.0,
                    'logit_codes': logits.tolist(),
                    'logits': (logits / float(1 << self.metadata['logit_fraction'])).tolist(),
                    'activation_clipped': bool(flags & 1), 'compute_cycles': words[6],
                    'wall_seconds': time.monotonic() - started, 'model_tag': words[7]}
        except Exception:
            self.faulted = True
            raise

    def probability(self, features):
        """Implement the reference Detector's NCHW probability interface."""
        values = np.asarray(features)
        if values.ndim != 4 or values.shape[1:] != (1, 40, 24):
            raise ValueError('Detector batches must have shape N x 1 x 40 x 24')
        return np.asarray([self.infer(value[0])['horn_probability'] for value in values])

    def close(self):
        """Reset DMA before freeing buffers, including after a timeout."""
        if self.closed:
            return
        self.dma.mmio.write(0x00, 4)
        deadline = time.monotonic() + self.timeout
        while self.dma.mmio.read(0x00) & 4:
            if time.monotonic() > deadline:
                self.faulted = True
                raise RuntimeError('DMA reset timed out; buffers remain allocated')
            time.sleep(0.001)
        self.inputs.freebuffer()
        self.outputs.freebuffer()
        self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    from pynq import Overlay, allocate
    with open(arguments.manifest, encoding='utf-8') as source:
        metadata = json.load(source)
    features = np.load(arguments.features, allow_pickle=False)
    if features.ndim != 3 or features.shape[1:] != (40, 24):
        raise ValueError('Feature file must contain N x 40 x 24 floating point windows')
    overlay = Overlay(arguments.overlay)
    device = CnnDevice(overlay, metadata, allocate)
    try:
        results = [device.infer(value) for value in features]
    finally:
        device.close()
    with open(arguments.output, 'w', encoding='utf-8') as output:
        json.dump({'results': results, 'evidence': 'FPGA inference; no source association or road claim'}, output, indent=2)


if __name__ == '__main__':
    main()
