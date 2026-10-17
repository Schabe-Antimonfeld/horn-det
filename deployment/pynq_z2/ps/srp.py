"""Optional batched SRP DMA backend; FFT, PHAT and delay lookup stay on PS."""

import time

import numpy as np


PAIRS = 91
MAX_DIRECTIONS = 128
GCC_SCALE = 1 << 22


def encode_batch(lower, upper, fraction):
    lower, upper, fraction = [np.asarray(value, dtype=float) for value in (lower, upper, fraction)]
    if lower.ndim != 2 or lower.shape != upper.shape or lower.shape != fraction.shape or lower.shape[0] != PAIRS or not 1 <= lower.shape[1] <= MAX_DIRECTIONS:
        raise ValueError('SRP needs 91 pairs and 1..128 directions')
    if not all(np.isfinite(value).all() for value in (lower, upper, fraction)) or np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError('SRP inputs must be finite with fractions in [0,1]')
    codes = []
    for value in (lower, upper):
        scaled = np.sign(value) * np.floor(np.abs(value) * GCC_SCALE + 0.5)
        if np.any(scaled < -(1 << 23)) or np.any(scaled >= (1 << 23)):
            raise ValueError('GCC exceeds signed24 Q2.22 range')
        codes.append(scaled.astype(np.int32).T.view(np.uint32))
    weights = np.floor(fraction.T * 65536).astype(np.uint32)
    endpoint = weights == 65536
    codes[0][endpoint] = codes[1][endpoint]
    weights[endpoint] = 0
    payload = np.stack((codes[0], codes[1], weights), axis=-1).reshape(-1)
    return np.r_[np.uint32(lower.shape[1]), payload]


def decode_batch(words, directions):
    words = np.asarray(words, dtype=np.uint32).reshape(directions, 4)
    if np.any(words[:, 0] != 0x31505253) or np.any(words[:, 3] != 0):
        raise RuntimeError('SRP output tag or packet error')
    high = words[:, 2].view(np.int32).astype(np.int64)
    if np.any(high < -32768) or np.any(high > 32767):
        raise RuntimeError('SRP result is not sign-extended signed48')
    total = (high << 32) + words[:, 1].astype(np.int64)
    return total / float(GCC_SCALE * 65536)


class SrpDevice:
    """Use one DMA transfer per direction block; faults require overlay reload."""

    def __init__(self, overlay, allocate, timeout=2.0):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('SRP timeout must be positive and finite')
        self.dma, self.timeout = overlay.srp_dma, timeout
        self.closed, self.faulted = False, False
        self.inputs = allocate(shape=(1 + MAX_DIRECTIONS * PAIRS * 3,), dtype=np.uint32)
        try:
            self.outputs = allocate(shape=(MAX_DIRECTIONS * 4,), dtype=np.uint32)
        except Exception:
            self.inputs.freebuffer()
            raise

    def score(self, lower, upper, fraction):
        if self.closed or self.faulted:
            raise RuntimeError('SRP unavailable; reload overlay')
        packet = encode_batch(lower, upper, fraction)
        directions = int(packet[0])
        self.inputs[:len(packet)] = packet
        try:
            self.dma.recvchannel.transfer(self.outputs, nbytes=directions * 16)
            self.dma.sendchannel.transfer(self.inputs, nbytes=packet.nbytes)
            deadline = time.monotonic() + self.timeout
            while True:
                sending, receiving = self.dma.mmio.read(0x04), self.dma.mmio.read(0x34)
                if (sending | receiving) & 0x70:
                    raise RuntimeError('SRP AXI DMA error')
                if sending & 2 and receiving & 2:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('SRP DMA timed out')
                time.sleep(0.0001)
            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()
            if self.dma.mmio.read(0x58) != directions * 16:
                raise RuntimeError('SRP output length mismatch')
            return decode_batch(self.outputs[:directions * 4], directions)
        except Exception:
            self.faulted = True
            raise

    def close(self):
        """Reset both DMA channels before releasing their physical buffers."""
        if self.closed:
            return
        self.dma.mmio.write(0x00, 4)
        deadline = time.monotonic() + self.timeout
        while self.dma.mmio.read(0x00) & 4:
            if time.monotonic() > deadline:
                self.faulted = True
                raise RuntimeError('SRP DMA reset timed out; buffers remain allocated')
            time.sleep(0.001)
        self.inputs.freebuffer()
        self.outputs.freebuffer()
        self.closed = True
