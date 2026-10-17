"""Continuous PCM packet validation and bounded history independent of PYNQ imports."""

import queue
import threading
import time

import numpy as np


class PacketDecoder:
    def __init__(self, channels=14, samples=512, sample_rate=48000):
        self.channels, self.samples, self.sample_rate = channels, samples, sample_rate
        self.expected = None
        self.dropped = None

    def decode(self, words):
        words = np.asarray(words)
        if words.dtype != np.uint32 or words.shape != (8 + self.channels * self.samples,):
            raise ValueError('Invalid PCM packet size or word type')
        header = [int(value) for value in words[:8]]
        if header[:2] != [0x314d4450, 0x00010800 | self.channels]:
            raise ValueError('PCM packet format mismatch')
        if header[5:7] != [self.samples, self.sample_rate] or header[7] & ~1:
            raise ValueError('PCM configuration or flags mismatch')
        payload = words[8:].view(np.int32)
        if np.any(payload < -32768) or np.any(payload > 32767):
            raise ValueError('PCM is not sign-extended INT16')
        first = header[2] | header[3] << 32
        gap = self.expected is not None and (first != self.expected or header[4] != self.dropped)
        self.expected = (first + self.samples) & ((1 << 64) - 1)
        self.dropped = header[4]
        return {'first': first, 'gap': gap, 'dropped': header[4], 'clipped': bool(header[7]),
                'audio': payload.astype(np.int16).reshape(self.samples, self.channels)}


class WindowBuffer:
    def __init__(self, settings, dtype=np.int16):
        self.capacity = settings['window_samples']
        self.update = settings['update_samples']
        self.memory = np.zeros((self.capacity, settings['channels']), dtype=dtype)
        self.reset()
        self.epoch = 0

    def reset(self):
        self.cursor, self.filled, self.since = 0, 0, 0
        self.expected = None
        self.saturation = False

    def feed(self, packet):
        audio = packet['audio']
        if audio.ndim != 2 or audio.shape[1] != self.memory.shape[1] or len(audio) > self.capacity:
            raise ValueError('Packet does not fit the history buffer')
        if packet['gap'] or (self.expected is not None and packet['first'] != self.expected):
            self.reset()
            self.epoch += 1
        self.expected = packet['first'] + len(audio)
        count = min(len(audio), self.capacity - self.cursor)
        self.memory[self.cursor:self.cursor + count] = audio[:count]
        self.memory[:len(audio) - count] = audio[count:]
        self.cursor = (self.cursor + len(audio)) % self.capacity
        previous = self.filled
        self.filled = min(self.capacity, self.filled + len(audio))
        self.since += len(audio)
        self.saturation = self.saturation or packet['clipped']
        if self.filled < self.capacity or (previous == self.capacity and self.since < self.update):
            return None
        self.since = 0
        snapshot = np.concatenate((self.memory[self.cursor:], self.memory[:self.cursor]))
        return {'audio': snapshot, 'first': self.expected - self.capacity,
                'end': self.expected, 'epoch': self.epoch, 'clipped': self.saturation}


class CaptureDevice:
    def __init__(self, overlay, allocate, settings, timeout=2.0):
        self.dma, self.control = overlay.audio_dma, overlay.capture_control
        self.timeout, self.started, self.closed, self.faulted = timeout, False, False, False
        self.decoder = PacketDecoder(settings['channels'], settings['packet_samples'], settings['sample_rate'])
        self.buffer = allocate(shape=(8 + settings['channels'] * settings['packet_samples'],), dtype=np.uint32)

    def receive(self):
        if self.closed or self.faulted:
            raise RuntimeError('Capture is unavailable; reload the overlay')
        try:
            self.dma.recvchannel.transfer(self.buffer)
            if not self.started:
                deadline = time.monotonic() + self.timeout
                while not self.control.read(0x08) & 2:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Capture clock/warmup is not ready')
                    time.sleep(0.001)
                self.control.write(0, 1)
                self.started = True
            deadline = time.monotonic() + self.timeout
            while True:
                status = self.dma.mmio.read(0x34)
                if status & 0x70:
                    raise RuntimeError('Audio DMA error')
                if status & 2:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Audio DMA timeout')
                time.sleep(0.0001)
            self.dma.recvchannel.wait()
            return self.decoder.decode(self.buffer)
        except Exception:
            self.faulted = True
            raise

    def close(self):
        if self.closed:
            return
        self.control.write(0, 0)
        self.dma.mmio.write(0x30, 4)
        deadline = time.monotonic() + self.timeout
        while self.dma.mmio.read(0x30) & 4:
            if time.monotonic() > deadline:
                raise RuntimeError('DMA reset timeout; buffer remains allocated')
            time.sleep(0.001)
        self.buffer.freebuffer()
        self.closed = True


class CapturePump:
    """Keep acquisition independent of inference; only discard stale analysis snapshots."""

    def __init__(self, device, settings, conditioning=None):
        self.conditioning = conditioning
        self.device, self.history = device, WindowBuffer(settings, np.float64 if conditioning else np.int16)
        self.pending = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.error = None
        self.skipped, self.gaps = 0, 0
        self.thread = threading.Thread(target=self.run, name='audio-capture', daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                packet = self.device.receive()
                self.gaps += int(packet['gap'])
                if self.conditioning is not None:
                    if packet['gap']:
                        self.conditioning.reset()
                    packet['audio'] = self.conditioning.feed(packet['audio'])
                snapshot = self.history.feed(packet)
                if snapshot is not None:
                    try:
                        self.pending.put_nowait(snapshot)
                    except queue.Full:
                        try:
                            self.pending.get_nowait()
                            self.skipped += 1
                        except queue.Empty:
                            pass
                        self.pending.put_nowait(snapshot)
        except Exception as error:
            self.error = error
            self.stop.set()

    def receive(self):
        while True:
            if self.error is not None:
                raise RuntimeError('Capture worker failed: ' + str(self.error))
            try:
                return self.pending.get(timeout=0.1)
            except queue.Empty:
                if self.stop.is_set():
                    raise RuntimeError('Capture worker stopped')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.device.timeout + 1)
        if self.thread.is_alive():
            raise RuntimeError('Capture worker still owns DMA buffers')
        self.device.close()
