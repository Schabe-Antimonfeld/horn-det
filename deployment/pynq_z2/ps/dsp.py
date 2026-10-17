"""NumPy-only causal conditioning, bounded SRP search and beam features."""

import numpy as np


class Conditioning:
    def __init__(self, channels, gains=None, delays=None):
        self.gains = np.ones(channels) if gains is None else np.asarray(gains, dtype=float)
        self.delays = np.zeros(channels) if delays is None else np.asarray(delays, dtype=float)
        if self.gains.shape != (channels,) or self.delays.shape != (channels,) or not np.isfinite(self.gains).all() or not np.isfinite(self.delays).all() or np.any(self.gains <= 0):
            raise ValueError('Calibration must specify finite gains and delays for every channel')
        self.latency = float(max(0, self.delays.max()) + 32) if np.any(self.delays) else 0.0
        shifts = max(0, float(self.delays.max())) - self.delays
        self.integers = np.floor(shifts).astype(int)
        self.kernels = []
        for shift, integer in zip(shifts, self.integers):
            kernel = np.sinc(np.arange(65) - 32 - (shift - integer)) * np.hanning(65)
            self.kernels.append(kernel / kernel.sum())
        self.reset()

    def reset(self):
        self.previous = np.zeros(len(self.gains))
        self.state = np.zeros(len(self.gains))
        self.histories = [np.zeros(64) for _ in self.gains]
        self.buffers = [np.zeros(value) for value in self.integers]

    def feed(self, audio):
        audio = np.asarray(audio, dtype=float)
        if audio.ndim != 2 or audio.shape[1] != len(self.gains) or not np.isfinite(audio).all():
            raise ValueError('Conditioning needs finite PCM blocks')
        audio = audio / 32768.0
        result = np.empty_like(audio)
        for start in range(0, len(audio), 512):
            block = audio[start:start + 512]
            differences = np.diff(np.vstack((self.previous, block)), axis=0)
            powers = 0.995 ** np.arange(1, len(block) + 1)
            filtered = powers[:, None] * (self.state + np.cumsum(differences / powers[:, None], axis=0))
            result[start:start + len(block)] = filtered
            if len(block):
                self.previous, self.state = block[-1].copy(), filtered[-1].copy()
        result /= self.gains
        if self.latency:
            for channel, kernel in enumerate(self.kernels):
                joined = np.r_[self.histories[channel], result[:, channel]]
                filtered = np.convolve(joined, kernel, mode='valid')
                self.histories[channel] = joined[-64:].copy()
                buffered = np.r_[self.buffers[channel], filtered]
                result[:, channel] = buffered[:len(audio)]
                self.buffers[channel] = buffered[len(audio):].copy()
        return result


def coordinates(settings):
    phase = np.arange(settings['channels']) * 2 * np.pi / settings['channels']
    return settings['diameter'] / 2 * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(len(phase))))


def frames(audio, frame=1024, hop=512):
    audio = np.ascontiguousarray(audio, dtype=float)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or len(audio) < frame or not np.isfinite(audio).all():
        raise ValueError('Need a complete finite audio frame')
    count = 1 + (len(audio) - frame) // hop
    return np.lib.stride_tricks.as_strided(audio, shape=(count, audio.shape[1], frame),
                                          strides=(hop * audio.strides[0], audio.strides[1], audio.strides[0]), writeable=False)


def spatial_map(audio, audio_settings, search, scorer=None):
    from ps.geometry import direction_vectors
    values = frames(audio)
    pairs = np.asarray([(first, second) for first in range(audio.shape[1]) for second in range(first + 1, audio.shape[1])])
    total = np.zeros((len(pairs), 513), dtype=complex)
    for start in range(0, len(values), 12):
        spectrum = np.fft.rfft(values[start:start + 12] * np.hanning(1024), axis=-1)
        cross = spectrum[:, pairs[:, 0]] * spectrum[:, pairs[:, 1]].conj()
        total += (cross / (np.abs(cross) + 1e-12)).sum(axis=0)
    correlations = np.fft.irfft(total / len(values), n=1024, axis=-1)
    azimuths = np.arange(-180, 180, search['azimuth_step'], dtype=float)
    depressions = np.arange(0, 90 + search['depression_step'] / 2, search['depression_step'], dtype=float)
    azimuth, depression = np.meshgrid(azimuths, depressions, indexing='ij')
    vectors = direction_vectors(azimuth.ravel(), depression.ravel())
    array = coordinates(audio_settings)
    baselines = array[pairs[:, 1]] - array[pairs[:, 0]]
    response = np.empty(len(vectors))
    rows = np.arange(len(pairs))[:, None]
    block = min(search['block_directions'], 128) if scorer is not None else search['block_directions']
    for start in range(0, len(vectors), block):
        delays = baselines @ vectors[start:start + block].T * audio_settings['sample_rate'] / audio_settings['speed']
        lower = np.floor(delays).astype(int)
        fraction = delays - lower
        first = correlations[rows, lower % 1024]
        second = correlations[rows, (lower + 1) % 1024]
        response[start:start + block] = (scorer.score(first, second, fraction) if scorer is not None
                                        else ((1 - fraction) * first + fraction * second).sum(axis=0))
    return azimuths, depressions, response.reshape(azimuth.shape)


def candidates(grid, count=2, separation=12):
    from ps.geometry import Bearing, direction_vectors
    azimuths, depressions, response = grid
    if np.ptp(response) <= 1e-12:
        return []
    curve = response.max(axis=1)
    if np.ptp(curve) <= 1e-12:
        index = np.unravel_index(np.argmax(response), response.shape)
        return [Bearing(None, float(depressions[index[1]]), float(response[index]))]
    extended = np.tile(curve, 3)
    peaks, index = [], 1
    while index < len(extended) - 1:
        if extended[index] > extended[index - 1]:
            end = index
            while end + 1 < len(extended) and extended[end + 1] == extended[index]:
                end += 1
            if end + 1 < len(extended) and extended[end] > extended[end + 1]:
                peak = (index + end) // 2
                if len(curve) <= peak < 2 * len(curve):
                    peaks.append(peak % len(curve))
            index = end
        index += 1
    chosen, directions = [], []
    for index in sorted(peaks, key=lambda value: (-curve[value], value)):
        depression = float(depressions[np.argmax(response[index])])
        vector = direction_vectors(float(azimuths[index]), depression)
        if all(np.rad2deg(np.arccos(np.clip(np.dot(vector, other), -1, 1))) >= separation for other in directions):
            chosen.append(Bearing(None if depression >= 85 else float(azimuths[index]), depression, float(curve[index])))
            directions.append(vector)
        if len(chosen) == count:
            break
    return chosen


def beamform(audio, bearing, settings):
    from ps.geometry import direction_vectors
    vector = direction_vectors(0 if bearing.azimuth is None else bearing.azimuth,
                               90 if bearing.azimuth is None else bearing.depression)
    delays = -coordinates(settings) @ vector * settings['sample_rate'] / settings['speed']
    common = int(np.ceil(delays.max())) + 32
    output = np.zeros(len(audio))
    for channel, delay in enumerate(delays):
        shift = common - delay
        integer = int(np.floor(shift))
        kernel = np.sinc(np.arange(65) - 32 - (shift - integer)) * np.hanning(65)
        kernel /= kernel.sum()
        filtered = np.convolve(audio[:, channel], kernel, mode='full')[32:32 + len(audio)]
        if integer < len(audio):
            output[integer:] += filtered[:len(audio) - integer]
    return output / audio.shape[1], common


def features(waveform):
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 24000 / 700), 42) / 2595) - 1)
    frequency = np.fft.rfftfreq(1024, 1 / 48000)
    bank = np.maximum(0, np.minimum((frequency[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                    (edges[2:, None] - frequency[None, :]) / (edges[2:, None] - edges[1:-1, None])))
    power = abs(np.fft.rfft(frames(waveform)[:, 0] * np.hanning(1024), axis=-1)) ** 2
    logarithms = np.log(power @ bank.T + 1e-12)
    if len(logarithms) < 24:
        return np.empty((0, 40, 24))
    starts = sorted(set(range(0, len(logarithms) - 24 + 1, 12)) | {len(logarithms) - 24})
    return np.asarray([logarithms[start:start + 24].T for start in starts])
