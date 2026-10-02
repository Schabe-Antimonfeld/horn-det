"""Gain, fixed-delay and DC correction for synchronized PCM."""

import numpy as np
from scipy.signal import convolve, lfilter


def audio_array(audio):
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 2 or not np.isfinite(audio).all():
        raise ValueError("Audio must be finite with shape (samples, channels)")
    return audio


def fractional_delay(signal, delay, taps=65):
    """Shift a finite waveform without circular wrap; positive delay means later arrival."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all() or not np.isfinite(delay):
        raise ValueError("Require a finite waveform and delay")
    if taps < 3 or taps % 2 != 1:
        raise ValueError("Filter length must be odd and at least three")
    if len(signal) == 0:
        return signal.copy()
    integer = int(np.floor(delay))
    fraction = delay - integer
    center = taps // 2
    kernel = np.sinc(np.arange(taps) - center - fraction) * np.hanning(taps)
    kernel /= kernel.sum()
    filtered = convolve(signal, kernel, mode="full")[center:center + len(signal)]
    output = np.zeros_like(signal)
    start = max(0, integer)
    end = min(len(signal), len(signal) + integer)
    if end > start:
        output[start:end] = filtered[start - integer:end - integer]
    return output


def calibrate(audio, gains, delays):
    """Return aligned PCM and common sample delay; the sinc reference needs filter buffering."""
    audio = audio_array(audio)
    gains, delays = np.asarray(gains, dtype=float), np.asarray(delays, dtype=float)
    if gains.shape != (audio.shape[1],) or delays.shape != gains.shape:
        raise ValueError("Calibration must specify every channel")
    if not np.isfinite(gains).all() or not np.isfinite(delays).all() or np.any(gains <= 0):
        raise ValueError("Gains must be positive and calibration finite")
    common = float(max(0, delays.max()))
    output = np.column_stack([fractional_delay(audio[:, index] / gains[index],
                             common - delays[index]) for index in range(audio.shape[1])])
    return output, common


def remove_dc(audio, coefficient=0.995, state=None):
    audio = audio_array(audio)
    if not 0 <= coefficient < 1:
        raise ValueError("DC coefficient must be in [0, 1)")
    if state is None:
        state = np.zeros((1, audio.shape[1]))
    state = np.asarray(state, dtype=float)
    if state.shape != (1, audio.shape[1]):
        raise ValueError("DC state must have shape (1, channels)")
    if not len(audio):
        return audio.copy(), state.copy()
    return lfilter([1, -1], [1, -coefficient], audio, axis=0, zi=state)


class StreamCalibration:
    """Causal sinc calibration; 32-tap-center buffering is included in processing delay."""

    def __init__(self, gains, delays, taps=65):
        gains, delays = np.asarray(gains, dtype=float), np.asarray(delays, dtype=float)
        if gains.ndim != 1 or gains.shape != delays.shape or not len(gains):
            raise ValueError("Calibration vectors must match")
        if not np.isfinite(gains).all() or not np.isfinite(delays).all() or np.any(gains <= 0):
            raise ValueError("Calibration must be finite with positive gains")
        if taps < 3 or taps % 2 == 0:
            raise ValueError("Calibration filter requires an odd length")
        self.gains = gains
        self.common = max(0, float(delays.max()))
        self.latency = self.common + taps // 2
        shifts = self.common - delays
        self.integers = np.floor(shifts).astype(int)
        self.kernels = []
        self.states = [np.zeros(taps - 1) for _ in gains]
        self.buffers = [np.zeros(integer) for integer in self.integers]
        for shift, integer in zip(shifts, self.integers):
            kernel = np.sinc(np.arange(taps) - taps // 2 - (shift - integer)) * np.hanning(taps)
            self.kernels.append(kernel / kernel.sum())

    def feed(self, audio):
        audio = audio_array(audio)
        if audio.shape[1] != len(self.gains):
            raise ValueError("Calibration channel count changed")
        if not len(audio):
            return audio.copy()
        output = np.empty_like(audio)
        for channel, kernel in enumerate(self.kernels):
            filtered, self.states[channel] = lfilter(kernel, [1], audio[:, channel] / self.gains[channel],
                                                      zi=self.states[channel])
            buffered = np.concatenate((self.buffers[channel], filtered))
            output[:, channel] = buffered[:len(audio)]
            self.buffers[channel] = buffered[len(audio):]
        return output
