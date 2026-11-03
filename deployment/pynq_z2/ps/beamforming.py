"""Shared-input FFT beamforming with independent exact-direction output reuse."""

from collections import OrderedDict
import math
import time

import numpy as np

from ps.dsp import coordinates
from ps.geometry import direction_vectors
from ps.streaming import bearing_key


def validate_options(options):
    if not isinstance(options, dict) or set(options) - {'implementation', 'fft_size', 'cache_size'}:
        raise ValueError('Unsupported beamforming options')
    if options.get('implementation') != 'fft' or options.get('fft_size') != 1024:
        raise ValueError('Latest beamforming requires the 1024-point FFT implementation')
    if type(options.get('cache_size', 64)) is not int or not 1 <= options.get('cache_size', 64) <= 512:
        raise ValueError('Invalid beam plan cache size')


class Beamformer:
    def __init__(self, audio, options=None, native=None):
        self.native = native
        self.options = dict(options or {})
        validate_options(self.options)
        self.audio = dict(audio)
        self.array = coordinates(audio)
        self.implementation = 'fft'
        self.fft_size = self.options.get('fft_size', 512)
        self.cache = OrderedDict()
        support = int(math.ceil(audio['diameter'] * audio['sample_rate'] / audio['speed'])) + 65
        common = math.gcd(audio['update_samples'], audio['window_samples'])
        self.hop = max(value for value in range(1, self.fft_size - support + 1) if common % value == 0)
        self.history = self.fft_size - self.hop
        self.spectra = None
        self.outputs = {}
        self.end = None

    def _plan(self, bearing):
        angle = 0 if bearing.azimuth is None else bearing.azimuth
        depression = 90 if bearing.azimuth is None else bearing.depression
        if not all(isinstance(value, (int, float, np.number)) and math.isfinite(value) for value in (angle, depression)):
            raise ValueError('Bearing must specify finite angles')
        key = (float(angle), float(depression))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key], True
        vector = direction_vectors(angle, depression)
        delays = -self.array @ vector * self.audio['sample_rate'] / self.audio['speed']
        common = int(np.ceil(delays.max())) + 32
        shifts = common - delays
        integers = np.floor(shifts).astype(np.int32)
        kernels = np.sinc(np.arange(65)[None] - 32 - (shifts - integers)[:, None]) * np.hanning(65)
        kernels /= kernels.sum(axis=1)[:, None]
        if np.any(integers < 32):
            raise ValueError('Beam FIR requires causal centered-array delays')
        plan = dict(kernels=kernels, integers=integers, common=common)
        if self.implementation == 'fft':
            if int(integers.max()) + 32 >= self.fft_size:
                raise ValueError('Beam FFT cannot cover the physical array delay and FIR support')
            coefficients = np.zeros((len(integers), self.fft_size))
            for channel, integer in enumerate(integers):
                coefficients[channel, integer - 32:integer + 33] = kernels[channel]
            plan['spectrum'] = np.fft.rfft(coefficients, axis=-1)
        self.cache[key] = plan
        if len(self.cache) > self.options.get('cache_size', 64):
            self.cache.popitem(last=False)
        return plan, False

    @staticmethod
    def _convolve(audio, plan):
        output = np.zeros(len(audio))
        for channel, integer in enumerate(plan['integers']):
            if integer < len(audio):
                filtered = np.convolve(audio[:, channel], plan['kernels'][channel], mode='full')[32:32 + len(audio)]
                output[integer:] += filtered[:len(audio) - integer]
        return output / audio.shape[1]

    def run_snapshot(self, snapshot, bearings, context):
        audio = np.asarray(snapshot['audio'], dtype=float)
        if audio.ndim != 2 or audio.shape[1] != self.audio['channels'] or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Streaming beam audio must be finite with configured channels')
        count = (len(audio) + self.hop - 1) // self.hop
        plans, hits = [], 0
        for bearing in bearings:
            plan, hit = self._plan(bearing)
            plans.append(plan)
            hits += int(hit)
        metric = dict(native_synthesis=self.native is not None, implementation='refined_stream_fft', beams=len(plans), taps=65, fft_size=self.fft_size,
                      cache_hits=hits, fft_hop=self.hop, fft_blocks_total=count, fft_blocks_computed=0,
                      fft_blocks_reused=0, output_reused=False, output_reused_beams=0, output_reuse_flags=[],
                      support_samples=0, fft_seconds=0.0, synthesis_seconds=0.0, spectrum_cache_bytes=0)
        if not plans:
            self.spectra, self.outputs, self.end = None, {}, context['end']
            return [], metric
        began = time.perf_counter()
        step = self.audio['update_samples'] // self.hop
        expected = max(0, count - step - 1)
        reusable = context['reusable'] and self.end == context['previous_end'] and len(audio) % self.hop == 0
        reused = expected if reusable and self.spectra is not None and len(self.spectra) == expected else 0
        if len(audio) % self.hop == 0 and self.hop >= self.history:
            prefix = np.zeros((1, audio.shape[1], self.fft_size))
            prefix[0, :, self.history:] = audio[:self.hop].T
            interior = np.lib.stride_tricks.as_strided(audio[self.hop - self.history:],
                       shape=(count - 1, audio.shape[1], self.fft_size),
                       strides=(self.hop * audio.strides[0], audio.strides[1], audio.strides[0]), writeable=False)
            fresh = np.fft.rfft(interior[reused:], axis=-1)
            pieces = [np.fft.rfft(prefix, axis=-1)] + ([self.spectra] if reused else []) + [fresh]
        else:
            padded = np.zeros((audio.shape[1], self.history + count * self.hop))
            padded[:, self.history:self.history + len(audio)] = audio.T
            blocks = np.lib.stride_tricks.as_strided(padded, shape=(count, audio.shape[1], self.fft_size),
                      strides=(self.hop * padded.strides[1], padded.strides[0], padded.strides[1]), writeable=False)
            fresh = np.fft.rfft(blocks, axis=-1)
            pieces = [fresh]
        self.spectra = (fresh[-expected:].copy() if len(fresh) > expected + 1 else fresh[-expected:]) if expected else None
        metric.update(fft_blocks_computed=count - reused, fft_blocks_reused=reused,
                      fft_seconds=time.perf_counter() - began,
                      spectrum_cache_bytes=(self.spectra.base.nbytes if isinstance(self.spectra.base, np.ndarray)
                                             else self.spectra.nbytes) if self.spectra is not None else 0)
        support = max(int(plan['integers'].max()) for plan in plans) + 32
        keys = [bearing_key(bearing) for bearing in bearings]
        flags = [bool(reusable and key in self.outputs and context['overlap'] > support) for key in keys]
        output = []
        began = time.perf_counter()
        for key, plan, stable in zip(keys, plans, flags):
            start = context['overlap'] // self.hop if stable else 0
            tails, cursor = [], 0
            for piece in pieces:
                if cursor + len(piece) > start:
                    selected = piece[max(0, start - cursor):]
                    combined = (self.native.beam(selected, plan['spectrum']) if self.native is not None else
                                np.einsum('bcf,cf->bf', selected, plan['spectrum'], optimize=False))
                    tails.append(np.fft.irfft(combined, n=self.fft_size, axis=-1)[:, self.history:].reshape(-1))
                cursor += len(piece)
            tail = np.concatenate(tails)
            waveform = np.zeros(len(audio))
            offset = start * self.hop
            waveform[offset:] = tail[:len(audio) - offset] / audio.shape[1]
            if stable:
                delta, overlap = context['delta'], context['overlap']
                waveform[support:overlap] = self.outputs[key][delta + support:delta + overlap]
            prefix = min(len(audio), support if stable else int(plan['integers'].max()))
            waveform[:prefix] = self._convolve(audio[:prefix], plan)
            waveform.setflags(write=False)
            output.append((waveform, plan['common']))
        if not all(np.isfinite(waveform).all() for waveform, _ in output):
            raise ValueError('Streaming beam result is not finite')
        self.outputs = {key: waveform for key, (waveform, _) in zip(keys, output)} if len(keys) <= 8 else {}
        self.end = context['end']
        metric.update(output_reused=all(flags), output_reused_beams=sum(flags), output_reuse_flags=flags,
                      support_samples=support, synthesis_seconds=time.perf_counter() - began,
                      output_cache_bytes=sum(waveform.nbytes for waveform in self.outputs.values()))
        return output, metric
