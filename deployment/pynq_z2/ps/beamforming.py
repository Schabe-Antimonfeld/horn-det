"""Batch the frozen 65-tap beam FIR without changing its window boundaries."""

from collections import OrderedDict
import ctypes
import math
from pathlib import Path
import sys

import numpy as np

from ps.dsp import beamform, coordinates
from ps.geometry import direction_vectors


def validate_options(options):
    """Validate optional beam settings independently of the selected model."""
    allowed = {'implementation', 'fft_size', 'cache_size', 'native_library'}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError('Unsupported beamforming options')
    if options.get('implementation', 'reference') not in ('reference', 'cached', 'fft', 'native'):
        raise ValueError('Unsupported beamforming implementation')
    if type(options.get('fft_size', 512)) is not int or options.get('fft_size', 512) not in (512, 1024, 2048):
        raise ValueError('Beam FFT size must be 512, 1024 or 2048')
    size = options.get('cache_size', 64)
    if type(size) is not int or not 1 <= size <= 512:
        raise ValueError('Beam cache size must be an integer in 1..512')
    if 'native_library' in options and (options.get('implementation') != 'native'
            or not isinstance(options['native_library'], str) or not Path(options['native_library']).is_absolute()):
        raise ValueError('Beam native library override requires an absolute path and native implementation')


class Native:
    """Load the independent double-precision beam FIR ABI on the current host."""

    def __init__(self, path=None):
        location = Path(path) if path else Path(__file__).with_name('beamforming.dll' if sys.platform == 'win32' else 'libbeam.so')
        if not location.is_file():
            raise ValueError('Native beam library missing; run python scripts/build_beam.py on the target')
        try:
            self.library = ctypes.CDLL(str(location.resolve()))
            self.library.beam_abi.restype = ctypes.c_int
            if self.library.beam_abi() != 1:
                raise ValueError('Unsupported native beam ABI')
            real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
            number = ctypes.c_int
            self.library.beam_fir.argtypes = [real, number, number, real, integer, number, real]
            self.library.beam_fir.restype = number
        except (OSError, AttributeError) as error:
            raise ValueError('Cannot load native beam library; rebuild with scripts/build_beam.py: ' + str(error))

    def run(self, audio, plans):
        values = np.ascontiguousarray(audio.T)
        kernels = np.ascontiguousarray([plan['kernels'] for plan in plans])
        integers = np.ascontiguousarray([plan['integers'] for plan in plans], dtype=np.int32)
        output = np.zeros((len(plans), len(audio)))
        if max(len(audio), audio.shape[1], len(plans)) > 2147483000:
            raise ValueError('Beam input exceeds native integer limits')
        if self.library.beam_fir(values, len(audio), audio.shape[1], kernels, integers, len(plans), output):
            raise ValueError('Native beam FIR failed')
        return output


class Beamformer:
    """Reuse FIR plans and channel FFTs; retain reference output length and latency."""

    def __init__(self, audio, options=None):
        self.options = dict(options or {})
        validate_options(self.options)
        self.audio = dict(audio)
        self.array = coordinates(audio)
        self.implementation = self.options.get('implementation', 'reference')
        self.fft_size = self.options.get('fft_size', 512)
        self.cache = OrderedDict()
        self.native = Native(self.options.get('native_library')) if self.implementation == 'native' else None

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

    def _fft(self, audio, plans):
        history = max(int(plan['integers'].max()) for plan in plans) + 32
        hop = self.fft_size - history
        count = (len(audio) + hop - 1) // hop
        padded = np.zeros((audio.shape[1], history + count * hop))
        padded[:, history:history + len(audio)] = audio.T
        blocks = np.lib.stride_tricks.as_strided(padded,
                    shape=(count, audio.shape[1], self.fft_size),
                    strides=(hop * padded.strides[1], padded.strides[0], padded.strides[1]), writeable=False)
        spectrum = np.fft.rfft(blocks, axis=-1)
        output = []
        for plan in plans:
            combined = np.einsum('bcf,cf->bf', spectrum, plan['spectrum'], optimize=False)
            waveform = np.fft.irfft(combined, n=self.fft_size, axis=-1)[:, history:].reshape(-1)[:len(audio)]
            waveform /= audio.shape[1]
            # The frozen implementation discards each channel until its integer shift.
            prefix = min(len(audio), int(plan['integers'].max()))
            waveform[:prefix] = self._convolve(audio[:prefix], plan)
            output.append(waveform)
        return output

    def run(self, audio, bearings):
        """Return ordered (waveform, common delay) pairs and actual backend diagnostics."""
        values = np.asarray(audio, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.audio['channels'] or not len(values) or not np.isfinite(values).all():
            raise ValueError('Beam audio must contain finite samples for every configured channel')
        bearings = list(bearings)
        metric = dict(implementation=self.implementation, beams=len(bearings), taps=65, cache_hits=0,
                      fft_size=self.fft_size if self.implementation == 'fft' else None)
        if not bearings:
            return [], metric
        if self.implementation == 'reference':
            return [beamform(values, bearing, self.audio) for bearing in bearings], metric
        plans = []
        for bearing in bearings:
            plan, hit = self._plan(bearing)
            plans.append(plan)
            metric['cache_hits'] += int(hit)
        if self.native is not None:
            waveforms = self.native.run(values, plans)
        elif self.implementation == 'fft':
            waveforms = self._fft(values, plans)
        else:
            waveforms = [self._convolve(values, plan) for plan in plans]
        if not all(np.isfinite(waveform).all() for waveform in waveforms):
            raise ValueError('Beam result exceeds finite arithmetic range')
        return [(waveform, plan['common']) for waveform, plan in zip(waveforms, plans)], metric
