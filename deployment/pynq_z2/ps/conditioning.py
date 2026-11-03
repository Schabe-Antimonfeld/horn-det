"""Caller-owned packet buffers and an independent stateful calibration backend."""

import ctypes
from pathlib import Path
import sys

import numpy as np

from ps.dsp import Conditioning


class State(ctypes.Structure):
    _fields_ = [('channels', ctypes.c_int), ('unity', ctypes.c_int), ('fir', ctypes.c_int), ('history', ctypes.c_int),
                ('stride', ctypes.c_size_t), ('gains', ctypes.POINTER(ctypes.c_double)),
                ('previous', ctypes.POINTER(ctypes.c_double)), ('state', ctypes.POINTER(ctypes.c_double)),
                ('kernels', ctypes.POINTER(ctypes.c_double)), ('delays', ctypes.POINTER(ctypes.c_int32)),
                ('memory', ctypes.POINTER(ctypes.c_double))]


def validate_options(options):
    if not isinstance(options, dict) or set(options) - {'implementation', 'fir', 'native_library'}:
        raise ValueError('Unsupported conditioning settings')
    if options.get('implementation', 'buffered') != 'buffered' or options.get('fir', 'numpy') not in ('numpy', 'native'):
        raise ValueError('Unsupported conditioning implementation or FIR')
    if 'native_library' in options and (not isinstance(options['native_library'], str)
                                       or not Path(options['native_library']).is_absolute()):
        raise ValueError('Conditioning library override must be an absolute path')


class Runtime:
    def __init__(self, path=None):
        location = Path(path) if path else Path(__file__).with_name('conditioning.dll' if sys.platform == 'win32' else 'libconditioning.so')
        if not location.is_file():
            raise ValueError('Conditioning library missing; run python scripts/build_conditioning.py on the target')
        try:
            self.library = ctypes.CDLL(str(location.resolve()))
            self.library.conditioning_abi.restype = ctypes.c_int
            if self.library.conditioning_abi() != 1:
                raise ValueError('Unsupported conditioning ABI')
            self.library.conditioning_state_size.restype = ctypes.c_int
            if self.library.conditioning_state_size() != ctypes.sizeof(State):
                raise ValueError('Conditioning state layout disagrees with the native ABI')
            real = np.ctypeslib.ndpointer(dtype=np.float64, flags=('C_CONTIGUOUS', 'ALIGNED'))
            pcm = np.ctypeslib.ndpointer(dtype=np.int16, flags=('C_CONTIGUOUS', 'ALIGNED'))
            number = ctypes.c_int
            for name, dtype in [('conditioning_pcm16', pcm), ('conditioning_float64', real)]:
                function = getattr(self.library, name)
                function.argtypes = [dtype, number, ctypes.POINTER(State), real]
                function.restype = number
            self.library.conditioning_shift.argtypes = [ctypes.POINTER(State), number]
            self.library.conditioning_shift.restype = number
        except (OSError, AttributeError) as error:
            raise ValueError('Cannot load conditioning library; rebuild scripts/build_conditioning.py: ' + str(error))


class Conditioner:
    """Preserve independent feed results; feed_into writes only caller-provided output."""

    implementation = 'buffered'

    def __init__(self, channels, gains=None, delays=None, fir='numpy', library=None):
        if type(channels) is not int or not 0 < channels <= 2147483000 or fir not in ('numpy', 'native'):
            raise ValueError('Invalid conditioning channel count or FIR')
        self._channels, self._fir = channels, fir
        self._gains = np.require(np.ones(channels) if gains is None else gains, dtype=float, requirements=['C', 'A']).copy()
        values = np.asarray(np.zeros(channels) if delays is None else delays, dtype=float)
        if (self._gains.shape != (channels,) or values.shape != (channels,) or not np.isfinite(self._gains).all()
                or not np.isfinite(values).all() or np.any(self._gains <= 0)):
            raise ValueError('Calibration needs finite gains/delays and positive gains for every channel')
        shifts = max(0, float(values.max())) - values
        if not np.isfinite(shifts).all() or np.max(shifts) > 2147483000:
            raise ValueError('Calibration delay exceeds the native integer range')
        self._integers = np.floor(shifts).astype(np.int32)
        self._latency = float(max(0, values.max()) + 32) if np.any(values) else 0.0
        self._kernels = np.sinc(np.arange(65)[None] - 32 - (shifts - self._integers)[:, None]) * np.hanning(65)
        self._kernels /= self._kernels.sum(axis=1)[:, None]
        self._reversed = np.ascontiguousarray(self._kernels[:, ::-1])
        self._unity = int(np.all(self._gains == 1))
        self._history = int(self._integers.max()) + 64 if self._latency else 0
        self._capacity = 512
        self._memory = np.zeros((channels, self._history + self._capacity)) if self._latency else np.empty((0, 0))
        self._previous, self._state = np.zeros(channels), np.zeros(channels)
        self._runtime = Runtime(library)
        for array in (self._gains, self._integers, self._kernels, self._reversed):
            array.setflags(write=False)
        real = ctypes.POINTER(ctypes.c_double)
        self._context = State(channels, self._unity, int(bool(self._latency) and fir == 'native'), self._history,
                             self._memory.shape[1], self._gains.ctypes.data_as(real), self._previous.ctypes.data_as(real),
                             self._state.ctypes.data_as(real), self._reversed.ctypes.data_as(real),
                             self._integers.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), self._memory.ctypes.data_as(real))
        self._pointer = ctypes.pointer(self._context)

    @property
    def channels(self):
        return self._channels

    @property
    def fir(self):
        return self._fir

    @property
    def latency(self):
        return self._latency

    @property
    def gains(self):
        return self._gains

    @property
    def integers(self):
        return self._integers

    @property
    def kernels(self):
        return self._kernels

    @property
    def previous(self):
        view = self._previous.view()
        view.setflags(write=False)
        return view

    @property
    def state(self):
        view = self._state.view()
        view.setflags(write=False)
        return view

    def reset(self):
        self._previous.fill(0)
        self._state.fill(0)
        self._memory.fill(0)

    def _input(self, audio):
        values = np.asarray(audio)
        if values.ndim != 2 or values.shape[1] != self._channels or len(values) > 2147483000:
            raise ValueError('Conditioning audio must have the configured channels')
        if values.dtype == np.int16:
            return np.require(values, dtype=np.int16, requirements=['C', 'A'])
        values = np.require(values, dtype=np.float64, requirements=['C', 'A'])
        if not np.isfinite(values).all():
            raise ValueError('Conditioning audio must be finite')
        return values

    def _process(self, values, output):
        count = len(values)
        if not count:
            return output
        if self._latency and count > self._capacity:
            capacity = max(count, self._capacity * 2)
            memory = np.zeros((self._channels, self._history + capacity))
            memory[:, :self._history] = self._memory[:, :self._history]
            self._memory, self._capacity = memory, capacity
            self._context.memory = self._memory.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            self._context.stride = self._memory.shape[1]
        native = int(bool(self._latency) and self._fir == 'native')
        function = (self._runtime.library.conditioning_pcm16 if values.dtype == np.int16
                    else self._runtime.library.conditioning_float64)
        if function(values, count, self._pointer, output):
            raise ValueError('Native conditioning failed')
        if self._latency and not native:
            self._memory[:, self._history:self._history + count] = output.T
            for channel, integer in enumerate(self._integers):
                filtered = np.convolve(self._memory[channel, :self._history + count], self._kernels[channel], mode='valid')
                start = self._history - 64 - int(integer)
                output[:, channel] = filtered[start:start + count]
            if self._runtime.library.conditioning_shift(self._pointer, count):
                raise ValueError('Calibration history update failed')
        return output

    def feed(self, audio):
        values = self._input(audio)
        return self._process(values, np.empty(values.shape, dtype=np.float64))

    def feed_into(self, audio, output):
        """Write a finite PCM block into aligned writable float64 storage without aliasing input."""
        original = np.asarray(audio)
        if (not isinstance(output, np.ndarray) or output.shape != original.shape or output.dtype != np.float64
                or not output.flags.c_contiguous or not output.flags.aligned or not output.flags.writeable
                or np.may_share_memory(original, output)):
            raise ValueError('Conditioning output needs separate aligned writable C-contiguous float64 storage')
        return self._process(self._input(original), output)


def create_conditioning(config, runtime=None):
    audio, options = config['audio'], config['conditioning']
    validate_options(options)
    return Conditioner(audio['channels'], audio.get('gains'), audio.get('delays'),
                       options.get('fir', 'native'), options.get('native_library'))
