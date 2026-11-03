"""Continuous-window PHAT, PCM continuity and log-Mel frame reuse."""

from collections import OrderedDict
import ctypes
from pathlib import Path
import sys

import numpy as np

from ps.dsp import frames


class Runtime:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('streaming.dll' if sys.platform == 'win32' else 'libstream.so')
        if not path.is_file():
            raise ValueError('Streaming library missing; run python scripts/build_stream.py on the target')
        try:
            self.library = ctypes.CDLL(str(path.resolve()))
            self.library.stream_abi.restype = ctypes.c_int
            if self.library.stream_abi() != 1:
                raise ValueError('Unsupported streaming ABI')
            number = ctypes.c_int
            real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
            complex_double = np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS')
            for suffix, dtype in [('32', np.complex64), ('64', np.complex128)]:
                values = np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
                phat = getattr(self.library, 'stream_phat' + suffix)
                phat.argtypes, phat.restype = [values, number, number, number, integer, number, values], number
                reduce = getattr(self.library, 'stream_reduce' + suffix)
                reduce.argtypes, reduce.restype = [values, number, values, number, number, complex_double], number
                fused = getattr(self.library, 'stream_fused' + suffix)
                fused.argtypes = [values, number, number, number, integer, number, values, number, number,
                                 complex_double, values, values]
                fused.restype = number
            self.library.stream_dc.argtypes = [real, number, number, real, real, real, real]
            self.library.stream_dc.restype = number
        except (OSError, AttributeError) as error:
            raise ValueError('Cannot load streaming library; rebuild scripts/build_stream.py: ' + str(error))

    def phat(self, spectrum, pairs, precision):
        if precision not in ('single', 'double'):
            raise ValueError('Unsupported streaming precision')
        dtype = np.complex64 if precision == 'single' else np.complex128
        values = np.ascontiguousarray(spectrum, dtype=dtype)
        if values.ndim != 3 or min(values.shape) < 1 or values.shape[1] < 2 or values.shape[2] < 2:
            raise ValueError('Streaming PHAT needs complete channel spectra')
        maximum = float(np.max(abs(values)))
        if not np.isfinite(maximum) or maximum > (1e8 if precision == 'single' else 1e70):
            raise ValueError('Streaming PHAT exceeds its safe finite range')
        pairs = np.ascontiguousarray(pairs, dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError('Streaming PHAT needs microphone pairs')
        output = np.empty((len(values), len(pairs), values.shape[-1]), dtype=dtype)
        function = self.library.stream_phat32 if precision == 'single' else self.library.stream_phat64
        if function(values, len(values), values.shape[1], values.shape[2], pairs, len(pairs), output):
            raise ValueError('Streaming PHAT failed')
        return output

    def reduce(self, previous, current, precision):
        if precision not in ('single', 'double'):
            raise ValueError('Unsupported streaming precision')
        dtype = np.complex64 if precision == 'single' else np.complex128
        previous, current = np.ascontiguousarray(previous, dtype=dtype), np.ascontiguousarray(current, dtype=dtype)
        if current.ndim != 3 or previous.ndim != 3 or previous.shape[1:] != current.shape[1:] or not len(current):
            raise ValueError('Streaming reduction shapes differ')
        output = np.empty(current.shape[1:], dtype=np.complex128)
        function = self.library.stream_reduce32 if precision == 'single' else self.library.stream_reduce64
        if function(previous, len(previous), current, len(current), output.size * 2, output):
            raise ValueError('Streaming PHAT reduction failed')
        return output

    def fused(self, spectrum, pairs, precision, previous, reused, retained):
        if precision not in ('single', 'double'):
            raise ValueError('Unsupported streaming precision')
        dtype = np.complex64 if precision == 'single' else np.complex128
        values = np.ascontiguousarray(spectrum, dtype=dtype)
        if values.ndim != 3 or len(values) < 1 or values.shape[1] < 2 or values.shape[2] < 2:
            raise ValueError('Streaming PHAT needs complete channel spectra')
        maximum = float(np.max(abs(values)))
        if not np.isfinite(maximum) or maximum > (1e8 if precision == 'single' else 1e70):
            raise ValueError('Streaming PHAT exceeds its safe finite range')
        pairs = np.ascontiguousarray(pairs, dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError('Streaming PHAT needs microphone pairs')
        shape = (len(pairs), values.shape[-1])
        old = np.empty((0,) + shape, dtype=dtype) if previous is None else np.ascontiguousarray(previous, dtype=dtype)
        if type(reused) is not int or type(retained) is not int or reused < 0 or not 0 <= retained <= len(values):
            raise ValueError('Invalid streaming frame counts')
        if old.shape != ((reused + 11) // 12,) + shape:
            raise ValueError('Streaming cached group shape differs from its frame count')
        future = np.empty(((retained + 11) // 12,) + shape, dtype=dtype)
        partial = np.empty(shape, dtype=dtype)
        output = np.empty(shape, dtype=np.complex128)
        function = self.library.stream_fused32 if precision == 'single' else self.library.stream_fused64
        if function(values, len(values), values.shape[1], values.shape[2], pairs, len(pairs), old,
                    reused, retained, output, future, partial):
            raise ValueError('Streaming fused PHAT failed')
        return output, future


class Continuity:
    def __init__(self, settings):
        self.update = settings['update_samples']
        self.previous = None
        self.tail = None

    def prepare(self, snapshot):
        values = np.asarray(snapshot['audio'])
        first, end, epoch = snapshot['first'], snapshot['end'], snapshot['epoch']
        if end - first != len(values) or values.ndim != 2:
            raise ValueError('Streaming sample counters disagree with the audio window')
        reason, delta, previous_end = 'first_window', 0, None
        if self.previous is not None:
            previous_first, previous_end, previous_epoch = self.previous
            delta = first - previous_first
            reason = ('epoch_changed' if epoch != previous_epoch else 'nonforward_window' if delta <= 0
                      else 'skipped_window' if delta != self.update else 'window_size_changed'
                      if end - previous_end != delta else 'continuous')
            if reason == 'continuous' and (self.tail is None or self.tail.shape != values[:len(values) - delta].shape
                                          or not np.array_equal(self.tail, values[:len(values) - delta])):
                reason = 'overlap_changed'
        overlap = max(0, len(values) - self.update)
        self.tail = values[-overlap:].copy() if overlap else None
        self.previous = first, end, epoch
        return dict(reusable=reason == 'continuous', reset_reason=reason, delta=delta,
                    overlap=max(0, len(values) - delta), previous_end=previous_end, first=first, end=end, epoch=epoch)


def bearing_key(bearing):
    return (0.0, 90.0) if bearing.azimuth is None else (float(bearing.azimuth), float(bearing.depression))


class Frontend:
    def __init__(self, frontend='logmel'):
        if frontend not in ('logmel', 'centered_logmel'):
            raise ValueError('Unsupported streaming frontend')
        self.frontend = frontend
        self.window = np.hanning(1024)
        edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 24000 / 700), 42) / 2595) - 1)
        frequency = np.fft.rfftfreq(1024, 1 / 48000)
        self.bank = np.maximum(0, np.minimum((frequency[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                    (edges[2:, None] - frequency[None]) / (edges[2:, None] - edges[1:-1, None])))
        self.cache, self.end = OrderedDict(), None

    def run(self, waveform, bearing, context, support):
        values = frames(waveform)[:, 0]
        key = bearing_key(bearing)
        if self.end != context['end']:
            if not context['reusable']:
                self.cache.clear()
            self.end = context['end']
        logarithms = np.empty((len(values), 40))
        previous = self.cache.get(key)
        reused = np.zeros(len(values), dtype=bool)
        if context['reusable'] and previous is not None and previous[0] == context['previous_end']:
            start = np.arange(len(values)) * 512
            reused = (start >= support) & (start + 1024 <= context['overlap'])
            ids = np.flatnonzero(reused) + context['delta'] // 512
            logarithms[reused] = previous[1][ids]
        missing = np.flatnonzero(~reused)
        selected = values if len(missing) == len(values) else values[missing]
        power = abs(np.fft.rfft(selected * self.window, axis=-1)) ** 2
        logarithms[missing] = np.log(power @ self.bank.T + 1e-12)
        self.cache[key] = (context['end'], logarithms.copy())
        self.cache.move_to_end(key)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        metric = dict(frames_computed=len(missing), frames_reused=int(reused.sum()), frames_total=len(values))
        if len(logarithms) < 24:
            return np.empty((0, 40, 24)), metric
        starts = sorted(set(range(0, len(logarithms) - 24 + 1, 12)) | {len(logarithms) - 24})
        contexts = np.asarray([logarithms[start:start + 24].T for start in starts])
        if self.frontend == 'centered_logmel':
            contexts -= contexts.mean(axis=(1, 2), keepdims=True)
        return contexts, metric


class Processing:
    def __init__(self, config, frontend='logmel'):
        from ps.search import Search
        from ps.beamforming import Beamformer
        self.runtime = Runtime(config['streaming'].get('native_library'))
        self.continuity = Continuity(config['audio'])
        self.search = Search(config['audio'], config['search'], self.runtime, config['refinement'])
        self.beams = Beamformer(config['audio'], config['beamforming'])
        self.frontend = Frontend(frontend)
