"""Reuse only identical overlapping PCM with explicit discontinuity invalidation."""

from collections import OrderedDict
import ctypes
import math
from pathlib import Path
import sys
import time

import numpy as np

from ps.beamforming import Beamformer
from ps.dsp import Conditioning, frames
from ps.search import Search, folded_spectrum


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


class StreamConditioning(Conditioning):
    def __init__(self, channels, runtime, gains=None, delays=None):
        self.runtime = runtime
        super().__init__(channels, gains, delays)

    def feed(self, audio):
        values = np.ascontiguousarray(audio, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.gains) or not np.isfinite(values).all():
            raise ValueError('Conditioning needs finite PCM blocks')
        output = np.empty_like(values)
        if self.runtime.library.stream_dc(values, len(values), values.shape[1], self.gains, self.previous, self.state, output):
            raise ValueError('Native conditioning failed')
        if self.latency and len(values):
            for channel, kernel in enumerate(self.kernels):
                joined = np.r_[self.histories[channel], output[:, channel]]
                filtered = np.convolve(joined, kernel, mode='valid')
                self.histories[channel] = joined[-64:].copy()
                buffered = np.r_[self.buffers[channel], filtered]
                output[:, channel] = buffered[:len(values)]
                self.buffers[channel] = buffered[len(values):].copy()
        return output


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


class StreamSearch(Search):
    def __init__(self, audio, settings, runtime, precision=None):
        super().__init__(audio, settings)
        self.runtime = runtime
        self.precision = precision or settings.get('native_precision', 'double')
        self.cached = None
        self.retained = 0
        self.context = None
        self.metrics = {}

    def correlations(self, audio):
        values = frames(audio)
        context = self.context or dict(reusable=False)
        step = self.audio['update_samples'] // 512
        retained = max(0, len(values) - step)
        reused = retained if context['reusable'] and self.cached is not None and self.retained == retained else 0
        timing = dict(fft_seconds=0.0, phat_seconds=0.0, reduction_seconds=0.0, ifft_seconds=0.0)
        began = time.perf_counter()
        spectrum = folded_spectrum(values[reused:], self.window, self.fold)
        timing['fft_seconds'] = time.perf_counter() - began
        began = time.perf_counter()
        keep = retained if retained <= len(spectrum) else 0
        total, self.cached = self.runtime.fused(spectrum, self.pairs, self.precision,
                                                self.cached if reused else None, reused, keep)
        self.retained = keep
        timing['phat_seconds'] = time.perf_counter() - began
        began = time.perf_counter()
        output = np.fft.irfft(total / len(values), n=self.length, axis=-1)
        timing['ifft_seconds'] = time.perf_counter() - began
        self.metrics = dict(timing, fft_frames_total=len(values), fft_frames_computed=len(spectrum), fft_frames_reused=reused,
                            phat_cache_bytes=self.cached.nbytes if self.cached is not None else 0)
        self.last_correlations = output
        return output

    def run_snapshot(self, snapshot, context):
        self.context = context
        try:
            grid, bearings, metric = super().run(snapshot['audio'])
            metric.update(self.metrics)
            metric['stream_precision'] = self.precision
            return grid, bearings, metric
        finally:
            self.context = None


def bearing_key(bearing):
    return (0.0, 90.0) if bearing.azimuth is None else (float(bearing.azimuth), float(bearing.depression))


class Beams(Beamformer):
    def __init__(self, audio, options):
        super().__init__(audio, options)
        support = int(math.ceil(audio['diameter'] * audio['sample_rate'] / audio['speed'])) + 65
        common = math.gcd(audio['update_samples'], audio['window_samples'])
        self.hop = max(value for value in range(1, self.fft_size - support + 1) if common % value == 0)
        self.history = self.fft_size - self.hop
        self.spectra = None
        self.outputs = {}
        self.end = None

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
        metric = dict(implementation='stream_fft', beams=len(plans), taps=65, fft_size=self.fft_size,
                      cache_hits=hits, fft_hop=self.hop, fft_blocks_total=count, fft_blocks_computed=0,
                      fft_blocks_reused=0, output_reused=False, support_samples=0, fft_seconds=0.0,
                      synthesis_seconds=0.0, spectrum_cache_bytes=0)
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
            if reused:
                fresh = np.fft.rfft(blocks[1 + reused:], axis=-1)
                pieces = [np.fft.rfft(blocks[:1], axis=-1), self.spectra, fresh]
            else:
                fresh = np.fft.rfft(blocks, axis=-1)
                pieces = [fresh]
        self.spectra = (fresh[-expected:].copy() if len(fresh) > expected + 1 else fresh[-expected:]) if expected else None
        metric.update(fft_blocks_computed=count - reused, fft_blocks_reused=reused,
                      fft_seconds=time.perf_counter() - began,
                      spectrum_cache_bytes=(self.spectra.base.nbytes if isinstance(self.spectra.base, np.ndarray)
                                             else self.spectra.nbytes) if self.spectra is not None else 0)
        support = max(int(plan['integers'].max()) for plan in plans) + 32
        keys = [bearing_key(bearing) for bearing in bearings]
        stable = reusable and all(key in self.outputs for key in keys) and context['overlap'] > support
        output = []
        began = time.perf_counter()
        for key, plan in zip(keys, plans):
            start = context['overlap'] // self.hop if stable else 0
            tails, cursor = [], 0
            for piece in pieces:
                if cursor + len(piece) > start:
                    selected = piece[max(0, start - cursor):]
                    combined = np.einsum('bcf,cf->bf', selected, plan['spectrum'], optimize=False)
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
        metric.update(output_reused=bool(stable), support_samples=support, synthesis_seconds=time.perf_counter() - began,
                      output_cache_bytes=sum(waveform.nbytes for waveform in self.outputs.values()))
        return output, metric


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
        self.runtime = Runtime(config['streaming'].get('native_library'))
        self.continuity = Continuity(config['audio'])
        self.search = StreamSearch(config['audio'], config['search'], self.runtime)
        self.beams = Beams(config['audio'], config['beamforming'])
        self.frontend = Frontend(frontend)


def conditioning(config, runtime=None):
    audio = config['audio']
    if config.get('streaming', {}).get('conditioning', 'numpy') == 'native':
        runtime = runtime or Runtime(config['streaming'].get('native_library'))
        return StreamConditioning(audio['channels'], runtime, audio.get('gains'), audio.get('delays'))
    return Conditioning(audio['channels'], audio.get('gains'), audio.get('delays'))
