"""Fused Hann/fold preparation, rolling PHAT and complete-grid native SRP."""

import ctypes
from pathlib import Path
import sys
import time

import numpy as np

from ps.dsp import candidates, coordinates
from ps.geometry import direction_vectors


class Preparation:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('refinement.dll' if sys.platform == 'win32' else 'librefinement.so')
        if not path.is_file():
            raise ValueError('Refinement library missing; run python scripts/build_refinement.py on the target')
        try:
            self.library = ctypes.CDLL(str(path.resolve()))
            self.library.refinement_abi.restype = ctypes.c_int
            if self.library.refinement_abi() != 1:
                raise ValueError('Unsupported refinement ABI')
            real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            number = ctypes.c_int
            self.library.refinement_prepare.argtypes = [real, number, number, real, number, number, number, real]
            self.library.refinement_prepare.restype = number
        except (OSError, AttributeError) as error:
            raise ValueError('Cannot load refinement library; rebuild scripts/build_refinement.py: ' + str(error))
        self.workspace = None

    def prepare(self, audio, window, fold, first=0):
        """Return borrowed folded input, valid until the next call on this runtime."""
        audio, window = np.ascontiguousarray(audio, dtype=float), np.ascontiguousarray(window, dtype=float)
        if (audio.ndim != 2 or not 1024 <= len(audio) <= 2147483647 or not 1 <= audio.shape[1] <= 64 or
                window.shape != (1024,) or not np.isfinite(audio).all() or not np.isfinite(window).all()):
            raise ValueError('Fused preparation requires finite PCM and a 1024-sample window')
        count = 1 + (len(audio) - 1024) // 512
        if type(fold) is not int or fold not in (1, 2, 4, 8) or type(first) is not int or not 0 <= first < count:
            raise ValueError('Invalid folded preparation factor or frame offset')
        shape = (count - first, audio.shape[1], 1024 // fold)
        if self.workspace is None or self.workspace.shape != shape:
            self.workspace = np.empty(shape)
        if self.library.refinement_prepare(audio, len(audio), audio.shape[1], window, fold, first,
                                            count - first, self.workspace):
            raise ValueError('Fused preparation failed')
        return self.workspace


class Native:
    def __init__(self, search, path=None):
        path = Path(path) if path else Path(__file__).with_name('srp_search.dll' if sys.platform == 'win32' else 'libsrp.so')
        if not path.is_file():
            raise ValueError('SRP library missing; run python scripts/build.py')
        self.library = ctypes.CDLL(str(path.resolve()))
        self.library.srp_abi.restype = ctypes.c_int
        if self.library.srp_abi() != 2:
            raise ValueError('Unsupported SRP ABI')
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.library.srp_scores.argtypes = [real, integer, integer, real, ctypes.c_int, integer, ctypes.c_int, real]
        self.library.srp_scores.restype = ctypes.c_int
        self.first = np.ascontiguousarray(search.first.T, dtype=np.int32)
        self.second = np.ascontiguousarray(search.second.T, dtype=np.int32)
        self.fraction = np.ascontiguousarray(search.fraction.T)
        self.precision = search.settings['native_precision']

    def score(self, correlations, indices):
        values = np.ascontiguousarray(correlations, dtype=float)
        ids = np.ascontiguousarray(indices, dtype=np.int32)
        if values.shape != (self.first.shape[1], 256) or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self.first)):
            raise ValueError('SRP score dimensions or direction indices are invalid')
        output = np.empty(len(ids))
        if self.library.srp_scores(values, self.first, self.second, self.fraction, self.first.shape[1], ids, len(ids), output):
            raise ValueError('SRP scoring failed')
        return output


class Search:
    def __init__(self, audio, settings, runtime, preparation):
        began = time.perf_counter()
        self.audio, self.settings = dict(audio), dict(settings)
        self.algorithm = settings.get('algorithm', 'cached')
        if self.algorithm != 'cached':
            raise ValueError('Unsupported cached search algorithm')
        self.fold = settings.get('fft_fold', 1)
        if type(self.fold) is not int or self.fold not in (1, 2, 4, 8):
            raise ValueError('FFT fold must be one of 1, 2, 4, 8')
        self.length = 1024 // self.fold
        self.window = np.hanning(1024)
        self.pairs = np.asarray([(first, second) for first in range(audio['channels']) for second in range(first + 1, audio['channels'])])
        self.azimuths = np.arange(-180, 180, settings['azimuth_step'], dtype=float)
        self.depressions = np.arange(0, 90 + settings['depression_step'] / 2, settings['depression_step'], dtype=float)
        azimuth, depression = np.meshgrid(self.azimuths, self.depressions, indexing='ij')
        self.shape = azimuth.shape
        self.vectors = direction_vectors(azimuth.ravel(), depression.ravel())
        array = coordinates(audio)
        baselines = array[self.pairs[:, 1]] - array[self.pairs[:, 0]]
        delays = baselines @ self.vectors.T * audio['sample_rate'] / audio['speed']
        self.lower = np.floor(delays).astype(np.int32)
        self.fraction = delays - self.lower
        offsets = np.arange(len(self.pairs))[:, None] * self.length
        self.first = offsets + self.lower % self.length
        self.second = offsets + (self.lower + 1) % self.length
        self.radius = int(np.ceil(np.max(np.linalg.norm(baselines, axis=1)) * audio['sample_rate'] / audio['speed'])) + 1
        if self.radius >= self.length // 2:
            raise ValueError('Folded GCC period cannot cover the physical array delay interval')
        self.rows = np.arange(len(self.pairs))[:, None]
        self.native = Native(self, settings.get('native_library'))
        self.runtime = runtime
        self.preparation = Preparation(preparation.get('native_library'))
        self.precision = settings['native_precision']
        self.cached, self.retained, self.context = None, 0, None
        self.metrics = {}
        self.setup_seconds = time.perf_counter() - began

    def correlations(self, audio):
        count = 1 + (len(audio) - 1024) // 512
        context = self.context or dict(reusable=False)
        retained = max(0, count - self.audio['update_samples'] // 512)
        reused = retained if context['reusable'] and self.cached is not None and self.retained == retained else 0
        timing = dict(fft_seconds=0.0, phat_seconds=0.0, reduction_seconds=0.0, ifft_seconds=0.0)
        began = time.perf_counter()
        prepared = self.preparation.prepare(audio, self.window, self.fold, reused)
        timing['preparation_seconds'] = time.perf_counter() - began
        spectrum = np.fft.rfft(prepared, axis=-1)
        timing['fft_seconds'] = time.perf_counter() - began
        began = time.perf_counter()
        keep = retained if retained <= len(spectrum) else 0
        total, self.cached = self.runtime.fused(spectrum, self.pairs, self.precision,
                                                self.cached if reused else None, reused, keep)
        self.retained = keep
        timing['phat_seconds'] = time.perf_counter() - began
        began = time.perf_counter()
        output = np.fft.irfft(total / count, n=self.length, axis=-1)
        timing['ifft_seconds'] = time.perf_counter() - began
        self.metrics = dict(timing, fft_frames_total=count, fft_frames_computed=len(spectrum), fft_frames_reused=reused,
                            phat_cache_bytes=self.cached.nbytes if self.cached is not None else 0,
                            fused_input=True, preparation_workspace_bytes=prepared.nbytes)
        self.last_correlations = output
        return output

    def run(self, audio):
        began = time.perf_counter()
        correlations = self.correlations(audio)
        transformed = time.perf_counter()
        scores = self.native.score(correlations, np.arange(np.prod(self.shape), dtype=np.int32))
        grid = self.azimuths, self.depressions, scores.reshape(self.shape)
        bearings = candidates(grid, self.settings['candidates'], self.settings['separation'])
        ended = time.perf_counter()
        metric = dict(gcc_seconds=transformed - began, search_seconds=ended - transformed, srp_seconds=ended - began,
                      setup_seconds=self.setup_seconds, pair_count=len(self.pairs), fft_length=self.length,
                      full_directions=len(scores), evaluated_directions=len(scores), omitted_directions=0,
                      evidence_grid='complete_fine', implementation='native', native_precision=self.precision,
                      global_search='full', algorithm='cached')
        return grid, bearings, metric

    def run_snapshot(self, snapshot, context):
        self.context = context
        try:
            grid, bearings, metric = self.run(snapshot['audio'])
            metric.update(self.metrics)
            metric['stream_precision'] = self.precision
            return grid, bearings, metric
        finally:
            self.context = None
