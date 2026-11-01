"""Optional fused FFT preparation and independent exact-direction beam reuse."""

import ctypes
from pathlib import Path
import sys
import time

import numpy as np

from ps.streaming import (Beams as StreamBeams, Continuity, Frontend, Runtime as StreamRuntime,
                          StreamSearch, bearing_key)


def validate_options(options, streaming):
    if not isinstance(options, dict) or set(options) - {'fused_input', 'independent_beams', 'native_library'}:
        raise ValueError('Unsupported refinement settings')
    if not streaming.get('enabled') or any(type(options.get(key, True)) is not bool
                                           for key in ('fused_input', 'independent_beams')):
        raise ValueError('Refinement requires streaming and boolean switches')
    if 'native_library' in options and (not options.get('fused_input', True) or
            not isinstance(options['native_library'], str) or not Path(options['native_library']).is_absolute()):
        raise ValueError('Refinement library override requires fused input and an absolute path')


class Runtime:
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


class Search(StreamSearch):
    def __init__(self, audio, settings, runtime, preparation):
        super().__init__(audio, settings, runtime)
        self.preparation = preparation

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


class Beams(StreamBeams):
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
        metric = dict(implementation='refined_stream_fft', beams=len(plans), taps=65, fft_size=self.fft_size,
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
        metric.update(output_reused=all(flags), output_reused_beams=sum(flags), output_reuse_flags=flags,
                      support_samples=support, synthesis_seconds=time.perf_counter() - began,
                      output_cache_bytes=sum(waveform.nbytes for waveform in self.outputs.values()))
        return output, metric


class Processing:
    def __init__(self, config, frontend='logmel'):
        options = config['refinement']
        validate_options(options, config.get('streaming', {}))
        self.runtime = StreamRuntime(config['streaming'].get('native_library'))
        self.continuity = Continuity(config['audio'])
        self.search = (Search(config['audio'], config['search'], self.runtime, Runtime(options.get('native_library')))
                       if options.get('fused_input', True) else StreamSearch(config['audio'], config['search'], self.runtime))
        self.beams = (Beams if options.get('independent_beams', True) else StreamBeams)(config['audio'], config['beamforming'])
        self.frontend = Frontend(frontend)
