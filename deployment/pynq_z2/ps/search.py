"""Cached SRP with mixed baselines, folded spectra and bounded local refinement."""

import time

import numpy as np

from ps.dsp import candidates, coordinates, frames
from ps.geometry import Bearing, direction_vectors, wrap_angle


def microphone_pairs(channels, mode):
    if mode == 'all':
        return np.asarray([(first, second) for first in range(channels) for second in range(first + 1, channels)])
    if mode != 'mixed35' or channels != 14:
        raise ValueError('Mixed baselines require the fixed fourteen-channel array')
    return np.asarray(sorted({tuple(sorted((first, (first + step) % channels)))
                              for first in range(channels) for step in (1, 3, 7)}))


def folded_spectrum(values, window, fold):
    """Compute every fold-th original FFT bin while retaining the full Hann window."""
    weighted = values * window
    if fold != 1:
        weighted = weighted.reshape(weighted.shape[:-1] + (fold, weighted.shape[-1] // fold)).sum(axis=-2)
    return np.fft.rfft(weighted, axis=-1)


class Search:
    """Keep measured coarse evidence distinct from omitted fine-grid directions."""

    def __init__(self, audio, settings):
        began = time.perf_counter()
        self.audio, self.settings = dict(audio), dict(settings)
        self.algorithm = settings.get('algorithm', 'cached')
        if self.algorithm not in ('cached', 'hierarchical'):
            raise ValueError('Unsupported cached search algorithm')
        self.fold = settings.get('fft_fold', 1)
        if type(self.fold) is not int or self.fold not in (1, 2, 4, 8):
            raise ValueError('FFT fold must be one of 1, 2, 4, 8')
        self.length = 1024 // self.fold
        self.window = np.hanning(1024)
        self.pairs = microphone_pairs(audio['channels'], settings.get('pair_mode', 'all'))
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
        self.bounds = settings.get('bounds', True)
        if self.algorithm == 'hierarchical':
            self._prepare_cells()
        self.setup_seconds = time.perf_counter() - began

    def _prepare_cells(self):
        ratios = [self.settings.get('coarse_azimuth', 6) / self.settings['azimuth_step'],
                  self.settings.get('coarse_depression', 15) / self.settings['depression_step']]
        if any(value < 1 or abs(value - round(value)) > 1e-9 for value in ratios):
            raise ValueError('Coarse steps must be integer multiples of the fine steps')
        self.groups = [list(range(0, length, int(round(step)))) for length, step in zip(self.shape, ratios)]
        self.parts = [[np.arange(start, min(start + int(round(step)), length)) for start in starts]
                      for starts, length, step in zip(self.groups, self.shape, ratios)]
        self.coarse_azimuths = self.azimuths[[part[len(part) // 2] for part in self.parts[0]]]
        self.coarse_depressions = self.depressions[[part[len(part) // 2] for part in self.parts[1]]]
        self.cells, anchors, first, final = [], [], [], []
        for azimuth in self.parts[0]:
            for depression in self.parts[1]:
                ids = (azimuth[:, None] * self.shape[1] + depression).ravel()
                self.cells.append(ids)
                anchors.append(azimuth[len(azimuth) // 2] * self.shape[1] + depression[len(depression) // 2])
                first.append(self.lower[:, ids].min(axis=1) + self.radius)
                final.append(self.lower[:, ids].max(axis=1) + 1 + self.radius)
        self.anchors = np.asarray(anchors)
        self.starts, self.ends = np.asarray(first).T, np.asarray(final).T
        lengths = self.ends - self.starts + 1
        self.levels = np.floor(np.log2(lengths)).astype(int)
        self.right = self.ends - (1 << self.levels) + 1

    def correlations(self, audio):
        values = frames(audio)
        if values.shape[1] != self.audio['channels']:
            raise ValueError('PCM channel count differs from the cached array')
        total = np.zeros((len(self.pairs), self.length // 2 + 1), dtype=complex)
        for start in range(0, len(values), 12):
            spectrum = folded_spectrum(values[start:start + 12], self.window, self.fold)
            cross = spectrum[:, self.pairs[:, 0]] * spectrum[:, self.pairs[:, 1]].conj()
            total += (cross / (np.abs(cross) + 1e-12)).sum(axis=0)
        return np.fft.irfft(total / len(values), n=self.length, axis=-1)

    def score(self, correlations, indices):
        ids = np.asarray(indices, dtype=int)
        values = correlations.ravel()
        first = values[self.first[:, ids]]
        second = values[self.second[:, ids]]
        first += self.fraction[:, ids] * (second - first)
        return first.sum(axis=0)

    def cell_bounds(self, correlations):
        """Bound linear interpolation by pairwise maxima over each cell's lag interval."""
        values = correlations[:, np.arange(-self.radius, self.radius + 1) % self.length]
        width = values.shape[1]
        maximum = np.full((int(np.floor(np.log2(width))) + 1, len(self.pairs), width), -np.inf)
        maximum[0] = values
        for level in range(1, len(maximum)):
            span = 1 << (level - 1)
            valid = width - 2 * span + 1
            maximum[level, :, :valid] = np.maximum(maximum[level - 1, :, :valid],
                                                   maximum[level - 1, :, span:span + valid])
        bound = np.maximum(maximum[self.levels, self.rows, self.starts],
                           maximum[self.levels, self.rows, self.right]).sum(axis=0)
        return bound + 1e-12 * len(self.pairs)

    def _measure(self, correlations, scores, indices):
        ids = np.asarray(indices, dtype=int)
        missing = ids[np.isnan(scores[ids])]
        if len(missing):
            scores[missing] = self.score(correlations, missing)
        return ids

    def _region(self, row, column, radius):
        azimuths = [(row + offset) % len(self.parts[0]) for offset in range(-radius, radius + 1)]
        depressions = range(max(0, column - radius), min(len(self.parts[1]), column + radius + 1))
        cells = np.asarray([first * len(self.parts[1]) + second for first in azimuths for second in depressions])
        return cells, azimuths

    def _refine(self, correlations, scores, bound, row, column, radius):
        cells, azimuths = self._region(row, column, radius)
        known = self.anchors[cells]
        winner = int(known[np.argmax(scores[known])])
        pending = cells.copy()
        pruned = 0
        while len(pending):
            if self.bounds:
                keep = bound[pending] >= scores[winner]
                pruned += int((~keep).sum())
                pending = pending[keep]
            if not len(pending):
                break
            order = np.argsort(-bound[pending], kind='stable')
            selected, pending = pending[order[:2]], pending[order[2:]]
            ids = self._measure(correlations, scores, np.concatenate([self.cells[index] for index in selected]))
            best = int(ids[np.argmax(scores[ids])])
            if scores[best] > scores[winner]:
                winner = best
        first, final = divmod(winner, self.shape[1])
        lower = self.parts[1][max(0, column - radius)][0]
        upper = self.parts[1][min(len(self.parts[1]) - 1, column + radius)][-1]
        edge = first in (self.parts[0][azimuths[0]][0], self.parts[0][azimuths[-1]][-1])
        edge = edge or (final == lower and lower > 0) or (final == upper and upper < self.shape[1] - 1)
        return winner, bool(edge), pruned

    def _bearing(self, index, response):
        row, column = divmod(int(index), self.shape[1])
        depression = float(self.depressions[column])
        return Bearing(None if depression >= 85 else float(self.azimuths[row]), depression, float(response))

    def run(self, audio):
        began = time.perf_counter()
        correlations = self.correlations(audio)
        transformed = time.perf_counter()
        scores = np.full(np.prod(self.shape), np.nan)
        count, separation = self.settings['candidates'], self.settings['separation']
        pruned, expansions, fallback = 0, 0, False
        if self.algorithm == 'cached':
            self._measure(correlations, scores, np.arange(len(scores)))
            grid = (self.azimuths, self.depressions, scores.reshape(self.shape))
            bearings = candidates(grid, count, separation)
            evidence = 'complete_fine'
        else:
            self._measure(correlations, scores, self.anchors)
            grid = (self.coarse_azimuths, self.coarse_depressions,
                    scores[self.anchors].reshape(len(self.parts[0]), len(self.parts[1])))
            peaks = candidates(grid, self.settings.get('regions', 6), separation)
            bound = self.cell_bounds(correlations)
            winners = []
            for peak in peaks:
                row = int(np.argmin(abs(wrap_angle(self.coarse_azimuths - (peak.azimuth or 0)))))
                column = int(np.argmin(abs(self.coarse_depressions - peak.depression)))
                winner, edge, rejected = self._refine(correlations, scores, bound, row, column, 1)
                pruned += rejected
                if edge:
                    expansions += 1
                    winner, edge, rejected = self._refine(correlations, scores, bound, row, column, 2)
                    pruned += rejected
                    fallback = fallback or edge
                winners.append(winner)
            bearings, vectors = [], []
            for index in sorted(set(winners), key=lambda value: (-scores[value], value)):
                vector = self.vectors[index]
                if all(np.rad2deg(np.arccos(np.clip(np.dot(vector, other), -1, 1))) >= separation for other in vectors):
                    bearings.append(self._bearing(index, scores[index]))
                    vectors.append(vector)
                if len(bearings) == count:
                    break
            fallback = fallback or (len(bearings) < count and len(peaks) >= count)
            evidence = 'measured_coarse'
            if fallback:
                self._measure(correlations, scores, np.arange(len(scores)))
                grid = (self.azimuths, self.depressions, scores.reshape(self.shape))
                bearings = candidates(grid, count, separation)
                evidence = 'complete_fine_fallback'
        ended = time.perf_counter()
        evaluated = int(np.isfinite(scores).sum())
        return grid, bearings, dict(gcc_seconds=transformed - began, search_seconds=ended - transformed,
                    srp_seconds=ended - began, setup_seconds=self.setup_seconds, pair_count=len(self.pairs),
                    fft_length=self.length, spectral_bins=self.length // 2 + 1, full_directions=len(scores),
                    evaluated_directions=evaluated, direction_pair_evaluations=evaluated * len(self.pairs),
                    omitted_directions=len(scores) - evaluated, regional_bound_prunes=pruned,
                    boundary_expansions=expansions, full_fallback=fallback, evidence_grid=evidence,
                    global_search='full' if self.algorithm == 'cached' or fallback else 'coarse_full_with_heuristic_regions')
