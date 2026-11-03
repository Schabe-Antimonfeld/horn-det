import ast
from pathlib import Path

import numpy as np

from ps.configuration import load_config
from ps.dsp import band_features, coordinates, frames, features
from ps.streaming import Runtime, Continuity
from ps.search import Search, Preparation

ROOT = Path(__file__).resolve().parents[1]


def test_symmetric_hann_and_complete_frame_boundaries():
    audio = np.arange(2055, dtype=float)
    actual = frames(audio)
    assert actual.shape == (3, 1, 1024)
    np.testing.assert_array_equal(actual[1, 0], audio[512:1536])
    assert features(np.zeros(48128)).shape == (7, 40, 24)
    for path in (ROOT / 'ps').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 6))


def test_srp_matches_independent_single_precision_grouped_phat_and_interpolation():
    config = load_config()
    search = Search(config['audio'], config['search'], Runtime(), config['refinement'])
    audio = np.random.RandomState(71).normal(0, 0.04, size=(48128 + 24064, 14))
    continuity = Continuity(config['audio'])
    for start in (0, 24064):
        pcm = audio[start:start + 48128]
        snapshot = dict(audio=pcm, first=start, end=start + 48128, epoch=0)
        grid, bearings, metric = search.run_snapshot(snapshot, continuity.prepare(snapshot))
        weighted = frames(pcm) * np.hanning(1024)
        folded = weighted.reshape(93, 14, 4, 256).sum(axis=-2)
        spectra = np.fft.rfft(folded, axis=-1).astype(np.complex64)
        cross = spectra[:, search.pairs[:, 0]] * spectra[:, search.pairs[:, 1]].conj()
        normalized = cross / (abs(cross) + np.float32(1e-12))
        total = np.zeros((91, 129), dtype=np.complex128)
        for first in range(0, 93, 12):
            partial = np.zeros((91, 129), dtype=np.complex64)
            for frame in normalized[first:first + 12]:
                partial += frame
            total += partial
        expected = np.fft.irfft(total / 93, n=256, axis=-1)
        np.testing.assert_allclose(search.last_correlations, expected, atol=3e-8, rtol=1e-6)
        array = coordinates(config['audio'])
        delays = (array[search.pairs[:, 1]] - array[search.pairs[:, 0]]) @ search.vectors.T * 48000 / 343
        lower = np.floor(delays).astype(int)
        fraction = delays - lower
        rows = np.arange(91)[:, None]
        first = search.last_correlations[rows, lower % 256]
        second = search.last_correlations[rows, (lower + 1) % 256]
        expected_scores = (first + fraction * (second - first)).sum(axis=0)
        np.testing.assert_allclose(grid[2].ravel(), expected_scores, atol=1e-11, rtol=1e-11)
        assert metric['fft_frames_reused'] == (46 if start else 0)
        assert len(bearings) == 2 and metric['full_directions'] == 5580


def test_band_statistics_preserve_mel_position_and_match_analytic_ramp():
    values = np.arange(40)[:, None] + np.arange(24)[None, :]
    actual = band_features(values)
    np.testing.assert_allclose(actual[:40], np.arange(40) - 19.5)
    np.testing.assert_allclose(actual[40:80], np.sqrt((24 ** 2 - 1) / 12))
    np.testing.assert_allclose(actual[80:120], 1)
    np.testing.assert_allclose(actual[120:], 23)
    np.testing.assert_allclose(band_features(values - 300), actual)
    assert not np.array_equal(band_features(values[::-1]), actual)
