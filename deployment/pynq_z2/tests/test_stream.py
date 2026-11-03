import importlib.util
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.configuration import load_config, validate_config
from ps.dsp import beamform, features, frames
from ps.geometry import Bearing
from ps.beamforming import Beamformer as Beams
from ps.search import Preparation as Runtime, Search
from ps.streaming import Continuity


@pytest.fixture(scope='module')
def preparation():
    from ps.search import Preparation
    return Preparation()


def snapshots(count=4):
    audio = np.random.default_rng(41).normal(0, 0.04, (48128 + (count - 1) * 24064, 14))
    return [dict(audio=audio[start:start + 48128].copy(), first=start, end=start + 48128, epoch=0, clipped=False)
            for start in range(0, len(audio) - 48128 + 1, 24064)]

@pytest.mark.parametrize('fold', [1, 2, 4, 8])
def test_fused_input_matches_full_hann_fold_with_offset_strides_and_workspace(preparation, fold):
    for audio in (snapshots(1)[0]['audio'], np.zeros((2048, 14)), np.full((1024, 14), -0.0),
                  np.random.RandomState(4).normal(size=(4099, 28))[::2, ::2]):
        window = np.hanning(1024)
        for first in (0, (len(audio) - 1024) // 512):
            weighted = frames(audio)[first:] * window
            expected = weighted if fold == 1 else weighted.reshape(weighted.shape[:-1] + (fold, 1024 // fold)).sum(axis=-2)
            actual = preparation.prepare(audio, window, fold, first)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            assert preparation.prepare(audio, window, fold, first) is actual
            np.testing.assert_array_equal(np.fft.rfft(actual, axis=-1), np.fft.rfft(expected, axis=-1))


@pytest.mark.parametrize('fold,first', [(0, 0), (True, 0), (4, -1), (4, 93), (4, True)])
def test_preparation_rejects_invalid_offsets_and_factors(preparation, fold, first):
    with pytest.raises(ValueError):
        preparation.prepare(np.zeros((48128, 14)), np.hanning(1024), fold, first)


def test_preparation_rejects_nonfinite_and_short_input(preparation):
    for audio in (np.zeros((1023, 14)), np.full((1024, 14), np.nan), np.zeros(2048), np.zeros((1024, 0))):
        with pytest.raises(ValueError):
            preparation.prepare(audio, np.hanning(1024), 4)


@pytest.mark.parametrize('scenario', ['stable', 'partial', 'all', 'reorder', 'add', 'remove', 'vertical'])
def test_independent_beams_match_analytic_fir_and_features_when_candidates_change(scenario):
    settings = load_config()['audio']
    engine = Beams(settings, dict(implementation='fft', fft_size=1024))
    state = Continuity(settings)
    for index, snapshot in enumerate(snapshots(3)):
        bearings = [Bearing(23 + (index * 2 if scenario == 'all' else 0), 33, 1),
                    Bearing(-113 - (index * 2 if scenario in ('partial', 'all') else 0), 42, 1)]
        if scenario == 'reorder' and index % 2:
            bearings.reverse()
        if scenario == 'add' and index == 0 or scenario == 'remove' and index > 0:
            bearings = bearings[:1]
        if scenario == 'vertical':
            bearings[0] = Bearing(None, 90, 1)
        context = state.prepare(snapshot)
        outputs, metric = engine.run_snapshot(snapshot, bearings, context)
        for bearing, (waveform, latency) in zip(bearings, outputs):
            expected, common = beamform(snapshot['audio'], bearing, settings)
            np.testing.assert_allclose(waveform, expected, atol=3e-14, rtol=3e-12)
            np.testing.assert_allclose(features(waveform), features(expected), atol=3e-10, rtol=2e-10)
            assert latency == common and not waveform[:32].any() and not waveform.flags.writeable
        expected = (0 if index == 0 or scenario == 'all' else 1 if scenario in ('partial', 'remove') or
                    scenario == 'add' and index == 1 else len(bearings))
        assert metric['output_reused_beams'] == expected
        assert metric['output_reused'] == (expected == len(bearings))


@pytest.mark.parametrize('change', ['epoch', 'skip', 'reverse', 'pcm', 'partial', 'empty', 'evicted'])
def test_independent_beam_cache_invalidates_on_untrusted_history(change):
    settings = load_config()['audio']
    engine = Beams(settings, dict(implementation='fft', fft_size=1024))
    state = Continuity(settings)
    rows = snapshots()
    bearings = [Bearing(23, 33, 1)]
    engine.run_snapshot(rows[0], bearings, state.prepare(rows[0]))
    row = dict(rows[1])
    if change == 'epoch':
        row['epoch'] = 1
    elif change == 'skip':
        row = rows[2]
    elif change == 'reverse':
        row = rows[0]
    elif change == 'pcm':
        row['audio'] = row['audio'].copy()
        row['audio'][11000, 13] += 0.001
    elif change == 'partial':
        row['audio'] = row['audio'][:-128]
        row['end'] -= 128
    elif change == 'empty':
        engine.run_snapshot(rows[0], [], dict(reusable=False, end=rows[0]['end']))
    else:
        many = [Bearing(index * 20, 30, 1) for index in range(9)]
        engine.run_snapshot(rows[0], many, dict(reusable=False, end=rows[0]['end']))
    outputs, metric = engine.run_snapshot(row, bearings, state.prepare(row))
    assert metric['output_reused_beams'] == 0
    expected = beamform(row['audio'], bearings[0], settings)[0]
    np.testing.assert_allclose(outputs[0][0], expected, atol=3e-14, rtol=3e-12)


def test_missing_native_library_fails_early(tmp_path):
    with pytest.raises(ValueError, match='build_refinement.py'):
        Runtime(tmp_path / 'missing.so')
