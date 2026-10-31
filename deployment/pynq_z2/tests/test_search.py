from pathlib import Path
import importlib.util
import json
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.configuration import load_config
from ps.dsp import candidates, frames, spatial_map
from ps.search import Search, folded_spectrum, microphone_pairs


def test_mixed_pairs_cover_every_channel_orientation_and_three_baseline_lengths():
    pairs = microphone_pairs(14, 'mixed35')
    assert pairs.shape == (35, 2) and np.all(pairs[:, 0] < pairs[:, 1])
    assert len(set(map(tuple, pairs))) == 35
    np.testing.assert_array_equal(np.bincount(pairs.ravel()), np.full(14, 5))
    distance = np.minimum(pairs[:, 1] - pairs[:, 0], 14 - pairs[:, 1] + pairs[:, 0])
    assert {step: int((distance == step).sum()) for step in (1, 3, 7)} == {1: 14, 3: 14, 7: 7}
    with pytest.raises(ValueError, match='fourteen'):
        microphone_pairs(8, 'mixed35')


@pytest.mark.parametrize('fold', [1, 2, 4, 8])
def test_folded_fft_is_exact_spectral_comb_of_full_window(fold):
    values = np.random.default_rng(3).normal(size=(3, 14, 1024))
    window = np.hanning(1024)
    expected = np.fft.rfft(values * window, axis=-1)[..., ::fold]
    np.testing.assert_allclose(folded_spectrum(values, window, fold), expected, atol=3e-14)


def test_cached_full_search_matches_reference_and_does_not_rebuild_delay_geometry(monkeypatch):
    import ps.search as module
    config = load_config()
    audio = np.random.default_rng(4).normal(size=(4096, 14))
    search = Search(config['audio'], config['search'])
    expected = spatial_map(audio, config['audio'], config['search'])
    monkeypatch.setattr(module, 'coordinates', lambda *args: pytest.fail('Geometry must be cached'))
    actual, bearings, metric = search.run(audio)
    for first, second in zip(actual, expected):
        np.testing.assert_allclose(first, second, atol=1e-12)
    selected = candidates(expected, 2, 12)
    assert len(bearings) == len(selected)
    assert all(a.azimuth == b.azimuth and a.depression == b.depression for a, b in zip(bearings, selected))
    assert metric['pair_count'] == 91 and metric['evaluated_directions'] == 5580


def test_interval_bounds_dominate_every_fine_score_and_local_refinement_matches_exhaustive():
    config = load_config()
    settings = dict(config['search'], algorithm='hierarchical', pair_mode='mixed35', fft_fold=4)
    search = Search(config['audio'], settings)
    correlations = np.random.default_rng(8).normal(size=(35, 256))
    scores = search.score(correlations, np.arange(5580))
    bounds = search.cell_bounds(correlations)
    for index, ids in enumerate(search.cells):
        assert bounds[index] >= scores[ids].max() - 1e-12
    for row, column in ((0, 0), (59, 6), (25, 3)):
        partial = np.full(5580, np.nan)
        search._measure(correlations, partial, search.anchors)
        best, edge, _ = search._refine(correlations, partial, bounds, row, column, 1)
        assert type(edge) is bool
        cells, _ = search._region(row, column, 1)
        ids = np.concatenate([search.cells[index] for index in cells])
        assert partial[best] == pytest.approx(scores[ids].max(), abs=1e-12)


def test_folded_gcc_uses_all_original_time_frames_and_respects_signed_delays():
    config = load_config()
    settings = dict(config['search'], pair_mode='mixed35', fft_fold=8)
    search = Search(config['audio'], settings)
    signal = np.random.default_rng(11).normal(size=4096)
    audio = np.column_stack([np.roll(signal, shift) for shift in range(14)])
    actual = search.correlations(audio)
    spectrum = np.fft.rfft(frames(audio) * np.hanning(1024), axis=-1)[..., ::8]
    cross = spectrum[:, search.pairs[:, 0]] * spectrum[:, search.pairs[:, 1]].conj()
    expected = np.fft.irfft((cross / (abs(cross) + 1e-12)).mean(axis=0), n=128, axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    for row, (first, second) in enumerate(search.pairs):
        assert int(np.argmax(actual[row])) == (first - second) % 128
    with pytest.raises(ValueError, match='physical array delay'):
        Search(dict(config['audio'], diameter=0.9), settings)


def test_hierarchical_evidence_contains_only_measured_values_and_silence_has_no_candidates():
    config = load_config()
    search = Search(config['audio'], dict(config['search'], algorithm='hierarchical', pair_mode='mixed35', fft_fold=4))
    audio = np.random.default_rng(12).normal(size=(4096, 14))
    grid, bearings, metric = search.run(audio)
    correlations = search.correlations(audio)
    if metric['evidence_grid'] == 'measured_coarse':
        np.testing.assert_allclose(grid[2].ravel(), search.score(correlations, search.anchors), atol=1e-12)
        assert grid[2].size == 420
    assert np.isfinite(grid[2]).all()
    grid, bearings, metric = search.run(np.zeros((2048, 14)))
    assert bearings == [] and metric['evaluated_directions'] == 420
    assert not grid[2].any()


def test_hierarchical_search_keeps_rear_wrap_and_weaker_second_source(monkeypatch):
    from ps.dsp import coordinates
    from ps.geometry import direction_vectors, wrap_angle
    config = load_config()
    search = Search(config['audio'], dict(config['search'], algorithm='hierarchical', pair_mode='mixed35', fft_fold=8))
    array = coordinates(config['audio'])
    baselines = array[search.pairs[:, 1]] - array[search.pairs[:, 0]]
    lag = np.arange(-64, 64)
    values = np.zeros((35, 128))
    for angle, depression, amplitude in ((179, 30, 1), (-50, 45, 0.6)):
        delays = baselines @ direction_vectors(angle, depression) * 48000 / 343
        values[:, lag % 128] += amplitude * np.exp(-0.5 * ((lag[None] - delays[:, None]) / 0.7) ** 2)
    monkeypatch.setattr(search, 'correlations', lambda audio: values)
    _, bearings, metric = search.run(np.zeros((2048, 14)))
    assert len(bearings) == 2
    for angle in (179, -50):
        assert min(abs(wrap_angle(bearing.azimuth - angle)) for bearing in bearings) <= 2
    assert metric['evaluated_directions'] <= 5580


@pytest.fixture(scope='module')
def native_library(tmp_path_factory):
    if not shutil.which('gcc'):
        pytest.skip('Optional GCC compiler absent')
    spec = importlib.util.spec_from_file_location('native_search_builder', ROOT / 'scripts/build_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp('native-search') / ('srp_search.dll' if sys.platform == 'win32' else 'libsrp.so')
    result = module.build(output)
    assert result['abi'] == 2 and result['result'] == 'passed'
    return output


@pytest.mark.parametrize('precision,tolerance', [('double', 1e-12), ('single', 1e-6)])
def test_native_phat_scores_and_local_pruning_match_independent_numpy(native_library, precision, tolerance):
    config = load_config()
    settings = dict(config['search'], algorithm='hierarchical', fft_fold=4)
    reference = Search(config['audio'], settings)
    native = Search(config['audio'], dict(settings, implementation='native', native_precision=precision,
                                         native_library=str(native_library)))
    audio = np.random.default_rng(21).normal(size=(4096, 14))
    expected, actual = reference.correlations(audio), native.correlations(audio)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    ids = np.arange(0, 5580, 7)
    np.testing.assert_allclose(native.score(actual, ids), reference.score(expected, ids), atol=tolerance * 91, rtol=tolerance)
    bound = native.cell_bounds(actual)
    scores = np.full(5580, np.nan)
    native._measure(actual, scores, native.anchors)
    winner, _, _ = native._refine(actual, scores, bound, 59, 3, 1)
    cells, _ = native._region(59, 3, 1)
    exhaustive = native.score(actual, np.concatenate([native.cells[index] for index in cells]))
    assert scores[winner] == pytest.approx(exhaustive.max(), abs=1e-12)
    _, _, metric = native.run(audio)
    json.dumps(metric, allow_nan=False)
    assert metric['implementation'] == 'native' and metric['native_precision'] == precision
    with pytest.raises(ValueError, match='safe arithmetic'):
        native.native.phat(np.full((2, 14, 129), np.inf, dtype=complex))


def test_native_library_failure_is_explicit_instead_of_silent_numpy_fallback(tmp_path):
    config = load_config()
    with pytest.raises(ValueError, match='build_search.py'):
        Search(config['audio'], dict(config['search'], implementation='native', native_library=str(tmp_path / 'absent.so')))


@pytest.mark.parametrize('change', [dict(algorithm='unknown'), dict(algorithm='cached', fft_fold=16),
                                   dict(algorithm='cached', pair_mode='unknown'),
                                   dict(algorithm='cached', backend='fpga'),
                                   dict(algorithm='hierarchical', coarse_azimuth=7),
                                   dict(algorithm='hierarchical', regions=1),
                                   dict(algorithm='cached', implementation='native', native_library='relative.so')])
def test_optimized_search_configuration_rejects_incompatible_contracts(change):
    from ps.configuration import validate_config
    config = load_config()
    config['search'].update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_optimized_pipeline_exposes_measured_grid_and_phase_timing(native_library, tmp_path):
    from ps.pipeline import Pipeline
    config = load_config()
    config['search'].update(algorithm='hierarchical', implementation='native', fft_fold=4, native_library=str(native_library))
    class Detector:
        def score(self, contexts):
            return dict(score=0.9, clipped=False)
    pipeline = Pipeline(config, Detector())
    event, evidence = pipeline.analyze(dict(audio=np.random.default_rng(31).normal(size=(4096, 14)),
                                            first=0, end=4096, epoch=0, clipped=False))
    assert event['srp_diagnostics']['implementation'] == 'native'
    assert set(event['phase_seconds']) == {'srp', 'beamforming', 'features', 'detector'}
    assert all(value >= 0 for value in event['phase_seconds'].values())
    assert sum(event['phase_seconds'].values()) <= event['processing_seconds']
    assert evidence is not None and np.isfinite(evidence.grid[2]).all()
    event, evidence = pipeline.analyze(dict(audio=np.random.default_rng(31).normal(size=(4096, 14)), first=0,
                                            end=4096 + config['audio']['update_samples'], epoch=0, clipped=False))
    capture = pipeline.capture(event, evidence, None, None, tmp_path, 0)
    assert capture is not None and capture['processing_seconds'] >= 0
    assert all(value >= 0 for value in capture['phase_seconds'].values())
    assert sum(capture['phase_seconds'].values()) == pytest.approx(capture['processing_seconds'])
