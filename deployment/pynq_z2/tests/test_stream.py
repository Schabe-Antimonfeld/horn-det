import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.configuration import load_config
from ps.dsp import Conditioning, beamform, features
from ps.geometry import Bearing
from ps.search import Search
from ps.streaming import Beams, Continuity, Frontend, Runtime, StreamSearch, StreamConditioning


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    compiler = shutil.which('gcc')
    if not compiler:
        pytest.skip('Optional GCC absent')
    spec = importlib.util.spec_from_file_location('stream_builder', ROOT / 'scripts/build_stream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp('stream') / ('streaming.dll' if sys.platform == 'win32' else 'libstream.so')
    report = module.build(path, compiler)
    assert report['result'] == 'passed' and report['abi'] == 1
    return Runtime(path)


@pytest.fixture(scope='module')
def search_library(tmp_path_factory):
    compiler = shutil.which('gcc')
    if not compiler:
        pytest.skip('Optional GCC absent')
    spec = importlib.util.spec_from_file_location('stream_search_builder', ROOT / 'scripts/build_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp('stream-search') / ('srp_search.dll' if sys.platform == 'win32' else 'libsrp.so')
    module.build(path, compiler)
    return path


def snapshots(count=4):
    audio = np.random.default_rng(41).normal(0, 0.04, (48128 + (count - 1) * 24064, 14))
    return [dict(audio=audio[start:start + 48128].copy(), first=start, end=start + 48128, epoch=0, clipped=False)
            for start in range(0, len(audio) - 48128 + 1, 24064)]


@pytest.mark.parametrize('precision', ['double', 'single'])
def test_rolling_phat_matches_full_frames_without_recomputing_overlap(runtime, precision):
    config = load_config()
    settings = dict(config['search'], algorithm='cached', fft_fold=4)
    search = StreamSearch(config['audio'], settings, runtime, precision)
    reference = Search(config['audio'], settings)
    continuity = Continuity(config['audio'])
    seen = 0
    for index, snapshot in enumerate(snapshots()):
        context = continuity.prepare(snapshot)
        grid, bearings, metric = search.run_snapshot(snapshot, context)
        expected = reference.correlations(snapshot['audio'])
        actual = search.last_correlations
        tolerance = 2e-7 if precision == 'single' else 2e-14
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
        assert metric['fft_frames_computed'] == (93 if index == 0 else 47)
        assert metric['fft_frames_reused'] == (0 if index == 0 else 46)
        seen += metric['fft_frames_computed']
        np.testing.assert_allclose(grid[2].ravel(), reference.score(actual, np.arange(5580)), atol=1e-11)
    assert seen == 93 + 3 * 47


@pytest.mark.parametrize('change,reason', [('gap', 'epoch_changed'), ('skip', 'skipped_window'),
                                        ('data', 'overlap_changed'), ('reverse', 'nonforward_window')])
def test_continuity_invalidates_untrusted_cache(change, reason):
    rows = snapshots()
    state = Continuity(load_config()['audio'])
    assert not state.prepare(rows[0])['reusable']
    current = dict(rows[1])
    if change == 'gap':
        current['epoch'] += 1
    elif change == 'skip':
        current = rows[2]
    elif change == 'reverse':
        current = rows[0]
    else:
        current['audio'] = current['audio'].copy()
        current['audio'][100, 3] += 0.001
    context = state.prepare(current)
    assert not context['reusable'] and context['reset_reason'] == reason


@pytest.mark.parametrize('directions', ['stable', 'changing'])
def test_rolling_beams_match_frozen_boundaries_for_stable_and_changed_directions(directions):
    settings = load_config()['audio']
    engine = Beams(settings, dict(implementation='fft', fft_size=1024, cache_size=64))
    continuity = Continuity(settings)
    for index, snapshot in enumerate(snapshots()):
        bearings = [Bearing(23 + (index * 2 if directions == 'changing' else 0), 33, 1), Bearing(-113, 42, 1)]
        outputs, metric = engine.run_snapshot(snapshot, bearings, continuity.prepare(snapshot))
        for bearing, (waveform, latency) in zip(bearings, outputs):
            expected, common = beamform(snapshot['audio'], bearing, settings)
            np.testing.assert_allclose(waveform, expected, atol=2e-14, rtol=2e-12)
            assert latency == common and not waveform[:32].any()
        assert metric['fft_hop'] == 752 and metric['fft_blocks_total'] == 64
        assert metric['fft_blocks_computed'] == (64 if index == 0 else 33)
        assert metric['output_reused'] == (directions == 'stable' and index > 0)


def test_cached_frontend_matches_both_trained_frontends_across_windows():
    settings = load_config()['audio']
    for frontend in ('logmel', 'centered_logmel'):
        engine = Frontend(frontend)
        beams = Beams(settings, dict(implementation='fft', fft_size=1024))
        continuity = Continuity(settings)
        for index, snapshot in enumerate(snapshots()):
            context = continuity.prepare(snapshot)
            bearing = Bearing(23, 33, 1)
            outputs, diagnostic = beams.run_snapshot(snapshot, [bearing], context)
            waveform = outputs[0][0]
            actual, metric = engine.run(waveform, bearing, context, diagnostic['support_samples'])
            np.testing.assert_allclose(actual, features(waveform, frontend), atol=3e-10, rtol=2e-10)
            assert actual.shape == (7, 40, 24)
            assert metric['frames_reused'] == (0 if index == 0 else 45)


def test_partial_last_fft_block_is_never_reused_as_observed_pcm():
    settings = load_config()['audio']
    state = Continuity(settings)
    engine = Beams(settings, dict(implementation='fft', fft_size=1024))
    audio = np.random.default_rng(88).normal(0, 0.04, size=(48000 + 24064, 14))
    bearing = Bearing(23, 33, 1)
    for start in (0, 24064):
        snapshot = dict(audio=audio[start:start + 48000], first=start, end=start + 48000, epoch=0)
        outputs, metric = engine.run_snapshot(snapshot, [bearing], state.prepare(snapshot))
        expected = beamform(snapshot['audio'], bearing, settings)[0]
        np.testing.assert_allclose(outputs[0][0], expected, atol=2e-14, rtol=2e-12)
        assert metric['fft_blocks_reused'] == 0 and not metric['output_reused']


def test_stateful_conditioning_matches_reference_partitions_calibration_and_reset(runtime):
    audio = np.random.default_rng(5).normal(1000, 3000, (7000, 14))
    for delays in (np.zeros(14), np.linspace(-0.3, 0.5, 14)):
        gains = np.linspace(0.9, 1.1, 14)
        first, second = Conditioning(14, gains, delays), StreamConditioning(14, runtime, gains, delays)
        for _ in range(2):
            first.reset()
            second.reset()
            for block in np.array_split(audio, [1, 73, 512, 2048]):
                np.testing.assert_allclose(second.feed(block), first.feed(block), atol=3e-14, rtol=1e-12)
            assert first.latency == second.latency


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_fused_groups_are_bit_exact_to_reference_twelve_frame_sums(runtime, precision):
    rng = np.random.default_rng(4)
    spectrum = rng.normal(size=(140, 14, 129)) + 1j * rng.normal(size=(140, 14, 129))
    pairs = np.asarray([(a, b) for a in range(14) for b in range(a + 1, 14)], dtype=np.int32)
    normalized = runtime.phat(spectrum, pairs, precision)
    first, cached = runtime.fused(spectrum[:93], pairs, precision, None, 0, 46)
    np.testing.assert_array_equal(first, runtime.reduce(normalized[:0], normalized[:93], precision))
    second, future = runtime.fused(spectrum[93:], pairs, precision, cached, 46, 46)
    np.testing.assert_array_equal(second, runtime.reduce(normalized[:0], normalized[47:], precision))
    assert cached.shape == future.shape == (4, 91, 129)


def test_native_fused_rejects_inconsistent_cached_lengths_before_call(runtime):
    spectrum = np.ones((47, 14, 129), dtype=complex)
    pairs = np.asarray([[0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match='cached group'):
        runtime.fused(spectrum, pairs, 'single', None, 46, 46)


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_full_native_reference_correlations_and_directions_match(runtime, search_library, precision):
    config = load_config()
    settings = dict(config['search'], algorithm='cached', implementation='native', fft_fold=4,
                    native_precision=precision, native_library=str(search_library))
    expected = Search(config['audio'], settings)
    actual = StreamSearch(config['audio'], settings, runtime)
    state = Continuity(config['audio'])
    for row in snapshots():
        first, bearings, _ = expected.run(row['audio'])
        second, selected, _ = actual.run_snapshot(row, state.prepare(row))
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        assert [(b.azimuth, b.depression) for b in bearings] == [(b.azimuth, b.depression) for b in selected]


def test_pipeline_matches_real_model_through_clipping_epoch_and_skips(runtime, search_library):
    from ps.band import load_spectral
    from ps.pipeline import Pipeline
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    if not model.exists():
        pytest.skip('Optional trained model absent')
    config = load_config(ROOT / 'config/band_fast.json')
    config['search']['native_library'] = str(search_library)
    optimized = deepcopy(config)
    optimized['streaming'] = dict(enabled=True, conditioning='native', native_library=runtime.library._name)
    reference, current = Pipeline(config, load_spectral(model)), Pipeline(optimized, load_spectral(model))
    rows = snapshots(6)
    rows[2]['clipped'] = True
    rows[4]['epoch'] = rows[5]['epoch'] = 1
    for index in (0, 1, 2, 4, 5):
        first, evidence = reference.analyze(rows[index])
        second, actual = current.analyze(rows[index])
        np.testing.assert_allclose(first['scores'], second['scores'], atol=2e-10)
        for key in ('state', 'horn_candidate', 'doa_valid', 'capture_candidate', 'azimuth', 'depression', 'history_reset',
                    'event_id', 'ended_event', 'input_clipped', 'activation_clipped', 'frontend_clipped', 'acoustic_state'):
            assert first[key] == second[key]
        assert (evidence is None) == (actual is None)
    assert second['stream_diagnostics']['reusable']


def test_capture_worker_records_parallel_conditioning_cost_without_changing_pcm(runtime):
    from ps.acquisition import CapturePump
    config = load_config()['audio']
    data = np.random.default_rng(12).integers(-1000, 1000, size=(512 * 141, 14), dtype=np.int16)
    class Device:
        timeout = 1
        index = 0
        def receive(self):
            start = self.index * 512
            self.index += 1
            if self.index == 141:
                pump.stop.set()
            return dict(audio=data[start:start + 512], first=start, gap=False, clipped=False)
    pump = CapturePump(Device(), config, StreamConditioning(14, runtime))
    pump.run()
    snapshot = pump.receive()
    assert snapshot['conditioning']['packets'] == 47 and snapshot['conditioning']['seconds'] > 0
    expected = Conditioning(14).feed(data)[24064:]
    np.testing.assert_allclose(snapshot['audio'], expected, atol=3e-14, rtol=1e-12)
    assert pump.skipped == 1 and pump.gaps == 0


def test_silence_and_epoch_change_clear_beam_and_frontend_state():
    settings = load_config()['audio']
    state, engine, frontend = Continuity(settings), Beams(settings, dict(implementation='fft', fft_size=1024)), Frontend()
    rows = snapshots(2)
    bearings = [Bearing(None, 90, 1)]
    context = state.prepare(rows[0])
    outputs, metric = engine.run_snapshot(rows[0], bearings, context)
    frontend.run(outputs[0][0], bearings[0], context, metric['support_samples'])
    silence = dict(rows[1], audio=np.zeros_like(rows[1]['audio']), epoch=1)
    context = state.prepare(silence)
    outputs, metric = engine.run_snapshot(silence, bearings, context)
    actual, diagnostic = frontend.run(outputs[0][0], bearings[0], context, metric['support_samples'])
    assert not outputs[0][0].any() and diagnostic['frames_reused'] == 0
    assert not metric['output_reused'] and np.isfinite(actual).all()


@pytest.mark.parametrize('change', [dict(enabled='true'), dict(enabled=True, conditioning='unknown'),
                                   dict(enabled=True, native_library='relative.so'),
                                   dict(enabled=False, conditioning='native'), dict(enabled=True, extra=True)])
def test_streaming_configuration_rejects_unsafe_modes(change):
    from ps.configuration import validate_config
    config = load_config(ROOT / 'config/band_stream.json')
    config['streaming'] = change
    with pytest.raises(ValueError, match='[Ss]tream|conditioning'):
        validate_config(config)


@pytest.mark.parametrize('change', ['fpga', 'reference', 'unequal_overlap', 'direct_fir'])
def test_streaming_rejects_incompatible_processing_contract(change):
    from ps.configuration import validate_config
    config = load_config(ROOT / 'config/band_stream.json')
    if change == 'fpga':
        config['search']['backend'] = 'fpga'
    elif change == 'reference':
        config['search'].update(algorithm='reference', implementation='numpy', fft_fold=1)
    elif change == 'unequal_overlap':
        config['audio']['update_samples'] = 512
    else:
        config['beamforming']['implementation'] = 'native'
    with pytest.raises(ValueError):
        validate_config(config)
