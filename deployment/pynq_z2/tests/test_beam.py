import importlib.util
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.beamforming import Beamformer
from ps.configuration import load_config, validate_config
from ps.dsp import beamform, features
from ps.geometry import Bearing


@pytest.fixture(scope='module')
def native_beam(tmp_path_factory):
    specification = importlib.util.spec_from_file_location('beam_builder', ROOT / 'scripts/build_beam.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    existing = os.environ.get('HORN_BEAM_LIBRARY')
    if existing:
        destination = Path(existing).resolve()
        result = module.check(destination)
    else:
        compiler = shutil.which('gcc')
        if not compiler:
            pytest.skip('Optional GCC compiler absent')
        destination = tmp_path_factory.mktemp('native-beam') / ('beamforming.dll' if sys.platform == 'win32' else 'libbeam.so')
        result = module.build(destination, compiler)
    assert result['result'] == 'passed' and result['abi'] == 1
    return destination


@pytest.fixture(params=['cached', 'fft512', 'fft1024', 'native'])
def engine(request):
    settings = dict(implementation='fft', fft_size=512)
    if request.param.startswith('fft'):
        settings['fft_size'] = int(request.param[3:])
    else:
        settings['implementation'] = request.param
    if request.param == 'native':
        settings['native_library'] = str(request.getfixturevalue('native_beam'))
    return Beamformer(load_config()['audio'], settings)


@pytest.mark.parametrize('length', [1, 31, 32, 65, 129, 1024, 4097])
def test_batch_matches_frozen_fir_sign_boundaries_and_short_windows(engine, length):
    audio = np.random.default_rng(length).normal(size=(length, 14))
    bearings = [Bearing(-179, 0, 1), Bearing(57, 33, 1), Bearing(None, 87, 1), Bearing(0, 90, 1)]
    actual, diagnostic = engine.run(audio, bearings)
    assert diagnostic['beams'] == len(bearings) and diagnostic['taps'] == 65
    for bearing, (waveform, latency) in zip(bearings, actual):
        expected, common = beamform(audio, bearing, engine.audio)
        assert latency == common and waveform.shape == (length,)
        np.testing.assert_allclose(waveform, expected, atol=3e-14, rtol=3e-13)
        assert np.count_nonzero(waveform[:32]) == 0


def test_each_channel_impulse_retains_window_start_gating_and_end_truncation(engine):
    length = 257
    bearing = Bearing(41.5, 27, 1)
    for channel in range(14):
        for offset in (0, 1, 64, length - 1):
            audio = np.zeros((length, 14))
            audio[offset, channel] = 1
            expected, common = beamform(audio, bearing, engine.audio)
            actual, _ = engine.run(audio, [bearing])
            np.testing.assert_allclose(actual[0][0], expected, atol=2e-15)
            assert actual[0][1] == common


def test_no_cross_window_state_silence_and_strided_audio(engine):
    bearings = [Bearing(16, 30, 1), Bearing(-95, 51, 1)]
    audio = np.random.default_rng(56).normal(size=(8192, 28))[::2, ::2]
    engine.run(audio, bearings)
    outputs, _ = engine.run(np.zeros((257, 14)), bearings)
    assert all(not waveform.any() for waveform, _ in outputs)
    outputs, _ = engine.run(audio, bearings)
    for bearing, (waveform, _) in zip(bearings, outputs):
        np.testing.assert_allclose(waveform, beamform(audio, bearing, engine.audio)[0], atol=3e-14)
    outputs, diagnostic = engine.run(audio, [])
    assert outputs == [] and diagnostic['beams'] == 0


def test_completed_window_features_and_frozen_model_scores_match(engine):
    from ps.band import load_spectral
    model = ROOT.parents[1] / 'prototype/outputs/band_improvements_v1/model-combined'
    detector = load_spectral(model) if model.is_dir() else None
    samples = np.arange(48128)[:, None] / 48000
    phases = np.arange(14)[None] * 0.2
    audio = 0.03 * np.sin(2 * np.pi * 700 * samples + phases)
    audio += np.random.default_rng(104).normal(0, 0.002, audio.shape)
    bearings = [Bearing(23, 30, 1), Bearing(-113, 42, 1)]
    outputs, _ = engine.run(audio, bearings)
    for bearing, (waveform, _) in zip(bearings, outputs):
        expected = beamform(audio, bearing, engine.audio)[0]
        np.testing.assert_allclose(waveform, expected, atol=3e-14)
        for frontend in ('logmel', 'centered_logmel'):
            first, second = features(expected, frontend), features(waveform, frontend)
            np.testing.assert_allclose(second, first, atol=2e-10, rtol=1e-10)
        if detector is not None:
            first, second = detector.score(features(expected)), detector.score(features(waveform))
            assert first['clipped'] == second['clipped']
            assert first['score'] == pytest.approx(second['score'], abs=2e-10)
            assert (first['score'] >= 0.9) == (second['score'] >= 0.9)
            assert (first['score'] >= 0.95) == (second['score'] >= 0.95)


def test_fft_cache_is_bounded_and_input_transforms_are_shared(monkeypatch):
    engine = Beamformer(load_config()['audio'], dict(implementation='fft', cache_size=2))
    audio = np.random.default_rng(1).normal(size=(4096, 14))
    bearings = [Bearing(23, 30, 1), Bearing(-113, 42, 1)]
    engine.run(audio, bearings)
    previous = np.fft.rfft
    calls = []
    def transform(values, *args, **kwargs):
        calls.append(values.shape)
        return previous(values, *args, **kwargs)
    monkeypatch.setattr(np.fft, 'rfft', transform)
    _, metric = engine.run(audio, bearings)
    assert len(calls) == 1 and metric['cache_hits'] == 2
    for angle in range(10):
        engine.run(audio, [Bearing(angle, 30, 1)])
    assert len(engine.cache) == 2


def test_fft_preserves_existing_integer_detector_scores():
    from ps.detector import Detector
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Optional local integer weights absent')
    detector = Detector(model)
    settings = load_config()['audio']
    engine = Beamformer(settings, dict(implementation='fft', fft_size=1024))
    audio = np.random.default_rng(27).normal(0, 500, size=(48128, 14))
    bearings = [Bearing(0, 20, 1), Bearing(144, 51, 1)]
    outputs, _ = engine.run(audio, bearings)
    for bearing, (waveform, _) in zip(bearings, outputs):
        expected = features(beamform(audio, bearing, settings)[0])
        actual = features(waveform)
        assert detector.score(actual) == detector.score(expected)


@pytest.mark.parametrize('options', [dict(implementation='unknown'), dict(fft_size=65), dict(cache_size=0),
                                     dict(cache_size=True), dict(native_library='relative.so'),
                                     dict(implementation='fft', native_library='C:/unused.dll')])
def test_invalid_options_fail_before_processing(options):
    config = load_config()
    config['beamforming'] = options
    with pytest.raises(ValueError, match='[Bb]eam'):
        validate_config(config)
    with pytest.raises(ValueError, match='[Bb]eam'):
        Beamformer(config['audio'], options)


def test_missing_native_library_and_invalid_audio_fail_explicitly(tmp_path):
    settings = load_config()['audio']
    with pytest.raises(ValueError, match='build_beam.py'):
        Beamformer(settings, dict(implementation='native', native_library=str(tmp_path / 'absent.so')))
    engine = Beamformer(settings, dict(implementation='fft'))
    for audio in (np.zeros((0, 14)), np.zeros((512, 13)), np.full((512, 14), np.nan), np.zeros(512)):
        with pytest.raises(ValueError, match='audio'):
            engine.run(audio, [Bearing(30, 20, 1)])
    with pytest.raises(ValueError, match='[Bb]earing|Azimuth'):
        engine.run(np.zeros((512, 14)), [Bearing(float('nan'), 20, 1)])


@pytest.mark.parametrize('implementation', ['cached', 'fft', 'native'])
def test_pipeline_preserves_event_resets_invalid_audio_and_all_decisions(implementation, request):
    from copy import deepcopy
    from ps.pipeline import Pipeline
    class Detector:
        def score(self, contexts):
            return dict(score=float(1 / (1 + np.exp(-contexts.mean()))), clipped=False)
    config = load_config()
    config['search']['algorithm'] = 'cached'
    reference = Pipeline(config, Detector())
    optimized = deepcopy(config)
    optimized['beamforming'] = dict(implementation=implementation)
    if implementation == 'native':
        optimized['beamforming']['native_library'] = str(request.getfixturevalue('native_beam'))
    pipeline = Pipeline(optimized, Detector())
    signal = np.random.default_rng(31).normal(0, 300, size=48128)
    audio = np.column_stack([np.roll(signal, index) for index in range(14)])
    for index, (epoch, clipped, gap) in enumerate(((0, False, 1), (0, False, 1), (0, True, 2), (1, False, 1))):
        end = (index + gap) * config['audio']['update_samples'] + len(audio)
        snapshot = dict(audio=audio, first=end - len(audio), end=end, epoch=epoch, clipped=clipped)
        expected, _ = reference.analyze(snapshot)
        actual, _ = pipeline.analyze(snapshot)
        for key in ('state', 'horn_candidate', 'doa_valid', 'capture_candidate', 'history_reset', 'acoustic_state',
                    'event_id', 'ended_event', 'azimuth', 'depression', 'input_clipped'):
            assert actual[key] == expected[key]
        np.testing.assert_allclose(actual['scores'], expected['scores'], atol=2e-10)
        assert actual['beamforming_diagnostics']['implementation'] == implementation
        assert actual['phase_seconds']['beamforming'] > 0
    silent, _ = pipeline.analyze(dict(snapshot, audio=np.zeros_like(audio), epoch=2))
    assert not silent['scores'] and silent['beamforming_diagnostics']['beams'] == 0
    assert silent['state'] == 'IDLE' and not silent['horn_candidate']
