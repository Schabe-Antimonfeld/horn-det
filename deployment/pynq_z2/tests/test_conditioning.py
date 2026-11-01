import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.conditioning import Conditioner
from ps.dsp import Conditioning


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    compiler = shutil.which('gcc')
    if not compiler:
        pytest.skip('Optional GCC absent')
    spec = importlib.util.spec_from_file_location('conditioning_builder', ROOT / 'scripts/build_conditioning.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp('conditioning') / ('conditioning.dll' if sys.platform == 'win32' else 'libconditioning.so')
    result = module.build(path, compiler)
    assert result['result'] == 'passed' and result['abi'] == 1
    return path


@pytest.mark.parametrize('fir', ['numpy', 'native'])
@pytest.mark.parametrize('dtype', [np.int16, np.float64])
@pytest.mark.parametrize('calibration', ['none', 'gain', 'fractional', 'long_delay'])
def test_stream_partitions_match_frozen_reference(library, fir, dtype, calibration):
    rng = np.random.default_rng(5)
    audio = rng.integers(-12000, 12000, size=(8193, 14), dtype=np.int16).astype(dtype)
    gains = None if calibration == 'none' else np.linspace(0.9, 1.1, 14)
    delays = np.linspace(-0.3, 0.5, 14) if calibration == 'fractional' else np.linspace(-520, 1, 14) if calibration == 'long_delay' else None
    expected, actual = Conditioning(14, gains, delays), Conditioner(14, gains, delays, fir, library)
    assert expected.latency == actual.latency
    for _ in range(2):
        expected.reset()
        actual.reset()
        for part in np.array_split(audio, [1, 73, 512, 1025, 3074]):
            first, second = expected.feed(part), actual.feed(part)
            np.testing.assert_allclose(second, first, atol=3e-14, rtol=1e-11)
            np.testing.assert_allclose(actual.previous, expected.previous, atol=0)
            np.testing.assert_allclose(actual.state, expected.state, atol=3e-14, rtol=1e-11)


@pytest.mark.parametrize('fir', ['numpy', 'native'])
def test_impulses_prove_causal_calibration_and_integer_delay_sign(library, fir):
    gains, delays = np.linspace(0.9, 1.1, 14), np.linspace(-3, 4, 14)
    expected, actual = Conditioning(14, gains, delays), Conditioner(14, gains, delays, fir, library)
    for channel in range(14):
        expected.reset()
        actual.reset()
        data = np.zeros((1024, 14), dtype=np.int16)
        data[511, channel] = 16000
        first = np.concatenate([expected.feed(data[:512]), expected.feed(data[512:])])
        second = np.concatenate([actual.feed(data[:512]), actual.feed(data[512:])])
        np.testing.assert_allclose(second, first, atol=3e-14, rtol=1e-11)
        assert not second[:511].any()


def test_feed_into_reuses_caller_storage_without_changing_owned_feed_results(library):
    actual = Conditioner(14, library=library)
    audio = np.full((512, 14), 2000, dtype=np.int16)
    first = actual.feed(audio)
    saved = first.copy()
    output = np.empty((512, 14), dtype=np.float64)
    assert actual.feed_into(audio, output) is output
    assert not np.shares_memory(first, output)
    actual.feed_into(audio, output)
    np.testing.assert_array_equal(first, saved)
    before = actual.state.copy()
    actual.feed_into(np.empty((0, 14), dtype=np.int16), output[:0])
    np.testing.assert_array_equal(actual.state, before)


def test_bad_inputs_fail_before_mutating_state(library):
    actual = Conditioner(14, library=library)
    output = np.full((512, 14), 12.0)
    for audio in (np.zeros((512, 13)), np.full((512, 14), np.nan), np.full((512, 14), np.inf)):
        with pytest.raises(ValueError):
            actual.feed_into(audio, output)
        assert not actual.state.any() and not actual.previous.any()
    data = np.zeros((512, 14))
    for target in (np.empty((512, 14), dtype=np.float32), np.empty((1024, 14))[::2], data):
        with pytest.raises(ValueError):
            actual.feed_into(data, target)
    target = np.empty_like(data)
    target.setflags(write=False)
    with pytest.raises(ValueError):
        actual.feed_into(data, target)
    assert not actual.state.any()


def test_strided_and_unaligned_pcm_and_float_inputs_are_supported(library):
    for dtype in (np.int16, np.float64):
        actual, reference = Conditioner(14, library=library), Conditioning(14)
        data = np.arange(1024 * 28).reshape(1024, 28).astype(dtype)[::2, ::2]
        np.testing.assert_allclose(actual.feed(data), reference.feed(data), atol=3e-14, rtol=1e-11)
        unaligned = np.ndarray(data.shape, dtype=dtype, buffer=bytearray(data.nbytes + 1), offset=1)
        unaligned[:] = data
        np.testing.assert_allclose(actual.feed(unaligned), reference.feed(data), atol=3e-14, rtol=1e-11)


def test_calibration_is_owned_and_immutable(library):
    gains = np.ones(14)
    actual = Conditioner(14, gains=gains, library=library)
    gains[:] = 2
    data = np.full((512, 14), 1000, dtype=np.int16)
    np.testing.assert_allclose(actual.feed(data), Conditioning(14).feed(data), atol=3e-14)
    with pytest.raises(ValueError):
        actual.gains[0] = 2
    for name, value in [('fir', 'native'), ('channels', 1), ('gains', np.ones(14)), ('latency', 1)]:
        with pytest.raises(AttributeError):
            setattr(actual, name, value)


@pytest.mark.parametrize('options', [dict(gains=[0] * 14), dict(delays=[float('nan')] * 14),
                                    dict(fir='unknown'), dict(delays=[-1e20] + [0] * 13)])
def test_invalid_calibration_fails_without_large_allocations(library, options):
    with pytest.raises(ValueError):
        Conditioner(14, library=library, **options)


def test_aligned_packet_feed_into_preserves_workspace_without_output_allocation(library, monkeypatch):
    actual = Conditioner(14, delays=np.linspace(-0.3, 0.5, 14), fir='native', library=library)
    audio, output = np.ones((512, 14), dtype=np.int16), np.empty((512, 14))
    memory = actual._memory.ctypes.data
    def unexpected(*args, **kwargs):
        raise AssertionError('Caller-buffer packet path must not allocate another NumPy output')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'empty', unexpected)
        patch.setattr(np, 'zeros', unexpected)
        for _ in range(3):
            assert actual.feed_into(audio, output) is output
    assert actual._memory.ctypes.data == memory


def test_signed_pcm_extrema_preserve_scale_and_causal_state(library):
    data = np.tile(np.array([-32768, 32767], dtype=np.int16), (14, 1024)).T.copy()
    data.setflags(write=False)
    reference = Conditioning(14)
    current = Conditioner(14, library=library)
    for part in np.array_split(data, [1, 511, 1025]):
        np.testing.assert_allclose(current.feed(part), reference.feed(part), atol=3e-14, rtol=1e-11)


@pytest.mark.parametrize('delays', [None, np.linspace(-0.3, 0.5, 14)])
def test_capture_worker_consumes_reusable_output_before_next_packet(library, delays, monkeypatch):
    from ps.acquisition import CapturePump
    from ps.configuration import load_config
    data = np.random.default_rng(41).integers(-5000, 5000, size=(512 * 141, 14), dtype=np.int16)
    engine = Conditioner(14, delays=delays, fir='native', library=library)
    monkeypatch.setattr(engine, 'feed', lambda *args: pytest.fail('Capture must use feed_into'))
    class Device:
        index = 0
        def receive(self):
            start = self.index * 512
            self.index += 1
            if self.index == 141:
                pump.stop.set()
            return dict(first=start, audio=data[start:start + 512], gap=False, clipped=False)
    pump = CapturePump(Device(), load_config()['audio'], engine)
    pump.run()
    assert pump.error is None
    snapshot = pump.receive()
    expected = Conditioning(14, delays=delays).feed(data)[24064:]
    np.testing.assert_allclose(snapshot['audio'], expected, atol=3e-14, rtol=1e-11)
    assert not np.shares_memory(snapshot['audio'], pump.scratch)
    assert snapshot['conditioning']['implementation'] == 'buffered'
    assert snapshot['conditioning']['fir'] == ('disabled' if delays is None else 'native')


def test_wav_replay_consumes_reusable_output_without_aliasing_windows(library, tmp_path, monkeypatch):
    import wave
    from run import replay
    from ps.configuration import load_config
    data = np.random.default_rng(31).integers(-5000, 5000, size=(72192, 14), dtype=np.int16)
    source = tmp_path / 'array.wav'
    with wave.open(str(source), 'wb') as stream:
        stream.setnchannels(14)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(data.astype('<i2').tobytes())
    engine = Conditioner(14, library=library)
    monkeypatch.setattr(engine, 'feed', lambda *args: pytest.fail('Replay must use feed_into'))
    rows = list(replay(source, load_config()['audio'], engine))
    assert len(rows) == 2 and not np.shares_memory(rows[0]['audio'], rows[1]['audio'])
    expected = Conditioning(14).feed(data)
    for index, row in enumerate(rows):
        np.testing.assert_allclose(row['audio'], expected[index * 24064:index * 24064 + 48128], atol=3e-14, rtol=1e-11)


@pytest.mark.parametrize('options', [dict(fir='unknown'), dict(implementation='unknown'),
                                    dict(native_library='relative.so'), dict(extra=True)])
def test_configuration_rejects_inconsistent_conditioning_options(options):
    from ps.configuration import load_config, validate_config
    config = load_config()
    config['conditioning'] = options
    with pytest.raises(ValueError, match='[Cc]onditioning'):
        validate_config(config)
