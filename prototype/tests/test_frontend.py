import numpy as np
import pytest
import soundfile as sf

from horn.config import Config
from horn.frontend import STFT, power, read_audio, stft


def test_streaming_matches_independent_dft():
    config = Config(frame=16, hop=8)
    audio = np.random.default_rng(7).normal(size=(53, 2))
    stream = STFT(config)
    frames = []
    for block in np.array_split(audio, [3, 15, 20, 45]):
        frames.extend(stream.feed(block))
    assert len(frames) == 5
    transform = np.exp(-2j * np.pi * np.outer(np.arange(9), np.arange(16)) / 16)
    for index, frame in enumerate(frames):
        expected = transform @ (audio[index * 8:index * 8 + 16] * np.hanning(16)[:, None])
        np.testing.assert_allclose(frame.spectrum, expected, atol=2e-14)
        assert frame.end == index * 8 + 15
        assert frame.timestamp(48000) == frame.end * 1000000 // 48000
    np.testing.assert_allclose(frames[-1].spectrum, stft(audio, config)[-1].spectrum)


def test_no_future_padding_and_no_phase_cancellation():
    config = Config(frame=16, hop=8)
    stream = STFT(config)
    assert stream.feed(np.ones((15, 2))) == []
    assert len(stream.feed(np.ones((1, 2)))) == 1
    spectrum = np.array([[3 + 4j, -3 - 4j]])
    assert power(spectrum)[0] == 25
    with pytest.raises(ValueError):
        stream.feed(np.zeros((2, 3)))


def test_stereo_resampling_preserves_selected_channel(tmp_path):
    time = np.arange(4410) / 44100
    signal = np.sin(2 * np.pi * 1000 * time)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((signal, -signal)), 44100, subtype="PCM_16")
    stereo = read_audio(path)
    mono = read_audio(path, channel=0)
    assert stereo.shape == (4800, 2)
    np.testing.assert_array_equal(stereo[:, :1], mono)
    assert np.sqrt(np.mean(mono[100:-100] ** 2)) == pytest.approx(2 ** -0.5, abs=0.002)
