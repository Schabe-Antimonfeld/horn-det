import json
import queue
from types import SimpleNamespace
import time

import numpy as np
import pytest

from ps.acquisition import WindowBuffer
from ps.configuration import load_config
import ps.vision as camera


def jpeg(width=1920, height=1080):
    return b'\xff\xd8\xff\xc0\x00\x11\x08' + height.to_bytes(2, 'big') + width.to_bytes(2, 'big') + b'\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9'


def test_saturation_expires_when_its_packet_leaves_the_window():
    config = load_config()['audio']
    history = WindowBuffer(config)
    snapshots = []
    for index in range(94 + 2 * 47):
        packet = dict(audio=np.zeros((512, 14), np.int16), first=index * 512, gap=False, clipped=index == 0)
        result = history.feed(packet)
        if result is not None:
            snapshots.append(result)
    assert [row['clipped'] for row in snapshots] == [True, False, False]


def test_jpeg_header_can_be_checked_without_pixel_decode():
    assert camera.jpeg_dimensions(jpeg()) == (1920, 1080)
    for data in (b'bad', b'\xff\xd8\xff\xc0\x00\x11', jpeg()[:12], b'\xff\xd8\xff\xd9'):
        with pytest.raises(ValueError):
            camera.jpeg_dimensions(data)


def test_camera_ready_wait_rejects_startup_errors():
    cache = camera.CameraCache([dict(id='wide-0')], None)
    cache.errors['wide-0'] = 'open failed'
    with pytest.raises(RuntimeError, match='startup'):
        cache.wait_ready(0.05)


def test_encoded_lookup_does_not_decode_pixels():
    cache = camera.CameraCache([dict(id='wide-0')], None)
    cache.history['wide-0'].append(dict(jpeg=jpeg(), time=1))
    assert cache.nearest('wide-0', 1, decode=False)['jpeg'] == jpeg()


def test_native_kernels_match_numpy_for_strides_and_temporal_features():
    from ps.dsp_native import NativeDSP
    from ps.detector import temporal_features
    from ps.streaming import Frontend
    native = NativeDSP()
    rng = np.random.RandomState(408)
    spectrum = rng.normal(size=(9, 14, 513)) + 1j * rng.normal(size=(9, 14, 513))
    weights = spectrum[0].copy()
    for values, filters in ((spectrum, weights), (spectrum[:, ::-1, ::2], weights[::-1, ::2]),
                             (spectrum.transpose(2, 1, 0), weights.T.T[:, :9])):
        np.testing.assert_allclose(native.beam(values, filters), np.einsum('bcf,cf->bf', values, filters), rtol=2e-13, atol=2e-12)
    bank = Frontend().bank.copy()
    for _ in range(2):
        np.testing.assert_allclose(native.mel(spectrum[:, 0], bank), np.log(abs(spectrum[:, 0]) ** 2 @ bank.T + 1e-12), atol=2e-12)
        bank[0] = 0
    contexts = rng.normal(-15, 7, (7, 40, 24))
    np.testing.assert_allclose(native.temporal(contexts), temporal_features(contexts), rtol=2e-13, atol=2e-12)
    for values in (contexts[:, :, ::-1], np.full((7, 40, 24), -27.631021115928547)):
        np.testing.assert_allclose(native.temporal(values), temporal_features(values), rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize('operation', ['beam', 'mel', 'temporal'])
def test_native_kernels_reject_nonfinite_or_incompatible_input(operation):
    from ps.dsp_native import NativeDSP
    native = NativeDSP()
    with pytest.raises(ValueError):
        if operation == 'beam':
            native.beam(np.zeros((2, 14, 10)), np.zeros((14, 9)))
        elif operation == 'mel':
            native.mel(np.zeros((2, 512)), np.zeros((40, 513)))
        else:
            native.temporal(np.full((7, 40, 24), np.nan))


def test_incremental_windows_match_ring_windows_and_own_their_pcm():
    from ps.acquisition import IncrementalWindowBuffer
    settings = load_config()['audio']
    ring, incremental = WindowBuffer(settings), IncrementalWindowBuffer(settings)
    old = None
    for index in range(250):
        packet = dict(audio=np.full((512, 14), index, np.int16), first=index * 512,
                      gap=index == 180, clipped=index in (20, 95, 185))
        first, second = ring.feed(packet), incremental.feed(packet)
        assert (first is None) == (second is None)
        if first is not None:
            np.testing.assert_array_equal(first['audio'], second['audio'])
            assert {key: value for key, value in first.items() if key != 'audio'} == {key: value for key, value in second.items() if key != 'audio'}
            if old is not None:
                assert not np.shares_memory(old['audio'], second['audio'])
            old = second


def test_raw_camera_preserves_jpeg_bytes_and_rejects_wrong_dimensions():
    source = dict(id='wide-0', uri=0, backend='usb', width=1920, height=1080, fps=30, fourcc='MJPG')
    values = {1: sum(ord(c) << (8*i) for i, c in enumerate('MJPG')), 2: 1920, 3: 1080, 4: 30}
    device = SimpleNamespace(isOpened=lambda: True, set=lambda key, value: True,
                             get=lambda key: values[key], release=lambda: None,
                             read=lambda: (True, np.frombuffer(jpeg(), dtype=np.uint8).reshape(1, -1)))
    cv2 = SimpleNamespace(CAP_V4L2=200, CAP_PROP_FOURCC=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
                          CAP_PROP_FPS=4, CAP_PROP_BUFFERSIZE=5, CAP_PROP_CONVERT_RGB=6,
                          VideoCapture=lambda *args: device,
                          VideoWriter_fourcc=lambda *args: values[1])
    reader = camera.UsbCamera(source, cv2)
    frame = reader.read()
    assert frame['jpeg'] == jpeg() and frame['jpeg_reencoded'] is False
    device.read = lambda: (True, np.frombuffer(jpeg(640, 480), dtype=np.uint8))
    with pytest.raises(RuntimeError, match='dimensions'):
        reader.read()
    reader.close()


def test_gc_guard_restores_only_its_own_freeze(monkeypatch):
    import ps.acquisition as acquisition
    calls = []
    monkeypatch.setattr(acquisition.gc, 'get_freeze_count', lambda: 0)
    monkeypatch.setattr(acquisition.gc, 'collect', lambda: calls.append('collect'))
    monkeypatch.setattr(acquisition.gc, 'freeze', lambda: calls.append('freeze'))
    monkeypatch.setattr(acquisition.gc, 'unfreeze', lambda: calls.append('unfreeze'))
    guard = acquisition.CaptureGCGuard()
    guard.prepare()
    guard.prepare()
    guard.close()
    guard.close()
    assert calls == ['collect', 'freeze', 'unfreeze']
    monkeypatch.setattr(acquisition.gc, 'get_freeze_count', lambda: 20)
    guard.prepare()
    guard.close()
    assert calls == ['collect', 'freeze', 'unfreeze']


def test_capture_wait_invalidates_before_decoding_cached_dma_buffer():
    from ps.acquisition import CaptureDevice
    actions = []
    class Buffer(np.ndarray):
        def freebuffer(self):
            actions.append('free')
    words = np.zeros(8 + 512 * 14, dtype=np.uint32).view(Buffer)
    words[:8] = [0x314d4450, 0x1080e, 0, 0, 0, 512, 48000, 0]
    words[8] = 65535
    def allocate(**options):
        assert options['cacheable'] is True
        return words
    class Registers:
        def read(self, address):
            return 2 if address in (0x08, 0x34) else 0
        def write(self, address, value):
            actions.append((address, value))
    def wait():
        actions.append('invalidate')
        words[8] = 0xffffffff
    registers = Registers()
    overlay = SimpleNamespace(audio_dma=SimpleNamespace(mmio=registers, recvchannel=SimpleNamespace(
        transfer=lambda value: None, wait=wait)), capture_control=registers)
    device = CaptureDevice(overlay, allocate, load_config()['audio'])
    assert device.receive()['audio'][0, 0] == -1 and actions.count('invalidate') == 1
    device.close()
    assert actions.index((0x30, 4)) < actions.index('free')


def test_near_threshold_native_statistics_use_reference_scoring(monkeypatch):
    from ps.configuration import ROOT
    from ps.detector import Detector, temporal_features
    detector = Detector(ROOT / 'model')
    calls = []
    detector.native = SimpleNamespace(temporal=lambda values: np.full(640, 1.0))
    contexts = np.full((7, 40, 24), -20.0)
    def score(inputs):
        calls.append(inputs.copy())
        return dict(score=0.9 + (1e-12 if len(calls) == 1 else -1e-12), clipped=False)
    monkeypatch.setattr(detector, 'score_inputs', score)
    assert detector.score(contexts)['score'] < 0.9
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], temporal_features(contexts))


def test_usb_worker_only_publishes_at_cache_period(monkeypatch):
    state = {'frames': 0}
    class Camera:
        def __init__(self, source, cv2):
            pass
        def read(self):
            state['frames'] += 1
            return dict(jpeg=jpeg(), time=state['frames'] / 100)
        def close(self):
            state['closed'] = True
    monkeypatch.setitem(__import__('sys').modules, 'cv2', SimpleNamespace())
    monkeypatch.setattr(camera, 'UsbCamera', Camera)
    monkeypatch.setattr(camera.time, 'monotonic', lambda: state['frames'] / 100)
    messages = queue.Queue(maxsize=100)
    camera.usb_worker({}, messages, SimpleNamespace(is_set=lambda: state['frames'] >= 100), period=0.25)
    assert 3 <= messages.qsize() <= 5 and state['closed']
