import ast
import importlib.util
import json
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config, validate_config
from ps.vision import CameraCache, UsbCamera, usb_worker


class Capture:
    def __init__(self, source, override=None):
        self.values, self.released = {}, False
        self.source, self.override = source, override or {}
        self.image = np.zeros((source['height'], source['width'], 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def set(self, key, value):
        self.values[key] = value
        return True

    def get(self, key):
        return self.override.get(key, self.values[key])

    def read(self):
        return True, self.image

    def release(self):
        self.released = True


class Cv2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    IMREAD_COLOR = 1

    def __init__(self, source, override=None):
        self.capture = Capture(source, override)

    def VideoCapture(self, path, backend):
        assert backend == self.CAP_V4L2
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*characters):
        return sum(ord(value) << (index * 8) for index, value in enumerate(characters))

    def imencode(self, extension, image):
        assert extension == '.jpg'
        return True, np.array([11, 12, 13], dtype=np.uint8)

    def imdecode(self, data, flag):
        return self.capture.image


def source_settings():
    return dict(load_config()['camera']['sources'][0], width=640, height=360)


def test_usb_negotiation_and_original_frame_metadata():
    source = source_settings()
    cv2 = Cv2(source)
    camera = UsbCamera(source, cv2)
    frame = camera.read()
    assert frame['mode']['fourcc'] == 'MJPG'
    assert frame['mode']['fps_reported'] == 30
    assert frame['timestamp_kind'] == 'host_read_midpoint' and not frame['timing_verified']
    assert frame['jpeg_reencoded']
    camera.close()
    assert cv2.capture.released


@pytest.mark.parametrize('property,value', [(2, 320), (3, 480), (4, 15), (1, 0)])
def test_usb_mode_fallback_is_rejected_and_device_released(property, value):
    source = source_settings()
    cv2 = Cv2(source, {property: value})
    with pytest.raises(RuntimeError, match='mismatch'):
        UsbCamera(source, cv2)
    assert cv2.capture.released


def test_usb_frame_size_change_is_rejected():
    source = source_settings()
    cv2 = Cv2(source)
    camera = UsbCamera(source, cv2)
    cv2.capture.image = np.zeros((180, 320, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='different image mode'):
        camera.read()
    camera.close()


def test_usb_worker_queue_is_bounded_and_error_is_reported(monkeypatch):
    source = source_settings()
    cv2 = Cv2(source, {4: 15})
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    messages = queue.Queue(maxsize=1)
    usb_worker(source, messages, SimpleNamespace(is_set=lambda: False))
    message = messages.get_nowait()
    assert message['kind'] == 'error' and 'mismatch' in message['message']
    assert cv2.capture.released


def test_actual_spawn_reports_missing_opencv_and_stops():
    if importlib.util.find_spec('cv2') is not None:
        pytest.skip('Missing-runtime startup case requires OpenCV to be absent')
    cache = CameraCache([source_settings()], None, timeout=1)
    try:
        cache.start()
        deadline = time.monotonic() + 8
        while not cache.health()['errors'] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert 'cv2' in str(cache.health()['errors'])
    finally:
        cache.close()
    assert not any(process.is_alive() for process, messages, stopped in cache.workers)


def test_stuck_usb_process_is_terminated_before_queue_cleanup():
    state = {'alive': True}
    process = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: state['alive'],
                              terminate=lambda: state.update(alive=False))
    closed = []
    messages = SimpleNamespace(close=lambda: closed.append('close'), join_thread=lambda: closed.append('join'))
    cache = CameraCache([], None)
    cache.workers.append((process, messages, SimpleNamespace(set=lambda: None)))
    cache.close()
    assert not state['alive'] and closed == ['close', 'join']


def test_original_jpeg_grid_is_used_and_history_is_bounded():
    source = source_settings()
    cv2 = Cv2(source)
    cache = CameraCache([source], SimpleNamespace(cv2=cv2), frames=2)
    for sequence in range(5):
        cache.history[source['id']].append({'jpeg': b'test-placeholder', 'time': sequence})
    frame = cache.nearest(source['id'], 3.1)
    assert frame['time'] == 3 and frame['image'].shape == (360, 640, 3)
    assert cache.health()['cached_frames'][source['id']] == 2


def test_high_usb_rate_preserves_history_at_the_configured_interval():
    source = source_settings()
    cache = CameraCache([source], None, frames=8, period=0.25)
    messages = queue.Queue()
    for index in range(60):
        messages.put({'kind': 'frame', 'frame': {'time': index / 30, 'mode': {}}})
    cache.run_usb(source, SimpleNamespace(is_alive=lambda: False), messages)
    times = [frame['time'] for frame in cache.history[source['id']]]
    assert len(times) == 8 and times[-1] - times[0] > 1.7


def calibration_record():
    return {'format': 1, 'model': 'fisheye', 'width': 1920, 'height': 1080, 'maximum_angle': 80,
            'matrix': [[600, 0, 960], [0, 600, 540], [0, 0, 1]], 'coefficients': [0, 0, 0, 0]}


def test_calibration_loading_and_usb_geometry_consistency(tmp_path):
    config = load_config()
    record = tmp_path / 'lens.json'
    record.write_text(json.dumps(calibration_record()))
    config['camera']['calibration'] = str(record)
    profile = tmp_path / 'config.json'
    profile.write_text(json.dumps(config))
    assert load_config(profile)['camera']['calibration']['model'] == 'fisheye'
    config = load_config()
    config['camera']['sources'][0]['width'] = 1280
    with pytest.raises(ValueError, match='USB mode'):
        validate_config(config)
    config = load_config()
    config['vision']['enabled'] = True
    with pytest.raises(ValueError, match='measured lens'):
        validate_config(config)
    config['camera']['calibration'] = dict(calibration_record(), simulation_only=True)
    with pytest.raises(ValueError, match='Simulation-only'):
        validate_config(config)


def test_ground_check_uses_independent_equidistant_controls(tmp_path):
    specification = importlib.util.spec_from_file_location('camera_tool', ROOT / 'scripts/camera.py')
    tool = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(tool)
    ast.parse((ROOT / 'scripts/camera.py').read_text(), feature_version=(3, 6))
    config = load_config()
    config['camera'].update(pitch=90, calibration=calibration_record())
    world = np.array([[1, 0, 0], [0, 2, 0], [-2, -1, 0], [1, -2, 0]], dtype=float)
    normalized = -world[:, [1, 0]] / 6
    radius = np.linalg.norm(normalized, axis=1)
    pixels = [960, 540] + 600 * normalized * (np.arctan(radius) / radius)[:, None]
    profile, controls = tmp_path / 'profile.json', tmp_path / 'controls.json'
    profile.write_text(json.dumps(config))
    controls.write_text(json.dumps({'world': world.tolist(), 'pixels': pixels.tolist()}))
    args = SimpleNamespace(config=str(profile), points=str(controls), maximum_pixels=3, maximum_metres=0.25)
    result = tool.check_ground(args)
    assert result['passed'] and result['metres_rms'] < 1e-8
    controls.write_text(json.dumps({'world': world.tolist(), 'pixels': (pixels + 10).tolist()}))
    with pytest.raises(ValueError, match='exceed'):
        tool.check_ground(args)


def test_checkerboard_coverage_is_measured_across_the_image():
    specification = importlib.util.spec_from_file_location('camera_quality', ROOT / 'scripts/camera.py')
    tool = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(tool)
    points = np.array([[x, y] for x in (100, 600, 1100, 1600) for y in (100, 500, 900)])
    assert tool.calibration_quality([points], 1920, 1080)['occupied_cells'] == 12
