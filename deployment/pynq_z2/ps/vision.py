"""Optional OpenCV camera I/O and acoustic point annotation; no neural network."""

from collections import deque
import multiprocessing
from queue import Empty, Full
import threading
import time
from urllib.request import urlopen

import numpy as np


def validate_source(source):
    backend = source.get('backend', 'http')
    if backend == 'http':
        if not isinstance(source['uri'], str) or not source['uri'].startswith(('http://', 'https://')):
            raise ValueError('HTTP cameras need a snapshot URL')
    elif backend == 'usb':
        if not isinstance(source['uri'], (str, int)) or isinstance(source['uri'], bool):
            raise ValueError('USB camera needs a device path or index')
        if any(type(source.get(key)) is not int or source[key] <= 0 for key in ('width', 'height')):
            raise ValueError('USB dimensions must be positive integers')
        if not np.isfinite(source.get('fps', 0)) or source['fps'] <= 0 or source.get('fourcc') not in ('MJPG', 'YUY2', 'YUYV'):
            raise ValueError('USB camera needs a positive FPS and MJPG/YUY2/YUYV format')
    else:
        raise ValueError('Camera backend must be http or usb')


def jpeg_dimensions(data):
    """Read bounded JPEG SOF dimensions without decompressing image pixels."""
    if not data.startswith(b'\xff\xd8') or len(data) > 8*1024*1024:
        raise ValueError('Invalid or oversized JPEG frame')
    offset = 2
    sof = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while offset < len(data):
        if data[offset] != 255:
            raise ValueError('Invalid JPEG marker')
        while offset < len(data) and data[offset] == 255:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xd9, 0xda):
            break
        if marker == 1 or 0xd0 <= marker <= 0xd7:
            continue
        if offset+2 > len(data):
            break
        length = int.from_bytes(data[offset:offset+2], 'big')
        if length < 2 or offset+length > len(data):
            raise ValueError('Truncated JPEG segment')
        if marker in sof:
            if length < 8 or data[offset+2] != 8 or length != 8+3*data[offset+7]:
                raise ValueError('Unsupported JPEG frame header')
            height = int.from_bytes(data[offset+3:offset+5], 'big')
            width = int.from_bytes(data[offset+5:offset+7], 'big')
            if width < 1 or height < 1:
                raise ValueError('Invalid JPEG dimensions')
            return width, height
        offset += length
    raise ValueError('JPEG frame has no dimensions')



class UsbCamera:
    """Read a strictly negotiated V4L2 mode inside an isolated capture process."""

    def __init__(self, source, cv2):
        validate_source(source)
        self.cv2, self.source = cv2, source
        self.capture = cv2.VideoCapture(source['uri'], cv2.CAP_V4L2)
        try:
            if not self.capture.isOpened():
                raise RuntimeError('USB camera could not open through V4L2')
            fourcc = 'YUYV' if source['fourcc'] == 'YUY2' else source['fourcc']
            for key, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
                               (cv2.CAP_PROP_FRAME_WIDTH, source['width']),
                               (cv2.CAP_PROP_FRAME_HEIGHT, source['height']),
                               (cv2.CAP_PROP_FPS, source['fps']), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                self.capture.set(key, value)
            code = self.capture.get(cv2.CAP_PROP_FOURCC)
            if not np.isfinite(code):
                raise RuntimeError('USB camera did not report its pixel format')
            actual_format = ''.join(chr((int(round(code)) >> (8 * index)) & 255) for index in range(4))
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            if (width, height) != (source['width'], source['height']) or actual_format != fourcc or not np.isfinite(fps) or abs(fps - source['fps']) > max(0.5, source['fps'] * 0.03):
                raise RuntimeError('USB mode mismatch: {}x{} {} at {}; requested {}x{} {} at {}'.format(
                    width, height, actual_format, fps, source['width'], source['height'], fourcc, source['fps']))
            self.mode = {'width': int(width), 'height': int(height), 'fourcc': actual_format,
                         'fps_reported': float(fps), 'fps_requested': source['fps'],
                         'sustained_rate_verified': False}
            # V4L2 MJPG already contains JPEG. Preserve its bytes instead of
            # decompressing and recompressing every captured frame on the ARM.
            self.raw_jpeg = bool(fourcc == 'MJPG' and hasattr(cv2, 'CAP_PROP_CONVERT_RGB')
                                 and self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0))
            self.mode['jpeg_passthrough'] = self.raw_jpeg
        except Exception:
            self.capture.release()
            raise

    def read(self):
        started = time.monotonic()
        success, image = self.capture.read()
        ended = time.monotonic()
        if self.raw_jpeg:
            if not success or image is None or image.dtype != np.uint8 or image.ndim > 2 or image.nbytes > 8*1024*1024:
                raise RuntimeError('USB raw MJPEG read failed')
            data = image.tobytes()
            if jpeg_dimensions(data) != (self.source['width'], self.source['height']):
                raise RuntimeError('USB JPEG dimensions differ from the requested mode')
            return {'jpeg': data, 'time': (started+ended)/2, 'timing_verified': False,
                    'timestamp_kind': 'host_read_midpoint', 'mode': self.mode,
                    'jpeg_reencoded': False}
        if not success or image is None or image.shape != (self.source['height'], self.source['width'], 3) or image.dtype != np.uint8:
            raise RuntimeError('USB read failed or returned a different image mode')
        success, encoded = self.cv2.imencode('.jpg', image)
        if not success or encoded.nbytes > 8 * 1024 * 1024:
            raise RuntimeError('USB JPEG encoding failed or exceeded 8 MiB')
        return {'jpeg': encoded.tobytes(), 'time': (started + ended) / 2,
                'timing_verified': False, 'timestamp_kind': 'host_read_midpoint', 'mode': self.mode,
                'jpeg_reencoded': True}

    def close(self):
        self.capture.release()


def usb_worker(source, messages, stopped, period=0.25):
    """Isolate potentially blocking V4L2 reads from the main service lifecycle."""
    camera = None
    def publish(message):
        try:
            messages.put_nowait(message)
        except Full:
            try:
                messages.get_nowait()
            except Empty:
                pass
            try:
                messages.put_nowait(message)
            except Full:
                pass
    try:
        import cv2
        camera = UsbCamera(source, cv2)
        sequence = 0
        next_publish = 0.0
        while not stopped.is_set():
            frame = camera.read()
            sequence += 1
            frame['sequence'] = sequence
            now = time.monotonic()
            if now >= next_publish:
                publish({'kind': 'frame', 'frame': frame})
                next_publish = now + period
    except Exception as error:
        publish({'kind': 'error', 'message': str(error)})
    finally:
        if camera is not None:
            camera.close()


class ImageIO:
    """Decode camera images and draw only acoustic candidate points."""

    def __init__(self):
        try:
            import cv2
        except ImportError:
            raise RuntimeError('OpenCV is not installed; use the approved board-compatible image runtime')
        self.cv2 = cv2

    def annotate(self, image, points, result, path):
        output = image.copy()
        confirmed = result['horn_confirmed']
        color = (60, 220, 60) if confirmed else (0, 180, 255)
        for point in points:
            if point['state'] != 'projected':
                continue
            pixel = tuple(int(round(value)) for value in point['pixel'])
            pixel = (min(image.shape[1] - 1, pixel[0]), min(image.shape[0] - 1, pixel[1]))
            self.cv2.circle(output, pixel, 7, color, -1)
            self.cv2.putText(output, 'source candidate ' + str(point['candidate'] + 1),
                             (max(0, pixel[0] - 75), max(18, pixel[1] - 12)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        label = 'horn evidence confirmed' if confirmed else 'UNCONFIRMED acoustic candidate'
        self.cv2.putText(output, label + '; timing/calibration and vehicle attribution unverified',
                         (10, 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not self.cv2.imwrite(str(path), output):
            raise RuntimeError('Failed to save point-annotated image')


class CameraCache:
    """Bounded JPEG history; USB reads use terminable child processes."""

    def __init__(self, sources, vision, frames=4, period=0.5, timeout=2.0):
        if frames < 1 or period <= 0 or timeout <= 0:
            raise ValueError('Invalid camera cache bounds')
        self.sources, self.vision = sources, vision
        self.history = {item['id']: deque(maxlen=frames) for item in sources}
        if len(self.history) != len(sources):
            raise ValueError('Camera IDs must be distinct')
        self.period, self.timeout = period, timeout
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.errors, self.threads = {}, []
        self.workers, self.modes = [], {}

    def start(self):
        for source in self.sources:
            validate_source(source)
        for source in self.sources:
            if source.get('backend', 'http') == 'usb':
                context = multiprocessing.get_context('spawn')
                messages, stopped = context.Queue(maxsize=1), context.Event()
                process = context.Process(target=usb_worker, args=(source, messages, stopped, self.period), daemon=True)
                process.start()
                self.workers.append((process, messages, stopped))
                thread = threading.Thread(target=self.run_usb, args=(source, process, messages), daemon=True)
            else:
                thread = threading.Thread(target=self.run, args=(source,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def run(self, source):
        while not self.stop.is_set():
            began = time.monotonic()
            try:
                with urlopen(source['uri'], timeout=self.timeout) as response:
                    data = response.read(8 * 1024 * 1024 + 1)
                if len(data) > 8 * 1024 * 1024:
                    raise ValueError('Snapshot exceeds configured 8 MiB bound')
                image = self.vision.cv2.imdecode(np.frombuffer(data, dtype=np.uint8), self.vision.cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Invalid JPEG snapshot')
                frame = {'image': image, 'jpeg': data, 'time': (began + time.monotonic()) / 2,
                         'timing_verified': False, 'timestamp_kind': 'host_request_midpoint'}
                with self.lock:
                    self.history[source['id']].append(frame)
                    self.errors.pop(source['id'], None)
            except Exception as error:
                with self.lock:
                    self.errors[source['id']] = str(error)
            self.stop.wait(max(0, self.period - (time.monotonic() - began)))

    def wait_ready(self, timeout=30.0):
        """Complete camera initialization before starting time-critical audio."""
        if timeout <= 0:
            raise ValueError('Camera startup timeout must be positive')
        deadline = time.monotonic()+timeout
        while True:
            with self.lock:
                if self.errors:
                    raise RuntimeError('Camera startup failed: '+str(self.errors))
                if all(self.history[source['id']] for source in self.sources):
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError('Camera did not produce its first frame')
            time.sleep(0.02)

    def run_usb(self, source, process, messages):
        # Keep draining during shutdown so the child queue feeder can finish.
        # Stopping the reader first makes process.join wait for its own full pipe.
        while not self.stop.is_set() or process.is_alive():
            try:
                message = messages.get(timeout=0.2)
            except Empty:
                if not process.is_alive():
                    if not self.stop.is_set():
                        with self.lock:
                            self.errors.setdefault(source['id'], 'USB capture process exited; restart after checking the device')
                    return
                continue
            with self.lock:
                if message['kind'] == 'error':
                    self.errors[source['id']] = message['message']
                else:
                    history = self.history[source['id']]
                    if not history or message['frame']['time'] - history[-1]['time'] >= self.period:
                        history.append(message['frame'])
                    self.modes[source['id']] = message['frame']['mode']
                    self.errors.pop(source['id'], None)

    def nearest(self, identity, timestamp, decode=True):
        with self.lock:
            frames = list(self.history.get(identity, []))
        if not frames:
            return None
        frame = dict(min(frames, key=lambda item: abs(item['time'] - timestamp)))
        if decode and 'image' not in frame:
            frame['image'] = self.vision.cv2.imdecode(np.frombuffer(frame['jpeg'], dtype=np.uint8), self.vision.cv2.IMREAD_COLOR)
            if frame['image'] is None:
                raise ValueError('Cached USB JPEG cannot be decoded')
        return frame

    def latest(self, identity):
        """Return the existing encoded frame without decoding or opening another camera."""
        with self.lock:
            frames = self.history.get(identity)
            return dict(frames[-1]) if frames else None

    def health(self):
        with self.lock:
            return {'errors': dict(self.errors), 'negotiated_modes': dict(self.modes),
                    'cached_frames': {key: len(value) for key, value in self.history.items()}}

    def close(self):
        self.stop.set()
        for process, messages, stopped in self.workers:
            stopped.set()
            process.join(timeout=self.timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.timeout)
            if process.is_alive():
                raise RuntimeError('USB capture process did not stop')
        for thread in self.threads:
            thread.join(timeout=self.timeout + 1)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError('Camera worker did not stop within the network timeout')
        for process, messages, stopped in self.workers:
            messages.close()
            messages.join_thread()
