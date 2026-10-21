"""Optional OpenCV model adapter; no dependency install or model download occurs."""

from collections import deque
import hashlib
import multiprocessing
from pathlib import Path
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
        except Exception:
            self.capture.release()
            raise

    def read(self):
        started = time.monotonic()
        success, image = self.capture.read()
        ended = time.monotonic()
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


def usb_worker(source, messages, stopped):
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
        while not stopped.is_set():
            frame = camera.read()
            sequence += 1
            frame['sequence'] = sequence
            publish({'kind': 'frame', 'frame': frame})
    except Exception as error:
        publish({'kind': 'error', 'message': str(error)})
    finally:
        if camera is not None:
            camera.close()


def decode_boxes(raw, width, height, settings):
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or values.shape[1] < 6 or not np.isfinite(values).all():
        raise ValueError('Model must emit finite decoded rows: cx,cy,w,h,objectness,class scores')
    if np.any(values[:, 4:] < 0) or np.any(values[:, 4:] > 1):
        raise ValueError('Model must emit probabilities, not unprocessed logits')
    labels = np.argmax(values[:, 5:], axis=1)
    scores = values[:, 4] * values[np.arange(len(values)), labels + 5]
    selected = (scores >= settings['threshold']) & np.isin(labels, settings['classes'])
    boxes = []
    for row, label, score in zip(values[selected], labels[selected], scores[selected]):
        if np.any(row[:2] < 0) or np.any(row[:2] > 1) or np.any(row[2:4] <= 0) or np.any(row[2:4] > 2):
            raise ValueError('Decoded coordinates must use normalized image units')
        first = np.maximum(0, (row[:2] - row[2:4] / 2) * [width, height])
        final = np.minimum([width, height], (row[:2] + row[2:4] / 2) * [width, height])
        if np.all(final > first):
            boxes.append({'box': list(np.r_[first, final]), 'class': int(label), 'score': float(score)})
    kept = []
    for box in sorted(boxes, key=lambda item: -item['score']):
        a = np.asarray(box['box'])
        overlap = False
        for previous in kept:
            if box['class'] != previous['class']:
                continue
            b = np.asarray(previous['box'])
            intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
            union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
            if intersection / union > settings['nms']:
                overlap = True
                break
        if not overlap:
            kept.append(box)
    return kept


class Vision:
    def __init__(self, settings):
        try:
            import cv2
        except ImportError:
            raise RuntimeError('OpenCV is not installed; install a board-compatible runtime after approval')
        model = Path(settings['model'])
        if not settings['sha256'] or hashlib.sha256(model.read_bytes()).hexdigest() != settings['sha256']:
            raise ValueError('Vision model must match an explicitly approved checksum')
        self.cv2, self.settings = cv2, settings
        self.network = cv2.dnn.readNet(str(model), settings['configuration'])
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, image):
        cv2 = self.cv2
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.settings['input_size'],) * 2,
                                   swapRB=self.settings['pixel_order'] == 'RGB', crop=False)
        self.network.setInput(blob)
        names = self.network.getUnconnectedOutLayersNames()
        outputs = self.network.forward(names)
        raw = np.concatenate([np.asarray(value).reshape(-1, np.asarray(value).shape[-1]) for value in outputs])
        return decode_boxes(raw, image.shape[1], image.shape[0], self.settings)

    def annotate(self, image, boxes, result, path):
        output = image.copy()
        accepted = {item['identity'] for group in result['groups'] for item in group}
        for box in boxes:
            left, top, right, bottom = [int(value) for value in box['box']]
            candidate = box.get('identity') in accepted
            self.cv2.rectangle(output, (left, top), (right, bottom), (0, 180, 255) if candidate else (160, 160, 160), 2)
            self.cv2.putText(output, 'possible source' if candidate else 'vehicle', (left, max(15, top - 5)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255) if candidate else (160, 160, 160), 1)
        self.cv2.putText(output, result['state'] + ': candidates, attribution unverified',
                         (10, 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if not self.cv2.imwrite(str(path), output):
            raise RuntimeError('Failed to save annotated image')


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
                process = context.Process(target=usb_worker, args=(source, messages, stopped), daemon=True)
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

    def run_usb(self, source, process, messages):
        while not self.stop.is_set():
            try:
                message = messages.get(timeout=0.2)
            except Empty:
                if not process.is_alive():
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

    def nearest(self, identity, timestamp):
        with self.lock:
            frames = list(self.history.get(identity, []))
        if not frames:
            return None
        frame = dict(min(frames, key=lambda item: abs(item['time'] - timestamp)))
        if 'image' not in frame:
            frame['image'] = self.vision.cv2.imdecode(np.frombuffer(frame['jpeg'], dtype=np.uint8), self.vision.cv2.IMREAD_COLOR)
            if frame['image'] is None:
                raise ValueError('Cached USB JPEG cannot be decoded')
        return frame

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
