"""Optional OpenCV model adapter; no dependency install or model download occurs."""

from collections import deque
import hashlib
from pathlib import Path
import threading
import time
from urllib.request import urlopen

import numpy as np


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
        for box in boxes:
            left, top, right, bottom = [int(value) for value in box['box']]
            self.cv2.rectangle(output, (left, top), (right, bottom), (0, 180, 255), 2)
        self.cv2.putText(output, result['state'] + ': candidates, attribution unverified',
                         (10, 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if not self.cv2.imwrite(str(path), output):
            raise RuntimeError('Failed to save annotated image')


class CameraCache:
    """Bounded HTTP snapshot history; host receive times are not verified exposure times."""

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

    def start(self):
        for source in self.sources:
            if not source['uri'].startswith(('http://', 'https://')):
                raise ValueError('CameraCache currently requires HTTP JPEG snapshots; USB/RTSP need an adapter')
        for source in self.sources:
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

    def nearest(self, identity, timestamp):
        with self.lock:
            frames = list(self.history.get(identity, []))
        return min(frames, key=lambda item: abs(item['time'] - timestamp)) if frames else None

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=self.timeout + 1)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError('Camera worker did not stop within the network timeout')
