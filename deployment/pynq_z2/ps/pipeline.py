"""Board-only candidate inference and conservative event/vehicle decisions."""

import time

import numpy as np

from ps.dsp import beamform, candidates, features, spatial_map
from ps.geometry import AcousticEvidence, Association, Detection, Rig, associate, route_cameras


class Events:
    def __init__(self, settings, update):
        self.settings, self.update = settings, update
        self.identity, self.state, self.on, self.off = 0, 'IDLE', 0, 0
        self.epoch, self.end = None, None

    def feed(self, positive, snapshot):
        reset = self.epoch is not None and (snapshot['epoch'] != self.epoch or snapshot['end'] - self.end > self.update)
        ended = self.identity if reset and self.state != 'IDLE' else None
        if reset:
            self.state, self.on, self.off = 'IDLE', 0, 0
        self.epoch, self.end = snapshot['epoch'], snapshot['end']
        if positive:
            if self.state == 'IDLE':
                self.identity += 1
                self.state = 'CANDIDATE'
            self.on += 1
            self.off = 0
            if self.on >= self.settings['confirm']:
                self.state = 'CONFIRMED'
        else:
            self.on = 0
            if self.state != 'IDLE':
                self.off += 1
                if self.off >= self.settings['release']:
                    ended = self.identity
                    self.state, self.off = 'IDLE', 0
        return {'state': self.state, 'event_id': self.identity if self.state != 'IDLE' else 0,
                'ended_event': ended, 'history_reset': reset}


class Pipeline:
    def __init__(self, config, detector):
        self.config, self.detector = config, detector
        self.events = Events(config['event'], config['audio']['update_samples'])
        camera = dict(config['camera'])
        camera.pop('sources')
        self.rig = Rig(**camera)
        self.captured = 0

    def analyze(self, snapshot):
        started = time.monotonic()
        grid = spatial_map(snapshot['audio'], self.config['audio'], self.config['search'])
        bearings = candidates(grid, self.config['search']['candidates'], self.config['search']['separation'])
        scores, clipped = [], []
        for bearing in bearings:
            waveform, latency = beamform(snapshot['audio'], bearing, self.config['audio'])
            result = self.detector.score(features(waveform))
            scores.append(result['score'])
            clipped.append(result['clipped'])
        order = np.argsort(-np.asarray(scores), kind='stable')
        selected = bearings[int(order[0])] if len(order) else None
        score = scores[int(order[0])] if len(order) else 0.0
        difference = score - scores[int(order[1])] if len(order) > 1 else score
        valid = bool(selected is not None and score >= self.config['detector']['threshold']
                     and difference >= self.config['detector']['margin'])
        event = self.events.feed(valid, snapshot)
        event.update({'sample_first': snapshot['first'], 'sample_end_exclusive': snapshot['end'],
                      'epoch': snapshot['epoch'], 'score': score, 'score_margin': difference,
                      'horn_candidate': valid, 'azimuth': selected.azimuth if valid else None,
                      'depression': selected.depression if valid else None, 'scores': scores,
                      'input_clipped': snapshot['clipped'], 'activation_clipped': any(clipped),
                      'processing_seconds': time.monotonic() - started,
                      'classifier': 'TinyCNN-1834; thresholds and beam-domain performance require validation'})
        evidence = AcousticEvidence(selected, grid) if valid else None
        return event, evidence

    def capture(self, event, evidence, cache, vision, output, representative_time):
        if event['state'] != 'CONFIRMED' or not event['horn_candidate'] or event['event_id'] == self.captured:
            return None
        selected = route_cameras(self.rig, event['azimuth'])
        detections, images = [], []
        for identity in selected:
            frame = cache.nearest(identity, representative_time) if cache is not None else None
            if frame is None:
                continue
            boxes = vision.infer(frame['image'])
            for box in boxes:
                left, top, right, bottom = box['box']
                detections.append(Detection('observation-' + str(len(detections)), identity, tuple(box['box']),
                                            ((left + right) / 2, bottom), frame['time']))
            images.append((identity, frame, boxes))
        result = associate(self.rig, event['azimuth'], representative_time, detections,
                           Association(**self.config['association']), evidence=evidence)
        if result['groups']:
            result['state'] = 'ambiguous'
        result['verification'] = 'unverified timing/calibration and heuristic box-bottom anchors; no unique vehicle attribution'
        for identity, frame, boxes in images:
            name = 'event-{}-{}'.format(event['event_id'], identity)
            (output / (name + '.jpg')).write_bytes(frame['jpeg'])
            vision.annotate(frame['image'], boxes, result, output / (name + '-annotated.jpg'))
        if images:
            self.captured = event['event_id']
        return {'selected_cameras': selected, 'association': result, 'photos': len(images)}
