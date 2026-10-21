"""Board-only candidate inference and conservative event/vehicle decisions."""

import time

import numpy as np

from ps.dsp import beamform, candidates, features, spatial_map
from ps.detector import FrontendOverflow
from ps.geometry import AcousticEvidence, Association, Bearing, Detection, Rig, associate, route_cameras


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
    def __init__(self, config, detector, scorer=None):
        self.config, self.detector = config, detector
        if config['search'].get('backend', 'numpy') == 'fpga' and scorer is None:
            raise ValueError('FPGA SRP requires an initialized DMA scorer')
        self.scorer = scorer
        self.events = Events(config['event'], config['audio']['update_samples'])
        camera = dict(config['camera'])
        camera.pop('sources')
        self.rig = Rig(**camera)
        self.captured = 0

    def analyze(self, snapshot):
        started = time.monotonic()
        grid = spatial_map(snapshot['audio'], self.config['audio'], self.config['search'], self.scorer)
        bearings = candidates(grid, self.config['search']['candidates'], self.config['search']['separation'])
        scores, clipped, frontend_clipped = [], [], []
        for bearing in bearings:
            waveform, latency = beamform(snapshot['audio'], bearing, self.config['audio'])
            try:
                result = self.detector.score(features(waveform, getattr(self.detector, 'frontend', 'logmel')))
            except FrontendOverflow:
                result = dict(score=0.0, clipped=False, frontend_clipped=True)
            scores.append(result['score'])
            clipped.append(result['clipped'])
            frontend_clipped.append(result.get('frontend_clipped', False))
        order = np.argsort(-np.asarray(scores), kind='stable')
        selected = bearings[int(order[0])] if len(order) else None
        score = scores[int(order[0])] if len(order) else 0.0
        difference = score - scores[int(order[1])] if len(order) > 1 else score
        reliable = not snapshot['clipped'] and not any(clipped) and not any(frontend_clipped)
        audible = bool(reliable and selected is not None and score >= self.config['detector']['threshold'])
        valid = bool(audible and (len(order) == 1 or difference > 0)
                     and difference >= self.config['detector']['margin'])
        hypotheses = [dict(azimuth=bearings[int(index)].azimuth, depression=bearings[int(index)].depression,
                           response=bearings[int(index)].response, score=scores[int(index)])
                      for index in order if audible and scores[int(index)] >= self.config['detector']['threshold']
                      and (scores[int(index)] == score or score - scores[int(index)] < self.config['detector']['margin'])]
        event = self.events.feed(audible, snapshot)
        event.update({'sample_first': snapshot['first'], 'sample_end_exclusive': snapshot['end'],
                      'epoch': snapshot['epoch'], 'score': score, 'score_margin': difference,
                      'horn_candidate': audible, 'doa_valid': valid,
                      'acoustic_state': ('invalid_audio' if not reliable else 'direction_selected' if valid
                                         else 'ambiguous_direction' if audible else 'no_horn_evidence'),
                      'direction_candidates': hypotheses, 'azimuth': selected.azimuth if valid else None,
                      'depression': selected.depression if valid else None, 'scores': scores,
                      'input_clipped': snapshot['clipped'], 'activation_clipped': any(clipped),
                      'frontend_clipped': any(frontend_clipped),
                      'processing_seconds': time.monotonic() - started,
                      'srp_backend': 'fpga' if self.scorer is not None else 'numpy',
                      'classifier': getattr(self.detector, 'metadata', {}).get('architecture', 'test-detector')})
        evidence = AcousticEvidence(selected, grid) if audible else None
        return event, evidence

    def capture(self, event, evidence, cache, vision, output, representative_time):
        if event['state'] != 'CONFIRMED' or not event['horn_candidate'] or event['event_id'] == self.captured:
            return None
        hypotheses = event.get('direction_candidates') or [dict(azimuth=event['azimuth'])]
        routes = [route_cameras(self.rig, hypothesis['azimuth']) for hypothesis in hypotheses]
        selected = sorted({identity for route in routes for identity in route})
        detections, images = [], []
        for identity in selected:
            frame = cache.nearest(identity, representative_time) if cache is not None else None
            if frame is None:
                continue
            camera = next(item for item in self.rig.cameras() if item.identity == identity)
            if frame['image'].shape[:2] != (camera.height, camera.width):
                raise ValueError('Captured image dimensions disagree with camera calibration')
            boxes = vision.infer(frame['image'])
            for box in boxes:
                left, top, right, bottom = box['box']
                box['identity'] = 'observation-' + str(len(detections))
                detections.append(Detection(box['identity'], identity, tuple(box['box']),
                                            ((left + right) / 2, bottom), frame['time']))
            images.append((identity, frame, boxes))
        comparisons = []
        for hypothesis in hypotheses:
            support = evidence
            if evidence is not None and 'depression' in hypothesis:
                support = AcousticEvidence(Bearing(hypothesis['azimuth'], hypothesis['depression'], hypothesis['response']), evidence.grid)
            comparisons.append(associate(self.rig, hypothesis['azimuth'], representative_time, detections,
                                          Association(**self.config['association']), evidence=support))
        result = dict(comparisons[0])
        if len(comparisons) > 1:
            groups = {}
            for comparison in comparisons:
                for group in comparison['groups']:
                    groups[tuple(sorted(item['identity'] for item in group))] = group
            result['groups'] = list(groups.values())
            result['state'] = 'ambiguous' if groups else 'unmatched'
            result['source_unexplained'] = any(comparison['source_unexplained'] for comparison in comparisons)
            result['acoustic_hypotheses'] = comparisons
        if result['groups']:
            result['state'] = 'ambiguous'
        result['verification'] = 'unverified timing/calibration and heuristic box-bottom anchors; no unique vehicle attribution'
        for identity, frame, boxes in images:
            name = 'event-{}-{}'.format(event['event_id'], identity)
            (output / (name + '.jpg')).write_bytes(frame['jpeg'])
            vision.annotate(frame['image'], boxes, result, output / (name + '-annotated.jpg'))
        if images:
            self.captured = event['event_id']
        return {'selected_cameras': selected, 'association': result, 'photos': len(images),
                'visibility': ('partly_outside_camera_view' if selected and any(not route for route in routes)
                               else 'candidate_in_nominal_or_calibrated_view' if selected else 'outside_camera_view')}
