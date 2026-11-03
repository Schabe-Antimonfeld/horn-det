"""Board-only acoustic inference, event confirmation and point-only camera output."""

import time

import numpy as np

from ps.dsp import beamform, candidates, features, spatial_map
from ps.detector import FrontendOverflow
from ps.geometry import AcousticEvidence, Rig
from ps.pointing import point_sources


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
    def __init__(self, config, detector, scorer=None, searcher=None, beamformer=None, processing=None):
        self.config, self.detector = config, detector
        if config['search'].get('backend', 'numpy') == 'fpga' and scorer is None:
            raise ValueError('FPGA SRP requires an initialized DMA scorer')
        self.scorer = scorer
        self.processing = processing
        if config.get('streaming', {}).get('enabled') and self.processing is None:
            from ps.streaming import Processing
            self.processing = Processing(config, getattr(detector, 'frontend', 'logmel'))
        self.searcher = None
        if self.processing is not None:
            self.searcher = self.processing.search
        elif config['search'].get('algorithm', 'reference') != 'reference':
            from ps.search import Search
            self.searcher = searcher if searcher is not None else Search(config['audio'], config['search'])
        self.beamformer = self.processing.beams if self.processing is not None else beamformer
        if self.beamformer is None and config.get('beamforming', {}).get('implementation', 'reference') != 'reference':
            from ps.beamforming import Beamformer
            self.beamformer = Beamformer(config['audio'], config['beamforming'])
        self.events = Events(config['event'], config['audio']['update_samples'])
        camera = dict(config['camera'])
        camera.pop('sources')
        self.rig = Rig(**camera)
        self.horn_events = Events(config['event'], config['audio']['update_samples'])
        self.capture_event, self.captured = 0, set()

    def analyze(self, snapshot):
        started = time.perf_counter()
        context = self.processing.continuity.prepare(snapshot) if self.processing is not None else None
        if self.searcher is None:
            grid = spatial_map(snapshot['audio'], self.config['audio'], self.config['search'], self.scorer)
            bearings = candidates(grid, self.config['search']['candidates'], self.config['search']['separation'])
            diagnostic = dict(algorithm='reference', pair_count=91, evidence_grid='complete_fine')
        else:
            grid, bearings, diagnostic = (self.searcher.run_snapshot(snapshot, context) if context is not None
                                         else self.searcher.run(snapshot['audio']))
            diagnostic['algorithm'] = self.searcher.algorithm
        spatial_done = time.perf_counter()
        phases = dict(srp=spatial_done - started, beamforming=0.0, features=0.0, detector=0.0)
        beam_metric = dict(implementation='reference', beams=len(bearings), taps=65, cache_hits=0, fft_size=None)
        beams = None
        if self.beamformer is not None:
            step = time.perf_counter()
            beams, beam_metric = (self.beamformer.run_snapshot(snapshot, bearings, context) if context is not None
                                  else self.beamformer.run(snapshot['audio'], bearings))
            phases['beamforming'] = time.perf_counter() - step
        scores, clipped, frontend_clipped, feature_metrics = [], [], [], []
        for index, bearing in enumerate(bearings):
            if beams is None:
                step = time.perf_counter()
                waveform, latency = beamform(snapshot['audio'], bearing, self.config['audio'])
                phases['beamforming'] += time.perf_counter() - step
            else:
                waveform, latency = beams[index]
            try:
                step = time.perf_counter()
                if context is not None:
                    contexts, feature_metric = self.processing.frontend.run(waveform, bearing, context, beam_metric['support_samples'])
                    feature_metrics.append(feature_metric)
                else:
                    contexts = features(waveform, getattr(self.detector, 'frontend', 'logmel'))
                phases['features'] += time.perf_counter() - step
                step = time.perf_counter()
                result = self.detector.score(contexts)
                phases['detector'] += time.perf_counter() - step
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
        capture_threshold = self.config['detector'].get('capture_threshold', self.config['detector']['threshold'])
        capture_candidate = bool(reliable and selected is not None and score >= capture_threshold)
        capture_hypotheses = hypotheses
        if 'capture_threshold' in self.config['detector']:
            capture_hypotheses = [dict(azimuth=bearings[int(index)].azimuth, depression=bearings[int(index)].depression,
                                      response=bearings[int(index)].response, score=scores[int(index)])
                                  for index in order if capture_candidate and scores[int(index)] >= capture_threshold]
        event = self.events.feed(capture_candidate, snapshot)
        horn_event = self.horn_events.feed(audible, snapshot)
        event.update(horn_event_state=horn_event['state'],
                     horn_confirmed=bool(audible and self.horn_events.on >= self.config['event']['confirm']))
        event.update({'sample_first': snapshot['first'], 'sample_end_exclusive': snapshot['end'],
                      'epoch': snapshot['epoch'], 'score': score, 'score_margin': difference,
                      'horn_candidate': audible, 'doa_valid': valid,
                      'capture_candidate': capture_candidate, 'capture_candidates': capture_hypotheses,
                      'capture_threshold': capture_threshold, 'direction_threshold': self.config['detector']['threshold'],
                      'event_basis': 'capture_candidate' if 'capture_threshold' in self.config['detector'] else 'horn_candidate',
                      'acoustic_state': ('invalid_audio' if not reliable else 'direction_selected' if valid
                                         else 'ambiguous_direction' if audible else 'capture_only' if capture_candidate
                                         else 'no_horn_evidence'),
                      'direction_candidates': hypotheses, 'azimuth': selected.azimuth if valid else None,
                      'depression': selected.depression if valid else None, 'scores': scores,
                      'input_clipped': snapshot['clipped'], 'activation_clipped': any(clipped),
                      'frontend_clipped': any(frontend_clipped),
                      'processing_seconds': time.perf_counter() - started,
                      'phase_seconds': phases, 'srp_diagnostics': diagnostic, 'beamforming_diagnostics': beam_metric,
                      'srp_backend': 'fpga' if self.scorer is not None else 'numpy',
                      'classifier': getattr(self.detector, 'metadata', {}).get('architecture', 'test-detector')})
        evidence = AcousticEvidence(selected, grid) if capture_candidate else None
        if context is not None:
            event['stream_diagnostics'] = dict(context, features=feature_metrics)
        return event, evidence

    def capture(self, event, evidence, cache, imaging, output, representative_time):
        """Retain one provisional and one confirmed image per camera and event, without DNN inference."""
        if (not event.get('capture_candidate', False) or not event.get('event_id')
                or any(event.get(key, False) for key in ('input_clipped', 'activation_clipped', 'frontend_clipped'))):
            return None
        if not np.isfinite(representative_time):
            raise ValueError('Camera reference time must be finite')
        if event['event_id'] != self.capture_event:
            self.capture_event, self.captured = event['event_id'], set()
        started = time.perf_counter()
        stage = 'confirmed' if event.get('horn_confirmed', False) else 'candidate'
        hypotheses = event.get('capture_candidates') or event.get('direction_candidates') or []
        projections = point_sources(self.rig, hypotheses)
        selected = sorted(identity for identity, points in projections.items()
                          if any(point['state'] != 'outside_camera_view' for point in points))
        pending = [identity for identity in selected if (identity, stage) not in self.captured]
        if selected and not pending:
            return None
        projected = any(point['state'] == 'projected' for points in projections.values() for point in points)
        visibility = ('partly_outside_camera_view' if projected and any(
            point['state'] == 'outside_camera_view' for points in projections.values() for point in points)
            else 'candidate_in_nominal_or_calibrated_view' if projected
            else 'projection_unresolved' if selected else 'outside_camera_view')
        result = dict(selected_cameras=selected, projections=projections, photos=0, frames=[],
                      stage=stage, horn_confirmed=event.get('horn_confirmed', False), visibility=visibility,
                      vehicle_attribution='unverified',
                      verification='Acoustic direction points only; no vehicle detection or unique vehicle attribution')
        projection_done = time.perf_counter()
        for identity in pending:
            frame = cache.nearest(identity, representative_time) if cache is not None else None
            status = dict(camera=identity, state='unavailable')
            result['frames'].append(status)
            if frame is None:
                continue
            skew = float(frame['time'] - representative_time)
            status.update(skew_seconds=skew if np.isfinite(skew) else None,
                          timing_verified=bool(frame.get('timing_verified', False)))
            if not np.isfinite(skew) or abs(skew) > self.config['association']['maximum_skew']:
                status['state'] = 'stale'
                continue
            camera = next(item for item in self.rig.cameras() if item.identity == identity)
            if frame['image'].shape[:2] != (camera.height, camera.width):
                raise ValueError('Captured image dimensions disagree with camera calibration')
            name = 'event-{}-{}-{}'.format(event['event_id'], identity, stage)
            (output / (name + '.jpg')).write_bytes(frame['jpeg'])
            imaging.annotate(frame['image'], projections[identity], result, output / (name + '-points.jpg'))
            status.update(state='saved', original=name + '.jpg', annotated=name + '-points.jpg')
            self.captured.add((identity, stage))
            result['photos'] += 1
        finished = time.perf_counter()
        result.update(processing_seconds=finished - started,
                      phase_seconds=dict(projection=projection_done - started,
                                         camera_and_save=finished - projection_done, vision=0.0))
        return result
