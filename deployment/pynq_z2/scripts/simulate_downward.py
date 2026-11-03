"""Simulate downward camera geometry and the frozen acoustic pipeline without a vision model."""

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import argparse
from copy import deepcopy
from html import escape
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.conditioning import create_conditioning
from ps.configuration import load_config
from ps.detector import Detector
from ps.dsp import coordinates
from ps.geometry import Rig, direction_vectors
from ps.pipeline import Pipeline
from ps.pointing import point_sources
from ps.evaluation import verify_assets
from scripts.build import build


CONDITIONS = ('nominal', 'height_error', 'tilt_error', 'lens_error', 'pixel_noise', 'occluded')
POSITIONS = ((1.8, 4), (-1.8, 4), (1.8, -4), (-1.8, -4), (1.8, 10), (-1.8, -10),
             (3.8, 5), (-3.8, -5), (8, 0), (-8, 0), (0.3, 0), (1.8, 4))





def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')



def configuration(height=6.0, hfov=144.0, vfov=77.6):
    """Construct a simulation-only fisheye matching the nominal central-axis fields of view."""
    if not np.isfinite([height, hfov, vfov]).all() or not height > 2.3 or not 0 < min(hfov, vfov) <= max(hfov, vfov) < 170:
        raise ValueError('Use height > 2.3 m and fields of view between zero and 170 degrees')
    config = load_config()
    angles = np.deg2rad([hfov, vfov]) / 2
    mapped = angles * (1 - 0.035 * angles ** 2)
    focal = np.array([960, 540]) / mapped
    calibration = dict(format=1, model='fisheye', width=1920, height=1080,
                       matrix=[[float(focal[0]), 0, 960], [0, float(focal[1]), 540], [0, 0, 1]],
                       coefficients=[-0.035, 0, 0, 0], maximum_angle=85.0, simulation_only=True)
    config['camera'].update(height=height, pitch=90.0, yaw=0.0, roll=0.0, hfov=hfov, vfov=vfov,
                            calibration=calibration)
    return config



def make_rig(config, condition='nominal'):
    camera = deepcopy(config['camera'])
    camera.pop('sources')
    if condition == 'height_error':
        camera['height'] += 0.3
    elif condition == 'tilt_error':
        camera.update(pitch=88.0, yaw=2.0)
    elif condition == 'lens_error':
        camera['calibration']['matrix'][0][0] *= 1.03
        camera['calibration']['matrix'][1][1] *= 1.03
        camera['calibration']['coefficients'][0] += 0.01
    return Rig(**camera)



def vehicles(position, index):
    point = np.asarray(position, dtype=float)
    angle = np.arctan2(point[1], point[0])
    other = -10 * np.array([np.cos(angle), np.sin(angle)])
    if index == 11:
        other = 2.4 * point
    third = 9 * np.array([-np.sin(angle), np.cos(angle)])
    return [dict(identity='vehicle-A', position=point.tolist(), height=(1.2, 1.5, 2.0)[index % 3], heading=90),
            dict(identity='vehicle-B', position=other.tolist(), height=1.5, heading=-90),
            dict(identity='vehicle-C', position=third.tolist(), height=1.6, heading=90)]



def body_points(vehicle):
    angle = np.deg2rad(vehicle['heading'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-2.25, -0.9], [2.25, -0.9], [2.25, 0.9], [-2.25, 0.9]]) @ rotation.T
    corners += vehicle['position']
    return np.vstack((np.column_stack((corners, np.zeros(4))),
                      np.column_stack((corners, np.full(4, vehicle['height'])))))



def recording(path, count):
    """Read a PCM16 clip, select channel zero, linearly resample and choose its highest-energy interval."""
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('Simulation recordings must be uncompressed PCM16 WAV')
        rate, channels = stream.getframerate(), stream.getnchannels()
        values = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').reshape(-1, channels)[:, 0] / 32768.0
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Source recording is empty or nonfinite')
    if rate != 48000:
        values = np.interp(np.arange(int(len(values) * 48000 / rate)) * rate / 48000, np.arange(len(values)), values)
    values = np.pad(values, (0, max(0, count - len(values))))
    starts = np.arange(0, len(values) - count + 1, 512)
    power = np.r_[0, np.cumsum(values ** 2)]
    start = int(starts[np.argmax(power[starts + count] - power[starts])])
    selected = values[start:start + count]
    rms = float(np.sqrt(np.mean(selected ** 2)))
    if rms < 1e-9:
        raise ValueError('Source recording has no usable energy')
    return selected / rms, dict(file=Path(path).name, sha256=digest(path), original_rate=rate,
                                crop_48k=start, samples=count, original_rms=rms, resampling='linear interpolation')



def propagate(waveform, position, array, height, source_height=0.8):
    """Use spherical absolute delays and relative channel gains; center RMS is a digital proxy."""
    source = np.r_[position, source_height - height]
    distances = np.linalg.norm(source - array, axis=1)
    reference = np.linalg.norm(source)
    samples = np.arange(len(waveform), dtype=float)
    output = np.empty((len(waveform), len(array)))
    for channel, distance in enumerate(distances):
        output[:, channel] = np.interp(samples - distance * 48000 / 343, samples, waveform, left=0, right=0) * reference / distance
    return output



def synthesize(config, target, other, cars, seed):
    array = coordinates(config['audio'])
    first = propagate(target, cars[0]['position'], array, config['camera']['height'])
    second = propagate(other, cars[1]['position'], array, config['camera']['height'])
    mixture = 0.025 * first + 0.025 * 10 ** (5 / 20) * second
    mixture += np.random.RandomState(seed).normal(0, 0.001, mixture.shape)
    gain = min(1.0, 0.95 / max(1e-12, float(np.max(abs(mixture)))))
    pcm = np.rint(mixture * gain * 32768).astype('<i2')
    return pcm, dict(gain=gain, peak=float(np.max(abs(pcm.astype(float))) / 32768),
                     sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), sir_db=-5.0,
                     scope='center digital RMS held fixed across positions, not calibrated SPL or an absolute distance attenuation test')



def analyze(config, detector, pcm, epoch, diagnostics=None):
    pipeline, conditioner = Pipeline(config, detector), create_conditioning(config)
    audio = np.vstack([conditioner.feed(block) for block in np.array_split(pcm, np.arange(512, len(pcm), 512))])
    events, first_capture, last = [], None, None
    for first in range(0, len(audio) - 48128 + 1, 24064):
        snapshot = dict(audio=audio[first:first + 48128], first=first, end=first + 48128, epoch=epoch, clipped=False)
        event, evidence = pipeline.analyze(snapshot)
        events.append(event)
        if diagnostics is not None:
            if first == 0:
                diagnostics.update(event=event, evidence=evidence)
            diagnostics.setdefault('candidates', []).append([
                dict(azimuth=None if depression >= 85 else azimuth, depression=depression)
                for azimuth, depression in pipeline.processing.beams.outputs])
        last = event, evidence
        if first_capture is None and event['capture_candidate']:
            first_capture = last
    selected, evidence = first_capture if first_capture is not None else last
    return selected, evidence, events, first_capture is not None



def text(x, y, value, size=22, color='#eaf0f7'):
    return '<text x="{}" y="{}" font-size="{}" fill="{}">{}</text>'.format(x, y, size, color, escape(str(value)))



def projected_line(camera, points, color, width=2):
    pixels, valid = camera.project(np.asarray(points, dtype=float))
    pieces = []
    for indices in np.split(np.arange(len(points)), np.flatnonzero(np.diff(valid.astype(int))) + 1):
        if len(indices) > 1 and valid[indices[0]]:
            data = ' '.join('{:.2f},{:.2f}'.format(*point) for point in pixels[indices])
            pieces.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="{}"/>'.format(data, color, width))
    return ''.join(pieces)



def coverage(path, config):
    camera = make_rig(config).cameras()[0]
    points = np.array([[x, y, 0] for x in np.arange(-15, 15, 0.5) for y in np.arange(-24, 24, 0.5)])
    _, visible = camera.project(points)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780">',
             '<rect width="1000" height="780" fill="#18212f"/>', text(20, 35, 'Downward synthetic fisheye | ground coverage, not detection accuracy', 23)]
    for point, valid in zip(points, visible):
        x, y = 500 + point[1] * 18, 350 - point[0] * 18
        parts.append('<rect x="{}" y="{}" width="8" height="8" fill="{}"/>'.format(x, y, '#5cb88f' if valid else '#714951'))
    for y in (-20, -10, 0, 10, 20):
        parts.append(text(490 + y * 18, 644, str(y), 17))
    for x in (-10, -5, 0, 5, 10):
        parts.append(text(22, 355 - x * 18, str(x), 17))
    parts += ['<circle cx="500" cy="350" r="7" fill="#fff"/>', text(940, 644, 'Y/m', 17), text(17, 78, 'X/m', 17),
              text(20, 695, 'Green: camera visibility only; red: outside. Center dot: camera.', 22),
              text(20, 742, 'height {} m | pitch 90 deg | nominal FOV {} x {} deg | 0.5 m grid'.format(config['camera']['height'], config['camera']['hfov'], config['camera']['vfov']), 22), '</svg>']
    Path(path).write_text('\n'.join(parts), encoding='utf-8')
    return dict(grid=len(points), visible=int(visible.sum()), fraction=float(visible.mean()),
                grid_extent_metres=dict(x=[-15, 15], y=[-24, 24]), spacing=0.5)



def geometry_checks(config):
    """Separate injected-bearing numerical checks from real-recording acoustic inference."""
    true_rig = make_rig(config)
    true = true_rig.cameras()[0]
    control = np.array([[x, y, 0] for x in (-3, -1, 1, 3) for y in (-12, -6, 0, 6, 12)], dtype=float)
    pixels, visible = true.project(control)
    control, pixels = control[visible], pixels[visible]
    controls, rows = {}, []
    for condition in CONDITIONS:
        rig = make_rig(config, condition)
        estimated = rig.cameras()[0]
        observed = pixels.copy()
        if condition == 'pixel_noise':
            observed += np.random.RandomState(91).normal(0, 2, observed.shape)
        recovered, valid = estimated.ground(observed)
        distance = np.linalg.norm(recovered[valid, :2] - control[valid, :2], axis=1)
        controls[condition] = dict(points=len(control), valid=int(valid.sum()),
                                   metres_rms=float(np.sqrt(np.mean(distance ** 2))) if len(distance) else None)
        for index, position in enumerate(POSITIONS):
            source = np.r_[position, 0.8]
            vector = source - [0, 0, config['camera']['height']]
            depression = float(np.rad2deg(np.arctan2(-vector[2], np.linalg.norm(vector[:2]))))
            hypothesis = dict(azimuth=float(np.rad2deg(np.arctan2(vector[1], vector[0]))), depression=depression)
            point = point_sources(rig, [hypothesis])[estimated.identity][0]
            truth_pixel, truth_visible = true.project([source])
            error = float(np.linalg.norm(point['pixel'] - truth_pixel[0])) if point['pixel'] is not None and truth_visible[0] else None
            rows.append(dict(position=index, condition=condition, point=point, truth_visible=bool(truth_visible[0]), pixel_error=error))
    return dict(scope='Injected exact bearings and synthetic independent ground controls; not horn detection or fitted camera calibration',
                independent_ground_controls=controls, rows=rows)


def render(path, camera, cars, points, caption, hidden=()):
    """Render projected cuboids and acoustic points, without vehicle detection boxes or truth dots."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1200" viewBox="0 0 1920 1200">',
             '<rect width="1920" height="1200" fill="#18212f"/>',
             '<defs><clipPath id="frame"><rect x="0" y="0" width="1920" height="1080"/></clipPath></defs>',
             '<g clip-path="url(#frame)"><rect width="1920" height="1080" fill="#626c70"/>']
    for x in np.arange(-7, 7.1, 3.5):
        for y in np.arange(-32, 32, 5):
            parts.append(projected_line(camera, [[x, item, 0] for item in np.linspace(y, y + 3, 41)], '#e4e3cd', 3))
    for index, car in enumerate(cars):
        if car['identity'] in hidden:
            continue
        pixels, valid = camera.project(body_points(car))
        for face in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
            vertices = pixels[list(face)]
            if np.isfinite(vertices).all() and valid[list(face)].any():
                data = ' '.join('{:.2f},{:.2f}'.format(*point) for point in vertices)
                parts.append('<polygon points="{}" fill="{}" stroke="#253545" stroke-width="3"/>'.format(data, ('#c6d2dc', '#8caabd', '#c0beb4')[index]))
    for point in points:
        if point['state'] == 'projected':
            u, v = point['pixel']
            parts.append('<circle class="acoustic-point" cx="{}" cy="{}" r="10" fill="#ffad32"/>'.format(u, v))
            parts.append(text(max(0, u - 70), max(80, v - 17), 'source candidate ' + str(point['candidate'] + 1), 24, '#ffad32'))
    states = ', '.join(str(point['candidate'] + 1) + ':' + point['state'] for point in points) or 'no capture candidate'
    parts += ['</g><rect width="1920" height="62" fill="#15324b"/>',
              text(22, 41, 'DOWNWARD FISHEYE SIMULATION | synthetic scene; acoustic points only; no vision model', 25),
              text(22, 1118, caption, 23), text(22, 1162, states + ' | vehicle attribution unverified', 22), '</svg>']
    Path(path).write_text('\n'.join(parts), encoding='utf-8')


def positive_control(config, detector, output):
    """Use the known repeated positive fixture only as a separate wiring regression control."""
    if config['camera']['height'] != 6.0:
        return dict(skipped='Packaged fixture uses six-metre array height', images=[])
    control = deepcopy(config)
    control['camera']['yaw'] = 90.0
    source = ROOT / 'examples/horn-array14.wav'
    with wave.open(str(source), 'rb') as stream:
        pcm = np.frombuffer(stream.readframes(96256), dtype='<i2').reshape(-1, 14)
    event, _, events, triggered = analyze(control, detector, pcm, 9999)
    provenance = json.loads((ROOT / 'examples/provenance.json').read_text(encoding='utf-8'))
    cars = vehicles(provenance['case']['points'][0], 0)
    cars[1]['position'] = provenance['case']['points'][1]
    rig = make_rig(control)
    points = point_sources(rig, event['capture_candidates'])['wide-0']
    name = 'positive-control-points.svg'
    render(Path(output) / name, rig.cameras()[0], cars, points,
           'KNOWN REPEATED CONTROL | first candidate image | NOT an independent road test')
    return dict(source_sha256=digest(source), scope='Known repeated fixture, excluded from the 48 scenes',
                camera_yaw=90.0, triggered=triggered, event=event, events=events, points=points,
                horn_confirmed=any(row['horn_confirmed'] for row in events), images=[name])


def simulate(output, source_directory=None, geometry_only=False, height=6.0, hfov=144.0, vfov=77.6):
    verify_assets(ROOT)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new simulation output directory')
    config = configuration(height, hfov, vfov)
    sources, audio = [], {}
    if not geometry_only:
        if source_directory is None:
            raise ValueError('Provide --source-directory with the original PCM16 source clips; no downloads occur')
        records = json.loads((ROOT / 'evaluation/sources.json').read_text(encoding='utf-8'))
        chosen = {label: sorted(row['file'] for row in records if row['dataset'] == 'ESC50' and row['label'] == label)[:2] for label in (0, 1)}
        for name in chosen[0] + chosen[1]:
            audio[name], record = recording(Path(source_directory) / name, 96256)
            if record['sha256'] != next(row['sha256'] for row in records if row['file'] == name):
                raise ValueError('Original source checksum mismatch: ' + name)
            sources.append(record)
        builds, detector = build(), Detector(ROOT / 'model')
    output.mkdir(parents=True)
    (output / 'simulation_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(output / 'plan.json', dict(configuration=config, positions=POSITIONS, conditions=CONDITIONS,
                source_recordings=sources, direction_threshold=config['detector']['threshold'], capture_threshold=config['detector']['capture_threshold'],
                model_sha256=digest(ROOT / 'model/weights.npz'), simulation_sha256=digest(__file__),
                policy=dict(confirm_windows=config['event']['confirm'], direction_threshold=config['detector']['threshold']),
                selection='Cases and filenames fixed before inference, without outcome tuning'))
    write_json(output / 'configuration.json', config)
    geometry = geometry_checks(config)
    write_json(output / 'geometry.json', geometry)
    coverage_report = coverage(output / 'coverage.svg', config)
    rows, acoustic, images = [], [], []
    control = positive_control(config, detector, output) if not geometry_only else dict(images=[])
    images.extend(control['images'])
    true = make_rig(config).cameras()[0]
    if not geometry_only:
        for index, position in enumerate(POSITIONS):
            cars = vehicles(position, index)
            for source_index in (0, 1):
                for label in (1, 0):
                    identity = len(acoustic)
                    pcm, metadata = synthesize(config, audio[chosen[label][source_index]], audio[chosen[0][1-source_index]], cars, 20261103 + index * 7 + source_index)
                    diagnostic = {}
                    event, _, events, triggered = analyze(config, detector, pcm, identity, diagnostic)
                    truth = np.r_[position, 0.8 - height]
                    truth /= np.linalg.norm(truth)
                    angular_errors = [[float(np.rad2deg(np.arccos(np.clip(direction_vectors(bearing['azimuth'] or 0, bearing['depression']) @ truth, -1, 1))))
                                       for bearing in frame] for frame in diagnostic['candidates']]
                    acoustic.append(dict(id=identity, position=index, cars=cars, label=label,
                                          target=chosen[label][source_index], interferer=chosen[0][1-source_index],
                                          pcm=metadata, triggered=triggered, horn_candidate=event['horn_candidate'],
                                          horn_confirmed=any(row['horn_confirmed'] for row in events), events=events,
                                          legacy_confirmed_capture=any(row['state'] == 'CONFIRMED' and row['capture_candidate'] for row in events),
                                          candidate_3d_errors=angular_errors))
                    truth_pixel, truth_visible = true.project([[*position, 0.8]])
                    for condition in CONDITIONS:
                        points = point_sources(make_rig(config, condition), event['capture_candidates'])['wide-0'] if triggered else []
                        errors = [float(np.linalg.norm(point['pixel'] - truth_pixel[0])) for point in points
                                  if point['state'] == 'projected' and truth_visible[0] and label]
                        projected = sum(point['state'] == 'projected' for point in points)
                        row = dict(scene=identity, label=label, condition=condition, points=points,
                                   candidate_image_eligible=bool(triggered and any(point['state'] != 'outside_camera_view' for point in points)),
                                   projected_points=projected, target_visible=bool(truth_visible[0]),
                                   target_occluded_proxy=condition == 'occluded',
                                   closest_pixel_error=min(errors) if errors else None,
                                   scope='Synthetic available frame at exact representative time; image eligibility, not verified camera I/O')
                        rows.append(row)
                        if label and source_index == 0 and condition == 'nominal':
                            name = 'scene-{:02d}-points.svg'.format(index)
                            render(output / name, true, cars, points,
                                   '{} | state={} | horn_confirmed={} | score={:.3f}'.format(
                                       'FIRST CANDIDATE IMAGE' if triggered else 'NO ELIGIBLE CAPTURE',
                                       event['state'], event['horn_confirmed'], event['score']))
                            images.append(name)
                    print('scene', identity + 1, '/ 48 | horn', label, '| candidate image', triggered, flush=True)
    nominal = [row for row in rows if row['condition'] == 'nominal']
    errors = [row['closest_pixel_error'] for row in nominal if row['closest_pixel_error'] is not None]
    summaries = dict(positive_scenes=sum(row['label'] for row in acoustic),
                     negative_scenes=sum(not row['label'] for row in acoustic),
                     positive_candidate_scenes=sum(row['label'] and row['triggered'] for row in acoustic),
                     legacy_positive_confirmed_captures=sum(row['label'] and row['legacy_confirmed_capture'] for row in acoustic),
                     strict_positive_confirmed=sum(row['label'] and row['horn_confirmed'] for row in acoustic),
                     positive_image_eligible=sum(row['label'] and row['candidate_image_eligible'] for row in nominal),
                     negative_image_eligible=sum(not row['label'] and row['candidate_image_eligible'] for row in nominal),
                     positive_point_images=sum(row['label'] and row['projected_points'] > 0 for row in nominal),
                     conditional_pixel_error_count=len(errors),
                     conditional_pixel_error_median=float(np.median(errors)) if errors else None,
                     conditional_pixel_error_max=max(errors) if errors else None)
    report = dict(format=2, scope='Synthetic fisheye/cuboids, acoustic points only; real-recording PCM through current SRP/beam/model. No vehicle detector, camera I/O or fitted calibration.',
                  source_recordings=sources, model_sha256=digest(ROOT / 'model/weights.npz'),
                  thresholds=dict(direction=config['detector']['threshold'], capture=config['detector']['capture_threshold'], margin=config['detector']['margin'], confirm_windows=config['event']['confirm']), geometry=geometry, coverage=coverage_report,
                  acoustic=acoustic, pointing=rows, summaries=summaries, positive_control=control,
                  native_builds=builds if not geometry_only else {}, main_config_sha256=digest(ROOT / 'config/default.json'),
                  simulation_sha256=digest(__file__), default_configuration_changed=False, board_validated=False,
                  caveats='No vehicle attribution. Occlusion proxy hides geometry only, not sound; pixel_noise changes ground control observations only. Conditional pixel errors exclude absent/ambiguous/outside points and cannot represent end-to-end recall.')
    write_json(output / 'report.json', report)
    sections = ''.join('<figure><img src="{}" style="width:100%"><figcaption>{}</figcaption></figure>'.format(name, escape(name)) for name in images)
    html = '<!doctype html><html><meta charset="utf-8"><title>Downward acoustic points</title><body style="background:#18212f;color:#eee;font-family:Arial;max-width:1500px;margin:auto"><h1>朝下广角：声源候选点模拟</h1><p>仅橙点为真实声学推理的候选方向投影；车体为计算生成。不运行YOLO、不画车辆框、不使用真值替代声学点。当前单窗0.90确认声学鸣笛证据；候选点仍不代表确认某辆车。</p><p>假设相机缓存中有同步图像，图像保存资格与实机拍摄分开。正对照为已有重复录音，不计48场景。</p><img src="coverage.svg" style="max-width:850px;width:100%">' + sections + '</body></html>'
    (output / 'index.html').write_text(html, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-directory')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--height', type=float, default=6.0)
    parser.add_argument('--hfov', type=float, default=144.0)
    parser.add_argument('--vfov', type=float, default=77.6)
    args = parser.parse_args()
    result = simulate(args.output, args.source_directory, args.geometry_only, args.height, args.hfov, args.vfov)
    print(json.dumps(result['summaries']))


if __name__ == '__main__':
    main()
