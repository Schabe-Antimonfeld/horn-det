"""Probe USB modes, collect checkerboards and fit/check camera calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps.configuration import load_config
from ps.geometry import Lens, Rig
from ps.vision import CameraCache


def require_opencv():
    try:
        import cv2
    except ImportError:
        raise RuntimeError('OpenCV is required for USB capture/calibration; no dependencies were installed')
    return cv2


def capture(args):
    cv2 = require_opencv()
    config = load_config(args.config)
    source = config['camera']['sources']
    if len(source) != 1 or source[0].get('backend') != 'usb':
        raise ValueError('Camera tool requires one USB source')
    if args.count < 2 or not np.isfinite([args.period, args.timeout]).all() or args.period < 0 or args.timeout <= 0:
        raise ValueError('Capture needs at least two frames, a nonnegative period and positive timeout')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cache = CameraCache(source, SimpleNamespace(cv2=cv2), timeout=args.timeout)
    records, previous, beginning = [], None, time.monotonic()
    status = 'starting'
    try:
        cache.start()
        deadline = time.monotonic() + args.timeout
        while len(records) < args.count:
            frame = cache.nearest(source[0]['id'], time.monotonic())
            if frame is not None and frame['sequence'] != previous and (not records or frame['time'] - records[-1]['time'] >= args.period):
                name = 'frame-{:04d}.jpg'.format(len(records))
                if args.command == 'collect':
                    (output / name).write_bytes(frame['jpeg'])
                records.append({key: frame[key] for key in ('time', 'sequence', 'mode', 'timestamp_kind', 'timing_verified', 'jpeg_reencoded')})
                records[-1]['file'] = name if args.command == 'collect' else None
                previous = frame['sequence']
                deadline = time.monotonic() + args.timeout + args.period
            if time.monotonic() > deadline:
                raise TimeoutError('USB did not provide another usable frame: ' + str(cache.health()))
            time.sleep(0.005)
        status = 'completed'
    finally:
        try:
            cache.close()
        finally:
            span = records[-1]['time'] - records[0]['time'] if len(records) > 1 else 0
            report = {'status': status, 'frames': records, 'health': cache.health(),
                      'host_sequence_rate': (records[-1]['sequence'] - records[0]['sequence']) / span if span > 0 else None,
                      'wall_seconds': time.monotonic() - beginning, 'board_validated': False,
                      'evidence': 'Host USB measurements; no verified exposure synchronization or road detection claim'}
            (output / 'capture.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'frames': len(records), 'host_sequence_rate': report['host_sequence_rate']}))


def calibration_quality(points, width, height):
    points = np.vstack(points).reshape(-1, 2)
    cells = np.floor(points / [width, height] * [4, 3]).astype(int)
    cells = np.clip(cells, 0, [3, 2])
    occupied = len({tuple(value) for value in cells})
    return {'occupied_cells': occupied, 'grid_cells': 12,
            'image_bounds': [points.min(axis=0).tolist(), points.max(axis=0).tolist()]}


def calibrate(args):
    cv2 = require_opencv()
    if min(args.columns, args.rows) < 3 or not np.isfinite([args.square, args.maximum_rms, args.maximum_angle]).all() or args.square <= 0 or args.maximum_rms <= 0:
        raise ValueError('Invalid checkerboard or error limit')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Calibration output already exists')
    object_points = np.zeros((args.rows * args.columns, 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square
    points, paths, checksums, rejected, size = [], [], set(), [], None
    for path in sorted(Path(args.images).glob('*.jpg')):
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum in checksums:
            rejected.append({'file': path.name, 'reason': 'duplicate_image'})
            continue
        checksums.add(checksum)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append({'file': path.name, 'reason': 'decode_failed'})
            continue
        current = (image.shape[1], image.shape[0])
        if size is not None and size != current:
            raise ValueError('Calibration images must use one unchanged sensor mode')
        size = current
        found, corners = cv2.findChessboardCorners(image, (args.columns, args.rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            rejected.append({'file': path.name, 'reason': 'checkerboard_not_found'})
            continue
        corners = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5))
        points.append(corners.astype(np.float64))
        paths.append({'file': path.name, 'sha256': checksum})
    if len(points) < 15:
        raise ValueError('Need at least 15 distinct usable checkerboard images, including tilted and edge views')
    quality = calibration_quality(points, *size)
    if quality['occupied_cells'] < 10:
        raise ValueError('Checkerboards cover fewer than 10/12 image cells; collect more edge/corner views')
    validation = list(range(0, len(points), 5))
    training = [index for index in range(len(points)) if index not in validation]
    training_quality = calibration_quality([points[index] for index in training], *size)
    if training_quality['occupied_cells'] < 10:
        raise ValueError('Training views alone must cover at least 10/12 image cells')
    quality['training_coverage'] = training_quality
    if args.model == 'fisheye':
        rms, matrix, coefficients, _, _ = cv2.fisheye.calibrate(
            [object_points.reshape(1, -1, 3) for index in training],
            [points[index].reshape(1, -1, 2) for index in training], size, np.eye(3), np.zeros((4, 1)),
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW)
    else:
        rms, matrix, coefficients, _, _ = cv2.calibrateCamera(
            [object_points.astype(np.float32) for index in training],
            [points[index].astype(np.float32) for index in training], size, None, None)
    errors = []
    for index in validation:
        observed = points[index]
        if args.model == 'fisheye':
            normalized = cv2.fisheye.undistortPoints(observed, matrix, coefficients)
            success, rotation, translation = cv2.solvePnP(object_points, normalized, np.eye(3), None)
            if not success:
                raise RuntimeError('Holdout pose estimation failed')
            predicted, _ = cv2.fisheye.projectPoints(object_points.reshape(1, -1, 3), rotation, translation, matrix, coefficients)
        else:
            success, rotation, translation = cv2.solvePnP(object_points, observed, matrix, coefficients)
            if not success:
                raise RuntimeError('Holdout pose estimation failed')
            predicted, _ = cv2.projectPoints(object_points, rotation, translation, matrix, coefficients)
        errors.extend(np.linalg.norm(predicted.reshape(-1, 2) - observed.reshape(-1, 2), axis=1).tolist())
    holdout_rms = float(np.sqrt(np.mean(np.square(errors))))
    if not np.isfinite([rms, holdout_rms]).all() or max(rms, holdout_rms) > args.maximum_rms:
        raise ValueError('Calibration exceeds the requested training/holdout RMS limit')
    record = {'format': 1, 'model': args.model, 'matrix': matrix.tolist(),
              'coefficients': coefficients.reshape(-1).tolist(), 'width': size[0], 'height': size[1],
              'maximum_angle': args.maximum_angle, 'quality': dict(quality, training_rms=float(rms), holdout_rms=holdout_rms),
              'training_images': [paths[index] for index in training], 'holdout_images': [paths[index] for index in validation],
              'rejected_images': rejected, 'extrinsics_verified': False,
              'evidence': 'Intrinsic fit only; maximum_angle and installation pose require independent ground checks'}
    Lens.from_record(record)
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'quality': record['quality']}))


def check_ground(args):
    config = load_config(args.config)
    settings = dict(config['camera'])
    settings.pop('sources')
    if settings.get('layout') != 'single':
        raise ValueError('Ground tool requires the single camera profile')
    if settings.get('calibration') is None:
        raise ValueError('Ground check requires measured lens calibration')
    camera = Rig(**settings).cameras()[0]
    record = json.loads(Path(args.points).read_text(encoding='utf-8'))
    world, pixels = np.asarray(record['world'], dtype=float), np.asarray(record['pixels'], dtype=float)
    if world.shape != (len(world), 3) or pixels.shape != (len(world), 2) or len(world) < 4 or not np.isfinite(pixels).all() or np.any(world[:, 2] != 0):
        raise ValueError('Ground check needs four or more matched original pixels and z=0 world points')
    predicted, visible = camera.project(world)
    recovered, valid = camera.ground(pixels)
    if not visible.all() or not valid.all():
        raise ValueError('Ground controls are outside the calibrated camera domain')
    pixels_rms = float(np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1))))
    metres_rms = float(np.sqrt(np.mean(np.sum((recovered - world) ** 2, axis=1))))
    if not np.isfinite([args.maximum_pixels, args.maximum_metres]).all() or min(args.maximum_pixels, args.maximum_metres) <= 0:
        raise ValueError('Ground error limits must be finite and positive')
    result = {'pixels_rms': pixels_rms, 'metres_rms': metres_rms,
              'passed': pixels_rms <= args.maximum_pixels and metres_rms <= args.maximum_metres,
              'evidence': 'Checks supplied lens/pose against supplied controls; does not verify exposure synchronization'}
    print(json.dumps(result))
    if not result['passed']:
        raise ValueError('Ground control errors exceed configured limits')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    for name in ('probe', 'collect'):
        command = commands.add_parser(name)
        command.add_argument('--config')
        command.add_argument('--output', required=True)
        command.add_argument('--count', type=int, default=30)
        command.add_argument('--period', type=float, default=0 if name == 'probe' else 1.0)
        command.add_argument('--timeout', type=float, default=10)
        command.set_defaults(action=capture)
    command = commands.add_parser('calibrate')
    command.add_argument('--images', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--model', choices=('fisheye', 'brown'), required=True)
    command.add_argument('--columns', type=int, default=9)
    command.add_argument('--rows', type=int, default=6)
    command.add_argument('--square', type=float, default=0.025)
    command.add_argument('--maximum-angle', type=float, default=80)
    command.add_argument('--maximum-rms', type=float, default=1.5)
    command.set_defaults(action=calibrate)
    command = commands.add_parser('check-ground')
    command.add_argument('--config')
    command.add_argument('--points', required=True)
    command.add_argument('--maximum-pixels', type=float, default=3)
    command.add_argument('--maximum-metres', type=float, default=0.25)
    command.set_defaults(action=check_ground)
    args = parser.parse_args(argv)
    if not hasattr(args, 'action'):
        parser.error('Choose probe, collect, calibrate or check-ground')
    return args.action(args)


if __name__ == '__main__':
    main()
