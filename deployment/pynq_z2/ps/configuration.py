"""Validate the board contract without importing PYNQ, Torch or the PC prototype."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(str(path or ROOT / 'config/default.json'), encoding='utf-8') as stream:
        config = json.load(stream)
    calibration = config['camera'].get('calibration')
    if isinstance(calibration, str):
        location = Path(calibration)
        if not location.is_absolute():
            location = ROOT / location
        with location.open(encoding='utf-8') as stream:
            config['camera']['calibration'] = json.load(stream)
    validate_config(config)
    return config


def validate_config(config):
    if config.get('format') != 1:
        raise ValueError('Unsupported deployment configuration')
    audio = config['audio']
    expected = {'channels': 14, 'sample_rate': 48000, 'packet_samples': 512, 'frame': 1024, 'hop': 512}
    if any(audio.get(key) != value for key, value in expected.items()):
        raise ValueError('Configuration must match the fixed capture and trained frontend')
    if audio['window_samples'] < 48000 or audio['update_samples'] <= 0 or any(
            audio[key] % 512 for key in ('window_samples', 'update_samples')):
        raise ValueError('Windows must contain one second and align with capture packets')
    if not 0 < audio['diameter'] <= 1 or audio['speed'] <= 0:
        raise ValueError('Invalid array geometry')
    from ps.beamforming import validate_options
    validate_options(config.get('beamforming', {}))
    search = config['search']
    if search.get('backend', 'numpy') not in ('numpy', 'fpga'):
        raise ValueError('Unsupported SRP backend')
    for span, key in ((360, 'azimuth_step'), (90, 'depression_step')):
        step = search[key]
        if not math.isfinite(step) or step <= 0 or abs(span / step - round(span / step)) > 1e-9:
            raise ValueError('Search steps must divide their spans')
    if any(type(search[key]) is not int or search[key] <= 0 for key in ('candidates', 'block_directions')):
        raise ValueError('Search sizes must be positive integers')
    if not 0 < search['separation'] <= 180:
        raise ValueError('Invalid candidate separation')
    algorithm = search.get('algorithm', 'reference')
    implementation = search.get('implementation', 'numpy')
    if algorithm not in ('reference', 'cached', 'hierarchical') or implementation not in ('numpy', 'native'):
        raise ValueError('Unsupported SRP algorithm or implementation')
    if search.get('pair_mode', 'all') not in ('all', 'mixed35'):
        raise ValueError('Unsupported microphone pair mode')
    fold = search.get('fft_fold', 1)
    if type(fold) is not int or fold not in (1, 2, 4, 8):
        raise ValueError('Invalid folded FFT factor')
    if algorithm == 'reference' and (implementation != 'numpy' or search.get('pair_mode', 'all') != 'all' or fold != 1):
        raise ValueError('Reference SRP does not apply cached/native/pair/fold options')
    if algorithm != 'reference' and search.get('backend', 'numpy') != 'numpy':
        raise ValueError('Optimized search requires PS execution; the existing FPGA protocol has 91 fixed pairs')
    if math.ceil(audio['diameter'] * audio['sample_rate'] / audio['speed']) + 1 >= 1024 // fold // 2:
        raise ValueError('Folded GCC cannot cover the physical array delays')
    if search.get('native_precision', 'double') not in ('single', 'double'):
        raise ValueError('Invalid native PHAT precision')
    if 'native_library' in search and (implementation != 'native' or not isinstance(search['native_library'], str)
                                      or not Path(search['native_library']).is_absolute()):
        raise ValueError('A native library override must be an absolute path for the native implementation')
    if algorithm == 'hierarchical':
        for key, fine, span, maximum in (('coarse_azimuth', 'azimuth_step', 360, 60),
                                         ('coarse_depression', 'depression_step', 90, 45)):
            step = search.get(key, 6 if key == 'coarse_azimuth' else 15)
            if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or not 0 < step <= maximum or abs(span / step - round(span / step)) > 1e-9 or step < search[fine] or abs(step / search[fine] - round(step / search[fine])) > 1e-9:
                raise ValueError('Coarse SRP steps must cover the grid and be multiples of fine steps')
        regions = search.get('regions', 6)
        if type(regions) is not int or not search['candidates'] <= regions <= 64 or type(search.get('bounds', True)) is not bool:
            raise ValueError('Invalid region count or regional bound switch')
    streaming = config.get('streaming', {})
    if not isinstance(streaming, dict) or set(streaming) - {'enabled', 'conditioning', 'native_library'}:
        raise ValueError('Unsupported streaming settings')
    if type(streaming.get('enabled', False)) is not bool or streaming.get('conditioning', 'numpy') not in ('numpy', 'native'):
        raise ValueError('Invalid streaming mode or conditioning')
    if 'native_library' in streaming and (not isinstance(streaming['native_library'], str)
                                         or not Path(streaming['native_library']).is_absolute()):
        raise ValueError('Streaming library override must be absolute')
    if streaming.get('enabled') and (algorithm not in ('cached', 'hierarchical') or search.get('backend', 'numpy') != 'numpy'
            or config.get('beamforming', {}).get('implementation') != 'fft'
            or audio['window_samples'] != 2 * audio['update_samples']):
        raise ValueError('Streaming requires PS search, FFT beams and fifty-percent aligned overlap')
    if not streaming.get('enabled', False) and (streaming.get('conditioning', 'numpy') != 'numpy' or 'native_library' in streaming):
        raise ValueError('Native streaming options require enabled streaming')
    if 'refinement' in config:
        from ps.refinement import validate_options as validate_refinement
        validate_refinement(config['refinement'], streaming)
    if 'conditioning' in config:
        from ps.conditioning import validate_options as validate_conditioning
        validate_conditioning(config['conditioning'])
    detector = config['detector']
    if detector['backend'] not in ('fpga', 'integer', 'spectral') or not all(0 <= detector[key] <= 1 for key in ('threshold', 'margin')):
        raise ValueError('Unsupported detector backend or gates')
    capture = detector.get('capture_threshold', detector['threshold'])
    if isinstance(capture, bool) or not isinstance(capture, (int, float)) or not math.isfinite(capture) or not 0 <= capture <= detector['threshold']:
        raise ValueError('Capture threshold must be finite and between zero and the direction threshold')
    if any(type(config['event'][key]) is not int or config['event'][key] <= 0 for key in ('confirm', 'release')):
        raise ValueError('Event counters must be positive integers')
    camera = config['camera']
    if type(camera['count']) is not int or camera['count'] < 1 or not 0 <= camera['horn_height'] < camera['height']:
        raise ValueError('Invalid camera count or height')
    if not 0 <= camera['pitch'] <= 90 or not 0 <= camera['minimum_range'] < camera['maximum_range']:
        raise ValueError('Invalid camera pitch or range')
    for key in ('hfov', 'vfov', 'down_hfov', 'down_vfov'):
        if not 0 < camera[key] < 179:
            raise ValueError('Invalid field of view')
    if config['association']['method'] != 'spatial':
        raise ValueError('Deployment requires spatial association; no azimuth-only fallback')
    if config['vision']['enabled'] and not config['camera']['sources']:
        raise ValueError('Vision requires configured cameras')
    identities = [source['id'] for source in camera['sources']]
    allowed = {'wide-0'} if camera.get('layout') == 'single' else {'ring-' + str(index) for index in range(camera['count'])} | {'down'}
    if len(set(identities)) != len(identities) or set(identities) - allowed:
        raise ValueError('Camera sources must reference distinct configured camera IDs')
    from ps.vision import validate_source
    for source in camera['sources']:
        validate_source(source)
        if camera.get('layout') == 'single' and source.get('backend') == 'usb' and (source['width'], source['height']) != (camera['width'], camera['image_height']):
            raise ValueError('USB mode must match the camera geometry and calibration')
    if config['vision']['enabled'] and camera.get('layout') == 'single' and camera.get('calibration') is None:
        raise ValueError('Wide-angle vision requires measured lens calibration')
    if config['vision']['enabled'] and isinstance(camera.get('calibration'), dict) and camera['calibration'].get('simulation_only'):
        raise ValueError('Simulation-only lens parameters cannot enable live vision')
    if config['hardware']['timeout'] <= 0:
        raise ValueError('Hardware timeout must be positive')
    cache = config.get('camera_cache', {'frames': 4, 'period': 0.5, 'timeout': 2.0})
    if type(cache['frames']) is not int or not 2 <= cache['frames'] <= 128 or any(
            not math.isfinite(cache[key]) or cache[key] <= 0 for key in ('period', 'timeout')):
        raise ValueError('Camera cache needs 2..128 frames and finite positive periods/timeouts')
    for key in ('gains', 'delays'):
        values = audio.get(key)
        if values is not None and (len(values) != 14 or not all(math.isfinite(value) for value in values)
                                   or (key == 'gains' and min(values) <= 0)):
            raise ValueError('Calibration needs 14 finite gains/delays and positive gains')
    if config['vision']['output'] != 'yolo_xywh_classes':
        raise ValueError('Only explicitly described YOLO decoded outputs are supported')
    if config['vision']['input_size'] <= 0 or not all(0 <= config['vision'][key] <= 1 for key in ('threshold', 'nms')):
        raise ValueError('Invalid vision dimensions or gates')
    from ps.geometry import Association, Rig
    geometry = dict(camera)
    geometry.pop('sources')
    Rig(**geometry)
    Association(**config['association'])
    for section in config.values():
        if isinstance(section, dict):
            for value in section.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('Configuration must be finite')
