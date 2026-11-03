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
    if (search.get('algorithm') != 'cached' or search.get('implementation') != 'native'
            or search.get('pair_mode') != 'all' or search.get('fft_fold') != 4
            or search.get('native_precision') != 'single' or search.get('backend') != 'numpy'):
        raise ValueError('Latest SRP requires complete-grid native single PHAT with all pairs and fold four')
    if audio['window_samples'] != 48128 or audio['update_samples'] != 24064:
        raise ValueError('Latest temporal streaming requires 48128/24064 samples')
    if math.ceil(audio['diameter'] * audio['sample_rate'] / audio['speed']) + 1 >= 128:
        raise ValueError('Folded GCC cannot cover physical array delays')
    streaming, refinement = config.get('streaming', {}), config.get('refinement', {})
    if (not isinstance(streaming, dict) or set(streaming) - {'enabled', 'native_library'}
            or streaming.get('enabled') is not True):
        raise ValueError('Latest runtime requires continuous-window processing')
    if (not isinstance(refinement, dict) or set(refinement) - {'fused_input', 'independent_beams', 'native_library'}
            or refinement.get('fused_input') is not True or refinement.get('independent_beams') is not True):
        raise ValueError('Latest runtime requires fused input and independent beam reuse')
    for section in (search, streaming, refinement):
        if 'native_library' in section and (not isinstance(section['native_library'], str)
                                          or not Path(section['native_library']).is_absolute()):
            raise ValueError('Native library overrides must be absolute paths')
    from ps.conditioning import validate_options as validate_conditioning
    validate_conditioning(config.get('conditioning', {}))
    detector = config['detector']
    if detector['backend'] != 'spectral' or not all(0 <= detector[key] <= 1 for key in ('threshold', 'margin')):
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
