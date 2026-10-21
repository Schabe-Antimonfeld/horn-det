import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps.configuration import load_config, validate_config


def test_default_contract():
    config = load_config()
    assert config['audio']['window_samples'] == 94 * 512
    assert config['audio']['update_samples'] == 47 * 512
    assert config['association']['method'] == 'spatial'
    assert config['camera']['layout'] == 'single' and config['camera']['count'] == 1
    assert config['camera']['calibration'] is None


@pytest.mark.parametrize('section,key,value', [('audio', 'channels', 8), ('audio', 'window_samples', 48000),
                                               ('search', 'azimuth_step', 7), ('camera', 'height', 0),
                                               ('search', 'backend', 'unknown'),
                                               ('association', 'method', 'azimuth'),
                                               ('audio', 'gains', [1] * 13),
                                               ('camera', 'sources', [{'id': 'unknown', 'uri': 'http://camera'}]),
                                               ('hardware', 'timeout', -1)])
def test_contract_rejects_incompatible_settings(section, key, value):
    config = copy.deepcopy(load_config())
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('capture', [-0.01, 0.66, float('nan'), float('inf'), True, None, '0.5'])
def test_capture_threshold_rejects_invalid_or_stricter_gates(capture):
    config = load_config()
    config['detector']['capture_threshold'] = capture
    with pytest.raises(ValueError, match='Capture threshold'):
        validate_config(config)


@pytest.mark.parametrize('capture', [0, 0.5, 0.65])
def test_optional_capture_threshold_leaves_direction_gate_unchanged(capture):
    config = load_config()
    assert 'capture_threshold' not in config['detector']
    original = dict(config['detector'])
    config['detector']['capture_threshold'] = capture
    validate_config(config)
    assert config['detector'] == dict(original, capture_threshold=capture)
