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


@pytest.mark.parametrize('section,key,value', [('audio', 'channels', 8), ('audio', 'window_samples', 48000),
                                               ('search', 'azimuth_step', 7), ('camera', 'height', 0),
                                               ('association', 'method', 'azimuth'),
                                               ('audio', 'gains', [1] * 13),
                                               ('camera', 'sources', [{'id': 'unknown', 'uri': 'http://camera'}]),
                                               ('hardware', 'timeout', -1)])
def test_contract_rejects_incompatible_settings(section, key, value):
    config = copy.deepcopy(load_config())
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
