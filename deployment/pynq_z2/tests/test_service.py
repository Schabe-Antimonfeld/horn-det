import importlib.util
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_complete_service_replays_silence_with_real_model(tmp_path):
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained weights are not available')
    specification = importlib.util.spec_from_file_location('board_service', ROOT / 'run.py')
    service = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(service)
    audio = tmp_path / 'array.wav'
    with wave.open(str(audio), 'wb') as output:
        output.setnchannels(14)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(np.zeros((72192, 14), dtype='<i2').tobytes())
    output = tmp_path / 'run'
    assert service.main(['--model', str(model), '--replay', str(audio), '--output', str(output)]) == 0
    records = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert len(records) == 2 and all(row['state'] == 'IDLE' for row in records)
    assert json.loads((output / 'health.json').read_text())['status'] == 'completed'
    with pytest.raises(ValueError, match='empty'):
        service.main(['--model', str(model), '--replay', str(audio), '--output', str(output)])


def test_cli_model_path_is_relative_to_working_directory(monkeypatch):
    model = ROOT.parents[1] / 'prototype/outputs/horn_detector_v1/fpga_model'
    if not model.is_dir():
        pytest.skip('Local trained weights are not available')
    specification = importlib.util.spec_from_file_location('board_check', ROOT / 'run.py')
    service = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(service)
    monkeypatch.chdir(ROOT.parents[1])
    assert service.main(['--model', 'prototype/outputs/horn_detector_v1/fpga_model', '--check']) == 0


def test_replay_rejects_fpga_srp_before_creating_output(tmp_path):
    from ps.configuration import load_config
    specification = importlib.util.spec_from_file_location('board_srp_service', ROOT / 'run.py')
    service = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(service)
    config = load_config()
    config['search']['backend'] = 'fpga'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='NumPy SRP'):
        service.main(['--config', str(path), '--replay', 'absent.wav', '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()
