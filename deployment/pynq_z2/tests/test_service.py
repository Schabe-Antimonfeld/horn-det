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
