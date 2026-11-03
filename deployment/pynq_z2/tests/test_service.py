import json
from pathlib import Path

import numpy as np
import pytest

from ps.configuration import load_config
from ps.evaluation import compare, evaluate
import run

ROOT = Path(__file__).resolve().parents[1]


def test_default_service_checks_and_replays_the_frozen_positive(tmp_path, capsys):
    assert run.main(['--check']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['model_tag'] == 405187002 and value['direction_threshold'] == 0.9
    assert value['capture_threshold'] == 0.9 and value['refinement']['independent_beams']
    output = tmp_path / 'events'
    args = ['--replay', str(ROOT / 'examples/horn-array14.wav'), '--output', str(output)]
    assert run.main(args) == 0
    actual = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    expected = json.loads((ROOT / 'examples/expected.json').read_text())['horn-array14.wav']
    current = [dict(row, state='CONFIRMED', direction_threshold=0.9, horn_confirmed=True) for row in expected]
    compare(current, actual)
    assert [row['state'] for row in actual] == ['CONFIRMED', 'CONFIRMED', 'CONFIRMED', 'CONFIRMED']
    assert actual[1]['srp_diagnostics']['fft_frames_reused'] == 46
    with pytest.raises(ValueError, match='empty'):
        run.main(args)


def test_wrong_model_tag_fails_before_output_creation(tmp_path):
    config = load_config()
    config['detector']['model_tag'] += 1
    path = tmp_path / 'wrong.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='model tag'):
        run.main(['--config', str(path), '--output', str(tmp_path / 'absent')])
    assert not (tmp_path / 'absent').exists()


def test_all_frozen_protocol_metrics_are_recomputed_from_features():
    report = evaluate(ROOT)
    assert report['result'] == 'passed' and report['scenes'] == 672
    assert report['protocols']['test-original']['historical_joint']['f1'] == pytest.approx(2/3)
    assert report['protocols']['test-visible']['historical_joint']['f1'] == pytest.approx(0.6307692307692307)
    assert report['protocols']['test-original']['joint']['f1'] == pytest.approx(0.7412587412587412)
    assert report['protocols']['test-visible']['joint']['f1'] == pytest.approx(0.7310344827586208)
    assert report['operating_point'] == dict(threshold=0.9, confirm=1)
    assert report['maximum_score_error'] < 2e-10
