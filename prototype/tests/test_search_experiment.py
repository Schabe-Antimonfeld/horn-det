from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def experiment():
    path = Path(__file__).resolve().parents[1] / 'optimize_srp.py'
    spec = importlib.util.spec_from_file_location('srp_optimization_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_quality_guards_cannot_hide_recall_or_false_capture_losses():
    module = experiment()
    baseline = dict(joint=dict(candidate_hits=83, tp=60, f1=0.7),
                    capture=dict(detection_recall=0.8, accepted_negative=3), visual_target_retained=30)
    assert module.quality_guard(deepcopy(baseline), baseline) == []
    candidate = deepcopy(baseline)
    candidate['joint'].update(candidate_hits=81, tp=57, f1=0.67)
    candidate['capture'].update(detection_recall=0.77, accepted_negative=6)
    candidate['visual_target_retained'] = 27
    assert set(module.quality_guard(candidate, baseline)) == {
        'candidate_recall', 'direction_recall', 'joint_f1', 'capture_recall', 'negative_capture', 'visual_target_retention'}


def test_search_summary_counts_ambiguous_negative_presence_and_paired_times():
    module = experiment()
    bearings = [dict(azimuth=0, depression=30, response=1), dict(azimuth=90, depression=30, response=1)]
    cases = [dict(label=label, angle=0) for label in (1, 0)]
    rows = [dict(bearings=bearings, scores=values, visual_target_retained=label,
                 timing=dict(gcc_seconds=0.01, search_seconds=0.005, srp_seconds=0.015,
                             evaluated_directions=500, direction_pair_evaluations=17500,
                             full_fallback=False, regional_bound_prunes=2))
            for label, values in ((True, [0.99, 0.1]), (False, [0.95, 0.95]))]
    baseline = deepcopy(rows)
    for row in baseline:
        row['timing'].update(gcc_seconds=0.1, search_seconds=0.05, srp_seconds=0.15)
    summary = module.summarize(cases, rows, baseline)
    assert summary['joint']['tp'] == 1 and summary['joint']['accepted_negative'] == 1
    assert summary['joint']['ambiguous_negative'] == 1
    assert summary['visual_target_retained'] == 1
    assert summary['relative_time']['srp_seconds']['median'] == pytest.approx(0.1)
    assert summary['bounded_prunes'] == 4


def test_search_experiment_blocks_test_before_freeze_and_selection_from_test(monkeypatch, tmp_path):
    module = experiment()
    monkeypatch.setattr(module, 'OUTPUT', tmp_path)
    monkeypatch.setattr(module, 'check_plan', lambda: {})
    with pytest.raises(ValueError, match='Freeze search choice'):
        module.prepare('test')
    monkeypatch.setattr(module, 'load_result', lambda key: ([dict(split='test')], {}))
    with pytest.raises(ValueError, match='validation sources'):
        module.validation_summary()


def test_producer_snapshot_does_not_allow_runtime_kernel_changes(monkeypatch, tmp_path):
    module = experiment()
    prototype = tmp_path / 'prototype'
    prototype.mkdir()
    output = tmp_path / 'output'
    (output / 'source-snapshot').mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'ps').mkdir()
    driver = tmp_path / 'scripts/build_search.py'
    kernel = tmp_path / 'ps/search_kernel.c'
    driver.write_text('original producer')
    kernel.write_text('unchanged runtime')
    checksums = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (driver, kernel)}
    (output / 'source-snapshot/build_search.py').write_bytes(driver.read_bytes())
    (output / 'plan.json').write_text(json.dumps(dict(settings=module.SETTINGS, frozen=checksums)))
    monkeypatch.setattr(module, 'PROTO', prototype)
    monkeypatch.setattr(module, 'OUTPUT', output)
    driver.write_text('self-check packaging fix; runtime unchanged')
    module.check_plan()
    kernel.write_text('changed runtime')
    with pytest.raises(ValueError, match='Frozen input changed'):
        module.check_plan()
