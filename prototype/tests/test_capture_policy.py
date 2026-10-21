import importlib.util
from pathlib import Path

import numpy as np
import pytest


def experiment():
    path = Path(__file__).resolve().parents[1] / 'improve_capture.py'
    spec = importlib.util.spec_from_file_location('capture_policy_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene(label, angles=(0, 90), domain='HornBase', split='validation'):
    return dict(label=label, angle=0, dataset=domain, split=split,
                bearings=[dict(azimuth=angle) for angle in angles])


def test_capture_counts_presence_and_keeps_second_direction_without_claiming_final_doa():
    module = experiment()
    cases = [scene(1, (90, 0)), scene(1), scene(0), scene(0)]
    values = [[0.7, 0.6], [0.1, 0.2], [0.6, 0.6], [0.1, 0.1]]
    metric = module.capture_metrics(cases, values, 0.6)
    assert metric['captured_positive'] == 1 and metric['captured_negative'] == 1
    assert metric['capture_recall'] == 0.5 and metric['negative_capture_rate'] == 0.5
    assert metric['retained_correct_direction'] == 1
    assert module.band.evaluation.outcomes(cases, values, 0.65, 0)['tp'] == 0
    with pytest.raises(ValueError, match='finite candidate'):
        module.capture_metrics(cases, np.full((4, 2), np.nan), 0.6)
    with pytest.raises(ValueError, match='threshold'):
        module.capture_metrics(cases, values, -0.1)


def test_policy_selection_uses_worst_source_budget_and_blocks_test_labels():
    module = experiment()
    cases, values = [], []
    for domain in ('HornBase', 'ESC50'):
        cases.extend([scene(1, domain=domain), scene(1, domain=domain)])
        values.extend([[0.8, 0.1], [0.6, 0.1]])
        for index in range(10):
            cases.append(scene(0, domain=domain))
            values.append([0.7 if index < (3 if domain == 'HornBase' else 1) else 0.1, 0.05])
    selected = module.choose_policies(cases, values, 0.9)
    conservative = selected['policies']['budget-10']['selected']
    permissive = selected['policies']['budget-30']['selected']
    assert conservative['worst_domain_recall'] == 0.5
    assert permissive['worst_domain_recall'] == 1
    assert permissive['worst_domain_acceptance'] == 0.3
    assert all(item['threshold'] <= 0.9 for item in selected['curve'])
    assert selected['recall_targets']['90']['selected']['worst_domain_acceptance'] == 0.3
    with pytest.raises(ValueError, match='validation sources'):
        module.choose_policies([dict(case, split='test') for case in cases], values, 0.9)
    with pytest.raises(ValueError, match='both labels'):
        module.choose_policies([scene(1)], [[0.9, 0.1]], 0.9)


def test_capture_budget_reports_infeasible_instead_of_breaking_strict_gate():
    module = experiment()
    cases = [scene(label, domain=domain) for domain in ('HornBase', 'ESC50') for label in (1, 0)]
    selected = module.choose_policies(cases, [[0.8, 0.1]] * 4, 0.7)
    assert not selected['policies']['budget-30']['feasible']
    assert selected['policies']['budget-30']['selected'] is None


def test_nominal_photo_opportunities_are_separate_from_audible_captures():
    module = experiment()
    cases = [dict(scene(1), visible=False), dict(scene(1), visible=True), dict(scene(0), visible=True)]
    requested = np.array([[False, True], [False, True], [True, True]])
    target = np.array([[False, False], [False, True], [True, True]])
    metric = module.photo_metrics(cases, [[0.8, 0.2], [0.8, 0.6], [0.7, 0.1]], 0.5, (requested, target))
    assert metric['positive_photo_opportunities'] == 1
    assert metric['negative_photo_opportunities'] == 1
    assert metric['visible_positive'] == metric['nominal_target_in_requested_camera'] == 1


def test_preparing_held_out_capture_scores_requires_gate_freeze(monkeypatch, tmp_path):
    module = experiment()
    monkeypatch.setattr(module, 'OUTPUT', tmp_path)
    monkeypatch.setattr(module, 'check_plan', lambda: {})
    with pytest.raises(ValueError, match='Freeze capture gates'):
        module.prepare('test')
    with pytest.raises(ValueError, match='Freeze capture gates'):
        module.prepare('stress')
