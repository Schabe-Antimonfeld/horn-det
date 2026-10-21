import numpy as np
import pytest
import importlib.util
from pathlib import Path

from horn.recovery import choose_gate, pcm_roundtrip, visible_target, camera_rig, scene_plan
from horn.surround import wrap_angle
from horn.recovery import joint_metrics


def test_common_pcm_gain_preserves_channels_and_has_bounded_quantization_error():
    signal = np.column_stack((np.linspace(-4, 4, 500), np.linspace(-2, 2, 500)))
    interference = signal[:, ::-1] * 0.3
    noise = np.ones_like(signal) * 0.01
    actual, metadata = pcm_roundtrip(signal, interference, noise, 0.15)
    expected = (signal + interference + noise) * metadata['gain']
    assert np.max(np.abs(actual)) <= 0.9001 and metadata['clipped_samples'] == 0
    np.testing.assert_allclose(actual, expected, atol=0.5 / 32768, rtol=0)
    assert metadata['target_rms_actual'] <= 0.15


def test_headroom_limits_peak_without_silent_clipping():
    signal = np.zeros((100, 14))
    signal[30] = 100
    actual, metadata = pcm_roundtrip(signal, signal, signal, 0.9)
    assert metadata['headroom_limited'] and np.max(actual) <= 0.9001
    with pytest.raises(ValueError):
        pcm_roundtrip(signal * np.nan, signal, signal, 0.1)


def test_visibility_uses_geometry_and_excludes_rear_ground_targets():
    rig = camera_rig()
    assert visible_target(rig, 0, 15)
    assert not visible_target(rig, 180, 15)


def test_gate_selection_rejects_test_labels():
    with pytest.raises(ValueError, match='validation'):
        choose_gate([], [], 'test')


def test_protocols_pair_audio_conditions_and_select_visibility_before_scoring():
    records = []
    for split, scenario in [('train', 1), ('validation', 4), ('test', 7)]:
        for style in ('S', 'L', 'V'):
            records.append(dict(file=f'{split}/{style}_{style}.wav', scenario=scenario, dataset='HornBase',
                                split=split, label=1, category='car_horn'))
        records.append(dict(file=f'{split}/esc.wav', dataset='ESC50', split=split, label=1, category='car_horn'))
        for dataset in ('HornBase', 'ESC50'):
            for category in ('engine', 'siren', 'train', 'church_bells'):
                records.append(dict(file=f'{split}/{dataset}/{category}.wav', dataset=dataset,
                                    split=split, label=0, category=category))
    original = scene_plan(records, 'test', 'original')
    visible = scene_plan(records, 'test', 'visible')
    assert len(original) == len(visible) == 48
    for first, second in zip(original, visible):
        for name in ('target', 'interferer', 'sir_db', 'target_rms', 'noise_seed', 'visual_seed', 'label', 'radius'):
            assert first[name] == second[name]
        assert wrap_angle(first['angle'] - first['other']) == pytest.approx(wrap_angle(second['angle'] - second['other']))
        assert visible_target(camera_rig(), second['angle'], second['radius'])


def test_wrong_direction_counts_as_false_positive_and_false_negative():
    case = dict(label=1, angle=0, bearings=[dict(azimuth=90, depression=20, response=1)])
    result = joint_metrics([case], [[0.9]], 0.5, 0)
    assert (result['tp'], result['fp'], result['fn'], result['wrong_direction']) == (0, 1, 1, 1)
    with pytest.raises(ValueError, match='score list'):
        joint_metrics([case], [], 0.5, 0)


def experiment():
    path = Path(__file__).resolve().parents[1] / 'improve_detector.py'
    spec = importlib.util.spec_from_file_location('recovery_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthesis_replays_pcm_and_never_uses_vision_random_numbers():
    module = experiment()
    time = np.arange(48000) / 48000
    sources = dict(horn=(np.sin(2 * np.pi * 700 * time), 0), engine=(np.sin(2 * np.pi * 160 * time), 0))
    profile = module.load_json(module.PROFILE)
    case = dict(target='horn', interferer='engine', angle=30, other=-70, radius=15,
                sir_db=-5, noise_seed=42, visual_seed=1, split='validation', pair=0, target_rms=0.075)
    first, pcm, _, augmented = module.synthesize(case, sources, profile)
    second, other, _, _ = module.synthesize(dict(case, visual_seed=999), sources, profile)
    np.testing.assert_array_equal(first, second)
    assert pcm == other and not augmented and first.shape == (48000, 14)
    assert pcm['clipped_samples'] == 0 and pcm['peak'] <= 0.9001
    assert not np.array_equal(first, module.synthesize(case, sources, profile, stress=True)[0])


def test_weak_scene_detection_is_separate_from_direction_correctness():
    module = experiment()
    cases = [dict(label=1), dict(label=0)]
    metric = module.detection_metrics(cases, np.array([[0.1, 0.9], [0.1, 0.2]]), 0.5)
    assert metric['f1'] == 1 and metric['tp'] == 1
