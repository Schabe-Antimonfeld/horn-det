import numpy as np
import pytest

from horn.recovery import choose_gate, pcm_roundtrip, visible_target, camera_rig, scene_plan
from horn.surround import wrap_angle


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
