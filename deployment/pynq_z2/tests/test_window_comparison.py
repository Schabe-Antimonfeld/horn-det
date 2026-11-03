import hashlib
import wave

import numpy as np
import pytest

from scripts.compare_windows import decide, load_audio, make_scene, segment, summarize
from ps.configuration import load_config


def frame(score, index, correct=True, reliable=True):
    return dict(first=index * 24064 / 48000, end=(48128 + index * 24064) / 48000,
                epoch=0, score=score, reliable=reliable, direction_correct=correct)


def test_isolated_high_window_is_detected_once_but_cannot_confirm_twice():
    frames = [frame(0.1, 0), frame(0.95, 1), frame(0.1, 2)]
    assert decide(frames, 0.95, 1, [1.1, 1.3])['detected']
    assert not decide(frames, 0.95, 2, [1.1, 1.3])['detected']


def test_short_event_in_the_overlap_can_pass_both_with_half_second_extra_delay():
    frames = [frame(0.99, 0), frame(0.99, 1)]
    single, double = [decide(frames, 0.95, count, [0.6, 0.8]) for count in (1, 2)]
    assert single['detected'] and double['detected']
    assert double['latency_seconds'] - single['latency_seconds'] == pytest.approx(24064 / 48000)


@pytest.mark.parametrize('interruption', ['invalid', 'below', 'epoch', 'gap'])
def test_two_window_confirmation_cannot_bridge_unreliable_or_nonconsecutive_evidence(interruption):
    frames = [frame(0.99, 0), frame(0.99, 1)]
    if interruption == 'invalid':
        frames[0]['reliable'] = False
    elif interruption == 'below':
        frames[0]['score'] = 0.949
    elif interruption == 'epoch':
        frames[1]['epoch'] = 1
    else:
        frames[1] = frame(0.99, 3)
    assert not decide(frames, 0.95, 2, [0.6, 3])['detected']


def test_negative_and_off_event_alarms_count_once_per_release_two_identity():
    scores = [0.99, 0.99, 0.1, 0.99, 0.99, 0.1, 0.1, 0.99, 0.99]
    frames = [frame(score, index) for index, score in enumerate(scores)]
    result = decide(frames, 0.9, 2)
    assert not result['detected'] and result['false_alarms'] == 2
    result = decide(frames, 0.9, 2, [4.6, 4.8])
    assert result['detected'] and result['false_alarms'] == 1


def test_temporal_hit_and_correct_direction_are_separate():
    result = decide([frame(0.99, 0, correct=False)], 0.9, 1, [0.1, 0.5])
    assert result['detected'] and not result['direction_correct']


def test_crops_are_not_repeated_or_padded_to_manufacture_long_horns():
    values = np.sin(np.arange(5 * 48000) * 2 * np.pi * 700 / 48000)
    assert segment(values, 1, 3, 3) is None
    crop, record = segment(values, 1, 3, 0.2)
    assert len(crop) == 9600 and record['first_48k'] == 91200
    assert crop[0] == 0 and crop[-1] == 0
    assert np.sqrt(np.mean(crop ** 2)) == pytest.approx(1)
    with pytest.raises(ValueError, match='span'):
        segment(values, -1, 2, 1)


def test_loader_verifies_original_bytes_and_resamples_channel_zero(tmp_path):
    path = tmp_path / 'source.wav'
    values = np.column_stack((np.full(4410, 1000), np.full(4410, 2000))).astype('<i2')
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
        stream.writeframes(values.tobytes())
    actual = load_audio(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(actual) == 4800
    np.testing.assert_allclose(actual, 1000 / 32768)
    with pytest.raises(ValueError, match='checksum'):
        load_audio(path, 'wrong')


def test_equal_background_and_noise_are_shared_across_target_durations():
    config = load_config()
    background = np.zeros(5 * 48000)
    values = np.sin(np.arange(2 * 48000) * 2 * np.pi * 700 / 48000)
    target, _ = segment(values, 0, 2, 0.2)
    one, first = make_scene(config, target, background, [1.8, 4], -5, 1.1, 7)
    two, second = make_scene(config, None, background, [1.8, 4], -5, None, 7)
    np.testing.assert_array_equal(one[:48000], two[:48000])
    assert first['headroom_gain'] == second['headroom_gain'] == 1
    assert one.dtype == np.dtype('<i2') and len(one) % 512 == 0


def test_summary_keeps_missing_detections_out_of_conditional_latency():
    cases = []
    for index, score in enumerate((0.99, 0.1)):
        frames = [frame(score, 0)]
        decisions = {'{:.2f}/{}'.format(threshold, count): decide(frames, threshold, count, [0.1, 0.5])
                     for threshold in (0.95, 0.9, 0.8) for count in (1, 2)}
        cases.append(dict(cohort='paired', duration=0.2, source_group='one-parent', frames=frames, decisions=decisions))
    row = next(row for row in summarize(cases) if row['group'] == 'paired-short' and row['threshold'] == 0.95 and row['windows'] == 1)
    assert row['positives'] == 2 and row['detected'] == 1 and row['source_groups'] == 1
    assert row['recall'] == 0.5 and row['conditional_latency_median'] > 0
    row = next(row for row in summarize(cases) if row['windows'] == 2)
    assert row['detected'] == 0 and row['conditional_latency_median'] is None
