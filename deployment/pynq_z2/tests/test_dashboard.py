import json
from pathlib import Path
import time

import pytest

from ps.dashboard import Dashboard, contained, utc_now
from ps.vision import CameraCache


@pytest.fixture
def dashboard(tmp_path):
    value = Dashboard(tmp_path / 'data')
    yield value
    value.close()


def wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError('Condition did not complete before timeout')


def add_record(dashboard, identity='fixture', saved=True, event_id=1):
    folder = dashboard.run_folder(identity)
    folder.mkdir(exist_ok=True)
    with dashboard.lock:
        dashboard.database.execute('INSERT OR IGNORE INTO runs(id,mode,status,started) VALUES(?,?,?,?)',
                                   (identity, 'live', 'completed', utc_now()))
        dashboard.database.commit()
    event = dict(event_id=event_id, horn_confirmed=True, score=0.92, azimuth=20,
                 sample_first=0, sample_end_exclusive=48128, recorded_at=utc_now())
    if saved:
        (folder / ('raw{}.jpg'.format(event_id))).write_bytes(b'original-test-bytes')
        (folder / ('point{}.jpg'.format(event_id))).write_bytes(b'point-test-bytes')
        event['capture'] = dict(stage='confirmed', frames=[dict(state='saved', camera='wide-0',
                                original='raw{}.jpg'.format(event_id), annotated='point{}.jpg'.format(event_id))], projections={})
    with (folder / 'events.jsonl').open('ab') as stream:
        stream.write((json.dumps(event) + '\n').encode())
    dashboard.ingest(identity)
    return event


def test_real_replay_is_indexed_without_stock_photos_and_metrics_deduplicate(dashboard):
    result = dashboard.start('replay', 'horn')
    with pytest.raises(RuntimeError, match='already active'):
        dashboard.start('replay', 'silence')
    wait_until(lambda: dashboard.status()['active'] is None)
    status = dashboard.status()
    assert status['run']['id'] == result['id'] and status['run']['status'] == 'completed'
    assert status['metrics']['horns_today'] == 1 and status['metrics']['photos_today'] == 0
    events = dashboard.list_events()
    assert len(events) == 4 and all(row['event']['horn_confirmed'] for row in events)
    assert all(row['mode'] == 'replay' for row in events)
    assert dashboard.list_photos()['items'] == [] and status['preview'] is None


def test_stop_and_repeat_replay_use_distinct_sessions(dashboard):
    first = dashboard.start('replay', 'horn')
    dashboard.stop()
    wait_until(lambda: dashboard.status()['active'] is None)
    assert dashboard.status()['run']['status'] == 'stopped'
    second = dashboard.start('replay', 'silence')
    assert first['id'] != second['id']
    wait_until(lambda: dashboard.status()['active'] is None)
    assert not dashboard.status()['latest']['horn_confirmed']


def test_partial_lines_and_duplicate_ingestion_are_safe(dashboard):
    add_record(dashboard)
    folder = dashboard.run_folder('fixture')
    with (folder / 'events.jsonl').open('ab') as stream:
        stream.write(b'{"event_id":2')
    dashboard.ingest('fixture')
    assert len(dashboard.list_events()) == 1
    assert dashboard.list_photos()['total'] == 1
    dashboard.ingest('fixture')
    assert dashboard.status()['metrics']['horns_today'] == 1


def test_photo_cursor_filters_and_media_are_backed_by_saved_files(dashboard):
    for event_id in range(1, 13):
        add_record(dashboard, event_id=event_id)
    first = dashboard.list_photos(kind='horn', period='today', query='wide-0', limit=9)
    second = dashboard.list_photos(cursor=first['next_cursor'], limit=9)
    assert len(first['items']) == 9 and len(second['items']) == 3
    assert not set(item['id'] for item in first['items']) & set(item['id'] for item in second['items'])
    assert dashboard.media(first['items'][0]['id'], 'points').read_bytes() == b'point-test-bytes'
    assert dashboard.list_photos(kind='candidate')['total'] == 0
    assert dashboard.list_photos(query="%' OR 1=1 --")['total'] == 0
    with pytest.raises(ValueError):
        dashboard.list_photos(kind='vehicle')
    with pytest.raises(ValueError):
        dashboard.media(1, '../../control')


def test_index_survives_restart_and_missing_files_are_not_replaced(dashboard):
    add_record(dashboard)
    root = dashboard.directory
    dashboard.close()
    other = Dashboard(root)
    try:
        assert other.list_photos()['total'] == 1
        assert len(other.list_events()) == 1
    finally:
        other.close()


def test_only_one_service_can_own_the_same_data_directory(dashboard):
    with pytest.raises(RuntimeError, match='already in use'):
        Dashboard(dashboard.directory)


@pytest.mark.parametrize('mode,source', [('unknown', None), ('replay', '../../secrets'), ('live', 'horn')])
def test_api_cannot_pass_arbitrary_launch_arguments(dashboard, mode, source):
    with pytest.raises(ValueError):
        dashboard.start(mode, source)


def test_index_rejects_escaped_photo_paths_transactionally(dashboard):
    add_record(dashboard, saved=False)
    folder = dashboard.run_folder('fixture')
    bad = dict(event_id=2, horn_confirmed=True, score=0.95, capture=dict(stage='confirmed', frames=[
        dict(state='saved', camera='wide-0', original='../../control/private.jpg', annotated='safe.jpg')]))
    with (folder / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(bad) + '\n')
    with pytest.raises(ValueError, match='outside'):
        dashboard.ingest('fixture')
    assert len(dashboard.list_events()) == 1


def test_preview_only_returns_existing_encoded_cache_and_expires(dashboard):
    cache = CameraCache([dict(id='wide-0')], None)
    cached = dict(time=time.monotonic(), jpeg=b'already-encoded', sequence=7)
    cache.history['wide-0'].append(cached)
    assert cache.latest('wide-0')['jpeg'] == b'already-encoded'
    assert cache.latest('unknown') is None
    folder = dashboard.run_folder('preview-test')
    folder.mkdir()
    record = dict(published_unix=time.time(), timestamp=utc_now(), sequence=7)
    (folder / 'preview.json').write_text(json.dumps(record))
    assert dashboard.preview('preview-test')['sequence'] == 7
    record['published_unix'] -= 5
    (folder / 'preview.json').write_text(json.dumps(record))
    assert dashboard.preview('preview-test') is None
    with pytest.raises(ValueError):
        contained(folder, folder / '../../control/private')


def test_preview_publication_uses_fresh_encoded_bytes_and_does_not_repeat(tmp_path):
    from run import publish_preview
    cache = CameraCache([dict(id='wide-0')], None)
    timestamp = time.monotonic()
    cache.history['wide-0'].append(dict(time=timestamp, jpeg=b'encoded-fixture'))
    assert publish_preview(cache, tmp_path, None, utc_now()) == timestamp
    assert (tmp_path / 'preview.jpg').read_bytes() == b'encoded-fixture'
    first = (tmp_path / 'preview.json').read_bytes()
    assert publish_preview(cache, tmp_path, timestamp, utc_now()) == timestamp
    assert (tmp_path / 'preview.json').read_bytes() == first
    cache.history['wide-0'].append(dict(time=timestamp - 10, jpeg=b'expired'))
    assert publish_preview(cache, tmp_path, timestamp, utc_now()) == timestamp
    assert (tmp_path / 'preview.jpg').read_bytes() == b'encoded-fixture'


def test_live_missing_overlay_is_reported_as_failure_without_importing_pynq(tmp_path):
    from ps.configuration import load_config
    config = load_config()
    config['hardware']['overlay'] = str(tmp_path / 'nonexistent.bit')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    value = Dashboard(tmp_path / 'live', profile)
    try:
        value.start('live')
        wait_until(lambda: value.status()['active'] is None)
        assert value.status()['run']['status'] == 'failed'
        assert 'bitstream and HWH are missing' in value.status()['run']['error']
        assert not value.list_events() and value.status()['preview'] is None
    finally:
        value.close()


def test_expired_owner_heartbeat_stops_before_any_inference(tmp_path):
    import os
    from run import main
    from ps.configuration import ROOT
    heartbeat = tmp_path / 'heartbeat'
    heartbeat.touch()
    os.utime(str(heartbeat), (time.time() - 20, time.time() - 20))
    output = tmp_path / 'stopped'
    assert main(['--replay', str(ROOT / 'examples/horn-array14.wav'), '--output', str(output),
                 '--heartbeat', str(heartbeat)]) == 0
    health = json.loads((output / 'health.json').read_text())
    assert health['status'] == 'stopped' and health['windows'] == 0
