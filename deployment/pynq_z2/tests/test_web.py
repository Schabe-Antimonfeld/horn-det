import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ps.dashboard import Dashboard
from web import Server, handler


@pytest.fixture
def service(tmp_path):
    dashboard = Dashboard(tmp_path / 'data')
    root = Path(__file__).resolve().parents[1]
    frontend = root / 'frontend' if (root / 'frontend').exists() else root.parents[1] / 'frontend'
    server = Server(('127.0.0.1', 0), handler(dashboard, frontend))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield 'http://127.0.0.1:' + str(server.server_port), dashboard
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    dashboard.close()


def test_frontend_is_served_without_fake_records_or_vehicle_boxes(service):
    base, dashboard = service
    with urlopen(base) as response:
        page = response.read().decode()
        assert 'script-src' in response.headers['Content-Security-Policy']
    assert 'src="/app.js"' in page and 'assets/stock/' not in page
    assert 'class="detect ' not in page and 'id="start-btn"' in page
    with urlopen(base + '/app.js') as response:
        assert b'/api/status' in response.read()
    with urlopen(base + '/api/status') as response:
        status = json.load(response)
    assert status['operating_point'] == dict(threshold=0.9, confirm=1, capture_threshold=0.9)
    assert status['run'] is None and status['metrics']['horns_today'] == 0


def test_mutations_require_token_and_same_origin(service):
    base, dashboard = service
    for headers in ({}, {'X-Control-Token': dashboard.token, 'Origin': 'http://unrelated.invalid'}):
        request = Request(base + '/api/stop', data=b'{}', headers=headers)
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
    request = Request(base + '/api/stop', data=b'{}', headers={'X-Control-Token': dashboard.token, 'Origin': base})
    with urlopen(request) as response:
        assert json.load(response)['status'] == 'idle'


@pytest.mark.parametrize('path,code', [('/api/photos?limit=no', 400), ('/api/photos?kind=vehicle', 400),
                                      ('/media/1/points', 404), ('/api/preview/unknown', 404),
                                      ('/../config/default.json', 404), ('/assets/stock/live.jpg', 404)])
def test_invalid_requests_do_not_expose_files_or_fake_images(service, path, code):
    base, _ = service
    with pytest.raises(HTTPError) as error:
        urlopen(base + path)
    assert error.value.code == code


def test_launch_body_is_validated(service):
    base, dashboard = service
    for body in (b'[]', b'{bad', b'{"mode":"replay","source":"horn","command":"anything"}'):
        request = Request(base + '/api/runs', data=body, headers={'X-Control-Token': dashboard.token})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
