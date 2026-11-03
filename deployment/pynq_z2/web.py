"""Serve the dashboard and a same-origin API using the Python standard library."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from ps.configuration import ROOT
from ps.dashboard import Dashboard, contained


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def handler(dashboard, frontend):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, kind='application/json; charset=utf-8'):
            body = json.dumps(data, allow_nan=False).encode('utf-8') if isinstance(data, (dict, list)) else data
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                route = urlparse(self.path)
                query = parse_qs(route.query)
                if route.path == '/api/status':
                    self.send(200, dashboard.status())
                elif route.path == '/api/events':
                    limit = max(1, min(100, int(query.get('limit', ['20'])[0])))
                    self.send(200, dict(items=dashboard.list_events(limit)))
                elif route.path == '/api/photos':
                    cursor = int(query['cursor'][0]) if 'cursor' in query else None
                    limit = max(1, min(100, int(query.get('limit', ['9'])[0])))
                    self.send(200, dashboard.list_photos(query.get('kind', ['all'])[0], query.get('range', ['all'])[0],
                                                        query.get('q', [''])[0][:200], cursor, limit))
                elif route.path.startswith('/api/preview/'):
                    identity = route.path.split('/')[-1]
                    if dashboard.active != identity or dashboard.preview(identity) is None:
                        raise FileNotFoundError('No fresh camera frame')
                    self.send(200, (dashboard.run_folder(identity) / 'preview.jpg').read_bytes(), 'image/jpeg')
                elif route.path.startswith('/media/'):
                    parts = route.path.split('/')
                    if len(parts) != 4:
                        raise FileNotFoundError('Unknown photo route')
                    self.send(200, dashboard.media(int(parts[2]), parts[3]).read_bytes(), 'image/jpeg')
                elif route.path in ('/', '/index.html', '/app.js'):
                    name = 'app.js' if route.path == '/app.js' else 'index.html'
                    self.send(200, contained(frontend, frontend / name).read_bytes(),
                              'text/javascript; charset=utf-8' if name == 'app.js' else 'text/html; charset=utf-8')
                else:
                    raise FileNotFoundError('Unknown route')
            except FileNotFoundError as error:
                self.send(404, dict(error=str(error)))
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, dict(error=str(error)))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self.send(500, dict(error='Service could not complete the request'))

        def do_POST(self):
            origin = self.headers.get('Origin')
            if ((origin is not None and (urlparse(origin).netloc != self.headers.get('Host') or urlparse(origin).scheme != 'http'))
                    or not secrets.compare_digest(self.headers.get('X-Control-Token', ''), dashboard.token)):
                self.send(403, dict(error='Same-origin control token required'))
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('JSON body must contain 1..4096 bytes')
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
                if self.path == '/api/runs':
                    if set(data) - {'mode', 'source'}:
                        raise ValueError('Unsupported launch settings')
                    self.send(201, dashboard.start(data.get('mode'), data.get('source')))
                elif self.path == '/api/stop':
                    self.send(200, dashboard.stop())
                else:
                    self.send(404, dict(error='Unknown control route'))
            except RuntimeError as error:
                self.send(409, dict(error=str(error)))
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                self.send(400, dict(error=str(error)))
            except Exception:
                self.send(500, dict(error='Could not start or stop inference; inspect the service log'))

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data', default=str(ROOT / 'outputs/dashboard'))
    parser.add_argument('--config')
    parser.add_argument('--frontend')
    args = parser.parse_args()
    bundled = ROOT / 'frontend'
    frontend = Path(args.frontend).resolve() if args.frontend else bundled if bundled.exists() else ROOT.parents[1] / 'frontend'
    if not (frontend / 'index.html').is_file() or not (frontend / 'app.js').is_file():
        raise ValueError('Frontend index.html/app.js missing; supply --frontend DIRECTORY')
    dashboard = Dashboard(args.data, args.config)
    try:
        server = Server((args.host, args.port), handler(dashboard, frontend))
        server.timeout = 0.5
        print('Dashboard: http://{}:{}'.format(args.host, server.server_port), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()


if __name__ == '__main__':
    main()
