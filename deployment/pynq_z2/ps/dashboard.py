"""Persistent dashboard records and a single owned inference subprocess."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from ps.configuration import ROOT, load_config


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def contained(root, path):
    """Resolve paths before exposing files, including symlink targets."""
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('Path is outside its allowed directory')
    return path


class Dashboard:
    """Index append-only event records without loading models in HTTP threads."""

    def __init__(self, directory, config=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.guard = (self.directory / 'service.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.guard.write(b'0')
                self.guard.flush()
                self.guard.seek(0)
                msvcrt.locking(self.guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.guard.close()
            raise RuntimeError('This dashboard data directory is already in use')
        (self.directory / 'control').mkdir(exist_ok=True)
        (self.directory / 'sessions').mkdir(exist_ok=True)
        self.config = load_config(config)
        self.lock = threading.RLock()
        self.token = secrets.token_hex(32)
        self.database = sqlite3.connect(str(self.directory / 'dashboard.sqlite3'), check_same_thread=False)
        self.database.row_factory = sqlite3.Row
        self.database.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, mode TEXT NOT NULL, source TEXT, status TEXT NOT NULL,
                started TEXT NOT NULL, ended TEXT, error TEXT, offset INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run TEXT NOT NULL, timestamp TEXT NOT NULL,
                event_id INTEGER NOT NULL, confirmed INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_run ON events(run, id);
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY, run TEXT NOT NULL, event_id INTEGER NOT NULL,
                camera TEXT NOT NULL, stage TEXT NOT NULL, timestamp TEXT NOT NULL,
                original TEXT NOT NULL, annotated TEXT NOT NULL, data TEXT NOT NULL,
                UNIQUE(run, event_id, camera, stage));
            CREATE INDEX IF NOT EXISTS photos_time ON photos(timestamp);
        ''')
        interrupted = self.database.execute("SELECT COUNT(*) FROM runs WHERE status IN ('starting','running','stopping')").fetchone()[0]
        self.recovery_until = time.time() + 12 if interrupted else 0
        self.database.execute("UPDATE runs SET status='interrupted', ended=? WHERE status IN ('starting','running','stopping')", (utc_now(),))
        self.database.commit()
        self.process, self.active, self.log = None, None, None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.monitor, name='dashboard-monitor', daemon=True)
        self.thread.start()

    def run_folder(self, identity):
        return contained(self.directory / 'sessions', self.directory / 'sessions' / identity)

    def start(self, mode, source=None):
        """Only launch fixed CLI arguments and known replay fixtures."""
        if mode not in ('replay', 'live'):
            raise ValueError('Mode must be replay or live')
        if mode == 'replay' and source not in ('horn', 'silence'):
            raise ValueError('Replay source must be horn or silence')
        if mode == 'live' and source is not None:
            raise ValueError('Live mode does not accept a replay source')
        with self.lock:
            if time.time() < self.recovery_until:
                raise RuntimeError('Waiting for the previous worker heartbeat to expire; retry in 12 seconds')
            if self.active is not None:
                raise RuntimeError('An inference session is already active')
            identity = uuid.uuid4().hex
            folder = self.run_folder(identity)
            control = self.directory / 'control'
            settings = json.loads(json.dumps(self.config))
            if mode == 'replay':
                settings['vision']['enabled'] = False
            profile = control / (identity + '.json')
            profile.write_text(json.dumps(settings, indent=2), encoding='utf-8')
            stop = control / (identity + '.stop')
            heartbeat = control / (identity + '.heartbeat')
            heartbeat.touch()
            command = [sys.executable, '-B', '-u', str(ROOT / 'run.py'), '--config', str(profile),
                       '--output', str(folder), '--stop-file', str(stop), '--heartbeat', str(heartbeat)]
            if mode == 'replay':
                command += ['--replay', str(ROOT / 'examples' / (source + '-array14.wav')), '--realtime']
            elif settings['vision']['enabled']:
                command += ['--preview']
            self.database.execute('INSERT INTO runs(id,mode,source,status,started) VALUES(?,?,?,?,?)',
                                  (identity, mode, source, 'starting', utc_now()))
            self.database.commit()
            self.log = (control / (identity + '.log')).open('wb')
            try:
                self.process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                                stdout=self.log, stderr=subprocess.STDOUT,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except Exception as error:
                self.log.close()
                self.database.execute('UPDATE runs SET status=?,error=?,ended=? WHERE id=?',
                                      ('failed', str(error), utc_now(), identity))
                self.database.commit()
                raise
            self.active = identity
            return dict(id=identity, mode=mode, status='starting')

    def stop(self):
        with self.lock:
            if self.active is None:
                return dict(status='idle')
            (self.directory / 'control' / (self.active + '.stop')).touch()
            self.database.execute("UPDATE runs SET status='stopping' WHERE id=?", (self.active,))
            self.database.commit()
            return dict(id=self.active, status='stopping')

    def ingest(self, identity):
        """Commit complete lines with their offset; unfinished writes are retried."""
        with self.lock, self.database:
            row = self.database.execute('SELECT offset FROM runs WHERE id=?', (identity,)).fetchone()
            path = self.run_folder(identity) / 'events.jsonl'
            if row is None or not path.is_file():
                return
            offset = row['offset']
            with path.open('rb') as stream:
                stream.seek(offset)
                for _ in range(64):
                    line = stream.readline(1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 1024 * 1024:
                        raise ValueError('Event line exceeds 1 MiB')
                    if not line.endswith(b'\n'):
                        break
                    event = json.loads(line.decode('utf-8'))
                    stamp = event.get('recorded_at', utc_now())
                    identity_number = int(event['event_id'])
                    self.database.execute('INSERT INTO events(run,timestamp,event_id,confirmed,data) VALUES(?,?,?,?,?)',
                                          (identity, stamp, identity_number, int(bool(event.get('horn_confirmed'))), json.dumps(event, allow_nan=False)))
                    capture = event.get('capture') or {}
                    for frame in capture.get('frames', []):
                        if frame.get('state') != 'saved':
                            continue
                        for key in ('original', 'annotated'):
                            contained(self.run_folder(identity), self.run_folder(identity) / frame[key])
                        self.database.execute('''INSERT OR IGNORE INTO photos
                            (run,event_id,camera,stage,timestamp,original,annotated,data) VALUES(?,?,?,?,?,?,?,?)''',
                            (identity, identity_number, frame['camera'], capture['stage'], stamp,
                             frame['original'], frame['annotated'], json.dumps(event, allow_nan=False)))
                    offset = stream.tell()
                    self.database.execute('UPDATE runs SET offset=? WHERE id=?', (offset, identity))
            self.database.commit()

    def monitor(self):
        while not self.closed.wait(0.25):
            with self.lock:
                if self.active is None:
                    continue
                identity = self.active
                try:
                    (self.directory / 'control' / (identity + '.heartbeat')).touch()
                    self.ingest(identity)
                    code = self.process.poll()
                    health = self.read_health(identity)
                    if code is None:
                        if health.get('status') == 'running':
                            self.database.execute("UPDATE runs SET status='running' WHERE id=? AND status='starting'", (identity,))
                            self.database.commit()
                        continue
                    self.ingest(identity)
                    stopping = self.database.execute('SELECT status FROM runs WHERE id=?', (identity,)).fetchone()['status'] == 'stopping'
                    status = 'stopped' if code == 0 and stopping else 'completed' if code == 0 else 'failed'
                    error = None if code == 0 else self.log_tail(identity)
                    self.database.execute('UPDATE runs SET status=?,ended=?,error=? WHERE id=?', (status, utc_now(), error, identity))
                    self.database.commit()
                    self.log.close()
                    self.active, self.process = None, None
                except Exception as error:
                    self.database.rollback()
                    self.database.execute('UPDATE runs SET error=? WHERE id=?', (str(error), identity))
                    self.database.commit()
                    (self.directory / 'control' / (identity + '.stop')).touch()
                    if self.process.poll() is not None:
                        self.database.execute("UPDATE runs SET status='failed', ended=? WHERE id=?", (utc_now(), identity))
                        self.database.commit()
                        self.log.close()
                        self.active, self.process = None, None

    def log_tail(self, identity):
        path = self.directory / 'control' / (identity + '.log')
        if not path.exists():
            return 'Process exited without a log'
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 4000))
            return stream.read().decode('utf-8', errors='replace')

    def read_health(self, identity):
        try:
            return json.loads((self.run_folder(identity) / 'health.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def status(self):
        with self.lock:
            row = self.database.execute('SELECT * FROM runs ORDER BY started DESC LIMIT 1').fetchone()
            run = dict(row) if row else None
            event = self.database.execute('SELECT data FROM events WHERE run=? ORDER BY id DESC LIMIT 1', (run['id'],)).fetchone() if run else None
            today = datetime.now(timezone.utc).date().isoformat()
            photos = self.database.execute('SELECT COUNT(*) FROM photos WHERE timestamp>=?', (today,)).fetchone()[0]
            horns = self.database.execute('SELECT COUNT(*) FROM (SELECT DISTINCT run,event_id FROM events WHERE confirmed=1 AND timestamp>=?)', (today,)).fetchone()[0]
            healthy = self.read_health(run['id']) if run else {}
            preview = self.preview(run['id']) if run and self.active == run['id'] else None
            return dict(connected=True, run=run, active=self.active, health=healthy,
                        latest=json.loads(event['data']) if event else None,
                        preview=preview, metrics=dict(photos_today=photos, horns_today=horns),
                        operating_point=dict(threshold=self.config['detector']['threshold'],
                                             confirm=self.config['event']['confirm'], capture_threshold=self.config['detector']['capture_threshold']),
                        vision_enabled=self.config['vision']['enabled'], control_token=self.token,
                        timezone='UTC', camera='wide-0', board_validated=False)

    def preview(self, identity):
        try:
            data = json.loads((self.run_folder(identity) / 'preview.json').read_text(encoding='utf-8'))
            age = max(0, time.time() - data['published_unix'])
            if age > 3:
                return None
            return dict(url='/api/preview/' + identity, age_seconds=age, sequence=data['sequence'],
                        timestamp=data['timestamp'], timing_verified=False)
        except (OSError, ValueError, KeyError):
            return None

    def list_events(self, limit=20):
        with self.lock:
            rows = self.database.execute('''SELECT events.id, events.run, events.timestamp, events.data, runs.mode
                FROM events JOIN runs ON runs.id=events.run ORDER BY events.id DESC LIMIT ?''', (limit,)).fetchall()
            return [dict(id=row['id'], run=row['run'], timestamp=row['timestamp'], mode=row['mode'], event=json.loads(row['data'])) for row in rows]

    def list_photos(self, kind='all', period='all', query='', cursor=None, limit=9):
        if kind not in ('all', 'horn', 'candidate') or period not in ('all', 'today', 'week'):
            raise ValueError('Unsupported photo filter')
        clauses, values = [], []
        if kind != 'all':
            clauses.append('p.stage=?')
            values.append('confirmed' if kind == 'horn' else 'candidate')
        if period != 'all':
            from datetime import timedelta
            date = datetime.now(timezone.utc)
            cutoff = date.replace(hour=0, minute=0, second=0, microsecond=0) if period == 'today' else date - timedelta(days=7)
            clauses.append('p.timestamp>=?')
            values.append(cutoff.isoformat())
        if query:
            clauses.append("(p.camera LIKE ? ESCAPE '!' OR p.run LIKE ? ESCAPE '!' OR CAST(p.event_id AS TEXT) LIKE ? ESCAPE '!')")
            pattern = '%' + query.replace('!', '!!').replace('%', '!%').replace('_', '!_') + '%'
            values += [pattern] * 3
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with self.lock:
            total = self.database.execute('SELECT COUNT(*) FROM photos p' + where, values).fetchone()[0]
            if cursor is not None:
                where += (' AND ' if clauses else ' WHERE ') + 'p.id<?'
                values.append(cursor)
            rows = self.database.execute('''SELECT p.*,r.mode FROM photos p JOIN runs r ON r.id=p.run''' + where + ' ORDER BY p.id DESC LIMIT ?', values + [limit + 1]).fetchall()
            items = []
            for row in rows[:limit]:
                event = json.loads(row['data'])
                items.append(dict(id=row['id'], run=row['run'], event_id=row['event_id'], camera=row['camera'],
                                  timestamp=row['timestamp'], stage=row['stage'], mode=row['mode'],
                                  score=event['score'], azimuth=event.get('azimuth'), depression=event.get('depression'),
                                  original='/media/{}/original'.format(row['id']), image='/media/{}/points'.format(row['id']),
                                  evidence=event.get('capture'), sample_first=event.get('sample_first'),
                                  sample_end=event.get('sample_end_exclusive')))
            return dict(items=items, total=total, next_cursor=items[-1]['id'] if len(rows) > limit else None)

    def media(self, identity, variant):
        if variant not in ('original', 'points'):
            raise ValueError('Unsupported image variant')
        with self.lock:
            row = self.database.execute('SELECT * FROM photos WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise FileNotFoundError('Photo not found')
            return contained(self.run_folder(row['run']), self.run_folder(row['run']) / row['original' if variant == 'original' else 'annotated'])

    def close(self):
        if self.closed.is_set():
            return
        self.stop()
        with self.lock:
            process = self.process
            identity = self.active
        forced = False
        if process is not None:
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                forced = True
                process.terminate()
                process.wait(timeout=5)
                with self.lock:
                    self.database.execute("UPDATE runs SET error=? WHERE id=?", ('Forced stop; verify FPGA/DMA state before restarting live acquisition', identity))
                    self.database.commit()
        self.closed.set()
        self.thread.join(timeout=2)
        with self.lock:
            if identity is not None:
                self.ingest(identity)
                if forced:
                    self.database.execute("UPDATE runs SET status='failed',ended=?,error=? WHERE id=?", (
                        utc_now(), 'Forced stop; verify FPGA/DMA state before restarting live acquisition', identity))
                else:
                    self.database.execute("UPDATE runs SET status='stopped', ended=? WHERE id=? AND status IN ('starting','running','stopping')", (utc_now(), identity))
                self.database.commit()
            if self.log is not None and not self.log.closed:
                self.log.close()
            self.database.close()
            self.guard.close()
