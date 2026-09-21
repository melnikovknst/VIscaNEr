import json
import sqlite3
from pathlib import Path

from backend.schemas import ScanResult


class History:
    def __init__(self, path: Path, limit: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.limit = limit
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS scans (id TEXT PRIMARY KEY, session TEXT NOT NULL, created_at TEXT NOT NULL, result TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS scans_session ON scans(session, created_at)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, session: str, result: ScanResult):
        with self.connect() as db:
            db.execute("INSERT INTO scans VALUES (?, ?, ?, ?)", (result.id, session, result.created_at, result.model_dump_json()))
            db.execute("DELETE FROM scans WHERE session=? AND id NOT IN (SELECT id FROM scans WHERE session=? ORDER BY created_at DESC LIMIT ?)", (session, session, self.limit))
            db.execute("DELETE FROM scans WHERE created_at < strftime('%Y-%m-%dT%H:%M:%S', 'now', '-30 days')")

    def list(self, session: str):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT result FROM scans WHERE session=? AND created_at >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-30 days') ORDER BY created_at DESC LIMIT ?", (session, self.limit))]

    def clear(self, session: str):
        with self.connect() as db:
            db.execute("DELETE FROM scans WHERE session=?", (session,))
