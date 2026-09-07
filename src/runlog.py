"""Local run logging to SQLite.

Deliberately dependency-free: anyone cloning the repo can reproduce a run and
plot the curves without creating a W&B account or exporting an API key. Set
`report_to: wandb` in a config if you want it in addition.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


class RunLogger:
    def __init__(self, db_path: str | Path = "runs/runs.sqlite", config: dict | None = None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.con = sqlite3.connect(self.path)
        self._init_schema()
        self.con.execute(
            "INSERT INTO runs (run_id, started_at, git_sha, config) VALUES (?,?,?,?)",
            (self.run_id, time.time(), git_sha(), json.dumps(config or {}, default=str)),
        )
        self.con.commit()

    def _init_schema(self) -> None:
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, started_at REAL, finished_at REAL,
                git_sha TEXT, config TEXT, notes TEXT);
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT, step INTEGER, wall REAL, key TEXT, value REAL);
            CREATE INDEX IF NOT EXISTS idx_metrics ON metrics(run_id, key, step);
            CREATE TABLE IF NOT EXISTS predictions (
                run_id TEXT, db_id TEXT, question TEXT, gold TEXT,
                pred TEXT, correct INTEGER, error_kind TEXT);
            """
        )
        self.con.commit()

    def log(self, step: int, **kv) -> None:
        now = time.time()
        self.con.executemany(
            "INSERT INTO metrics (run_id, step, wall, key, value) VALUES (?,?,?,?,?)",
            [
                (self.run_id, step, now, k, float(v))
                for k, v in kv.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ],
        )
        self.con.commit()

    def log_predictions(self, rows) -> None:
        self.con.executemany(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
            [
                (self.run_id, r.db_id, r.question, r.gold, r.pred, int(r.correct), r.error_kind)
                for r in rows
            ],
        )
        self.con.commit()

    def finish(self, notes: str = "") -> None:
        self.con.execute(
            "UPDATE runs SET finished_at=?, notes=? WHERE run_id=?",
            (time.time(), notes, self.run_id),
        )
        self.con.commit()
        self.con.close()
