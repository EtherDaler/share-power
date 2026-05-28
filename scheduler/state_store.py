from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .job_queue import Job, JobStatus


def normalize_job_after_restart(job: Job) -> None:
    """После рестарта координатора нет живых WebSocket — активные задачи снова в очереди."""
    if job.status in (JobStatus.ASSIGNED, JobStatus.RUNNING):
        job.status = JobStatus.QUEUED
        job.assigned_worker_id = None
        job.backup_worker_id = None
        job.cancel_requested = False
    job.shard_worker_ids = []
    job.shard_exit_codes = {}
    job.shard_abort_requested = False


class JobStateStore:
    """SQLite-хранилище задач координатора (скрипт, статусы, чекпоинты, логи)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(jobs)")
        cols = {str(r[1]) for r in cur.fetchall()}
        if "shard_world_size" not in cols:
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN shard_world_size INTEGER NOT NULL DEFAULT 1"
            )

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    script TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    assigned_worker_id TEXT,
                    backup_worker_id TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    last_checkpoint TEXT,
                    last_step INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    logs_json TEXT NOT NULL DEFAULT '[]',
                    shard_world_size INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self._migrate(conn)
            conn.commit()

    def upsert(self, job: Job) -> None:
        logs = getattr(job, "logs", []) or []
        sw = max(1, int(getattr(job, "shard_world_size", 1) or 1))
        row = (
            job.id,
            job.script,
            job.owner_id,
            int(job.priority),
            float(job.created_at),
            job.status.value,
            job.assigned_worker_id,
            job.backup_worker_id,
            int(job.retry_count),
            int(job.max_retries),
            job.last_checkpoint,
            job.last_step if job.last_step is not None else None,
            1 if job.cancel_requested else 0,
            json.dumps(logs, ensure_ascii=False),
            sw,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    id, script, owner_id, priority, created_at, status,
                    assigned_worker_id, backup_worker_id,
                    retry_count, max_retries, last_checkpoint, last_step,
                    cancel_requested, logs_json, shard_world_size
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            conn.commit()

    def load_all(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at ASC"
            ).fetchall()
        out: list[Job] = []
        for r in rows:
            raw_logs = r["logs_json"] or "[]"
            try:
                logs = json.loads(raw_logs)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
            st = r["status"]
            try:
                status = JobStatus(st)
            except ValueError:
                status = JobStatus.QUEUED
            try:
                sw = max(1, int(r["shard_world_size"]))
            except (KeyError, ValueError, TypeError):
                sw = 1
            job = Job(
                id=r["id"],
                script=r["script"] or "",
                owner_id=r["owner_id"] or "anon",
                priority=int(r["priority"] or 0),
                created_at=float(r["created_at"] or time.time()),
                status=status,
                assigned_worker_id=r["assigned_worker_id"],
                backup_worker_id=r["backup_worker_id"],
                retry_count=int(r["retry_count"] or 0),
                max_retries=int(r["max_retries"] or 3),
                last_checkpoint=r["last_checkpoint"],
                last_step=(
                    int(r["last_step"])
                    if r["last_step"] is not None
                    else None
                ),
                cancel_requested=bool(r["cancel_requested"]),
                logs=[str(x) for x in logs],
                subscribers=[],
                shard_world_size=sw,
            )
            normalize_job_after_restart(job)
            out.append(job)
        return out
