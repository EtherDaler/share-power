import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time

class JobStatus(str, Enum):
    QUEUED       = "queued"
    ASSIGNED     = "assigned"
    RUNNING      = "running"
    DONE         = "done"
    FAILED       = "failed"
    CANCELLED    = "cancelled"
    RESCHEDULED  = "rescheduled"
    DEAD         = "dead"

@dataclass
class Job:
    id: str
    script: str
    owner_id: str                          # кто отправил задачу
    priority: int = 0                      # выше = важнее
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.QUEUED
    assigned_worker_id: Optional[str] = None
    backup_worker_id: Optional[str] = None # замена при падении
    retry_count: int = 0
    max_retries: int = 3
    last_checkpoint: Optional[str] = None  # путь к последнему чекпоинту
    last_step: Optional[int] = None
    cancel_requested: bool = False
    logs: list[str] = field(default_factory=list)
    subscribers: list = field(default_factory=list)  # WebSocket браузеров
    # FSDP / multi-node: число рангов; >1 — группа воркеров (не резерв)
    shard_world_size: int = 1
    shard_worker_ids: list[str] = field(default_factory=list)
    shard_exit_codes: dict[str, int] = field(default_factory=dict)
    shard_abort_requested: bool = False
    # Pipeline + subspace: YAML целиком в сообщении run_job (не в SQLite-схеме)
    pipeline_enabled: bool = False
    pipeline_config_yaml: Optional[str] = None


class TrainingExitResult(NamedTuple):
    finalized: bool
    job: Optional[Job]
    old_backup: Optional[str]
    release_worker_ids: list[str]
    peers_to_cancel: list[str]


class JobQueue:
    def __init__(self, store: Optional[Any] = None):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._jobs: dict[str, Job] = {}
        self._on_changed: Callable[[], None] = lambda: None
        self._store = store

    async def restore_from_store(self) -> None:
        """Загружает задачи с диска; активные (assigned/running) переводит в queued."""
        if not self._store:
            return
        jobs = await asyncio.to_thread(self._store.load_all)
        for job in jobs:
            self._jobs[job.id] = job
            if job.status == JobStatus.QUEUED:
                await self._queue.put((-job.priority, job.created_at, job.id))
        if jobs:
            self._wake()

    async def save_job(self, job: Job) -> None:
        if not self._store:
            return
        await asyncio.to_thread(self._store.upsert, job)

    async def maybe_batch_persist_logs(self, job: Job) -> None:
        if not self._store or not job.logs:
            return
        if len(job.logs) % 50 == 0:
            await asyncio.to_thread(self._store.upsert, job)

    def set_on_changed(self, fn: Callable[[], None]) -> None:
        self._on_changed = fn

    def _wake(self) -> None:
        self._on_changed()

    async def push(self, job: Job):
        self._jobs[job.id] = job
        await self._queue.put((-job.priority, job.created_at, job.id))
        self._wake()
        if self._store:
            await asyncio.to_thread(self._store.upsert, job)

    async def pop(self) -> Optional[Job]:
        """Возвращает следующую задачу из очереди."""
        while not self._queue.empty():
            _, _, job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            # Пропускаем задачи которые уже не в очереди (переназначены или мертвы)
            if job and job.status == JobStatus.QUEUED:
                return job
        return None

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    async def reschedule(self, job: Job):
        """Ставит упавшую задачу обратно в очередь."""
        job.retry_count += 1
        if job.retry_count >= job.max_retries:
            job.status = JobStatus.DEAD
            await self._notify(job, {"type": "job_dead", "job_id": job.id})
            if self._store:
                await asyncio.to_thread(self._store.upsert, job)
            return
        job.status = JobStatus.QUEUED
        job.assigned_worker_id = None
        job.backup_worker_id = None
        job.cancel_requested = False
        job.shard_worker_ids = []
        job.shard_exit_codes = {}
        job.shard_abort_requested = False
        await self._queue.put((-job.priority, time.time(), job.id))
        self._wake()
        if self._store:
            await asyncio.to_thread(self._store.upsert, job)

    def record_training_exit(
        self, job_id: str, reporting_worker_id: str, exit_code: int
    ) -> TrainingExitResult:
        """Учёт завершения subprocess: одна машина или последний ранг FSDP-группы."""
        job = self.get(job_id)
        if not job:
            return TrainingExitResult(False, None, None, [reporting_worker_id], [])
        if job.status not in (JobStatus.RUNNING, JobStatus.ASSIGNED):
            return TrainingExitResult(False, job, None, [reporting_worker_id], [])

        sw = max(1, int(job.shard_world_size or 1))
        if sw <= 1:
            old_backup = job.backup_worker_id
            job.backup_worker_id = None
            if job.cancel_requested:
                job.status = JobStatus.CANCELLED
                job.cancel_requested = False
            else:
                job.status = JobStatus.DONE if exit_code == 0 else JobStatus.FAILED
            self._wake()
            if self._store:
                self._store.upsert(job)
            return TrainingExitResult(
                True, job, old_backup, [reporting_worker_id], []
            )

        if reporting_worker_id not in job.shard_worker_ids:
            return TrainingExitResult(
                False, job, None, [reporting_worker_id], []
            )
        if reporting_worker_id in job.shard_exit_codes:
            return TrainingExitResult(
                False, job, None, [reporting_worker_id], []
            )

        job.shard_exit_codes[reporting_worker_id] = exit_code
        peers_to_cancel: list[str] = []
        if exit_code != 0:
            job.shard_abort_requested = True
            for oid in job.shard_worker_ids:
                if oid != reporting_worker_id and oid not in job.shard_exit_codes:
                    peers_to_cancel.append(oid)

        total = len(job.shard_worker_ids)
        if len(job.shard_exit_codes) < total:
            # Воркеры остаются BUSY, пока все ранги не завершились (FSDP).
            return TrainingExitResult(False, job, None, [], peers_to_cancel)

        agg = 0 if all(c == 0 for c in job.shard_exit_codes.values()) else 1
        old_backup = job.backup_worker_id
        job.backup_worker_id = None
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.cancel_requested = False
        elif agg == 0:
            job.status = JobStatus.DONE
        else:
            job.status = JobStatus.FAILED
        release_ids = list(job.shard_worker_ids)
        job.shard_worker_ids = []
        job.shard_exit_codes = {}
        job.shard_abort_requested = False
        self._wake()
        if self._store:
            self._store.upsert(job)
        return TrainingExitResult(True, job, old_backup, release_ids, [])

    async def _notify(self, job: Job, msg: dict):
        import json
        dead = []
        for ws in job.subscribers:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            job.subscribers.remove(ws)
