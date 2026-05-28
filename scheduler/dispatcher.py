from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Optional, Set

from .job_queue import Job, JobQueue, JobStatus
from .worker_registry import Worker, WorkerRegistry, WorkerStatus

class Dispatcher:
    """
    Постоянно крутится в фоне.
    Берёт задачи из очереди и назначает свободным воркерам.
    """
    def __init__(self, queue: JobQueue, registry: WorkerRegistry):
        self.queue = queue
        self.registry = registry
        self._new_job_event = asyncio.Event()   # сигнал: появилась новая задача
        self._new_worker_event = asyncio.Event() # сигнал: появился новый воркер

    def notify_new_job(self):
        self._new_job_event.set()

    def notify_new_worker(self):
        self._new_worker_event.set()

    def wake(self) -> None:
        """Разбудить цикл диспетчера (новая задача или освободился воркер)."""
        self._new_job_event.set()
        self._new_worker_event.set()

    def reserved_backup_ids_except_job(self, except_job_id: Optional[str]) -> Set[str]:
        """ID воркеров, зарезервированных как backup для других RUNNING-задач."""
        out: Set[str] = set()
        for j in self.queue.all():
            if j.status != JobStatus.RUNNING or not j.backup_worker_id:
                continue
            if except_job_id is not None and j.id == except_job_id:
                continue
            out.add(j.backup_worker_id)
        return out

    def _vram_per_rank(self, job: Job) -> int:
        base = self._estimate_vram(job)
        sw = max(1, int(getattr(job, "shard_world_size", 1) or 1))
        if sw <= 1:
            return base
        return max(0, base // sw)

    async def run(self):
        print("[Dispatcher] Запущен")
        while True:
            job = await self.queue.pop()

            if job is None:
                # Очередь пуста — ждём нового задания
                self._new_job_event.clear()
                await self._new_job_event.wait()
                continue

            sw = max(1, int(getattr(job, "shard_world_size", 1) or 1))
            if sw > 1:
                await self._dispatch_fsdp_group(job, sw)
                continue

            required_vram = self._estimate_vram(job)

            # Ищем подходящего воркера
            worker = self.registry.get_idle(
                min_vram_mb=required_vram,
                exclude_ids=self.reserved_backup_ids_except_job(None),
            )
            if worker is None:
                # Воркеров нет — возвращаем задачу в очередь и ждём
                await self.queue.push(job)
                self._new_worker_event.clear()
                print(f"[Dispatcher] Нет воркеров для задачи {job.id}, ждём...")
                await self._new_worker_event.wait()
                continue

            # Назначаем задачу
            job.status = JobStatus.ASSIGNED
            job.assigned_worker_id = worker.id
            worker.status = WorkerStatus.BUSY
            worker.current_job_id = job.id

            for w in self.registry.all():
                if w.id != worker.id and w.current_job_id == job.id:
                    w.current_job_id = None

            msg = {
                "type": "run_job",
                "job_id": job.id,
                "script": job.script,
            }
            if job.last_checkpoint:
                msg["resume"] = True
                msg["checkpoint_path"] = job.last_checkpoint
                msg["start_step"] = job.last_step or 0

            await worker.ws.send_text(json.dumps(msg))
            job.status = JobStatus.RUNNING
            print(f"[Dispatcher] Задача {job.id} → воркер {worker.id} ({worker.gpu_name})")
            await self.assign_backup_for_running_job(job)
            await self.queue.save_job(job)

    async def _dispatch_fsdp_group(self, job: Job, sw: int) -> None:
        """Назначает sw воркеров одной FSDP-задаче (rank 0 = tcp master)."""
        required_vram = self._vram_per_rank(job)
        ex = self.reserved_backup_ids_except_job(None)
        workers = self.registry.get_idle_n(
            sw, min_vram_mb=required_vram, exclude_ids=ex
        )
        if not workers:
            await self.queue.push(job)
            self._new_worker_event.clear()
            print(
                f"[Dispatcher] Недостаточно свободных воркеров для FSDP "
                f"(нужно {sw}) задача {job.id}, ждём..."
            )
            await self._new_worker_event.wait()
            return

        workers_sorted = sorted(workers, key=lambda w: w.id)
        master_addr = (
            (workers_sorted[0].advertise_host or "").strip()
            or os.getenv("DISTGPU_RENDEZVOUS_HOST", "").strip()
        )
        if not master_addr:
            print(
                f"[Dispatcher] FSDP {job.id}: у воркера {workers_sorted[0].id} "
                f"нет advertise_host и не задан DISTGPU_RENDEZVOUS_HOST — отложено"
            )
            await self.queue.push(job)
            self.notify_new_job()
            await asyncio.sleep(0.3)
            return

        master_port = random.randint(29600, 29999)
        job.shard_worker_ids = [w.id for w in workers_sorted]
        job.assigned_worker_id = workers_sorted[0].id
        job.backup_worker_id = None
        job.shard_exit_codes = {}
        job.shard_abort_requested = False

        ws_ids = {w.id for w in workers_sorted}
        for w in self.registry.all():
            if w.id not in ws_ids and w.current_job_id == job.id:
                w.current_job_id = None

        for w in workers_sorted:
            w.status = WorkerStatus.BUSY
            w.current_job_id = job.id

        job.status = JobStatus.ASSIGNED
        for rank, w in enumerate(workers_sorted):
            msg = {
                "type": "run_job",
                "job_id": job.id,
                "script": job.script,
                "distributed": {
                    "rank": rank,
                    "world_size": sw,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "local_rank": 0,
                },
            }
            if getattr(job, "pipeline_enabled", False) and getattr(
                job, "pipeline_config_yaml", None
            ):
                msg["pipeline_enabled"] = True
                msg["pipeline_config_yaml"] = job.pipeline_config_yaml
                msg["pipeline_stage_idx"] = rank
            if job.last_checkpoint:
                msg["resume"] = True
                msg["checkpoint_path"] = job.last_checkpoint
                msg["start_step"] = job.last_step or 0
            await w.ws.send_text(json.dumps(msg))

        job.status = JobStatus.RUNNING
        print(
            f"[Dispatcher] FSDP {job.id}: {sw} рангов, rendezvous tcp://"
            f"{master_addr}:{master_port}"
        )
        await self.queue.save_job(job)

    def _standby_sync(self, job: Job, primary_id: str) -> dict:
        return {
            "type": "sync",
            "has_active_job": True,
            "job_id": job.id,
            "current_worker_id": primary_id,
            "current_step": job.last_step or 0,
            "checkpoint_path": job.last_checkpoint,
            "role": "standby",
        }

    def _pick_idle_backup(
        self, exclude: Set[str], min_vram_mb: int
    ) -> Optional[Worker]:
        cands = [
            w
            for w in self.registry.all()
            if w.id not in exclude
            and w.status == WorkerStatus.IDLE
            and w.vram_free_mb >= min_vram_mb
        ]
        if not cands:
            return None
        return max(cands, key=lambda w: w.vram_free_mb)

    async def assign_backup_for_running_job(self, job: Job) -> None:
        """Назначает резервного воркера (hot standby) и шлёт ему sync."""
        if job.status != JobStatus.RUNNING:
            return
        if max(1, int(getattr(job, "shard_world_size", 1) or 1)) > 1:
            return
        primary_id = job.assigned_worker_id
        if not primary_id:
            return
        req = self._estimate_vram(job)
        exclude = {primary_id} | self.reserved_backup_ids_except_job(job.id)
        backup = self._pick_idle_backup(exclude, req)
        job.backup_worker_id = backup.id if backup else None
        if backup and backup.ws:
            try:
                await backup.ws.send_text(
                    json.dumps(self._standby_sync(job, primary_id))
                )
                print(f"[Dispatcher] Резерв для {job.id} → {backup.id}")
            except Exception as e:
                print(f"[Dispatcher] Не удалось уведомить резерв {backup.id}: {e}")

    def _estimate_vram(self, job) -> int:
        """
        Простая эвристика: ищем в скрипте подсказки о размере модели.
        В будущем — пользователь указывает явно при загрузке.
        """
        script_lower = job.script.lower()
        if "llama" in script_lower or "7b" in script_lower:
            return 6000
        if "bert" in script_lower:
            return 2000
        return 0  # нет ограничений
