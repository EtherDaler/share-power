import asyncio
import json
import time
from typing import TYPE_CHECKING

from .worker_registry import WorkerRegistry, WorkerStatus, STALE_TIMEOUT, REMOVED_TIMEOUT
from .job_queue import JobQueue, JobStatus

if TYPE_CHECKING:
    from .dispatcher import Dispatcher


class HealthMonitor:
    def __init__(self, registry: WorkerRegistry, queue: JobQueue, dispatcher: "Dispatcher"):
        self.registry = registry
        self.queue = queue
        self.dispatcher = dispatcher
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._stale_check_loop())

    async def _heartbeat_loop(self):
        """Пингует всех воркеров каждые 10 секунд."""
        while self._running:
            await asyncio.sleep(10)
            for worker in self.registry.alive():
                if worker.ws is None:
                    continue
                try:
                    await worker.ws.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    self.registry.disconnect(worker.id)
                    await self._handle_worker_failure(worker.id)

    async def _stale_check_loop(self):
        """Каждые 5 секунд проверяет дисконнекченных воркеров."""
        while self._running:
            await asyncio.sleep(5)
            now = time.time()
            for worker in self.registry.all():

                if worker.status == WorkerStatus.DISCONNECTED:
                    elapsed = now - (worker.disconnected_at or now)

                    if elapsed > STALE_TIMEOUT:
                        worker.status = WorkerStatus.STALE
                        print(f"[Monitor] Воркер {worker.id} → STALE")
                        # Задача была на этом воркере — ищем замену
                        if worker.current_job_id:
                            await self._replace_worker(worker)

                if worker.status == WorkerStatus.STALE:
                    elapsed = now - (worker.disconnected_at or now)
                    if elapsed > REMOVED_TIMEOUT:
                        worker.status = WorkerStatus.REMOVED
                        print(f"[Monitor] Воркер {worker.id} → REMOVED")

    async def _handle_worker_failure(self, worker_id: str):
        """Вызывается сразу при обрыве соединения."""
        worker = self.registry.get(worker_id)
        if not worker or not worker.current_job_id:
            return
        job = self.queue.get(worker.current_job_id)
        if not job:
            return
        if max(1, int(getattr(job, "shard_world_size", 1) or 1)) > 1:
            return
        # Не помечаем FAILED сразу: воркер может переподключиться за STALE_TIMEOUT
        # и получить run_job снова (см. coordinator worker_ws reconnect).
        print(
            f"[Monitor] Воркер {worker_id} отключён во время задачи {job.id}, "
            f"ждём reconnect ({STALE_TIMEOUT}s)…"
        )

    async def _abort_shard_job_reschedule(self, job, failed_worker_id: str) -> None:
        """Падение участника FSDP-группы: отмена остальных и возврат задачи в очередь."""
        peers = list(getattr(job, "shard_worker_ids", []) or [])
        for wid in peers:
            if wid == failed_worker_id:
                continue
            w = self.registry.get(wid)
            if w and w.ws:
                try:
                    await w.ws.send_text(
                        json.dumps({"type": "cancel_job", "job_id": job.id})
                    )
                except Exception:
                    pass
        await self.queue.reschedule(job)
        self.registry.release_worker(failed_worker_id)
        print(f"[Monitor] FSDP задача {job.id} прервана (воркер {failed_worker_id}), reschedule")

    async def _replace_worker(self, failed_worker):
        """Находит замену для задачи упавшего воркера."""
        job = self.queue.get(failed_worker.current_job_id)
        if not job or job.status in (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.DEAD):
            self.registry.release_worker(failed_worker.id)
            return

        self.registry.release_worker(failed_worker.id)

        if max(1, int(getattr(job, "shard_world_size", 1) or 1)) > 1:
            await self._abort_shard_job_reschedule(job, failed_worker.id)
            return

        # Есть ли уже назначенная замена?
        if job.backup_worker_id:
            backup = self.registry.get(job.backup_worker_id)
            if (
                backup
                and backup.status == WorkerStatus.IDLE
                and backup.ws is not None
            ):
                print(f"[Monitor] Передаём задачу {job.id} резервному воркеру {backup.id}")
                await self._assign_job_to_worker(job, backup, checkpoint=job.last_checkpoint)
                return

        # Ищем любого свободного воркера
        replacement = self.registry.get_idle(
            exclude_ids=self.dispatcher.reserved_backup_ids_except_job(job.id),
        )
        if replacement:
            print(f"[Monitor] Заменяем воркер {failed_worker.id} → {replacement.id}")
            await self._assign_job_to_worker(job, replacement, checkpoint=job.last_checkpoint)
        else:
            # Свободных нет — ставим обратно в очередь
            print(f"[Monitor] Нет свободных воркеров, задача {job.id} → обратно в очередь")
            await self.queue.reschedule(job)

    async def _assign_job_to_worker(self, job, worker, checkpoint=None):
        """Отправляет задачу воркеру, опционально с чекпоинтом."""
        from .job_queue import JobStatus
        job.backup_worker_id = None
        job.status = JobStatus.ASSIGNED
        job.assigned_worker_id = worker.id
        worker.status = WorkerStatus.BUSY
        worker.current_job_id = job.id

        msg = {
            "type": "run_job",
            "job_id": job.id,
            "script": job.script,
        }
        if checkpoint:
            msg["checkpoint_path"] = checkpoint
            msg["resume"] = True
            msg["start_step"] = job.last_step or 0

        for w in self.registry.all():
            if w.id != worker.id and w.current_job_id == job.id:
                w.current_job_id = None

        await worker.ws.send_text(json.dumps(msg))
        job.status = JobStatus.RUNNING
        print(f"[Monitor] Задача {job.id} запущена на воркере {worker.id}"
              + (f" с чекпоинта {checkpoint}" if checkpoint else ""))
        if max(1, int(getattr(job, "shard_world_size", 1) or 1)) <= 1:
            await self.dispatcher.assign_backup_for_running_job(job)
        await self.queue.save_job(job)