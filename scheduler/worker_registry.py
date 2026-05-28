import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

class WorkerStatus(str, Enum):
    IDLE         = "idle"
    BUSY         = "busy"
    DISCONNECTED = "disconnected"
    STALE        = "stale"    # был disconnected > 30 сек
    REMOVED      = "removed"  # был disconnected > 5 мин

STALE_TIMEOUT   = 30    # сек до перевода в STALE и замены задачи
REMOVED_TIMEOUT = 300   # сек до полного удаления из реестра

@dataclass
class Worker:
    id: str
    ws: object                              # WebSocket соединение
    gpu_name: str
    vram_total_mb: int
    vram_free_mb: int
    status: WorkerStatus = WorkerStatus.IDLE
    current_job_id: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    disconnected_at: Optional[float] = None
    last_heartbeat: float = field(default_factory=time.time)
    # Публичный IP/hostname для tcp:// rendezvous (rank 0 в FSDP-группе)
    advertise_host: Optional[str] = None

class WorkerRegistry:
    def __init__(self):
        self._workers: dict[str, Worker] = {}

    def register(self, worker: Worker):
        existing = self._workers.get(worker.id)
        if existing and existing.status in (WorkerStatus.DISCONNECTED, WorkerStatus.STALE):
            # Воркер переподключился — восстанавливаем
            worker.current_job_id = existing.current_job_id
            if not worker.advertise_host and existing.advertise_host:
                worker.advertise_host = existing.advertise_host
            print(f"[Registry] Воркер {worker.id} переподключился, job={worker.current_job_id}")
        self._workers[worker.id] = worker

    def disconnect(self, worker_id: str):
        w = self._workers.get(worker_id)
        if w:
            w.status = WorkerStatus.DISCONNECTED
            w.disconnected_at = time.time()
            w.ws = None

    def release_worker(self, worker_id: str) -> None:
        """Воркер завершил задачу или простаивает с активным соединением."""
        w = self._workers.get(worker_id)
        if not w:
            return
        w.status = WorkerStatus.IDLE
        w.current_job_id = None

    def get_idle(
        self,
        min_vram_mb: int = 0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> Optional[Worker]:
        """Возвращает лучшего свободного воркера под задачу."""
        ex = exclude_ids or set()
        candidates = [
            w for w in self._workers.values()
            if w.status == WorkerStatus.IDLE
            and w.ws is not None
            and w.vram_free_mb >= min_vram_mb
            and w.id not in ex
        ]
        if not candidates:
            return None
        # Выбираем воркера с наибольшим свободным VRAM
        return max(candidates, key=lambda w: w.vram_free_mb)

    def get_idle_n(
        self,
        n: int,
        min_vram_mb: int = 0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> Optional[List[Worker]]:
        """Возвращает n различных свободных воркеров или None."""
        if n <= 0:
            return []
        taken: Set[str] = set(exclude_ids or set())
        out: List[Worker] = []
        for _ in range(n):
            block = taken | {w.id for w in out}
            w = self.get_idle(min_vram_mb=min_vram_mb, exclude_ids=block)
            if w is None:
                return None
            out.append(w)
        return out

    def get(self, worker_id: str) -> Optional[Worker]:
        return self._workers.get(worker_id)

    def all(self) -> list[Worker]:
        return list(self._workers.values())

    def alive(self) -> list[Worker]:
        return [w for w in self._workers.values()
                if w.status not in (WorkerStatus.REMOVED,)]