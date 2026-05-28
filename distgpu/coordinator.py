"""
DistGPU coordinator: HTTP + WS браузера (PORT), отдельный ASGI для воркеров (WORKER_PORT).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distgpu.executor import notebook_to_ddp_script
from scheduler.dispatcher import Dispatcher
from scheduler.health_monitor import HealthMonitor
from scheduler.job_queue import Job, JobQueue, JobStatus
from scheduler.state_store import JobStateStore
from scheduler.worker_registry import Worker, WorkerRegistry, WorkerStatus

STATIC_DIR = Path(__file__).resolve().parent / "static"

_DATA_DIR = Path(
    os.getenv("DISTGPU_DATA_DIR", str(_REPO / ".distgpu_data"))
)
_state_store = JobStateStore(_DATA_DIR / "coordinator.sqlite")
queue = JobQueue(store=_state_store)
registry = WorkerRegistry()
dispatch = Dispatcher(queue, registry)
monitor = HealthMonitor(registry, queue, dispatch)
queue.set_on_changed(dispatch.wake)


def _worker_token() -> str:
    return os.getenv("WORKER_TOKEN") or os.getenv("TOKEN") or "secret-token-123"


def _run_job_payload(job: Job) -> dict:
    msg = {
        "type": "run_job",
        "job_id": job.id,
        "script": job.script,
    }
    if job.last_checkpoint:
        msg["resume"] = True
        msg["checkpoint_path"] = job.last_checkpoint
        msg["start_step"] = job.last_step or 0
    return msg


async def _send_run_job(ws: WebSocket, job: Job) -> None:
    await ws.send_text(json.dumps(_run_job_payload(job)))


async def _broadcast_log(job: Optional[Job], text: str) -> None:
    if not job:
        return
    job.logs.append(text)
    await queue.maybe_batch_persist_logs(job)
    blob = json.dumps({"type": "log", "job_id": job.id, "text": text})
    dead: list[WebSocket] = []
    for sub in job.subscribers:
        try:
            await sub.send_text(blob)
        except Exception:
            dead.append(sub)
    for d in dead:
        if d in job.subscribers:
            job.subscribers.remove(d)


async def _notify_backup_standby_off(backup_id: Optional[str]) -> None:
    if not backup_id:
        return
    ob = registry.get(backup_id)
    if ob and ob.ws:
        try:
            await ob.ws.send_text(
                json.dumps({"type": "sync", "has_active_job": False})
            )
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await queue.restore_from_store()
    n_jobs = len(queue.all())
    if n_jobs:
        print(
            f"[Coordinator] Задач в хранилище: {n_jobs} "
            f"(каталог данных: {_DATA_DIR})"
        )
    await monitor.start()
    asyncio.create_task(dispatch.run())

    worker_host = os.getenv("WORKER_HOST", os.getenv("HOST", "0.0.0.0"))
    worker_port = int(os.getenv("WORKER_PORT", "8765"))
    log_level = os.getenv("LOG_LEVEL", "info")
    wcfg = uvicorn.Config(
        worker_app,
        host=worker_host,
        port=worker_port,
        log_level=log_level,
    )
    wsrv = uvicorn.Server(wcfg)
    wtask = asyncio.create_task(wsrv.serve())
    print(
        f"[Coordinator] Воркеры: ws://{worker_host}:{worker_port}/worker"
    )
    try:
        yield
    finally:
        wsrv.should_exit = True
        try:
            await asyncio.wait_for(wtask, timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            wtask.cancel()
            try:
                await wtask
            except asyncio.CancelledError:
                pass


app = FastAPI(lifespan=lifespan)

worker_router = APIRouter()


@worker_router.websocket("/worker")
async def worker_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    worker_id: Optional[str] = None
    try:
        raw = await ws.receive_text()
        reg = json.loads(raw)
        if reg.get("token") != _worker_token():
            await ws.close(code=4001, reason="Неверный токен")
            return

        wid = reg.get("worker_id")
        if isinstance(wid, str) and len(wid) == 8 and wid.isalnum():
            worker_id = wid
        else:
            worker_id = str(uuid.uuid4())[:8]

        gpu = reg.get("gpu") or {}
        name = gpu.get("name") or (
            "N/A" if gpu.get("available") is False else "unknown"
        )
        vram_t = int(gpu.get("vram_total_mb") or 0)
        vram_f = int(gpu.get("vram_free_mb") or vram_t or 0)

        adv_raw = reg.get("advertise_host") or reg.get("host")
        if isinstance(adv_raw, str):
            adv = adv_raw.strip() or None
        else:
            adv = None

        worker = Worker(
            id=worker_id,
            ws=ws,
            gpu_name=name,
            vram_total_mb=vram_t,
            vram_free_mb=vram_f,
            status=WorkerStatus.IDLE,
            advertise_host=adv,
        )
        registry.register(worker)

        job_for_reconnect = None
        if worker.current_job_id:
            job_for_reconnect = queue.get(worker.current_job_id)
            if job_for_reconnect and job_for_reconnect.status in (
                JobStatus.RUNNING,
                JobStatus.ASSIGNED,
            ):
                sh = max(1, int(getattr(job_for_reconnect, "shard_world_size", 1) or 1))
                if sh > 1:
                    if worker.id in getattr(
                        job_for_reconnect, "shard_worker_ids", []
                    ):
                        worker.status = WorkerStatus.BUSY
                elif job_for_reconnect.assigned_worker_id == worker.id:
                    worker.status = WorkerStatus.BUSY

        dispatch.notify_new_worker()
        print(f"[Coordinator] Воркер {worker_id} подключён: {name}")

        if job_for_reconnect:
            assigned = job_for_reconnect.assigned_worker_id
            sync = {
                "type": "sync",
                "has_active_job": True,
                "job_id": job_for_reconnect.id,
                "current_worker_id": assigned,
                "current_step": job_for_reconnect.last_step or 0,
                "checkpoint_path": job_for_reconnect.last_checkpoint,
                "role": (
                    "standby"
                    if assigned and assigned != worker.id
                    else "active"
                ),
            }
            await ws.send_text(json.dumps(sync))
            sh = max(1, int(getattr(job_for_reconnect, "shard_world_size", 1) or 1))
            if (
                job_for_reconnect.status in (JobStatus.RUNNING, JobStatus.ASSIGNED)
                and sh <= 1
            ):
                if assigned == worker.id:
                    job_for_reconnect.status = JobStatus.RUNNING
                    await _send_run_job(ws, job_for_reconnect)
                    await queue.save_job(job_for_reconnect)

        while True:
            raw_msg = await ws.receive_text()
            msg = json.loads(raw_msg)

            if msg["type"] == "pong":
                continue

            if msg["type"] == "log":
                j = queue.get(msg["job_id"])
                await _broadcast_log(j, msg["text"])

            elif msg["type"] == "checkpoint":
                j = queue.get(msg["job_id"])
                if j:
                    j.last_checkpoint = msg["path"]
                    j.last_step = int(msg["step"])
                    await queue.save_job(j)
                    print(
                        f"[Coordinator] Чекпоинт {msg['job_id']}: шаг {msg['step']}"
                    )

            elif msg["type"] == "job_done":
                jid = msg["job_id"]
                code = int(msg.get("exit_code", 1))
                ex = queue.record_training_exit(jid, worker_id, code)
                for peer in ex.peers_to_cancel:
                    pw = registry.get(peer)
                    if pw and pw.ws:
                        try:
                            await pw.ws.send_text(
                                json.dumps({"type": "cancel_job", "job_id": jid})
                            )
                        except Exception:
                            pass
                for rid in ex.release_worker_ids:
                    registry.release_worker(rid)
                if ex.finalized and ex.job:
                    await _notify_backup_standby_off(ex.old_backup)
                    await _broadcast_log(
                        ex.job,
                        f"\n--- завершение обучения (exit {code}) ---\n",
                    )
                print(f"[Coordinator] job_done {jid} worker={worker_id} code={code}")

    except WebSocketDisconnect:
        print(f"[Coordinator] Воркер {worker_id} отключён (WS)")
    except Exception as e:
        print(f"[Coordinator] Ошибка worker_ws: {e}")
    finally:
        if worker_id:
            w = registry.get(worker_id)
            had_running_job = (
                w
                and w.current_job_id
                and (j := queue.get(w.current_job_id))
                and j.status in (JobStatus.RUNNING, JobStatus.ASSIGNED)
            )
            registry.disconnect(worker_id)
            if had_running_job:
                await monitor._handle_worker_failure(worker_id)
            dispatch.wake()


worker_app = FastAPI()
worker_app.include_router(worker_router)


def _worker_is_online(w: Worker) -> bool:
    return w.ws is not None and w.status not in (
        WorkerStatus.REMOVED,
        WorkerStatus.DISCONNECTED,
        WorkerStatus.STALE,
    )


@app.get("/api/health")
def api_health():
    """Проверка, что координатор отвечает (удобно с VPS: curl http://127.0.0.1:8000/api/health)."""
    return {"ok": True, "service": "distgpu-coordinator"}


@app.get("/api/workers")
def api_workers():
    workers_out = []
    online = idle = busy = 0
    total_vram = free_vram_idle = 0

    for w in registry.all():
        if w.status == WorkerStatus.REMOVED:
            continue
        is_online = _worker_is_online(w)
        workers_out.append(
            {
                "id": w.id,
                "online": is_online,
                "gpu": {
                    "name": w.gpu_name,
                    "available": w.vram_total_mb > 0,
                    "vram_total_mb": w.vram_total_mb,
                    "vram_free_mb": w.vram_free_mb,
                },
                "status": w.status.value,
                "current_job_id": w.current_job_id,
            }
        )
        if not is_online:
            continue
        online += 1
        total_vram += w.vram_total_mb
        if w.status == WorkerStatus.IDLE:
            idle += 1
            free_vram_idle += w.vram_free_mb
        elif w.status == WorkerStatus.BUSY:
            busy += 1

    return {
        "summary": {
            "online": online,
            "idle": idle,
            "busy": busy,
            "total_vram_mb": total_vram,
            "available_vram_mb": free_vram_idle,
        },
        "workers": workers_out,
    }


@app.post("/api/submit")
async def submit_job(
    file: UploadFile = File(...),
    shard_world_size: int = Form(1),
    pipeline_enabled: str = Form("0"),
    pipeline_config: Optional[UploadFile] = File(None),
):
    nb_bytes = await file.read()
    script = notebook_to_ddp_script(nb_bytes)
    job_id = str(uuid.uuid4())[:8]
    sw = max(1, min(int(shard_world_size), 32))
    pe = str(pipeline_enabled).strip().lower() in ("1", "true", "yes", "on")
    yaml_text: Optional[str] = None
    if pipeline_config is not None and (
        getattr(pipeline_config, "filename", None) or ""
    ).strip():
        raw = await pipeline_config.read()
        yaml_text = raw.decode("utf-8", errors="replace")
    job = Job(
        id=job_id,
        script=script,
        owner_id="anon",
        shard_world_size=sw,
        pipeline_enabled=bool(pe and yaml_text),
        pipeline_config_yaml=yaml_text,
    )
    await queue.push(job)
    return {
        "job_id": job_id,
        "shard_world_size": sw,
        "pipeline_enabled": job.pipeline_enabled,
    }


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    j = queue.get(job_id)
    if not j:
        return {"error": "не найдено"}
    return {
        "status": j.status.value,
        "logs": j.logs,
        "assigned_worker_id": j.assigned_worker_id,
        "backup_worker_id": j.backup_worker_id,
        "cancel_requested": j.cancel_requested,
        "shard_world_size": j.shard_world_size,
        "shard_worker_ids": list(j.shard_worker_ids),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str):
    job = queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="задача не найдена")
    if job.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="отмена возможна только для задач в статусе running",
        )
    targets: list[str] = []
    if max(1, int(getattr(job, "shard_world_size", 1) or 1)) > 1:
        targets = list(getattr(job, "shard_worker_ids", []) or [])
        if not targets:
            raise HTTPException(
                status_code=409, detail="нет списка воркеров FSDP-группы"
            )
    else:
        wid = job.assigned_worker_id
        if not wid:
            raise HTTPException(status_code=409, detail="нет назначенного воркера")
        targets = [wid]

    job.cancel_requested = True
    any_ok = False
    last_err: Optional[Exception] = None
    for tw in targets:
        w = registry.get(tw)
        if not w or not w.ws:
            continue
        try:
            await w.ws.send_text(
                json.dumps({"type": "cancel_job", "job_id": job.id})
            )
            any_ok = True
        except Exception as e:
            last_err = e
    if not any_ok:
        job.cancel_requested = False
        raise HTTPException(
            status_code=503,
            detail=(
                str(last_err)
                if last_err
                else "не удалось отправить cancel ни одному воркеру"
            ),
        ) from last_err
    await _broadcast_log(job, "\n[сервер] Запрошена отмена задачи (cancel_job).\n")
    await queue.save_job(job)
    return {"ok": True, "job_id": job.id}


@app.websocket("/logs/{job_id}")
async def log_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    job = queue.get(job_id)
    if not job:
        await ws.close()
        return
    job.subscribers.append(ws)
    try:
        for line in job.logs:
            await ws.send_text(
                json.dumps({"type": "log", "job_id": job_id, "text": line})
            )
        while job.status not in (
            JobStatus.DONE,
            JobStatus.FAILED,
            JobStatus.DEAD,
            JobStatus.CANCELLED,
        ):
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in job.subscribers:
            job.subscribers.remove(ws)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsProactorEventLoopPolicy()
        )
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    worker_port = int(os.getenv("WORKER_PORT", "8765"))
    log_level = os.getenv("LOG_LEVEL", "info")
    bind = host if host not in ("0.0.0.0", "::") else "0.0.0.0"
    print("[Coordinator] Запуск…")
    print(f"[Coordinator] UI/API:     http://{bind}:{port}/")
    print(f"[Coordinator] Health:     http://{bind}:{port}/api/health")
    print(f"[Coordinator] WS логов:   ws://{bind}:{port}/logs/{{job_id}}")
    print(f"[Coordinator] WS воркеры: ws://{bind}:{worker_port}/worker")
    if host in ("127.0.0.1", "localhost", "::1"):
        print(
            "[Coordinator] ВНИМАНИЕ: HOST=127.0.0.1 — с других машин веб не откроется. "
            "На VPS задайте: export HOST=0.0.0.0"
        )
    elif host == "0.0.0.0":
        print(
            "[Coordinator] С браузера откройте http://<публичный-IP-VPS>:{port}/ "
            "(не 127.0.0.1). Откройте порты {port} и {wp} в firewall / security group.".format(
                port=port, wp=worker_port
            )
        )
    uvicorn.run(app, host=host, port=port, log_level=log_level)
