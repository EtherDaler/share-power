#!/usr/bin/env python3
"""
DistGPU worker — один файл для распространения.

При первом запуске создаёт .venv рядом со скриптом, ставит websockets и PyTorch,
затем подключается к координатору по WebSocket.

Пример:
  set SERVER_URL=ws://192.168.1.10:8765/worker
  set WORKER_TOKEN=secret-token-123
  python worker.py

Или:
  python worker.py --server ws://host:8765/worker --token secret-token-123

Переменные: SERVER_URL, WORKER_TOKEN / TOKEN, DISTGPU_TORCH (cu124, nightly-cu128 для RTX 50xx, …),
DISTGPU_ADVERTISE_HOST, CHECKPOINT_EVERY, PYTHON (базовый Python для venv),
DISTGPU_DATA_ROOT (каталог данных; по умолчанию worker/.distgpu_runtime на диске worker.py).

RTX 5060/5070/5080/5090 (sm_120) требуют nightly-cu128 — worker подберёт пресет сам по nvidia-smi.

Данные задач (CIFAR, output/, чекпоинты) не пишутся в репозиторий и не на C: — только в DISTGPU_DATA_ROOT.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

_BOOTSTRAP_ENV = "DISTGPU_WORKER_BOOTSTRAPPED"
_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_DIR = _SCRIPT_DIR / ".venv"
_RUNTIME_ROOT: Optional[Path] = None


def runtime_root() -> Path:
    """Корень данных воркера — на том же диске, что и worker.py (не %USERPROFILE% на C:)."""
    global _RUNTIME_ROOT
    if _RUNTIME_ROOT is not None:
        return _RUNTIME_ROOT
    custom = os.environ.get("DISTGPU_DATA_ROOT", "").strip()
    if custom:
        root = Path(custom).expanduser().resolve()
    else:
        root = (_SCRIPT_DIR / ".distgpu_runtime").resolve()
    root.mkdir(parents=True, exist_ok=True)
    _RUNTIME_ROOT = root
    return root


def _worker_id_file() -> Path:
    return runtime_root() / "worker_id"
_REQUIREMENTS = ("websockets>=12.0", "numpy>=1.26", "pillow>=10.0")

_TORCH_INDEX = {
    "cpu": None,
    "mps": None,
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "nightly-cu128": "https://download.pytorch.org/whl/nightly/cu128",
}


def _venv_python() -> Path:
    if sys.platform == "win32":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"


def _default_torch_preset() -> str:
    if sys.platform == "darwin":
        return "mps"
    return "cu124"


def _nvidia_gpu_names_and_caps() -> list[tuple[str, Optional[float]]]:
    """Имена GPU и compute capability (например 12.0 для RTX 5060)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if not out or out.returncode != 0:
        return []
    rows: list[tuple[str, Optional[float]]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        name = parts[0]
        cap: Optional[float] = None
        if len(parts) > 1:
            try:
                cap = float(parts[1])
            except ValueError:
                pass
        rows.append((name, cap))
    return rows


def _gpu_needs_blackwell_torch() -> bool:
    for name, cap in _nvidia_gpu_names_and_caps():
        upper = name.upper()
        if "RTX 50" in upper or "RTX 60" in upper:
            return True
        if cap is not None and cap >= 12.0:
            return True
    return False


def _resolve_torch_preset() -> str:
    explicit = os.environ.get("DISTGPU_TORCH", "").strip()
    if explicit:
        return explicit
    if _gpu_needs_blackwell_torch():
        print(
            "[worker] GPU Blackwell / sm_12.x (например RTX 5060) — "
            "пресет PyTorch: nightly-cu128"
        )
        return "nightly-cu128"
    return _default_torch_preset()


def _find_base_python() -> str:
    if env_py := os.environ.get("PYTHON", "").strip():
        if Path(env_py).is_file():
            return env_py
        raise SystemExit(f"[worker] PYTHON указывает на несуществующий файл: {env_py}")

    if sys.platform == "win32":
        for ver in ("3.12", "3.11", "3.10"):
            try:
                out = subprocess.run(
                    ["py", f"-{ver}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            except FileNotFoundError:
                break
    for name in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        if shutil.which(name):
            return name
    raise SystemExit(
        "[worker] Не найден Python 3.10+. Установите Python или задайте PYTHON=..."
    )


def _run_pip(py: Path, *args: str) -> None:
    cmd = [str(py), "-m", "pip", *args]
    print("[worker] ", " ".join(cmd))
    subprocess.check_call(cmd)


def _deps_ok(py: Path) -> bool:
    code = (
        "import websockets, torch, torchvision, numpy\n"
        "import sys; sys.exit(0)"
    )
    return subprocess.run([str(py), "-c", code], capture_output=True).returncode == 0


_TORCH_STACK = ("torch", "torchvision")


def _torch_cuda_mismatch(py: Path) -> bool:
    """True, если GPU sm_12.x, а установленный torch её не поддерживает."""
    script = r"""
import sys
try:
    import torch
except ImportError:
    sys.exit(0)
if not torch.cuda.is_available():
    sys.exit(0)
major, minor = torch.cuda.get_device_capability(0)
if major < 12:
    sys.exit(0)
arch_fn = getattr(torch.cuda, "get_arch_list", None)
if arch_fn is None:
    sys.exit(1)
archs = arch_fn() or []
ok = any("120" in a or "12.0" in a for a in archs)
sys.exit(0 if ok else 1)
"""
    r = subprocess.run([str(py), "-c", script], capture_output=True)
    return r.returncode == 1


def _install_torch(py: Path, preset: str, *, reinstall: bool = False) -> None:
    """torch + torchvision с одного индекса (версии должны совпадать)."""
    if reinstall:
        _run_pip(py, "uninstall", "-y", *_TORCH_STACK)

    custom_url = os.environ.get("DISTGPU_PYTORCH_INDEX_URL", "").strip()
    extra = os.environ.get("DISTGPU_PYTORCH_EXTRA", "").strip().split()

    if custom_url:
        _run_pip(py, "install", *extra, *_TORCH_STACK, "--index-url", custom_url)
        return

    if preset == "cpu":
        _run_pip(py, "install", *_TORCH_STACK)
        return
    if preset == "mps":
        _run_pip(py, "install", *_TORCH_STACK)
        return
    if preset == "nightly-cu128":
        _run_pip(
            py,
            "install",
            *extra,
            "--pre",
            *_TORCH_STACK,
            "--index-url",
            _TORCH_INDEX["nightly-cu128"],
        )
        return

    url = _TORCH_INDEX.get(preset)
    if not url:
        raise SystemExit(
            f"[worker] Неизвестный DISTGPU_TORCH={preset!r}. "
            f"Допустимо: {', '.join(sorted(_TORCH_INDEX))}, cpu, mps"
        )
    _run_pip(py, "install", *extra, *_TORCH_STACK, "--index-url", url)


def _print_cuda_fix_hint(preset: str) -> None:
    print(
        "[worker] Для RTX 5060/50xx переустановите PyTorch:\n"
        f"  set DISTGPU_TORCH={preset}\n"
        f"  python worker.py --reinstall-torch\n"
        "  (или удалите папку worker\\.venv и запустите worker.py снова)"
    )


def _ensure_torchvision_if_missing() -> None:
    """После перезапуска из venv: torchvision мог отсутствовать в старом .venv."""
    if os.environ.get(_BOOTSTRAP_ENV) != "1":
        return
    vpy = _venv_python()
    if not vpy.is_file() or _deps_ok(vpy):
        return
    preset = _resolve_torch_preset()
    print(f"[worker] Нет torchvision — ставим с пресетом {preset}…")
    _install_torch(vpy, preset, reinstall=False)


def ensure_environment(
    *, install_only: bool = False, reinstall_torch: bool = False
) -> None:
    """Создаёт venv и ставит зависимости при необходимости; перезапускает себя из venv."""
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        _ensure_torchvision_if_missing()
        return

    vpy = _venv_python()
    preset = _resolve_torch_preset()

    if vpy.is_file() and _deps_ok(vpy):
        if reinstall_torch or _torch_cuda_mismatch(vpy):
            print(f"[worker] Переустановка PyTorch/torchvision (пресет {preset})…")
            _install_torch(vpy, preset, reinstall=True)
            if _torch_cuda_mismatch(vpy):
                _print_cuda_fix_hint(preset)
                raise SystemExit(
                    "[worker] PyTorch всё ещё без поддержки sm_120. "
                    "Попробуйте nightly-cu128 вручную."
                )
        if Path(sys.executable).resolve() != vpy.resolve():
            _reexec(vpy)
        return

    if vpy.is_file():
        _run_pip(vpy, "install", *_REQUIREMENTS)
        if not _deps_ok(vpy):
            print(f"[worker] Доустановка PyTorch/torchvision (пресет {preset})…")
            _install_torch(vpy, preset, reinstall=reinstall_torch)
        if not _deps_ok(vpy):
            raise SystemExit("[worker] Не удалось установить зависимости в существующий .venv")
        if Path(sys.executable).resolve() != vpy.resolve():
            _reexec(vpy)
        return

    base = _find_base_python()
    subprocess.check_call(
        [base, "-c", "import sys; assert sys.version_info >= (3, 10)"],
    )
    print(f"[worker] Базовый Python: {base}")

    if not _VENV_DIR.is_dir():
        print(f"[worker] Создаю venv: {_VENV_DIR}")
        subprocess.check_call([base, "-m", "venv", str(_VENV_DIR)])

    vpy = _venv_python()
    if not vpy.is_file():
        raise SystemExit(f"[worker] Не найден интерпретатор venv: {vpy}")

    _run_pip(vpy, "install", "-U", "pip", "wheel", "setuptools")
    _run_pip(vpy, "install", *_REQUIREMENTS)

    print(f"[worker] Установка PyTorch + torchvision (пресет {preset})…")
    _install_torch(vpy, preset, reinstall=reinstall_torch)

    if not _deps_ok(vpy):
        raise SystemExit("[worker] Проверка зависимостей не прошла после установки")
    if _torch_cuda_mismatch(vpy):
        _print_cuda_fix_hint(preset)
        raise SystemExit(
            "[worker] Установленный PyTorch не поддерживает вашу GPU (sm_120). "
            "Задайте DISTGPU_TORCH=nightly-cu128 и --reinstall-torch."
        )

    print("[worker] Окружение готово.")
    if install_only:
        subprocess.check_call([str(vpy), __file__, "--verify-env"])
        return

    _reexec(vpy)


def _reexec(vpy: Path) -> None:
    env = {**os.environ, _BOOTSTRAP_ENV: "1"}
    argv = [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]]
    if sys.platform == "win32":
        raise SystemExit(subprocess.call(argv, env=env))
    os.execve(str(vpy), argv, env)


def _verify_env() -> int:
    import numpy  # noqa: F401
    import torch
    import torchvision
    import websockets  # noqa: F401

    print("[worker] websockets OK")
    print("[worker] numpy OK")
    print(f"[worker] torch {torch.__version__}")
    print(f"[worker] torchvision {torchvision.__version__}")
    print(f"[worker] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[worker] GPU: {name} (capability {cap[0]}.{cap[1]})")
        arch_fn = getattr(torch.cuda, "get_arch_list", None)
        if arch_fn:
            print(f"[worker] torch arch list: {arch_fn()}")
        if cap[0] >= 12:
            archs = arch_fn() if arch_fn else []
            if not any("120" in a for a in (archs or [])):
                print(
                    "[worker] ВНИМАНИЕ: GPU sm_12x, но torch без sm_120. "
                    "Нужен DISTGPU_TORCH=nightly-cu128 и --reinstall-torch."
                )
                return 1
    return 0


def _warn_torch_cuda_at_runtime() -> None:
    vpy = _venv_python()
    if not vpy.is_file():
        return
    if _torch_cuda_mismatch(vpy):
        _print_cuda_fix_hint(_resolve_torch_preset())


def _load_or_create_worker_id() -> str:
    wid_path = _worker_id_file()
    try:
        if wid_path.exists():
            wid = wid_path.read_text(encoding="utf-8").strip()
            if len(wid) == 8 and wid.isalnum():
                return wid
    except OSError:
        pass
    wid = uuid.uuid4().hex[:8]
    try:
        wid_path.write_text(wid, encoding="utf-8")
    except OSError:
        pass
    return wid


def _gpu_via_nvidia_smi() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = None
    if not out or out.returncode != 0 or not out.stdout.strip():
        return {
            "available": False,
            "name": None,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "device_count": 0,
        }
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return {
            "available": False,
            "name": None,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "device_count": 0,
        }
    try:
        total = int(float(parts[1]))
        free = int(float(parts[2]))
    except ValueError:
        total, free = 0, 0
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    return {
        "available": True,
        "name": parts[0],
        "vram_total_mb": total,
        "vram_free_mb": max(0, free),
        "device_count": len(lines),
    }


async def get_gpu_info() -> dict:
    gpu = _gpu_via_nvidia_smi()
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory // (1024**2)
            reserved = torch.cuda.memory_reserved(0) // (1024**2)
            gpu = {
                "available": True,
                "name": props.name,
                "vram_total_mb": total,
                "vram_free_mb": max(0, total - reserved),
                "device_count": torch.cuda.device_count(),
            }
    except ImportError:
        pass
    return gpu


async def _ws_send(ws: Any, payload: dict) -> bool:
    """Отправка на координатор; False если WS уже закрыт."""
    try:
        await ws.send(json.dumps(payload))
        return True
    except Exception as e:
        print(f"[worker] Не удалось отправить {payload.get('type')}: {e}")
        return False


class TrainingSession:
    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._user_cancelled: bool = False
        self._active_job_id: Optional[str] = None

    def _job_running(self, job_id: str) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._active_job_id == job_id
        )

    async def _stop_subprocess(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

    async def cancel(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        self._user_cancelled = True
        await self._stop_subprocess()

    async def run_job(self, ws: Any, msg: dict) -> None:
        job_id = msg["job_id"]
        if self._job_running(job_id):
            print(f"[worker] Задача {job_id} уже выполняется, повторный run_job пропущен")
            return
        await self._stop_subprocess()
        self._user_cancelled = False
        self._active_job_id = job_id
        script = msg["script"]
        resume = bool(msg.get("resume"))
        ckpt_path = msg.get("checkpoint_path")
        start_step = int(msg.get("start_step", 0))

        root = runtime_root()
        tmp_dir = root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"distgpu_{job_id}_{uuid.uuid4().hex[:6]}.py"
        tmp.write_text(script, encoding="utf-8")
        job_work_dir = root / "jobs" / job_id
        job_work_dir.mkdir(parents=True, exist_ok=True)

        torch_home = job_work_dir / ".torch"
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "JOB_ID": job_id,
            "DISTGPU_JOB_WORK_DIR": str(job_work_dir),
            "CHECKPOINT_DIR": str(job_work_dir / "checkpoints"),
            "CHECKPOINT_EVERY": os.environ.get("CHECKPOINT_EVERY", "100"),
            "TORCH_HOME": str(torch_home),
            "HF_HOME": str(job_work_dir / ".hf"),
        }
        _repo_root = _SCRIPT_DIR.parent
        if (_repo_root / "distgpu").is_dir():
            pp = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = str(_repo_root) + (os.pathsep + pp if pp else "")

        if resume and ckpt_path:
            env["CHECKPOINT_PATH"] = str(ckpt_path)
            env["START_STEP"] = str(start_step)

        dist = msg.get("distributed")
        if isinstance(dist, dict) and dist.get("master_addr") and dist.get("world_size"):
            env["MASTER_ADDR"] = str(dist["master_addr"])
            env["MASTER_PORT"] = str(int(dist["master_port"]))
            env["RANK"] = str(int(dist["rank"]))
            env["WORLD_SIZE"] = str(int(dist["world_size"]))
            env["LOCAL_RANK"] = str(int(dist.get("local_rank", 0)))
            env["DISTGPU_MANUAL_INIT"] = "1"
            env["DISTGPU_USE_FSDP"] = "1"

        if msg.get("pipeline_enabled") and msg.get("pipeline_config_yaml"):
            cfg_tmp = runtime_root() / "tmp" / (
                f"distgpu_pipe_{job_id}_{uuid.uuid4().hex[:8]}.yaml"
            )
            cfg_tmp.parent.mkdir(parents=True, exist_ok=True)
            cfg_tmp.write_text(str(msg["pipeline_config_yaml"]), encoding="utf-8")
            env["DISTGPU_USE_PIPELINE"] = "1"
            env["DISTGPU_CONFIG_PATH"] = str(cfg_tmp)
            _rank_hint = (
                int(dist["rank"]) if isinstance(dist, dict) and "rank" in dist else 0
            )
            env["DISTGPU_PIPELINE_STAGE_IDX"] = str(
                int(msg.get("pipeline_stage_idx", _rank_hint))
            )
            if v := os.environ.get("DISTGPU_SUBSPACE_RANK"):
                env["DISTGPU_SUBSPACE_RANK"] = str(v)
            if v := os.environ.get("DISTGPU_CONTEXT_PARALLEL"):
                env["DISTGPU_CONTEXT_PARALLEL"] = str(v)

        py = os.environ.get("PYTHON", sys.executable).strip() or sys.executable
        print(f"[worker] Запуск: {py} {tmp}")
        print(f"[worker] Рабочая папка задачи: {job_work_dir}")
        self._proc = await asyncio.create_subprocess_exec(
            py,
            str(tmp),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(job_work_dir),
        )

        assert self._proc.stdout is not None
        exit_code = 1
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")

                if line.startswith("__CHECKPOINT__:"):
                    rest = line[len("__CHECKPOINT__:") :]
                    parts = rest.rsplit(":", 1)
                    if len(parts) == 2:
                        path, step_s = parts[0], parts[1]
                        try:
                            step = int(step_s)
                        except ValueError:
                            continue
                        await _ws_send(
                            ws,
                            {
                                "type": "checkpoint",
                                "job_id": job_id,
                                "path": path,
                                "step": step,
                            },
                        )
                    continue

                if line.startswith("__RESUMED__:"):
                    await _ws_send(ws, {"type": "log", "job_id": job_id, "text": line})
                    continue

                await _ws_send(ws, {"type": "log", "job_id": job_id, "text": line})

            exit_code = await self._proc.wait()
        finally:
            self._proc = None
            self._active_job_id = None
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

        if self._user_cancelled:
            exit_code = 130
        self._user_cancelled = False

        await _ws_send(
            ws,
            {"type": "job_done", "job_id": job_id, "exit_code": int(exit_code)},
        )
        print(f"[worker] Задача {job_id} завершена, exit_code={exit_code}")


async def connect(server_url: str, token: str, advertise_host: Optional[str]) -> None:
    import websockets

    session = TrainingSession()
    worker_id = _load_or_create_worker_id()
    job_task: Optional[asyncio.Task] = None

    async def _run_job_task(ws_conn: Any, job_msg: dict) -> None:
        try:
            await session.run_job(ws_conn, job_msg)
        except asyncio.CancelledError:
            await session.cancel()
            raise
        except Exception as e:
            print(f"[worker] Ошибка выполнения задачи: {e}")

    while True:
        try:
            root = runtime_root()
            print(f"[worker] Подключение к {server_url}…")
            print(f"[worker] Данные и кэши: {root}")
            # Долгий run_job не должен блокировать pong — иначе keepalive рвёт WS
            async with websockets.connect(
                server_url,
                ping_interval=30,
                ping_timeout=300,
                close_timeout=10,
            ) as ws:
                gpu = await get_gpu_info()
                reg_body: dict = {
                    "type": "register",
                    "token": token,
                    "worker_id": worker_id,
                    "gpu": gpu,
                    "platform": platform.system(),
                }
                if advertise_host:
                    reg_body["advertise_host"] = advertise_host
                await ws.send(json.dumps(reg_body))
                name = gpu.get("name") or "нет GPU"
                vram = gpu.get("vram_total_mb", 0)
                print(f"[worker] Подключён id={worker_id} · {name} · {vram} MB VRAM")

                async for raw in ws:
                    msg = json.loads(raw)

                    if msg["type"] == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif msg["type"] == "run_job":
                        jid = msg["job_id"]
                        print(f"[worker] Задача {jid} (фоновый процесс)")
                        if job_task and not job_task.done():
                            print(f"[worker] Отмена предыдущей задачи перед {jid}")
                            job_task.cancel()
                            try:
                                await job_task
                            except asyncio.CancelledError:
                                pass
                        job_task = asyncio.create_task(_run_job_task(ws, msg))

                    elif msg["type"] == "cancel_job":
                        print(f"[worker] cancel_job {msg.get('job_id')}")
                        if job_task and not job_task.done():
                            job_task.cancel()
                            try:
                                await job_task
                            except asyncio.CancelledError:
                                pass
                        else:
                            await session.cancel()

                    elif msg["type"] == "sync":
                        role = msg.get("role", "standby")
                        print(
                            f"[worker] sync job={msg.get('job_id')} role={role} "
                            f"step={msg.get('current_step')}"
                        )

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[worker] Соединение потеряно: {e}. Переподключение через 5 с…")
            if job_task and not job_task.done():
                job_task.cancel()
                try:
                    await job_task
                except asyncio.CancelledError:
                    pass
            await session.cancel()
            await asyncio.sleep(5)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DistGPU worker (один файл)")
    p.add_argument(
        "--server",
        default=os.environ.get("SERVER_URL", "ws://127.0.0.1:8765/worker"),
        help="WebSocket координатора (или SERVER_URL)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("WORKER_TOKEN") or os.environ.get("TOKEN") or "secret-token-123",
        help="Токен (или WORKER_TOKEN / TOKEN)",
    )
    p.add_argument(
        "--advertise-host",
        default=(os.environ.get("DISTGPU_ADVERTISE_HOST") or "").strip() or None,
        help="Публичный IP для FSDP rendezvous (rank 0)",
    )
    p.add_argument(
        "--install-only",
        action="store_true",
        help="Только установить зависимости в .venv и выйти",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Не проверять/ставить зависимости (уже в venv)",
    )
    p.add_argument(
        "--reinstall-torch",
        action="store_true",
        help="Переустановить torch в .venv (для RTX 50xx: nightly-cu128)",
    )
    p.add_argument("--verify-env", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_env:
        raise SystemExit(_verify_env())

    if not args.skip_install:
        ensure_environment(
            install_only=args.install_only,
            reinstall_torch=args.reinstall_torch,
        )
    _ensure_torchvision_if_missing()
    if args.install_only:
        return

    _warn_torch_cuda_at_runtime()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(connect(args.server, args.token, args.advertise_host))


if __name__ == "__main__":
    main()
