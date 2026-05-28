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

Переменные: SERVER_URL, WORKER_TOKEN / TOKEN, DISTGPU_TORCH (cu124, cu121, cpu, …),
DISTGPU_ADVERTISE_HOST, CHECKPOINT_EVERY, PYTHON (базовый Python для venv).
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
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

_BOOTSTRAP_ENV = "DISTGPU_WORKER_BOOTSTRAPPED"
_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_DIR = _SCRIPT_DIR / ".venv"
_REQUIREMENTS = "websockets>=12.0"

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
        "import websockets, torch\n"
        "import sys; sys.exit(0)"
    )
    return subprocess.run([str(py), "-c", code], capture_output=True).returncode == 0


def _install_torch(py: Path, preset: str) -> None:
    custom_url = os.environ.get("DISTGPU_PYTORCH_INDEX_URL", "").strip()
    extra = os.environ.get("DISTGPU_PYTORCH_EXTRA", "").strip().split()

    if custom_url:
        _run_pip(py, "install", *extra, "torch", "--index-url", custom_url)
        return

    if preset == "cpu":
        _run_pip(py, "install", "torch")
        return
    if preset == "mps":
        _run_pip(py, "install", "torch")
        return
    if preset == "nightly-cu128":
        _run_pip(
            py,
            "install",
            *extra,
            "--pre",
            "torch",
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
    _run_pip(py, "install", *extra, "torch", "--index-url", url)


def ensure_environment(*, install_only: bool = False) -> None:
    """Создаёт venv и ставит зависимости при необходимости; перезапускает себя из venv."""
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        return

    vpy = _venv_python()
    if vpy.is_file() and _deps_ok(vpy):
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
    _run_pip(vpy, "install", _REQUIREMENTS)

    preset = os.environ.get("DISTGPU_TORCH", "").strip() or _default_torch_preset()
    print(f"[worker] Установка PyTorch (пресет {preset})…")
    _install_torch(vpy, preset)

    if not _deps_ok(vpy):
        raise SystemExit("[worker] Проверка зависимостей не прошла после установки")

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
    import torch
    import websockets  # noqa: F401

    print("[worker] websockets OK")
    print(f"[worker] torch {torch.__version__}")
    print(f"[worker] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[worker] GPU: {torch.cuda.get_device_name(0)}")
    return 0


WORKER_ID_FILE = Path.home() / ".distgpu_worker_id"


def _load_or_create_worker_id() -> str:
    try:
        if WORKER_ID_FILE.exists():
            wid = WORKER_ID_FILE.read_text(encoding="utf-8").strip()
            if len(wid) == 8 and wid.isalnum():
                return wid
    except OSError:
        pass
    wid = uuid.uuid4().hex[:8]
    try:
        WORKER_ID_FILE.write_text(wid, encoding="utf-8")
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


class TrainingSession:
    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._user_cancelled: bool = False

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
        await self._stop_subprocess()
        self._user_cancelled = False
        job_id = msg["job_id"]
        script = msg["script"]
        resume = bool(msg.get("resume"))
        ckpt_path = msg.get("checkpoint_path")
        start_step = int(msg.get("start_step", 0))

        tmp = Path(tempfile.gettempdir()) / f"distgpu_{job_id}_{uuid.uuid4().hex[:6]}.py"
        tmp.write_text(script, encoding="utf-8")

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "JOB_ID": job_id,
            "CHECKPOINT_DIR": str(Path.home() / "distgpu_checkpoints"),
            "CHECKPOINT_EVERY": os.environ.get("CHECKPOINT_EVERY", "100"),
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
            cfg_tmp = Path(tempfile.gettempdir()) / (
                f"distgpu_pipe_{job_id}_{uuid.uuid4().hex[:8]}.yaml"
            )
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
        self._proc = await asyncio.create_subprocess_exec(
            py,
            str(tmp),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
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
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "checkpoint",
                                    "job_id": job_id,
                                    "path": path,
                                    "step": step,
                                }
                            )
                        )
                    continue

                if line.startswith("__RESUMED__:"):
                    await ws.send(
                        json.dumps({"type": "log", "job_id": job_id, "text": line})
                    )
                    continue

                await ws.send(
                    json.dumps({"type": "log", "job_id": job_id, "text": line})
                )

            exit_code = await self._proc.wait()
        finally:
            self._proc = None
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

        if self._user_cancelled:
            exit_code = 130
        self._user_cancelled = False

        await ws.send(
            json.dumps(
                {"type": "job_done", "job_id": job_id, "exit_code": int(exit_code)}
            )
        )


async def connect(server_url: str, token: str, advertise_host: Optional[str]) -> None:
    import websockets

    session = TrainingSession()
    worker_id = _load_or_create_worker_id()

    while True:
        try:
            print(f"[worker] Подключение к {server_url}…")
            async with websockets.connect(server_url) as ws:
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
                        print(f"[worker] Задача {msg['job_id']}")
                        await session.run_job(ws, msg)

                    elif msg["type"] == "cancel_job":
                        await session.cancel()

                    elif msg["type"] == "sync":
                        role = msg.get("role", "standby")
                        print(
                            f"[worker] sync job={msg.get('job_id')} role={role} "
                            f"step={msg.get('current_step')}"
                        )

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[worker] Соединение потеряно: {e}. Переподключение через 5 с…")
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
    p.add_argument("--verify-env", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_env:
        raise SystemExit(_verify_env())

    if not args.skip_install:
        ensure_environment(install_only=args.install_only)
    if args.install_only:
        return

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(connect(args.server, args.token, args.advertise_host))


if __name__ == "__main__":
    main()
