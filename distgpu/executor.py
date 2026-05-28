import nbformat

# Единый блок: torchrun (DDP), multi-node от воркера DistGPU (DISTGPU_MANUAL_INIT), FSDP при DISTGPU_USE_FSDP=1
TRAINING_INIT_CODE = '''
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

USE_FSDP = os.environ.get("DISTGPU_USE_FSDP") == "1"
USE_PIPELINE = os.environ.get("DISTGPU_USE_PIPELINE") == "1"

def setup_distributed():
    """True если запущен distributed (torchrun или переменные от координатора)."""
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("Distributed training requires CUDA on this worker")
    if os.environ.get("DISTGPU_MANUAL_INIT") == "1":
        addr = os.environ["MASTER_ADDR"]
        port = os.environ["MASTER_PORT"]
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{addr}:{port}",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    else:
        dist.init_process_group(backend="nccl")
    return True

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def wrap_model(model):
    if not dist.is_initialized():
        return model
    if USE_FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return FSDP(
            model.cuda(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    return DDP(model.cuda(), device_ids=[local_rank])

def get_device():
    if dist.is_initialized():
        return torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ddp_active = setup_distributed()
DEVICE = get_device()
_mode = "FSDP" if (_ddp_active and USE_FSDP) else ("DDP" if _ddp_active else "нет")
if USE_PIPELINE:
    _mode = _mode + "+PIPELINE"
print(f"[Worker] DEVICE={DEVICE}, distributed={_ddp_active}, режим={_mode}")
'''

CHECKPOINT_HELPERS_CODE = '''
import os
from pathlib import Path

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", str(Path.home() / "distgpu_checkpoints"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100"))
JOB_ID = os.environ.get("JOB_ID", "unknown")
START_STEP = int(os.environ.get("START_STEP", "0"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, step: int):
    import torch
    import torch.distributed as dist
    path = f"{CHECKPOINT_DIR}/{JOB_ID}_step{step}.pt"
    if not dist.is_initialized():
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)
        print(f"__CHECKPOINT__:{path}:{step}")
        return path
    if USE_FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            m_state = model.state_dict()
        optim_sd = FSDP.full_optim_state_dict(model, optimizer)
        if dist.get_rank() == 0:
            torch.save({
                "step": step,
                "model_state": m_state,
                "optimizer_state": optim_sd,
            }, path)
            print(f"__CHECKPOINT__:{path}:{step}")
        dist.barrier()
        return path
    if dist.get_rank() == 0:
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)
        print(f"__CHECKPOINT__:{path}:{step}")
    dist.barrier()
    return path

def load_checkpoint(model, optimizer, path: str) -> int:
    import torch
    import torch.distributed as dist
    ckpt = torch.load(path, map_location="cpu" if dist.is_initialized() else DEVICE)
    if not dist.is_initialized():
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"__RESUMED__:{ckpt['step']}")
        return ckpt["step"]
    if USE_FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(ckpt["model_state"])
        shard_optim = FSDP.shard_full_optim_state_dict(
            model, optimizer, ckpt["optimizer_state"]
        )
        optimizer.load_state_dict(shard_optim)
        print(f"__RESUMED__:{ckpt['step']}")
        return ckpt["step"]
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"__RESUMED__:{ckpt['step']}")
    return ckpt["step"]
'''

PIPELINE_GATE_CODE = '''
import os as __dg_os
if __dg_os.environ.get("DISTGPU_USE_PIPELINE") == "1":
    from distgpu.pipeline.script_entry import setup_and_run_hybrid
    setup_and_run_hybrid()
    raise SystemExit(0)
'''

# Защита окружения до user code: CUDA, Windows DataLoader, headless matplotlib
RUNTIME_GUARDS_CODE = '''
import os as _dg_os
import sys as _dg_sys

_dg_os.environ.setdefault("MPLBACKEND", "Agg")

if not _ddp_active:
    import torch as _dg_torch
    if not _dg_torch.cuda.is_available():
        raise RuntimeError(
            "[DistGPU] CUDA недоступна на воркере. "
            "Для RTX 50xx установите PyTorch nightly-cu128 (DISTGPU_TORCH=nightly-cu128)."
        )

if _dg_sys.platform == "win32":
    import torch.utils.data as _dg_tud
    _BaseDataLoader = _dg_tud.DataLoader

    class _WinSafeDataLoader(_BaseDataLoader):
        def __init__(self, *args, num_workers=0, pin_memory=False, **kwargs):
            if num_workers > 0:
                print("[DistGPU] Windows: DataLoader num_workers сброшен в 0")
                num_workers = 0
                pin_memory = False
            super().__init__(
                *args, num_workers=num_workers, pin_memory=pin_memory, **kwargs
            )

    _dg_tud.DataLoader = _WinSafeDataLoader

_dg_work = _dg_os.environ.get("DISTGPU_JOB_WORK_DIR", "").strip()
if _dg_work:
    print(f"[DistGPU] Рабочая папка задачи: {_dg_work}")
'''

DDP_CLEANUP_CODE = '''
cleanup_ddp()
print("[Worker] Обучение завершено.")
'''


def notebook_to_ddp_script(nb_bytes: bytes) -> str:
    """Превращает .ipynb в Python скрипт с DDP/FSDP и чекпоинтами."""
    nb = nbformat.reads(nb_bytes.decode("utf-8"), as_version=4)

    cells_code = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            lines = [l for l in cell.source.split("\n") if not l.startswith("%")]
            cells_code.append("\n".join(lines))

    user_code = "\n\n".join(cells_code)

    return f"""
# ===== AUTO-GENERATED TRAINING WRAPPER (DDP / FSDP) =====
{TRAINING_INIT_CODE}

# ===== CHECKPOINT HELPERS =====
{CHECKPOINT_HELPERS_CODE}

# ===== PIPELINE (гибридный режим; при включении — выход до user code) =====
{PIPELINE_GATE_CODE}

# ===== RUNTIME GUARDS (CUDA / Windows / matplotlib) =====
{RUNTIME_GUARDS_CODE}

# ===== USER CODE =====
{user_code}

# ===== CLEANUP =====
{DDP_CLEANUP_CODE}
"""
