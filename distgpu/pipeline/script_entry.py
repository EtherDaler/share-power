"""
Точка входа для DISTGPU_USE_PIPELINE=1 (вставляется в начало сгенерированного скрипта).
Ожидает уже инициализированный process group (как после FSDP-блока в executor) —
поэтому вызывается ПОСЛЕ TRAINING_INIT_CODE в notebook_to_ddp_script.
"""

from __future__ import annotations

import os

import torch.distributed as dist

from distgpu.config.hybrid import HybridConfig
from distgpu.pipeline.executor import HybridParallelExecutor


def setup_and_run_hybrid() -> None:
    if os.environ.get("DISTGPU_USE_PIPELINE") != "1":
        return
    if not dist.is_initialized():
        raise RuntimeError(
            "Pipeline-режим: сначала должен быть инициализирован distributed "
            "(RANK/WORLD_SIZE/MASTER_* от воркера)."
        )
    dist.barrier()
    cfg = HybridConfig.from_env()
    exe = HybridParallelExecutor(cfg)
    exe.setup_demo()
    for step in range(3):
        out = exe.train_step({})
        if dist.get_rank() == 0:
            print(f"[DistGPU pipeline] step={step} {out}")
    dist.barrier()
    print("[DistGPU pipeline] smoke OK")
    dist.destroy_process_group()
