from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from distgpu.compression.context_parallel import MixtureOfSubspacesCP
from distgpu.config.hybrid import HybridConfig
from distgpu.pipeline.comm import PipelineCommunicator
from distgpu.pipeline.splitter import split_model


class HybridParallelExecutor:
    """
    Гибрид: FSDP (опционально) + pipeline-стадия + сжатие на границе.
    Полный backward через границы — поэтапно в train_step (ручная пересылка градиентов).
    """

    def __init__(self, cfg: HybridConfig) -> None:
        self.cfg = cfg
        self._model: Optional[nn.Module] = None
        self._comm: Optional[PipelineCommunicator] = None
        self._cp: Optional[MixtureOfSubspacesCP] = None

    def setup(self, model: nn.Module) -> nn.Module:
        if dist.is_initialized():
            self.cfg.pipeline.num_stages = dist.get_world_size()
            self.cfg.pipeline.stage_idx = dist.get_rank()
        split_model(model, self.cfg.pipeline)
        if torch.cuda.is_available():
            model = model.cuda()
        if self.cfg.fsdp.enabled and dist.is_initialized():
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            strat = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }.get(self.cfg.fsdp.sharding_strategy, ShardingStrategy.FULL_SHARD)
            model = FSDP(model, sharding_strategy=strat, use_orig_params=True)
        sub_cfg = self.cfg.subspace if self.cfg.subspace.enabled else None
        self._comm = PipelineCommunicator(
            dist.get_rank() if dist.is_initialized() else 0,
            dist.get_world_size() if dist.is_initialized() else 1,
            self.cfg.d_model,
            sub_cfg,
        )
        if self.cfg.context_parallel.enabled:
            self._cp = MixtureOfSubspacesCP(
                self.cfg.d_model // 32,
                self.cfg.context_parallel,
            )
        self._model = model
        return model

    def setup_demo(self) -> None:
        """Две Linear в ModuleList — по одной на стадию при world_size=2."""
        d = min(64, self.cfg.d_model)
        self.cfg.d_model = d

        class Demo(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList(
                    [nn.Linear(d, d), nn.Linear(d, d)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.model.layers:
                    x = layer(x)
                return x

        self.setup(Demo())

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not dist.is_initialized() or self._comm is None or self._model is None:
            return {"loss": 0.0, "note": "no distributed"}
        return self._train_step_manual(batch)

    def _train_step_manual(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r = dist.get_rank()
        ws = dist.get_world_size()
        dev = next(self._model.parameters()).device
        dist.barrier()
        if ws < 2:
            x = torch.randn(2, 8, self.cfg.d_model, device=dev, requires_grad=True)
            y = self._model(x)
            loss = y.pow(2).mean()
            loss.backward()
            return {"loss": float(loss.item())}
        if r == 0:
            x = torch.randn(2, 8, self.cfg.d_model, device=dev, requires_grad=True)
            h = self._model(x)
            self._comm.send_activations(h)
            g = self._comm.recv_gradients(tuple(h.shape))
            h.backward(g)
            return {"loss": 0.0}
        if r == ws - 1:
            h_in = self._comm.recv_activations()
            out = self._model(h_in)
            loss = out.pow(2).mean()
            loss.backward()
            if h_in.grad is not None:
                self._comm.send_gradients(h_in.grad)
            return {"loss": float(loss.item())}
        h_in = self._comm.recv_activations()
        h_out = self._model(h_in)
        self._comm.send_activations(h_out)
        g_out = self._comm.recv_gradients(tuple(h_out.shape))
        h_out.backward(g_out)
        if h_in.grad is not None:
            self._comm.send_gradients(h_in.grad)
        return {"loss": 0.0}

    def save_checkpoint(self, path: str) -> None:
        r = dist.get_rank() if dist.is_initialized() else 0
        torch.save({"rank": r, "cfg": self.cfg.__dict__}, f"{path}.rank{r}.pt")

    def load_checkpoint(self, path: str) -> None:
        r = dist.get_rank() if dist.is_initialized() else 0
        torch.load(f"{path}.rank{r}.pt", map_location="cpu")
