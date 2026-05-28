from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from distgpu.compression.subspace import SubspaceCompressor, SubspaceConfig


class PipelineCommunicator:
    """Передача активаций/градиентов между соседними pipeline-рангами."""

    TAG_META = 901
    TAG_ACT = 902
    TAG_GRAD_META = 903
    TAG_GRAD = 904

    def __init__(
        self,
        rank: int,
        world_size: int,
        d_model: int,
        subspace_cfg: Optional[SubspaceConfig],
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("PipelineCommunicator требует init_process_group")
        self.rank = rank
        self.world_size = world_size
        self.d_model = d_model
        self._use_subspace = bool(
            subspace_cfg and getattr(subspace_cfg, "enabled", True)
        )
        self._sub_cfg = subspace_cfg
        self.compressor: Optional[SubspaceCompressor] = None
        if self._use_subspace and subspace_cfg is not None:
            self.compressor = SubspaceCompressor(d_model, subspace_cfg)
        self._dev = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _dtype_wire(self) -> torch.dtype:
        if not self._sub_cfg:
            return torch.float32
        m = getattr(self._sub_cfg, "dtype", "float16")
        if m == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def send_activations(self, x: torch.Tensor, requires_grad: bool = True) -> None:
        dst = (self.rank + 1) % self.world_size
        if dst == self.rank:
            return
        if self.compressor is not None:
            self.compressor.reset_error_buf(tuple(x.shape), x.device)
            z = self.compressor.encode(x)
            wire = z.to(self._dtype_wire())
        else:
            wire = x
        meta = torch.zeros(8, dtype=torch.long, device=wire.device)
        sh = list(wire.shape)
        meta[0] = len(sh)
        for i, s in enumerate(sh[:7]):
            meta[1 + i] = s
        try:
            dist.send(meta, dst=dst, tag=self.TAG_META)
            dist.send(wire.contiguous(), dst=dst, tag=self.TAG_ACT)
        except Exception as e:
            raise RuntimeError(f"send_activations failed rank={self.rank} -> {dst}: {e}") from e

    def recv_activations(self) -> torch.Tensor:
        src = (self.rank - 1) % self.world_size
        if src == self.rank:
            raise RuntimeError("recv_activations на rank 0 без источника")
        meta = torch.zeros(8, dtype=torch.long, device=self._dev)
        try:
            dist.recv(meta, src=src, tag=self.TAG_META)
        except Exception as e:
            raise RuntimeError(f"recv meta failed rank={self.rank} <- {src}: {e}") from e
        nd = int(meta[0].item())
        shape = tuple(int(meta[i].item()) for i in range(1, 1 + nd))
        buf = torch.empty(shape, dtype=self._dtype_wire(), device=self._dev)
        try:
            dist.recv(buf, src=src, tag=self.TAG_ACT)
        except Exception as e:
            raise RuntimeError(f"recv tensor failed rank={self.rank} <- {src}: {e}") from e
        if self.compressor is not None:
            out = self.compressor.decode(buf.float())
        else:
            out = buf.float()
        return out.requires_grad_(requires_grad)

    def send_gradients(self, grad: torch.Tensor) -> None:
        dst = (self.rank - 1) % self.world_size
        if dst == self.rank:
            return
        wire = grad.to(self._dtype_wire()).contiguous()
        meta = torch.zeros(8, dtype=torch.long, device=wire.device)
        sh = list(wire.shape)
        meta[0] = len(sh)
        for i, s in enumerate(sh[:7]):
            meta[1 + i] = s
        try:
            dist.send(meta, dst=dst, tag=self.TAG_GRAD_META)
            dist.send(wire, dst=dst, tag=self.TAG_GRAD)
        except Exception as e:
            raise RuntimeError(f"send_gradients failed: {e}") from e

    def recv_gradients(self, shape: Tuple[int, ...]) -> torch.Tensor:
        src = (self.rank + 1) % self.world_size
        if src == self.rank:
            raise RuntimeError("recv_gradients на последнем ранге без источника")
        meta = torch.zeros(8, dtype=torch.long, device=self._dev)
        try:
            dist.recv(meta, src=src, tag=self.TAG_GRAD_META)
        except Exception as e:
            raise RuntimeError(f"recv_grad meta failed: {e}") from e
        nd = int(meta[0].item())
        sh = tuple(int(meta[i].item()) for i in range(1, 1 + nd))
        buf = torch.empty(sh, dtype=self._dtype_wire(), device=self._dev)
        try:
            dist.recv(buf, src=src, tag=self.TAG_GRAD)
        except Exception as e:
            raise RuntimeError(f"recv_grad tensor failed: {e}") from e
        return buf.float()


class PipelineBoundaryFn(torch.autograd.Function):
    """Заготовка: явный градиент через comm остаётся в HybridParallelExecutor.train_step."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, comm: PipelineCommunicator) -> torch.Tensor:
        ctx.comm = comm
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None
