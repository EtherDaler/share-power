from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from distgpu.config.hybrid import ContextParallelConfig


@dataclass
class _CP:
    """Совместимость с именем из ТЗ; конфиг берётся из hybrid.ContextParallelConfig."""

    pass


class MixtureOfSubspacesCP(nn.Module):
    """
    Заготовка под Mixtures of Subspaces (NeurIPS 2025).
    При `enabled=False` методы no-op; при True — упрощённое сжатие K/V.
    """

    def __init__(self, d_head: int, cfg: ContextParallelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_head = d_head
        r = max(1, cfg.rank)
        self.bases = nn.ParameterList(
            [
                nn.Parameter(torch.randn(d_head, r) * 0.02)
                for _ in range(cfg.n_subspaces)
            ]
        )
        self.gate = nn.Linear(d_head, cfg.n_subspaces)

    def compress_kv(
        self, k: torch.Tensor, v: torch.Tensor, q_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.cfg.enabled:
            return k, v, torch.zeros(1, device=k.device)
        # (B, H, T, Dh) -> усреднение по T для весов смеси
        g_in = q_context.mean(dim=2)
        w = torch.softmax(self.gate(g_in), dim=-1)
        basis_weights = w
        # упрощённо: проекция по первому базису (полная MoS — в roadmap)
        b0 = self.bases[0]
        k_c = torch.matmul(k, b0)
        v_c = torch.matmul(v, b0)
        return k_c, v_c, basis_weights

    def decompress_kv(
        self,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        basis_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enabled:
            return k_c, v_c
        b0 = self.bases[0]
        k = torch.matmul(k_c, b0.T)
        v = torch.matmul(v_c, b0.T)
        return k, v
