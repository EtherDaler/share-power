from __future__ import annotations

import gc
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from distgpu.config.hybrid import PipelineConfig


def _get_layers_by_attr(model: nn.Module, attr_path: str) -> nn.ModuleList:
    cur: Any = model
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"Нет атрибута {part!r} у {type(cur).__name__} в пути {attr_path!r}"
            )
        cur = getattr(cur, part)
    if not isinstance(cur, nn.ModuleList):
        raise TypeError(
            f"По пути {attr_path!r} ожидался nn.ModuleList, получено {type(cur)}"
        )
    return cur


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _compute_boundaries_equal(n_layers: int, n_stages: int) -> list[tuple[int, int]]:
    if n_stages <= 0 or n_layers <= 0:
        return []
    base = n_layers // n_stages
    rem = n_layers % n_stages
    out: list[tuple[int, int]] = []
    start = 0
    for s in range(n_stages):
        w = base + (1 if s < rem else 0)
        end = start + w
        out.append((start, end))
        start = end
    return out


def _compute_boundaries_by_params(
    layers: Sequence[nn.Module], n_stages: int
) -> list[tuple[int, int]]:
    sizes = [_count_params(l) for l in layers]
    total = sum(sizes) or 1
    targets = [total * (i + 1) / n_stages for i in range(n_stages)]
    bounds: list[tuple[int, int]] = []
    cum = 0
    start = 0
    t_idx = 0
    for i, sz in enumerate(sizes):
        cum += sz
        if t_idx < n_stages - 1 and cum >= targets[t_idx]:
            bounds.append((start, i + 1))
            start = i + 1
            t_idx += 1
    bounds.append((start, len(layers)))
    while len(bounds) < n_stages:
        le = bounds[-1][1]
        bounds.append((le, le))
    return bounds[:n_stages]


def get_stage_boundaries(model: nn.Module, cfg: PipelineConfig) -> list[tuple[int, int]]:
    """Возвращает [(start, end), ...] для каждой стадии (end exclusive)."""
    if cfg.custom_split:
        bounds = []
        idx = 0
        for group in cfg.custom_split:
            n = len(group)
            bounds.append((idx, idx + n))
            idx += n
        return bounds
    layers = _get_layers_by_attr(model, cfg.layer_attr)
    n = len(layers)
    if cfg.balance_by == "params":
        return _compute_boundaries_by_params(list(layers), cfg.num_stages)
    return _compute_boundaries_equal(n, cfg.num_stages)


def split_model(model: nn.Module, cfg: PipelineConfig) -> nn.Module:
    """
    Заменяет ModuleList по `cfg.layer_attr` на подсписок [lo:hi) для стадии `stage_idx`.
    """
    if cfg.custom_split:
        raise NotImplementedError(
            "custom_split: используйте layer_attr на nn.ModuleList и balance_by"
        )
    bounds_all = get_stage_boundaries(model, cfg)
    if cfg.stage_idx < 0 or cfg.stage_idx >= len(bounds_all):
        raise ValueError(f"stage_idx {cfg.stage_idx} вне [0, {len(bounds_all)})")
    lo, hi = bounds_all[cfg.stage_idx]
    parts = cfg.layer_attr.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr_name = parts[-1]
    holder = getattr(parent, attr_name)
    if not isinstance(holder, nn.ModuleList):
        raise TypeError("layer_attr должен указывать на nn.ModuleList")
    full_list = list(holder)
    setattr(parent, attr_name, nn.ModuleList(full_list[lo:hi]))
    for i, mod in enumerate(full_list):
        if i < lo or i >= hi:
            del mod
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _demo_model() -> nn.Module:
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [nn.Linear(64, 64) for _ in range(12)]
            )
            self.lm_head = nn.Linear(64, 100)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embed(x)
            for layer in self.model.layers:
                h = layer(h)
            return self.lm_head(h)

    return M()


def _main() -> None:
    m0 = _demo_model()
    total_before = _count_params(m0)
    cfg_probe = PipelineConfig(
        num_stages=3,
        stage_idx=0,
        model_type="transformer",
        layer_attr="model.layers",
        balance_by="count",
    )
    bounds = get_stage_boundaries(m0, cfg_probe)
    print("Границы стадий:", bounds)
    for s in range(3):
        mc = _demo_model()
        c = PipelineConfig(
            num_stages=3,
            stage_idx=s,
            model_type="transformer",
            layer_attr="model.layers",
            balance_by="count",
        )
        split_model(mc, c)
        n_layers = len(mc.model.layers)
        p = _count_params(mc)
        lo, hi = bounds[s]
        print(
            f"Stage {s}: layers [{lo},{hi}) -> {n_layers} слоёв, "
            f"{p / 1e6:.2f}M params (всего до split {total_before / 1e6:.2f}M)"
        )


if __name__ == "__main__":
    _main()
