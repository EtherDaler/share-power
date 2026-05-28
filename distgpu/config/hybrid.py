from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class FSDPConfig:
    enabled: bool = True
    sharding_strategy: str = "FULL_SHARD"


@dataclass
class PipelineConfig:
    num_stages: int = 2
    stage_idx: int = 0
    model_type: str = "transformer"
    layer_attr: str = "model.layers"
    balance_by: str = "count"
    custom_split: Optional[list] = field(default=None)


@dataclass
class SubspaceConfig:
    enabled: bool = True
    rank: int = 64
    learn_basis: bool = True
    error_feedback: bool = True
    dtype: str = "float16"
    sync_basis_every: int = 100


@dataclass
class ContextParallelConfig:
    enabled: bool = False
    seq_chunk_size: int = 4096
    n_subspaces: int = 4
    rank: int = 32
    min_seq_len: int = 16384


@dataclass
class HybridConfig:
    d_model: int = 4096
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    context_parallel: ContextParallelConfig = field(
        default_factory=ContextParallelConfig
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> HybridConfig:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Для HybridConfig.from_yaml установите PyYAML: pip install pyyaml"
            ) from e
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls._from_dict(raw or {})

    @classmethod
    def from_env(cls) -> HybridConfig:
        path = os.environ.get("DISTGPU_CONFIG_PATH", "").strip()
        if path and Path(path).is_file():
            cfg = cls.from_yaml(path)
        else:
            cfg = cls()
        cfg = cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> HybridConfig:
        if v := os.environ.get("DISTGPU_PIPELINE_STAGES"):
            self.pipeline.num_stages = int(v)
        if v := os.environ.get("DISTGPU_PIPELINE_STAGE_IDX"):
            self.pipeline.stage_idx = int(v)
        if v := os.environ.get("DISTGPU_SUBSPACE_RANK"):
            self.subspace.rank = int(v)
        if v := os.environ.get("DISTGPU_CONTEXT_PARALLEL"):
            self.context_parallel.enabled = v.strip() in ("1", "true", "True")
        if v := os.environ.get("DISTGPU_D_MODEL"):
            self.d_model = int(v)
        if v := os.environ.get("DISTGPU_SUBSPACE_ENABLED"):
            self.subspace.enabled = v.strip() in ("1", "true", "True")
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HybridConfig:
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> HybridConfig:
        def sub(dc_type, key: str, default_factory):
            sub_d = d.get(key) or {}
            kw = {
                f.name: sub_d.get(f.name, getattr(default_factory(), f.name))
                for f in fields(dc_type)
            }
            return dc_type(**kw)

        return cls(
            d_model=int(d.get("d_model", 4096)),
            fsdp=sub(FSDPConfig, "fsdp", FSDPConfig),
            pipeline=sub(PipelineConfig, "pipeline", PipelineConfig),
            subspace=sub(SubspaceConfig, "subspace", SubspaceConfig),
            context_parallel=sub(
                ContextParallelConfig, "context_parallel", ContextParallelConfig
            ),
        )

    @classmethod
    def from_yaml_string(cls, yaml_text: str) -> HybridConfig:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("pip install pyyaml") from e
        raw = yaml.safe_load(yaml_text) or {}
        return cls._from_dict(raw)
