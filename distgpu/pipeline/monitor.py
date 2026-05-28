from __future__ import annotations

from typing import Any, Dict, List

import torch

from distgpu.compression.subspace import SubspaceCompressor


class CompressionMonitor:
    def __init__(self) -> None:
        self._ratios: List[float] = []
        self._errors: List[float] = []
        self._buf_norms: List[float] = []

    def log_step(self, step: int, compressor: SubspaceCompressor) -> None:
        if step % 50 != 0:
            return
        ratio = compressor.compression_ratio()
        err = 0.0
        try:
            x = torch.randn(2, 32, compressor.d_model, device=compressor.U.device)
            compressor.reset_error_buf(tuple(x.shape), x.device)
            err = compressor.reconstruction_error(x)
        except Exception:
            pass
        buf_n = float(compressor.error_buf.norm().item()) if compressor.error_buf.numel() else 0.0
        self._ratios.append(ratio)
        self._errors.append(err)
        self._buf_norms.append(buf_n)
        print(
            f"[distgpu/compression] step={step} ratio={ratio:.1f}x "
            f"recon_err={err:.4f} err_buf_norm={buf_n:.4f}"
        )

    def summary(self) -> Dict[str, Any]:
        if not self._ratios:
            return {}
        import statistics as st

        return {
            "mean_ratio": st.mean(self._ratios),
            "mean_recon_err": st.mean(self._errors),
            "mean_err_buf_norm": st.mean(self._buf_norms),
        }
