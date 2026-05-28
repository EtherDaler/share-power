from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SubspaceConfig:
    rank: int = 64
    learn_basis: bool = True
    error_feedback: bool = True
    dtype: str = "float16"
    sync_basis_every: int = 100


class SubspaceCompressor(nn.Module):
    """
    Проекция (B, T, D) или (B, C, H, W) в ранг r и обратно.
    error_buf — register_buffer, не участвует в autograd к loss.
    """

    def __init__(self, d_model: int, cfg: SubspaceConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.cfg = cfg
        r = max(1, cfg.rank)
        self.r = r
        U = torch.empty(d_model, r)
        nn.init.orthogonal_(U)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(U.T.clone())
        self.register_buffer("error_buf", torch.zeros(1))
        self._last_shape: tuple[int, ...] = ()

    def reset_error_buf(self, shape: tuple[int, ...], device: torch.device) -> None:
        """Вызывать в начале нового forward-батча."""
        t = torch.zeros(shape, device=device, dtype=torch.float32)
        self.error_buf = t
        self._last_shape = shape

    def _flatten(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.dim() == 3:
            b, t, d = x.shape
            return x.reshape(b * t, d), (b, t, d)
        if x.dim() == 4:
            b, c, h, w = x.shape
            return x.reshape(b, c * h * w), (b, c, h, w)
        raise ValueError(f"Ожидаются 3D или 4D тензоры, shape={tuple(x.shape)}")

    def _unflatten(self, y: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
        if len(orig_shape) == 3:
            b, t, d = orig_shape
            return y.reshape(b, t, d)
        b, c, h, w = orig_shape
        return y.reshape(b, c, h, w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xf, shp = self._flatten(x)
        if self.cfg.error_feedback and self.error_buf.numel() == xf.numel():
            xf = xf + self.error_buf.reshape_as(xf)
        z = xf @ self.U
        recon = z @ self.V
        if self.cfg.error_feedback:
            err = (xf - recon).detach()
            self.error_buf = err
        return z.reshape(*shp[:-1], self.r) if len(shp) == 3 else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            b, t, r = z.shape
            zf = z.reshape(b * t, r)
            out = zf @ self.V
            return out.reshape(b, t, self.d_model)
        raise NotImplementedError("decode для 4D пока не реализован")

    def init_basis_from_data(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            xf, _ = self._flatten(x.float())
            if xf.shape[0] < self.r:
                return
            # SVD на подвыборке строк
            idx = torch.randperm(xf.shape[0], device=xf.device)[: min(512, xf.shape[0])]
            xs = xf[idx]
            _, _, Vh = torch.linalg.svd(xs, full_matrices=False)
            r = min(self.r, Vh.shape[0])
            basis = Vh[:r, :].T.contiguous()
            self.U.data[:, :r] = basis[:, :r]
            self.V.data[:r, :] = self.U.data[:, :r].T

    def compression_ratio(self) -> float:
        return self.d_model / max(1, self.r)

    def reconstruction_error(self, x: torch.Tensor) -> float:
        """Относительная ошибка линейной реконструкции U,V без error_feedback (мониторинг)."""
        with torch.no_grad():
            xf, shp = self._flatten(x.float())
            recon = (xf @ self.U) @ self.V
            if len(shp) == 3:
                recon3 = recon.reshape(*shp)
            else:
                recon3 = self._unflatten(recon, shp)
            num = (x.float() - recon3).norm()
            den = x.float().norm().clamp(min=1e-8)
            return float((num / den).item())


def _main() -> None:
    d = 1024
    cfg = SubspaceConfig(rank=64, learn_basis=True, error_feedback=True)
    comp = SubspaceCompressor(d, cfg)
    x = torch.randn(2, 512, d)
    comp.reset_error_buf((2, 512, d), x.device)
    z = comp.encode(x)
    _ = comp.decode(z)
    print(f"Compression ratio: {comp.compression_ratio():.1f}x")
    print(f"Reconstruction error: {comp.reconstruction_error(x):.4f}")


if __name__ == "__main__":
    _main()
