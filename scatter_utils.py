
"""
A tiny fallback for `torch_scatter.scatter` (dim=0 only).

This repo originally depends on `torch-scatter`. If you don't have it installed,
this module provides a compatible subset used by ZEN and the causal extension.

Note:
- Only supports dim=0.
- Supports reduce='sum' and reduce='mean'.
"""
from typing import Optional
import torch


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("scatter_utils.scatter only supports dim=0")

    if index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        out_shape = (dim_size,) + tuple(src.shape[1:])
        return torch.zeros(out_shape, device=src.device, dtype=src.dtype)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)

    if reduce == "sum":
        return out
    if reduce == "mean":
        # counts
        counts = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        ones = torch.ones(index.size(0), device=src.device, dtype=src.dtype)
        counts.index_add_(0, index, ones)
        counts = counts.clamp(min=1.0)
        # broadcast
        view = (dim_size,) + (1,) * (src.dim() - 1)
        return out / counts.view(view)

    raise ValueError(f"Unsupported reduce='{reduce}'")
