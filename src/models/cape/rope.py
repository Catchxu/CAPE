from __future__ import annotations

import torch
from torch import Tensor


def priority_to_rank_positions(priority: Tensor, max_positions: int) -> Tensor:
    """Convert priority scores into integer rank positions in ``[0, max_positions - 1]``."""

    if priority.dim() != 2:
        raise ValueError(f"Expected priority shape (B, L), got {tuple(priority.shape)}")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    order = torch.argsort(priority, dim=-1, descending=False)
    ranks = torch.empty_like(order)
    base = torch.arange(priority.size(-1), device=priority.device, dtype=order.dtype)
    ranks.scatter_(1, order, base.unsqueeze(0).expand_as(order))
    if priority.size(-1) == 1:
        return torch.zeros_like(ranks)

    scaled = ranks.float() * float(max_positions - 1) / float(priority.size(-1) - 1)
    return scaled.round().long()


def build_rope_cache(positions: Tensor, dim: int, base: float = 10000.0) -> Tensor:
    """Build sinusoidal RoPE cache for batch-specific positions."""

    if positions.dim() != 2:
        raise ValueError(f"Expected positions shape (B, L), got {tuple(positions.shape)}")
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    device = positions.device
    dtype = torch.float32
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / max(half_dim, 1)))
    angles = positions.to(dtype).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return torch.cat([sin, cos], dim=-1)


def apply_batched_rotary_pos_emb(q: Tensor, k: Tensor, rope_cache: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors shaped ``(B, H, L, D)``."""

    if rope_cache.dim() != 3:
        raise ValueError(f"Expected rope cache shape (B, L, D), got {tuple(rope_cache.shape)}")

    sin, cos = rope_cache.chunk(2, dim=-1)
    sin = torch.repeat_interleave(sin.unsqueeze(1), repeats=2, dim=-1)
    cos = torch.repeat_interleave(cos.unsqueeze(1), repeats=2, dim=-1)

    q_rot = (q * cos) + (_rotate_every_two(q) * sin)
    k_rot = (k * cos) + (_rotate_every_two(k) * sin)
    return q_rot, k_rot


def _rotate_every_two(x: Tensor) -> Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
