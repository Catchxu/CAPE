from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
    except ImportError:
        _flash_attn_qkvpacked_func = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.value(x) * F.silu(self.gate(x))
        x = self.dropout(x)
        return self.out(x)


class FlashAttentionBlock(nn.Module):
    """FlashAttention-preferred self-attention with RMSNorm and residual updates."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.flash_attn_fn = _flash_attn_qkvpacked_func if use_flash_attn else None
        self.last_attention_backend = "flash" if self.flash_attn_fn is not None else "sdpa"

        self.norm = RMSNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def _can_use_flash(self, x: Tensor) -> bool:
        if self.flash_attn_fn is None:
            return False
        if not x.is_cuda:
            return False
        return x.dtype in (torch.float16, torch.bfloat16)

    def _flash_attention(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x).view(x.size(0), x.size(1), 3, self.num_heads, self.head_dim)
        attn = self.flash_attn_fn(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )
        self.last_attention_backend = "flash"
        return attn.reshape(x.size(0), x.size(1), self.embed_dim)

    def _fallback_attention(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x).view(x.size(0), x.size(1), 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        self.last_attention_backend = "sdpa"
        return attn.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        if self._can_use_flash(x_norm):
            attn_out = self._flash_attention(x_norm)
        else:
            # Fallback path uses PyTorch scaled dot-product attention when flash_attn
            # is unavailable or the current device/dtype is incompatible.
            attn_out = self._fallback_attention(x_norm)
        return x + self.resid_dropout(self.out_proj(attn_out))


def zero_diagonal(matrix: Tensor) -> Tensor:
    """Set self-loop entries to zero for a batch of square matrices."""

    size = matrix.size(-1)
    eye = torch.eye(size, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    return matrix * (1.0 - eye)
