from typing import Dict

import torch
from torch import Tensor, nn

from .modules import FlashAttentionBlock, RMSNorm, SwiGLU, zero_diagonal


class PredictorBlock(nn.Module):
    """One predictor block: FlashAttention + RMSNorm + SwiGLU."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.attn = FlashAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        self.norm = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x)
        return x + self.dropout(self.ffn(self.norm(x)))


class PairwiseEdgeScorer(nn.Module):
    """Predict edge logits with adj[b, i, j] interpreted as j -> i."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.src_proj = nn.Linear(embed_dim, embed_dim)
        self.tgt_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        src = self.src_proj(x)
        tgt = self.tgt_proj(x)
        return torch.einsum("bih,bjh->bij", tgt, src) * self.scale


class PriorityHead(nn.Module):
    """Predict node priority scores used to form the soft order mask."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x)).squeeze(-1)


class GraphPredictor(nn.Module):
    """Transformer-style predictor that outputs contextual states and graph tensors."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 2,
        tau: float = 1.0,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_layers)
            ]
        )
        self.edge_block = FlashAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        self.edge_scorer = PairwiseEdgeScorer(embed_dim=embed_dim)
        self.priority_head = PriorityHead(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    @property
    def last_attention_backend(self) -> str:
        if self.edge_block.last_attention_backend:
            return self.edge_block.last_attention_backend
        if self.blocks:
            return self.blocks[-1].attn.last_attention_backend
        return "sdpa"

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        attn_out = h
        for block in self.blocks:
            attn_out = block(attn_out)

        edge_features = self.edge_block(attn_out)
        edge_logits = self.edge_scorer(edge_features)
        edge_prob = torch.sigmoid(edge_logits)

        priority = self.priority_head(attn_out)
        priority_diff = priority.unsqueeze(-1) - priority.unsqueeze(-2)
        soft_mask = torch.sigmoid(priority_diff / max(self.tau, 1e-6))

        # The soft priority mask biases directional ordering but does not strictly
        # guarantee a DAG during training because the mask is continuous.
        soft_mask = zero_diagonal(soft_mask)
        adj = zero_diagonal(edge_prob * soft_mask)

        return {
            "attn_out": attn_out,
            "edge_logits": edge_logits,
            "edge_prob": edge_prob,
            "priority": priority,
            "priority_diff": priority_diff,
            "soft_mask": soft_mask,
            "adj": adj,
        }
