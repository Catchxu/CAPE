from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from .encoder import FoundationEncoder
from .loss import elbo_loss, graph_sparsity_loss
from .modules import zero_diagonal
from .predictor import GraphPredictor
from .sem import NonlinearLatentSem


class DagSemModel(nn.Module):
    """DAG-constrained latent SEM with foundation-model input embeddings."""

    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        tau: float = 1.0,
        dropout: float = 0.0,
        beta_kl: float = 1.0,
        lambda_sparse: float = 1e-4,
        use_flash_attn: bool = True,
        num_predictor_layers: int = 2,
        encoder_backend: str = "scbert",
        encoder_pretrained_model_name_or_path: str | None = None,
        freeze_encoder_backbone: bool = False,
        encoder_gene_token_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.beta_kl = beta_kl
        self.lambda_sparse = lambda_sparse
        self.tau = tau

        self.encoder = FoundationEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            dropout=dropout,
            backend=encoder_backend,
            pretrained_model_name_or_path=encoder_pretrained_model_name_or_path,
            freeze_backbone=freeze_encoder_backbone,
            gene_token_ids=encoder_gene_token_ids,
        )
        self.predictor = GraphPredictor(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_predictor_layers,
            tau=tau,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        self.sem = NonlinearLatentSem(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.predictor.set_tau(tau)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.encoder(x)
        graph_outputs = self.predictor(h)
        sem_outputs = self.sem(graph_outputs["attn_out"], graph_outputs["adj"])

        return {
            "input": x,
            "node_embeddings": h,
            "attn_out": graph_outputs["attn_out"],
            "edge_logits": graph_outputs["edge_logits"],
            "edge_prob": graph_outputs["edge_prob"],
            "priority": graph_outputs["priority"],
            "priority_diff": graph_outputs["priority_diff"],
            "soft_mask": graph_outputs["soft_mask"],
            "adj": graph_outputs["adj"],
            "mu": sem_outputs["mu"],
            "logvar": sem_outputs["logvar"],
            "z": sem_outputs["z"],
            "z_recon": sem_outputs["z_recon"],
            "x_recon": sem_outputs["x_recon"],
        }

    def compute_loss(self, x: Tensor, outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Total loss = ELBO + sparse regularization on masked graph strength.
        elbo, recon_loss, kl_loss = elbo_loss(
            x,
            outputs["x_recon"],
            outputs["mu"],
            outputs["logvar"],
            reduction="mean",
            beta=self.beta_kl,
        )
        sparse_loss = graph_sparsity_loss(outputs["adj"], reduction="mean")
        total = elbo + self.lambda_sparse * sparse_loss
        return {
            "loss": total,
            "elbo": elbo,
            "recon": recon_loss,
            "kl": kl_loss,
            "sparse": sparse_loss,
        }

    def binarize_adjacency(self, adj: Tensor, threshold: float = 0.5) -> Tensor:
        binary = (adj >= threshold).to(dtype=adj.dtype)
        return zero_diagonal(binary)

    def topk_prune(self, adj: Tensor, k: int) -> Tensor:
        if k <= 0:
            return torch.zeros_like(adj)

        k = min(k, adj.size(-1) - 1)
        adj = zero_diagonal(adj)
        topk_values, topk_indices = torch.topk(adj, k=k, dim=-1)
        pruned = torch.zeros_like(adj)
        pruned.scatter_(-1, topk_indices, topk_values)
        return zero_diagonal(pruned)
